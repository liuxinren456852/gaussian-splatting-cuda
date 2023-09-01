#include "debug_utils.cuh"
#include "gaussian.cuh"
#include "read_utils.cuh"
#include <exception>
#include <thread>

GaussianModel::GaussianModel(int sh_degree) : _max_sh_degree(sh_degree) {
}

/**
 * @brief Fetches the features of the Gaussian model
 *
 * This function concatenates _features_dc and _features_rest along the second dimension.
 *
 * @return Tensor of the concatenated features
 */
torch::Tensor GaussianModel::Get_features() const {
    auto features_dc = _features_dc;
    auto features_rest = _features_rest;
    return torch::cat({features_dc, features_rest}, 1);
}

/**
 * @brief Increment the SH degree by 1
 *
 * This function increments the active_sh_degree by 1, up to a maximum of max_sh_degree.
 */
void GaussianModel::One_up_sh_degree() {
    if (_active_sh_degree < _max_sh_degree) {
        _active_sh_degree++;
    }
}

/**
 * @brief Initialize Gaussian Model from a Point Cloud.
 *
 * This function creates a Gaussian model from a given PointCloud object. It also sets
 * the spatial learning rate scale. The model's features, scales, rotations, and opacities
 * are initialized based on the input point cloud.
 *
 * @param pcd The input point cloud
 * @param spatial_lr_scale The spatial learning rate scale
 */
void GaussianModel::Create_from_pcd(PointCloud& pcd, float spatial_lr_scale) {
    _spatial_lr_scale = spatial_lr_scale;

    const auto pointType = torch::TensorOptions().dtype(torch::kFloat32);
    _xyz = torch::from_blob(pcd._points.data(), {static_cast<long>(pcd._points.size()), 3}, pointType).to(torch::kCUDA);
    auto dist2 = torch::clamp_min(distCUDA2(_xyz), 0.0000001);
    _scaling = torch::log(torch::sqrt(dist2)).unsqueeze(-1).repeat({1, 3}).to(torch::kCUDA, true);
    _rotation = torch::zeros({_xyz.size(0), 4}).index_put_({torch::indexing::Slice(), 0}, 1).to(torch::kCUDA, true);
    _opacity = inverse_sigmoid(0.5 * torch::ones({_xyz.size(0), 1})).to(torch::kCUDA, true);
    _max_radii2D = torch::zeros({_xyz.size(0)}).to(torch::kCUDA, true);

    // colors
    auto colorType = torch::TensorOptions().dtype(torch::kUInt8);
    auto fused_color = RGB2SH(torch::from_blob(pcd._colors.data(), {static_cast<long>(pcd._colors.size()), 3}, colorType).to(pointType) / 255.f).to(torch::kCUDA);

    // features
    auto features = torch::zeros({fused_color.size(0), 3, static_cast<long>(std::pow((_max_sh_degree + 1), 2))}).to(torch::kCUDA);
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), 0}, fused_color);
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(3, torch::indexing::None), torch::indexing::Slice(1, torch::indexing::None)}, 0.0);
    _features_dc = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous();
    _features_rest = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}).transpose(1, 2).contiguous();
}

/**
 * @brief Setup the Gaussian Model for training
 *
 * This function sets up the Gaussian model for training by initializing several
 * parameters and settings based on the provided OptimizationParameters object.
 *
 * @param params The OptimizationParameters object providing the settings for training
 */
void GaussianModel::Training_setup(const OptimizationParameters& params) {
    this->_percent_dense = params.percent_dense;
    this->_xyz_gradient_accum = torch::zeros({this->_xyz.size(0), 1}).to(torch::kCUDA);
    this->_denom = torch::zeros({this->_xyz.size(0), 1}).to(torch::kCUDA);
    this->_xyz_scheduler_args = Expon_lr_func(params.position_lr_init * this->_spatial_lr_scale,
                                              params.position_lr_final * this->_spatial_lr_scale,
                                              params.position_lr_delay_mult,
                                              params.position_lr_max_steps);

    _optimizer = std::make_unique<gs::optim::Adam>();
    _optimizer->AddParameter(std::make_shared<gs::optim::AdamParameter>(gs::optim::ParamType::Pos,
                                                                        _xyz,
                                                                        params.position_lr_init * this->_spatial_lr_scale,
                                                                        nullptr));
    _optimizer->AddParameter(std::make_shared<gs::optim::AdamParameter>(gs::optim::ParamType::Features_dc,
                                                                        _features_dc,
                                                                        params.feature_lr,
                                                                        nullptr));
    _optimizer->AddParameter(std::make_shared<gs::optim::AdamParameter>(gs::optim::ParamType::Features_rest,
                                                                        _features_rest,
                                                                        params.feature_lr / 20.f,
                                                                        nullptr));
    _optimizer->AddParameter(std::make_shared<gs::optim::AdamParameter>(gs::optim::ParamType::Scaling,
                                                                        _scaling,
                                                                        params.scaling_lr * this->_spatial_lr_scale,
                                                                        nullptr));
    _optimizer->AddParameter(std::make_shared<gs::optim::AdamParameter>(gs::optim::ParamType::Rotation,
                                                                        _rotation,
                                                                        params.rotation_lr,
                                                                        nullptr));
    _optimizer->AddParameter(std::make_shared<gs::optim::AdamParameter>(gs::optim::ParamType::Opacity,
                                                                        _opacity,
                                                                        params.opacity_lr,
                                                                        nullptr));
}

void GaussianModel::Update_learning_rate(float iteration) {
    // This is hacky because you cant change in libtorch individual parameter learning rate
    // xyz is added first, since _optimizer->param_groups() return a vector, we assume that xyz stays first
    auto lr = _xyz_scheduler_args(iteration);
    _optimizer->GetAdamParameter(gs::optim::ParamType::Pos)->UpdateLearningRate(lr);
}

void GaussianModel::Reset_opacity() {
    // opacitiy activation
    _opacity = inverse_sigmoid(torch::ones_like(_opacity, torch::TensorOptions().dtype(torch::kFloat32)) * 0.01f);
    _optimizer->GetAdamParameter(gs::optim::ParamType::Opacity)->Set_Exp_Avg(torch::zeros_like(_opacity));
    _optimizer->GetAdamParameter(gs::optim::ParamType::Opacity)->Set_Exp_Avg_Sq(torch::zeros_like(_opacity));
    _optimizer->GetAdamParameter(gs::optim::ParamType::Opacity)->Set_Step(torch::zeros({_opacity.size(0), 1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)));
    _optimizer->GetAdamParameter(gs::optim::ParamType::Opacity)->Set_Param(_opacity);
}

void prune_optimizer(gs::optim::Adam* optimizer, const torch::Tensor& mask, torch::Tensor& old_tensor, gs::optim::ParamType param_type) {

    auto adam_param = optimizer->GetAdamParameter(param_type);
    old_tensor = old_tensor.index_select(0, mask);
    adam_param->Set_Exp_Avg(adam_param->Get_Exp_Avg().index_select(0, mask));
    adam_param->Set_Exp_Avg_Sq(adam_param->Get_Exp_Avg_Sq().index_select(0, mask));

    //    std::cout << "prune_optimizer: " << gs::optim::Map_param_type_to_string(param_type);
    adam_param->Set_Step(adam_param->Get_Step().index_select(0, mask));
    adam_param->Set_Param(old_tensor);
}

void GaussianModel::prune_points(torch::Tensor mask) {
    // reverse to keep points
    auto valid_point_mask = ~mask;
    int true_count = valid_point_mask.sum().item<int>();
    auto indices = torch::nonzero(valid_point_mask == true).index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None), torch::indexing::Slice(torch::indexing::None, 1)}).squeeze(-1);
    prune_optimizer(_optimizer.get(), indices, _xyz, gs::optim::ParamType::Pos);
    prune_optimizer(_optimizer.get(), indices, _features_dc, gs::optim::ParamType::Features_dc);
    prune_optimizer(_optimizer.get(), indices, _features_rest, gs::optim::ParamType::Features_rest);
    prune_optimizer(_optimizer.get(), indices, _scaling, gs::optim::ParamType::Scaling);
    prune_optimizer(_optimizer.get(), indices, _rotation, gs::optim::ParamType::Rotation);
    prune_optimizer(_optimizer.get(), indices, _opacity, gs::optim::ParamType::Opacity);

    _xyz_gradient_accum = _xyz_gradient_accum.index_select(0, indices);
    _denom = _denom.index_select(0, indices);
    _max_radii2D = _max_radii2D.index_select(0, indices);
}

void cat_tensors_to_optimizer(gs::optim::Adam* optimizer,
                              torch::Tensor& extension_tensor,
                              torch::Tensor& old_tensor,
                              gs::optim::ParamType param_type) {

    auto adam_param = optimizer->GetAdamParameter(param_type);
    old_tensor = torch::cat({old_tensor, extension_tensor}, 0);
    adam_param->Set_Param(old_tensor);
    adam_param->Set_Exp_Avg(torch::cat({adam_param->Get_Exp_Avg(), torch::zeros_like(extension_tensor)}, 0));
    //    std::cout << "cat_tensors_to_optimizer: " << gs::optim::Map_param_type_to_string(param_type);
    const auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    adam_param->Set_Step(torch::cat({adam_param->Get_Step(), torch::zeros({extension_tensor.size(0), 1}, options)}, 0));
    adam_param->Set_Exp_Avg_Sq(torch::cat({adam_param->Get_Exp_Avg_Sq(), torch::zeros_like(extension_tensor)}, 0));
}

void GaussianModel::densification_postfix(torch::Tensor& new_xyz,
                                          torch::Tensor& new_features_dc,
                                          torch::Tensor& new_features_rest,
                                          torch::Tensor& new_scaling,
                                          torch::Tensor& new_rotation,
                                          torch::Tensor& new_opacity) {
    cat_tensors_to_optimizer(_optimizer.get(), new_xyz, _xyz, gs::optim::ParamType::Pos);
    cat_tensors_to_optimizer(_optimizer.get(), new_features_dc, _features_dc, gs::optim::ParamType::Features_dc);
    cat_tensors_to_optimizer(_optimizer.get(), new_features_rest, _features_rest, gs::optim::ParamType::Features_rest);
    cat_tensors_to_optimizer(_optimizer.get(), new_scaling, _scaling, gs::optim::ParamType::Scaling);
    cat_tensors_to_optimizer(_optimizer.get(), new_rotation, _rotation, gs::optim::ParamType::Rotation);
    cat_tensors_to_optimizer(_optimizer.get(), new_opacity, _opacity, gs::optim::ParamType::Opacity);

    _xyz_gradient_accum = torch::zeros({_xyz.size(0), 1}).to(torch::kCUDA);
    _denom = torch::zeros({_xyz.size(0), 1}).to(torch::kCUDA);
    _max_radii2D = torch::zeros({_xyz.size(0)}).to(torch::kCUDA);
}

void GaussianModel::densify_and_split(torch::Tensor& grads, float grad_threshold, float scene_extent, float min_opacity, float max_screen_size) {
    static const int N = 2;
    const int n_init_points = _xyz.size(0);
    // Extract points that satisfy the gradient condition
    torch::Tensor padded_grad = torch::zeros({n_init_points}).to(torch::kCUDA);
    padded_grad.slice(0, 0, grads.size(0)) = grads.squeeze();
    torch::Tensor selected_pts_mask = torch::where(padded_grad >= grad_threshold, torch::ones_like(padded_grad).to(torch::kBool), torch::zeros_like(padded_grad).to(torch::kBool));
    selected_pts_mask = torch::logical_and(selected_pts_mask, std::get<0>(Get_scaling().max(1)) > _percent_dense * scene_extent);
    auto indices = torch::nonzero(selected_pts_mask.squeeze(-1) == true).index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None), torch::indexing::Slice(torch::indexing::None, 1)}).squeeze(-1);

    torch::Tensor stds = Get_scaling().index_select(0, indices).repeat({N, 1});
    torch::Tensor means = torch::zeros({stds.size(0), 3}).to(torch::kCUDA);
    torch::Tensor samples = torch::randn({stds.size(0), stds.size(1)}).to(torch::kCUDA) * stds + means;
    torch::Tensor rots = build_rotation(_rotation.index_select(0, indices)).repeat({N, 1, 1});

    torch::Tensor new_xyz = torch::bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + _xyz.index_select(0, indices).repeat({N, 1});
    torch::Tensor new_scaling = torch::log(Get_scaling().index_select(0, indices).repeat({N, 1}) / (0.8 * N));
    torch::Tensor new_rotation = _rotation.index_select(0, indices).repeat({N, 1});
    torch::Tensor new_features_dc = _features_dc.index_select(0, indices).repeat({N, 1, 1});
    torch::Tensor new_features_rest = _features_rest.index_select(0, indices).repeat({N, 1, 1});
    torch::Tensor new_opacity = _opacity.index_select(0, indices).repeat({N, 1});

    densification_postfix(new_xyz, new_features_dc, new_features_rest, new_scaling, new_rotation, new_opacity);

    torch::Tensor prune_filter = torch::cat({selected_pts_mask.squeeze(-1), torch::zeros({N * selected_pts_mask.sum().item<int>()}).to(torch::kBool).to(torch::kCUDA)});
    // torch::Tensor prune_filter = torch::cat({selected_pts_mask.squeeze(-1), torch::zeros({N * selected_pts_mask.sum().item<int>()})}).to(torch::kBool).to(torch::kCUDA);
    prune_filter = torch::logical_or(prune_filter, (Get_opacity() < min_opacity).squeeze(-1));
    prune_points(prune_filter);
}

void GaussianModel::densify_and_clone(torch::Tensor& grads, float grad_threshold, float scene_extent) {
    // Extract points that satisfy the gradient condition
    torch::Tensor selected_pts_mask = torch::where(torch::linalg::vector_norm(grads, {2}, 1, true, torch::kFloat32) >= grad_threshold,
                                                   torch::ones_like(grads.index({torch::indexing::Slice()})).to(torch::kBool),
                                                   torch::zeros_like(grads.index({torch::indexing::Slice()})).to(torch::kBool))
                                          .to(torch::kLong);

    selected_pts_mask = torch::logical_and(selected_pts_mask, std::get<0>(Get_scaling().max(1)).unsqueeze(-1) <= _percent_dense * scene_extent);

    auto indices = torch::nonzero(selected_pts_mask.squeeze(-1) == true).index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None), torch::indexing::Slice(torch::indexing::None, 1)}).squeeze(-1);
    torch::Tensor new_xyz = _xyz.index_select(0, indices);
    torch::Tensor new_features_dc = _features_dc.index_select(0, indices);
    torch::Tensor new_features_rest = _features_rest.index_select(0, indices);
    torch::Tensor new_opacity = _opacity.index_select(0, indices);
    torch::Tensor new_scaling = _scaling.index_select(0, indices);
    torch::Tensor new_rotation = _rotation.index_select(0, indices);

    densification_postfix(new_xyz, new_features_dc, new_features_rest, new_scaling, new_rotation, new_opacity);
}

void GaussianModel::Densify_and_prune(float max_grad, float min_opacity, float extent, float max_screen_size) {
    torch::Tensor grads = _xyz_gradient_accum / _denom;
    grads.index_put_({grads.isnan()}, 0.0);

    densify_and_clone(grads, max_grad, extent);
    densify_and_split(grads, max_grad, extent, min_opacity, max_screen_size);
}

void GaussianModel::Add_densification_stats(torch::Tensor& grad_means2D, torch::Tensor& update_filter) {
    auto filtered_grad = grad_means2D.index_select(0, update_filter.nonzero().squeeze()).slice(1, 0, 2).norm(2, -1, true);
    _xyz_gradient_accum.index_put_({update_filter}, _xyz_gradient_accum.index_select(0, update_filter.nonzero().squeeze()) + filtered_grad);
    _denom.index_put_({update_filter}, _denom.index_select(0, update_filter.nonzero().squeeze()) + 1);
}

std::vector<std::string> GaussianModel::construct_list_of_attributes() {
    std::vector<std::string> attributes = {"x", "y", "z", "nx", "ny", "nz"};

    for (int i = 0; i < _features_dc.size(1) * _features_dc.size(2); ++i)
        attributes.push_back("f_dc_" + std::to_string(i));

    for (int i = 0; i < _features_rest.size(1) * _features_rest.size(2); ++i)
        attributes.push_back("f_rest_" + std::to_string(i));

    attributes.emplace_back("opacity");

    for (int i = 0; i < _scaling.size(1); ++i)
        attributes.push_back("scale_" + std::to_string(i));

    for (int i = 0; i < _rotation.size(1); ++i)
        attributes.push_back("rot_" + std::to_string(i));

    return attributes;
}

void GaussianModel::Save_ply(const std::filesystem::path& file_path, int iteration, bool isLastIteration) {
    //    std::cout << "Saving at " << std::to_string(iteration) << " iterations\n";
    auto folder = file_path / ("point_cloud/iteration_" + std::to_string(iteration));
    std::filesystem::create_directories(folder);

    auto xyz = _xyz.cpu().contiguous();
    auto normals = torch::zeros_like(xyz);
    auto f_dc = _features_dc.transpose(1, 2).flatten(1).cpu().contiguous();
    auto f_rest = _features_rest.transpose(1, 2).flatten(1).cpu().contiguous();
    auto opacities = _opacity.cpu();
    auto scale = _scaling.cpu();
    auto rotation = _rotation.cpu();

    std::vector<torch::Tensor> tensor_attributes = {xyz.clone(),
                                                    normals.clone(),
                                                    f_dc.clone(),
                                                    f_rest.clone(),
                                                    opacities.clone(),
                                                    scale.clone(),
                                                    rotation.clone()};
    auto attributes = construct_list_of_attributes();
    std::thread t = std::thread([folder, tensor_attributes, attributes]() {
        Write_output_ply(folder / "point_cloud.ply", tensor_attributes, attributes);
    });

    if (isLastIteration) {
        t.join();
    } else {
        t.detach();
    }
}
void GaussianModel::Update_Grads(const torch::Tensor& grad_means3D,
                                 const torch::Tensor& grad_sh,
                                 const torch::Tensor& grad_opacities,
                                 const torch::Tensor& grad_scales,
                                 const torch::Tensor& grad_rotations) {
    auto grad_features_dc = grad_sh.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1), torch::indexing::Slice()}).contiguous();
    auto grad_features_rest = grad_sh.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()}).contiguous();

    if (grad_means3D.sizes() != _xyz.sizes() || grad_means3D.size(1) != 3) {
        throw std::runtime_error("grad_means3D and xyz have different sizes");
    }
    _optimizer->GetAdamParameter(gs::optim::ParamType::Pos)->Set_Gradient(grad_means3D);

    _optimizer->GetAdamParameter(gs::optim::ParamType::Features_dc)->Set_Gradient(grad_features_dc);
    _optimizer->GetAdamParameter(gs::optim::ParamType::Features_rest)->Set_Gradient(grad_features_rest);
    _optimizer->GetAdamParameter(gs::optim::ParamType::Scaling)->Set_Gradient(grad_scales);
    _optimizer->GetAdamParameter(gs::optim::ParamType::Rotation)->Set_Gradient(grad_rotations);
    _optimizer->GetAdamParameter(gs::optim::ParamType::Opacity)->Set_Gradient(grad_opacities);
}

void GaussianModel::Set_Params() {
    _optimizer->GetAdamParameter(gs::optim::ParamType::Pos)->Set_Param(_xyz);
    _optimizer->GetAdamParameter(gs::optim::ParamType::Features_dc)->Set_Param(_features_dc);
    _optimizer->GetAdamParameter(gs::optim::ParamType::Features_rest)->Set_Param(_features_rest);
    _optimizer->GetAdamParameter(gs::optim::ParamType::Scaling)->Set_Param(_scaling);
    _optimizer->GetAdamParameter(gs::optim::ParamType::Rotation)->Set_Param(_rotation);
    _optimizer->GetAdamParameter(gs::optim::ParamType::Opacity)->Set_Param(_opacity);
}
