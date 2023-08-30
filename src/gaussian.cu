#include "debug_utils.cuh"
#include "gaussian.cuh"
#include "read_utils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <exception>
#include <memory>
#include <thread>

GaussianModel::GaussianModel(int sh_degree) : _max_sh_degree(sh_degree) {
    cudaStreamCreate(&_stream1);
    cudaStreamCreate(&_stream2);
    cudaStreamCreate(&_stream3);
    cudaStreamCreate(&_stream4);
    cudaStreamCreate(&_stream5);
    cudaStreamCreate(&_stream6);
}

GaussianModel::~GaussianModel() {
    cudaStreamDestroy(_stream1);
    cudaStreamDestroy(_stream2);
    cudaStreamDestroy(_stream3);
    cudaStreamDestroy(_stream4);
    cudaStreamDestroy(_stream5);
    cudaStreamDestroy(_stream6);
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

    const std::vector<int> xyz_shape = {static_cast<int>(_xyz.size(0)), static_cast<int>(_xyz.size(1))};
    const std::vector<int> features_dc_shape = {static_cast<int>(_features_dc.size(0)), static_cast<int>(_features_dc.size(1)), static_cast<int>(_features_dc.size(2))};
    const std::vector<int> features_rest_shape = {static_cast<int>(_features_rest.size(0)), static_cast<int>(_features_rest.size(1)), static_cast<int>(_features_rest.size(2))};
    const std::vector<int> scaling_shape = {static_cast<int>(_scaling.size(0)), static_cast<int>(_scaling.size(1))};
    const std::vector<int> rotation_shape = {static_cast<int>(_rotation.size(0)), static_cast<int>(_rotation.size(1))};
    const std::vector<int> opacity_shape = {static_cast<int>(_opacity.size(0)), static_cast<int>(_opacity.size(1))};

    _optimizer = std::make_unique<gs::optim::Adam>();
    _optimizer->AddParameter(std::make_shared<gs::optim::AdamParameter<gs::optim::pos_param_t>>(gs::optim::ParamType::Pos,
                                                                                                xyz_shape,
                                                                                                params.position_lr_init * this->_spatial_lr_scale));
    _optimizer->AddParameter(std::make_shared<gs::optim::AdamParameter<gs::optim::feature_dc_param_t>>(gs::optim::ParamType::Features_dc,
                                                                                                       features_dc_shape,
                                                                                                       params.feature_lr));
    _optimizer->AddParameter(std::make_shared<gs::optim::AdamParameter<gs::optim::feature_rest_param_t>>(gs::optim::ParamType::Features_rest,
                                                                                                         features_rest_shape,
                                                                                                         params.feature_lr / 20.f));
    _optimizer->AddParameter(std::make_shared<gs::optim::AdamParameter<gs::optim::scaling_param_t>>(gs::optim::ParamType::Scaling,
                                                                                                    scaling_shape,
                                                                                                    params.scaling_lr * this->_spatial_lr_scale));
    _optimizer->AddParameter(std::make_shared<gs::optim::AdamParameter<gs::optim::rotation_param_t>>(gs::optim::ParamType::Rotation,
                                                                                                     rotation_shape,
                                                                                                     params.rotation_lr));
    _optimizer->AddParameter(std::make_shared<gs::optim::AdamParameter<gs::optim::opacity_param_t>>(gs::optim::ParamType::Opacity,
                                                                                                    opacity_shape,
                                                                                                    params.opacity_lr));

    _optimizer->Sync();
}

void GaussianModel::Update_learning_rate(float iteration) {
    // This is hacky because you cant change in libtorch individual parameter learning rate
    // xyz is added first, since _optimizer->param_groups() return a vector, we assume that xyz stays first
    auto lr = _xyz_scheduler_args(iteration);
    _optimizer->GetParameters(gs::optim::ParamType::Pos)->UpdateLearningRate(lr);
}

void GaussianModel::Update_Params_and_Grads(const torch::Tensor& grad_means3D,
                                            const torch::Tensor& grad_sh, // needs to be splitted
                                            const torch::Tensor& grad_opacities,
                                            const torch::Tensor& grad_scales,
                                            const torch::Tensor& grad_rotations) {
    auto xyz = reinterpret_cast<gs::optim::pos_param_t*>(_xyz.data_ptr<float>());
    auto features_dc = reinterpret_cast<gs::optim::feature_dc_param_t*>(_features_dc.data_ptr<float>());
    auto features_rest = reinterpret_cast<gs::optim::feature_rest_param_t*>(_features_rest.data_ptr<float>());
    auto scaling = reinterpret_cast<gs::optim::scaling_param_t*>(_scaling.data_ptr<float>());
    auto rotation = reinterpret_cast<gs::optim::rotation_param_t*>(_rotation.data_ptr<float>());
    auto opacity = reinterpret_cast<gs::optim::opacity_param_t*>(_opacity.data_ptr<float>());

    _optimizer->GetAdamParameter<gs::optim::pos_param_t>(gs::optim::ParamType::Pos)->Update_Parameter_Pointer(xyz);
    _optimizer->GetAdamParameter<gs::optim::feature_dc_param_t>(gs::optim::ParamType::Features_dc)->Update_Parameter_Pointer(features_dc);
    _optimizer->GetAdamParameter<gs::optim::feature_rest_param_t>(gs::optim::ParamType::Features_rest)->Update_Parameter_Pointer(features_rest);
    _optimizer->GetAdamParameter<gs::optim::scaling_param_t>(gs::optim::ParamType::Scaling)->Update_Parameter_Pointer(scaling);
    _optimizer->GetAdamParameter<gs::optim::rotation_param_t>(gs::optim::ParamType::Rotation)->Update_Parameter_Pointer(rotation);
    _optimizer->GetAdamParameter<gs::optim::opacity_param_t>(gs::optim::ParamType::Opacity)->Update_Parameter_Pointer(opacity);

    auto grad_features_dc = grad_sh.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1), torch::indexing::Slice()}).contiguous();
    auto grad_features_rest = grad_sh.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()}).contiguous();

    _optimizer->GetAdamParameter<gs::optim::pos_param_t>(gs::optim::ParamType::Pos)->Set_Gradient(reinterpret_cast<gs::optim::pos_param_t*>(grad_means3D.data_ptr<float>()), {static_cast<int>(grad_means3D.size(0), 3)});
    _optimizer->GetAdamParameter<gs::optim::feature_dc_param_t>(gs::optim::ParamType::Features_dc)->Set_Gradient(reinterpret_cast<gs::optim::feature_dc_param_t*>(grad_features_dc.data_ptr<float>()), {static_cast<int>(grad_features_dc.size(0)), static_cast<int>(grad_features_dc.size(1)), static_cast<int>(grad_features_dc.size(2))});
    _optimizer->GetAdamParameter<gs::optim::feature_rest_param_t>(gs::optim::ParamType::Features_rest)->Set_Gradient(reinterpret_cast<gs::optim::feature_rest_param_t*>(grad_features_rest.data_ptr<float>()), {static_cast<int>(grad_features_rest.size(0)), static_cast<int>(grad_features_rest.size(1)), static_cast<int>(grad_features_rest.size(2))});
    _optimizer->GetAdamParameter<gs::optim::scaling_param_t>(gs::optim::ParamType::Scaling)->Set_Gradient(reinterpret_cast<gs::optim::scaling_param_t*>(grad_scales.data_ptr<float>()), {static_cast<int>(grad_scales.size(0)), static_cast<int>(grad_scales.size(1))});
    _optimizer->GetAdamParameter<gs::optim::rotation_param_t>(gs::optim::ParamType::Rotation)->Set_Gradient(reinterpret_cast<gs::optim::rotation_param_t*>(grad_rotations.data_ptr<float>()), {static_cast<int>(grad_rotations.size(0)), static_cast<int>(grad_rotations.size(1))});
    _optimizer->GetAdamParameter<gs::optim::opacity_param_t>(gs::optim::ParamType::Opacity)->Set_Gradient(reinterpret_cast<gs::optim::opacity_param_t*>(grad_opacities.data_ptr<float>()), {static_cast<int>(grad_opacities.size(0)), static_cast<int>(grad_opacities.size(1))});
}

void GaussianModel::Reset_opacity() {
    // opacitiy activation
    _opacity = inverse_sigmoid(torch::ones_like(_opacity, torch::TensorOptions().dtype(torch::kFloat32)) * 0.01f);
    auto updateTensor = torch::zeros_like(_opacity);
    // new optimizer
    auto opacity_params = _optimizer->GetAdamParameter<gs::optim::opacity_param_t>(gs::optim::ParamType::Opacity);
    opacity_params->Set_Exp_Avg(reinterpret_cast<gs::optim::opacity_param_t*>(updateTensor.data_ptr<float>()), {static_cast<int>(_opacity.size(0))});
    opacity_params->Set_Exp_Avg_Sq(reinterpret_cast<gs::optim::opacity_param_t*>(updateTensor.data_ptr<float>()), {static_cast<int>(_opacity.size(0))});
}

void copy3DAsync(const float* src,
                 const std::vector<long>& src_size,
                 float* dst,
                 const std::vector<long>& dst_size,
                 cudaStream_t stream) {
    cudaMemcpy3DParms copyParams = {0};
    copyParams.kind = cudaMemcpyDeviceToDevice;

    copyParams.srcPtr = make_cudaPitchedPtr(
        (void*)src,
        (size_t)src_size[2] * sizeof(float),
        (size_t)src_size[2],
        (size_t)src_size[1]);

    copyParams.dstPtr = make_cudaPitchedPtr(
        (void*)dst,
        (size_t)dst_size[2] * sizeof(float),
        (size_t)dst_size[2],
        (size_t)dst_size[1]);

    copyParams.extent = make_cudaExtent(
        (size_t)src_size[2] * sizeof(float),
        (size_t)src_size[1],
        (size_t)src_size[0]);
    CHECK_CUDA_ERROR(cudaMemcpy3DAsync(&copyParams, stream));
}

void copy2DAsync(const float* src, const std::vector<long>& src_size, float* dst, cudaStream_t stream) {
    //    float *new_dst;
    //    cudaMalloc(&new_dst, dst.size(0) * dst.size(1) * sizeof(float));
    if (src_size.size() != 2) {
        return;
    }
    size_t width_in_bytes = src_size[1] * sizeof(float);
    size_t height_in_elements = src_size[0];
    size_t src_pitch = src_size[1] * sizeof(float); // provide stride
    size_t dst_pitch = src_size[1] * sizeof(float); // make stride

    cudaMemcpy2DAsync(
        dst,                      // Destination pointer
        dst_pitch,                // Destination pitch
        src,                      // Source pointer
        src_pitch,                // Source pitch
        width_in_bytes,           // Width of the 2D region in bytes
        height_in_elements,       // Height of the 2D region in elements
        cudaMemcpyDeviceToDevice, // Specifies the kind of copy (Device to Device)
        stream                    // Stream to perform the copy
    );
    CHECK_LAST_CUDA_ERROR();
}

template <typename T>
__global__ void index_select_2D_kernel(const T* __restrict__ A,
                                       T* __restrict__ B,
                                       const long* __restrict__ indices,
                                       int size_B) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= size_B) {
        return;
    }

    B[row] = A[indices[row]];
}

template <typename T>
__global__ void index_select_3D_kernel(const T* __restrict__ A,
                                       T* __restrict__ B,
                                       const long* __restrict__ indices,
                                       int size_B,
                                       int dim1) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= size_B) {
        return;
    }

    for (int i = 0; i < dim1; ++i) {
        B[row * dim1 + i] = A[indices[row] * dim1 + i];
    }
}

template <typename param_t>
void prune_optimizer(const torch::Tensor& mask,
                     torch::Tensor& old_tensor,
                     gs::optim::Adam* new_optimizer,
                     gs::optim::ParamType param_type,
                     const std::vector<int>& shape) {

    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const int threads = 1024;
    const int blocks = (mask.size(0) + threads - 1) / threads;

    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStream_t stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    switch (param_type) {
    case gs::optim::ParamType::Pos:
    case gs::optim::ParamType::Scaling:
    case gs::optim::ParamType::Rotation:
    case gs::optim::ParamType::Opacity: {
        auto masked_param = torch::zeros({mask.size(0), shape[1]}, options);
        auto exp_avg = torch::zeros({mask.size(0), shape[1]}, options);
        auto exp_avg_sq = torch::zeros({mask.size(0), shape[1]}, options);
        auto param = new_optimizer->GetAdamParameter<param_t>(param_type);
        index_select_2D_kernel<param_t><<<blocks, threads, 0, stream1>>>(reinterpret_cast<param_t*>(old_tensor.data_ptr<float>()),
                                                                         reinterpret_cast<param_t*>(masked_param.data_ptr<float>()),
                                                                         mask.data_ptr<long>(),
                                                                         static_cast<int>(mask.size(0)));

        index_select_2D_kernel<param_t><<<blocks, threads, 0, stream2>>>(param->Get_Exp_Avg(),
                                                                         reinterpret_cast<param_t*>(exp_avg.data_ptr<float>()),
                                                                         mask.data_ptr<long>(),
                                                                         static_cast<int>(mask.size(0)));
        index_select_2D_kernel<param_t><<<blocks, threads, 0, stream3>>>(param->Get_Exp_Avg_Sq(),
                                                                         reinterpret_cast<param_t*>(exp_avg_sq.data_ptr<float>()),
                                                                         mask.data_ptr<long>(),
                                                                         static_cast<int>(mask.size(0)));
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
        cudaStreamSynchronize(stream3);
        param->Set_Exp_Avg(reinterpret_cast<param_t*>(exp_avg_sq.data_ptr<float>()), shape);
        param->Set_Exp_Avg_Sq(reinterpret_cast<param_t*>(exp_avg_sq.data_ptr<float>()), shape);
        old_tensor = masked_param;
    } break;
    case gs::optim::ParamType::Features_dc:
    case gs::optim::ParamType::Features_rest: {
        auto masked_param = torch::zeros({mask.size(0), shape[1], shape[2]}, options);
        auto exp_avg = torch::zeros({mask.size(0), shape[1], shape[2]}, options);
        auto exp_avg_sq = torch::zeros({mask.size(0), shape[1], shape[2]}, options);
        auto param = new_optimizer->GetAdamParameter<param_t>(param_type);

        index_select_3D_kernel<param_t><<<blocks, threads, 0, stream1>>>(reinterpret_cast<param_t*>(old_tensor.data_ptr<float>()),
                                                                         reinterpret_cast<param_t*>(masked_param.data_ptr<float>()),
                                                                         mask.data_ptr<long>(),
                                                                         static_cast<int>(mask.size(0)),
                                                                         static_cast<int>(old_tensor.size(1)));

        index_select_3D_kernel<param_t><<<blocks, threads, 0, stream2>>>(param->Get_Exp_Avg(),
                                                                         reinterpret_cast<param_t*>(exp_avg.data_ptr<float>()),
                                                                         mask.data_ptr<long>(),
                                                                         static_cast<int>(mask.size(0)),
                                                                         static_cast<int>(old_tensor.size(1)));
        index_select_3D_kernel<param_t><<<blocks, threads, 0, stream3>>>(param->Get_Exp_Avg_Sq(),
                                                                         reinterpret_cast<param_t*>(exp_avg_sq.data_ptr<float>()),
                                                                         mask.data_ptr<long>(),
                                                                         static_cast<int>(mask.size(0)),
                                                                         static_cast<int>(old_tensor.size(1)));

        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
        cudaStreamSynchronize(stream3);
        param->Set_Exp_Avg(reinterpret_cast<param_t*>(exp_avg_sq.data_ptr<float>()), shape);
        param->Set_Exp_Avg_Sq(reinterpret_cast<param_t*>(exp_avg_sq.data_ptr<float>()), shape);
        old_tensor = masked_param;
    } break;
    default:
        throw std::runtime_error("Not implemented cast in tensors_to_optimizer_new");
    }

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
}

void GaussianModel::prune_points(torch::Tensor mask) {
    // reverse to keep points
    auto valid_point_mask = ~mask;
    auto indices = torch::nonzero(valid_point_mask == true).index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None), torch::indexing::Slice(torch::indexing::None, 1)}).squeeze(-1);
    prune_optimizer<gs::optim::pos_param_t>(indices, _xyz, _optimizer.get(), gs::optim::ParamType::Pos, {static_cast<int>(_xyz.size(0)), static_cast<int>(_xyz.size(1))});
    prune_optimizer<gs::optim::feature_dc_param_t>(indices, _features_dc, _optimizer.get(), gs::optim::ParamType::Features_dc, {static_cast<int>(_features_dc.size(0)), static_cast<int>(_features_dc.size(1)), static_cast<int>(_features_dc.size(2))});
    prune_optimizer<gs::optim::feature_rest_param_t>(indices, _features_rest, _optimizer.get(), gs::optim::ParamType::Features_rest, {static_cast<int>(_features_rest.size(0)), static_cast<int>(_features_rest.size(1)), static_cast<int>(_features_rest.size(2))});
    prune_optimizer<gs::optim::scaling_param_t>(indices, _scaling, _optimizer.get(), gs::optim::ParamType::Scaling, {static_cast<int>(_scaling.size(0)), static_cast<int>(_scaling.size(1))});
    prune_optimizer<gs::optim::rotation_param_t>(indices, _rotation, _optimizer.get(), gs::optim::ParamType::Rotation, {static_cast<int>(_rotation.size(0)), static_cast<int>(_rotation.size(1))});
    prune_optimizer<gs::optim::opacity_param_t>(indices, _opacity, _optimizer.get(), gs::optim::ParamType::Opacity, {static_cast<int>(_opacity.size(0)), static_cast<int>(_opacity.size(1))});

    _xyz_gradient_accum = _xyz_gradient_accum.index_select(0, indices);
    _denom = _denom.index_select(0, indices);
    _max_radii2D = _max_radii2D.index_select(0, indices);
}

template <typename param_t>
void tensors_to_optimizer_new(torch::Tensor& extended_tensor,
                              torch::Tensor& old_tensor,
                              torch::Tensor& exp_avg,
                              torch::Tensor exp_avg_sq,
                              gs::optim::Adam* new_optimizer,
                              gs::optim::ParamType param_type) {

    // new optimizer
    std::vector<int> shape;
    for (int i = 0; i < extended_tensor.sizes().size(); ++i) {
        shape.push_back(extended_tensor.size(i));
    }

    auto param = new_optimizer->GetAdamParameter<param_t>(param_type);
    param->Set_Exp_Avg(reinterpret_cast<param_t*>(exp_avg_sq.data_ptr<float>()), shape);
    param->Set_Exp_Avg_Sq(reinterpret_cast<param_t*>(exp_avg_sq.data_ptr<float>()), shape);
    old_tensor = extended_tensor;
}

template <typename param_t>
void cat_tensors_to_optimizer(torch::Tensor& extension_tensor,
                              torch::Tensor& old_tensor,
                              gs::optim::Adam* new_optimizer,
                              gs::optim::ParamType param_type) {
    cudaStream_t _stream1;
    cudaStreamCreate(&_stream1);

    auto new_exp_avg = torch::tensor({});
    auto new_exp_avg_sq = torch::tensor({});
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto param = new_optimizer->GetAdamParameter<param_t>(param_type);

    torch::Tensor new_tensor;
    if (param_type == gs::optim::ParamType::Features_rest || param_type == gs::optim::ParamType::Features_dc) {
        std::vector<long> old_shape = {static_cast<int>(old_tensor.size(0)),
                                       static_cast<int>(old_tensor.size(1)),
                                       static_cast<int>(old_tensor.size(2))};
        std::vector<long> extension_shape = {static_cast<int>(extension_tensor.size(0)),
                                             static_cast<int>(extension_tensor.size(1)),
                                             static_cast<int>(extension_tensor.size(2))};
        std::vector<long> new_shape = old_shape;
        new_shape[0] += extension_tensor.size(0);

        new_exp_avg = torch::zeros({new_shape[0], new_shape[1], new_shape[2]}, options);
        new_exp_avg_sq = torch::zeros({new_shape[0], new_shape[1], new_shape[2]}, options);
        // TODO need to copy form new optimizer
        copy3DAsync(reinterpret_cast<float*>(param->Get_Exp_Avg()), old_shape, new_exp_avg.data_ptr<float>(), new_shape, _stream1);
        copy3DAsync(reinterpret_cast<float*>(param->Get_Exp_Avg_Sq()), old_shape, new_exp_avg_sq.data_ptr<float>(), new_shape, _stream1);

        new_tensor = torch::zeros({new_shape[0], new_shape[1], new_shape[2]}, options);
        copy3DAsync(old_tensor.data_ptr<float>(), old_shape, new_tensor.data_ptr<float>(), new_shape, _stream1);
        auto shifted_ptr = new_tensor.data_ptr<float>() + old_shape[0] * old_shape[1] * old_shape[2];
        copy3DAsync(extension_tensor.data_ptr<float>(), extension_shape, shifted_ptr, extension_shape, _stream1);

        // TODO: somehow slipped in. Verfiy that it works
        old_tensor = torch::cat({old_tensor, extension_tensor}, 0);
    } else {
        std::vector<long> old_shape = {static_cast<int>(old_tensor.size(0)), static_cast<int>(old_tensor.size(1))};
        std::vector<long> new_shape = old_shape;
        std::vector<long> extension_shape = {static_cast<int>(extension_tensor.size(0)), static_cast<int>(extension_tensor.size(1))};
        new_shape[0] += extension_tensor.size(0);

        new_exp_avg = torch::zeros({new_shape[0], new_shape[1]}, options);
        new_exp_avg_sq = torch::zeros({new_shape[0], new_shape[1]}, options);

        // TODO: eed to copy form new optimizer
        copy2DAsync(reinterpret_cast<float*>(param->Get_Exp_Avg()), old_shape, new_exp_avg.data_ptr<float>(), _stream1);
        copy2DAsync(reinterpret_cast<float*>(param->Get_Exp_Avg_Sq()), old_shape, new_exp_avg_sq.data_ptr<float>(), _stream1);

        new_tensor = torch::zeros({new_shape[0], new_shape[1]}, options);
        copy2DAsync(old_tensor.data_ptr<float>(), old_shape, new_tensor.data_ptr<float>(), _stream1);
        auto shifted_ptr = new_tensor.data_ptr<float>() + old_shape[0] * old_shape[1];
        copy2DAsync(extension_tensor.data_ptr<float>(), extension_shape, shifted_ptr, _stream1);
        old_tensor = new_tensor;
    }

    cudaStreamSynchronize(_stream1);
    cudaStreamDestroy(_stream1);

    std::vector<int> shape;
    for (int i = 0; i < old_tensor.sizes().size(); ++i) {
        shape.push_back(old_tensor.size(i));
    }

    param->Set_Exp_Avg(reinterpret_cast<param_t*>(new_exp_avg_sq.data_ptr<float>()), shape);
    param->Set_Exp_Avg_Sq(reinterpret_cast<param_t*>(new_exp_avg_sq.data_ptr<float>()), shape);
}

void GaussianModel::densification_postfix(torch::Tensor& new_xyz,
                                          torch::Tensor& new_features_dc,
                                          torch::Tensor& new_features_rest,
                                          torch::Tensor& new_scaling,
                                          torch::Tensor& new_rotation,
                                          torch::Tensor& new_opacity) {
    cat_tensors_to_optimizer<gs::optim::pos_param_t>(new_xyz, _xyz, _optimizer.get(), gs::optim::ParamType::Pos);
    cat_tensors_to_optimizer<gs::optim::feature_dc_param_t>(new_features_dc, _features_dc, _optimizer.get(), gs::optim::ParamType::Features_dc);
    cat_tensors_to_optimizer<gs::optim::feature_rest_param_t>(new_features_rest, _features_rest, _optimizer.get(), gs::optim::ParamType::Features_rest);
    cat_tensors_to_optimizer<gs::optim::scaling_param_t>(new_scaling, _scaling, _optimizer.get(), gs::optim::ParamType::Scaling);
    cat_tensors_to_optimizer<gs::optim::rotation_param_t>(new_rotation, _rotation, _optimizer.get(), gs::optim::ParamType::Rotation);
    cat_tensors_to_optimizer<gs::optim::opacity_param_t>(new_opacity, _opacity, _optimizer.get(), gs::optim::ParamType::Opacity);

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
    // TODO I strongly belief that this split is totaly unneccessary. Just makes overhead.
    torch::Tensor new_features_dc = _features_dc.index_select(0, indices).repeat({N, 1, 1});
    torch::Tensor new_features_rest = _features_rest.index_select(0, indices).repeat({N, 1, 1});
    torch::Tensor new_opacity = _opacity.index_select(0, indices).repeat({N, 1});

    densification_postfix(new_xyz, new_features_dc, new_features_rest, new_scaling, new_rotation, new_opacity);

    torch::Tensor prune_filter = torch::cat({selected_pts_mask.squeeze(-1), torch::zeros({N * selected_pts_mask.sum().item<int>()}).to(torch::kBool).to(torch::kCUDA)});
    // torch::Tensor prune_filter = torch::cat({selected_pts_mask.squeeze(-1), torch::zeros({N * selected_pts_mask.sum().item<int>()})}).to(torch::kBool).to(torch::kCUDA);
    prune_filter = torch::logical_or(prune_filter, (Get_opacity() < min_opacity).squeeze(-1));
    prune_points(prune_filter);
}

__global__ void concat_selection_float3_kernel(
    const float3* __restrict__ src,
    const int64_t* __restrict__ indices,
    float3* __restrict__ dst,
    const size_t extension_size,
    const size_t orig_size) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= extension_size) {
        return;
    }

    const int64_t src_index = indices[idx];
    const int64_t dest_idx = orig_size + idx;

    // Single memory read operation for each 3D point
    dst[dest_idx] = src[src_index];
}

__global__ void concat_selection_float4_kernel(
    const float4* __restrict__ src,
    const int64_t* __restrict__ indices,
    float4* __restrict__ dst,
    const size_t extension_size,
    const size_t orig_size) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= extension_size) {
        return;
    }

    const int64_t src_index = indices[idx];
    const int64_t dest_idx = orig_size + idx;

    // Single memory read operation for each 3D point
    dst[dest_idx] = src[src_index];
}

__global__ void concat_elements_kernel_opacity(
    const float* __restrict__ opacity,
    const int64_t* __restrict__ indices,
    float* __restrict__ new_opacity,
    const size_t N,
    const size_t orig_N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    const int64_t index = indices[idx];
    const int64_t dest_idx = orig_N + idx;
    new_opacity[dest_idx] = opacity[index];
}

__global__ void concat_elements_kernel_features_dc(
    const float3* __restrict__ features_dc,
    const int64_t* __restrict__ indices,
    float3* __restrict__ new_features_dc,
    const size_t N,
    const size_t orig_N,
    const size_t F1) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    // Copy selected elements to new tensors, at positions after the original elements
    const int64_t index = indices[idx];
    const int64_t dest_idx = orig_N + idx;

    for (int j = 0; j < F1; j++) {
        new_features_dc[dest_idx * F1 + j] = features_dc[index * F1 + j];
    }
}

__global__ void concat_elements_kernel_features_rest(
    const float3* __restrict__ features_rest,
    const int64_t* __restrict__ indices,
    float3* __restrict__ new_features_rest,
    const size_t N,
    const size_t orig_N,
    const size_t F1) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    // Copy selected elements to new tensors, at positions after the original elements
    const int64_t index = indices[idx];
    const int64_t dest_idx = orig_N + idx;

    for (int j = 0; j < F1; j++) {
        new_features_rest[dest_idx * F1 + j] = features_rest[index * F1 + j];
    }
}

void copy1DAsync(const torch::Tensor& src, torch::Tensor& dst, cudaStream_t stream) {
    // assert(src.size(0) <= dst.size(0));

    size_t bytes_to_copy = src.size(0) * sizeof(float);

    CHECK_CUDA_ERROR(cudaMemcpyAsync(
        dst.data_ptr(),           // Destination pointer
        src.data_ptr(),           // Source pointer
        bytes_to_copy,            // Number of bytes to copy
        cudaMemcpyDeviceToDevice, // Specifies the kind of copy (Device to Device)
        stream                    // Stream to perform the copy
        ));
}

void GaussianModel::select_elements_and_cat(
    const float* xyz,
    const std::vector<long>& xyz_size,
    const float* features_dc,
    const std::vector<long>& features_dc_size,
    const float* features_rest,
    const std::vector<long>& features_rest_size,
    const float* opacity,
    const std::vector<long>& opacity_size,
    const float* scaling,
    const std::vector<long>& scaling_size,
    const float* rotation,
    const std::vector<long>& rotation_size,
    int64_t* indices,
    long original_size,
    long extension_size) {

    long threads = 256;
    long blocks = std::max(1L, (extension_size + threads - 1L) / threads);
    {

        auto xyz_paramstates = _optimizer->GetAdamParameter<gs::optim::pos_param_t>(gs::optim::ParamType::Pos);
        auto features_dc_paramstates = _optimizer->GetAdamParameter<gs::optim::feature_dc_param_t>(gs::optim::ParamType::Features_dc);
        auto features_rest_paramstates = _optimizer->GetAdamParameter<gs::optim::feature_rest_param_t>(gs::optim::ParamType::Features_rest);
        auto scaling_paramstates = _optimizer->GetAdamParameter<gs::optim::scaling_param_t>(gs::optim::ParamType::Scaling);
        auto rotation_paramstates = _optimizer->GetAdamParameter<gs::optim::rotation_param_t>(gs::optim::ParamType::Rotation);
        auto opacity_paramstates = _optimizer->GetAdamParameter<gs::optim::opacity_param_t>(gs::optim::ParamType::Opacity);

        const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        auto new_xyz = torch::zeros({_xyz.size(0) + extension_size, _xyz.size(1)}, options);
        auto new_features_dc = torch::zeros({original_size + extension_size, _features_dc.size(1), _features_dc.size(2)}, options);
        auto new_features_rest = torch::zeros({original_size + extension_size, _features_rest.size(1), _features_rest.size(2)}, options);
        auto new_opacity = torch::zeros({original_size + extension_size, 1}, options);
        auto new_scaling = torch::zeros({original_size + extension_size, 3}, options);
        auto new_rotation = torch::zeros({original_size + extension_size, 4}, options);

        const int total_extension_count = original_size + extension_size;
        auto xyz_exp_avg = torch::zeros({total_extension_count, 3}, options);
        auto xyz_exp_avg_sq = torch::zeros({total_extension_count, 3}, options);
        auto features_dc_avg = torch::zeros({total_extension_count, _features_dc.size(1), _features_dc.size(2)}, options);
        auto features_dc_avg_sq = torch::zeros({total_extension_count, _features_dc.size(1), _features_dc.size(2)}, options);
        auto features_rest_avg = torch::zeros({total_extension_count, _features_rest.size(1), _features_rest.size(2)}, options);
        auto features_rest_avg_sq = torch::zeros({total_extension_count, _features_rest.size(1), _features_rest.size(2)}, options);
        auto opacity_avg = torch::zeros({total_extension_count, _opacity.size(1)}, options);
        auto opacity_avg_sq = torch::zeros({total_extension_count, _opacity.size(1)}, options);
        auto scaling_avg = torch::zeros({total_extension_count, _scaling.size(1)}, options);
        auto scaling_avg_sq = torch::zeros({total_extension_count, _scaling.size(1)}, options);
        auto rotation_avg = torch::zeros({total_extension_count, _rotation.size(1)}, options);
        auto rotation_avg_sq = torch::zeros({total_extension_count, _rotation.size(1)}, options);

        // TODO: no reinterpret_cast -> refactor accordingly. Should be also faster if we take the native floatx type?
        copy2DAsync(reinterpret_cast<float*>(xyz_paramstates->Get_Exp_Avg()), {original_size, 3}, xyz_exp_avg.data_ptr<float>(), _stream1);
        copy2DAsync(reinterpret_cast<float*>(xyz_paramstates->Get_Exp_Avg_Sq()), {original_size, 3}, xyz_exp_avg_sq.data_ptr<float>(), _stream1);

        copy3DAsync(reinterpret_cast<float*>(features_dc_paramstates->Get_Exp_Avg()), features_dc_size, features_dc_avg.data_ptr<float>(), {original_size + extension_size, features_dc_size[1], features_dc_size[2]}, _stream2);
        copy3DAsync(reinterpret_cast<float*>(features_dc_paramstates->Get_Exp_Avg_Sq()), features_dc_size, features_dc_avg_sq.data_ptr<float>(), {original_size + extension_size, features_dc_size[1], features_dc_size[2]}, _stream2);

        copy3DAsync(reinterpret_cast<float*>(features_rest_paramstates->Get_Exp_Avg()), features_rest_size, features_rest_avg.data_ptr<float>(), {original_size + extension_size, features_rest_size[1], features_rest_size[2]}, _stream3);
        copy3DAsync(reinterpret_cast<float*>(features_rest_paramstates->Get_Exp_Avg_Sq()), features_rest_size, features_rest_avg_sq.data_ptr<float>(), {original_size + extension_size, features_rest_size[1], features_rest_size[2]}, _stream3);

        // opacity is already float*
        copy2DAsync(opacity_paramstates->Get_Exp_Avg(), {original_size, 1}, opacity_avg.data_ptr<float>(), _stream4);
        copy2DAsync(opacity_paramstates->Get_Exp_Avg_Sq(), {original_size, 1}, opacity_avg_sq.data_ptr<float>(), _stream4);

        copy2DAsync(reinterpret_cast<float*>(scaling_paramstates->Get_Exp_Avg()), {original_size, 3}, scaling_avg.data_ptr<float>(), _stream5);
        copy2DAsync(reinterpret_cast<float*>(scaling_paramstates->Get_Exp_Avg_Sq()), {original_size, 3}, scaling_avg_sq.data_ptr<float>(), _stream5);

        copy2DAsync(reinterpret_cast<float*>(rotation_paramstates->Get_Exp_Avg()), {original_size, 4}, rotation_avg.data_ptr<float>(), _stream6);
        copy2DAsync(reinterpret_cast<float*>(rotation_paramstates->Get_Exp_Avg_Sq()), {original_size, 4}, rotation_avg_sq.data_ptr<float>(), _stream6);

        copy2DAsync(xyz, xyz_size, new_xyz.data_ptr<float>(), _stream1);
        const auto* xyz3_ptr = reinterpret_cast<const float3*>(xyz);
        auto* new_xyz3_ptr = reinterpret_cast<float3*>(new_xyz.data_ptr<float>());
        concat_selection_float3_kernel<<<blocks, threads, 0, _stream1>>>(
            xyz3_ptr,
            indices,
            new_xyz3_ptr,
            extension_size,
            xyz_size[0]);

        copy2DAsync(scaling, scaling_size, new_scaling.data_ptr<float>(), _stream5);
        const auto* scaling3_ptr = reinterpret_cast<const float3*>(scaling);
        auto* new_scaling3_ptr = reinterpret_cast<float3*>(new_scaling.data_ptr<float>());
        concat_selection_float3_kernel<<<blocks, threads, 0, _stream5>>>(
            scaling3_ptr,
            indices,
            new_scaling3_ptr,
            extension_size,
            scaling_size[0]);
        copy3DAsync(features_dc, features_dc_size, new_features_dc.data_ptr<float>(), {original_size + extension_size, features_dc_size[1], features_dc_size[2]}, _stream2);
        const auto* features_dc_ptr = reinterpret_cast<const float3*>(features_dc);
        auto* new_features_dc_ptr = reinterpret_cast<float3*>(new_features_dc.data_ptr<float>());
        concat_elements_kernel_features_dc<<<blocks, threads, 0, _stream2>>>(
            features_dc_ptr,
            indices,
            new_features_dc_ptr,
            extension_size,
            xyz_size[0], features_dc_size[1]);
        copy2DAsync(opacity, opacity_size, new_opacity.data_ptr<float>(), _stream4);
        concat_elements_kernel_opacity<<<blocks, threads, 0, _stream4>>>(
            opacity,
            indices,
            new_opacity.data_ptr<float>(),
            extension_size,
            xyz_size[0]);
        copy3DAsync(features_rest, features_rest_size, new_features_rest.data_ptr<float>(), {original_size + extension_size, features_rest_size[1], features_rest_size[2]}, _stream3);
        const auto* features_rest_ptr = reinterpret_cast<const float3*>(features_rest);
        auto* new_features_rest_ptr = reinterpret_cast<float3*>(new_features_rest.data_ptr<float>());
        concat_elements_kernel_features_rest<<<blocks, threads, 0, _stream3>>>(
            features_rest_ptr,
            indices,
            new_features_rest_ptr,
            extension_size,
            features_rest_size[0], features_rest_size[1]);
        CHECK_LAST_CUDA_ERROR();

        copy2DAsync(rotation, rotation_size, new_rotation.data_ptr<float>(), _stream6);
        auto* rotation_ptr = reinterpret_cast<const float4*>(rotation);
        auto* new_rotation_ptr = reinterpret_cast<float4*>(new_rotation.data_ptr<float>());
        concat_selection_float4_kernel<<<blocks, threads, 0, _stream6>>>(
            rotation_ptr,
            indices,
            new_rotation_ptr,
            extension_size,
            rotation_size[0]);
        CHECK_LAST_CUDA_ERROR();

        cudaStreamSynchronize(_stream1);
        CHECK_LAST_CUDA_ERROR();
        cudaStreamSynchronize(_stream2);
        CHECK_LAST_CUDA_ERROR();
        cudaStreamSynchronize(_stream3);
        CHECK_LAST_CUDA_ERROR();
        cudaStreamSynchronize(_stream4);
        CHECK_LAST_CUDA_ERROR();
        cudaStreamSynchronize(_stream5);
        CHECK_LAST_CUDA_ERROR();
        cudaStreamSynchronize(_stream6);

        tensors_to_optimizer_new<gs::optim::pos_param_t>(new_xyz, _xyz, xyz_exp_avg, xyz_exp_avg_sq, _optimizer.get(), gs::optim::ParamType::Pos);
        tensors_to_optimizer_new<gs::optim::feature_dc_param_t>(new_features_dc, _features_dc, features_dc_avg, features_dc_avg_sq, _optimizer.get(), gs::optim::ParamType::Features_dc);
        tensors_to_optimizer_new<gs::optim::feature_rest_param_t>(new_features_rest, _features_rest, features_rest_avg, features_rest_avg_sq, _optimizer.get(), gs::optim::ParamType::Features_rest);
        tensors_to_optimizer_new<gs::optim::opacity_param_t>(new_opacity, _opacity, opacity_avg, opacity_avg_sq, _optimizer.get(), gs::optim::ParamType::Opacity);
        tensors_to_optimizer_new<gs::optim::scaling_param_t>(new_scaling, _scaling, scaling_avg, scaling_avg_sq, _optimizer.get(), gs::optim::ParamType::Scaling);
        tensors_to_optimizer_new<gs::optim::rotation_param_t>(new_rotation, _rotation, rotation_avg, rotation_avg_sq, _optimizer.get(), gs::optim::ParamType::Rotation);
    }

    CHECK_LAST_CUDA_ERROR();
}

void GaussianModel::densify_and_clone(torch::Tensor& grads, float grad_threshold, float scene_extent) {
    // Extract points that satisfy the gradient condition
    torch::Tensor selected_pts_mask = torch::where(torch::linalg::vector_norm(grads, {2}, 1, true, torch::kFloat32) >= grad_threshold,
                                                   torch::ones_like(grads.index({torch::indexing::Slice()})).to(torch::kBool),
                                                   torch::zeros_like(grads.index({torch::indexing::Slice()})).to(torch::kBool))
                                          .to(torch::kLong);

    selected_pts_mask = torch::logical_and(selected_pts_mask, std::get<0>(Get_scaling().max(1)).unsqueeze(-1) <= _percent_dense * scene_extent);

    auto indices = torch::nonzero(selected_pts_mask.squeeze(-1) == true).index({torch::indexing::Slice(torch::indexing::None, torch::indexing::None), torch::indexing::Slice(torch::indexing::None, 1)}).squeeze(-1);
    const auto extension_count = torch::sum(selected_pts_mask).item<int>();
    const auto total_extension_count = _xyz.size(0) + extension_count;
    at::cuda::CUDAStream stream1 = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(stream1);
    _xyz_gradient_accum = torch::zeros({total_extension_count, 1}).to(torch::kCUDA);
    _denom = torch::zeros({total_extension_count, 1}).to(torch::kCUDA);
    _max_radii2D = torch::zeros({total_extension_count}).to(torch::kCUDA);
    at::cuda::setCurrentCUDAStream(at::cuda::getDefaultCUDAStream());

    select_elements_and_cat(_xyz.data_ptr<float>(),
                            {_xyz.size(0), _xyz.size(1)},
                            _features_dc.data_ptr<float>(),
                            {_features_dc.size(0), _features_dc.size(1), _features_dc.size(2)},
                            _features_rest.data_ptr<float>(),
                            {_features_rest.size(0), _features_rest.size(1), _features_rest.size(2)},
                            _opacity.data_ptr<float>(),
                            {_opacity.size(0), _opacity.size(1)},
                            _scaling.data_ptr<float>(),
                            {_scaling.size(0), _scaling.size(1)},
                            _rotation.data_ptr<float>(),
                            {_rotation.size(0), _rotation.size(1)},
                            indices.data_ptr<int64_t>(),
                            _xyz.size(0),
                            extension_count);
    //    stream1.synchronize();
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