#include "debug_utils.cuh"
#include "gaussian.cuh"
#include "loss_monitor.cuh"
#include "loss_utils.cuh"
#include "parameters.cuh"
#include "render_utils.cuh"
#include "scene.cuh"
#include <args.hxx>
#include <c10/cuda/CUDACachingAllocator.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <torch/torch.h>

void Write_model_parameters_to_file(const ModelParameters& params) {
    std::filesystem::path outputPath = params.output_path;
    std::filesystem::create_directories(outputPath); // Make sure the directory exists

    std::ofstream cfg_log_f(outputPath / "cfg_args");
    if (!cfg_log_f.is_open()) {
        std::cerr << "Failed to open file for writing!" << std::endl;
        return;
    }

    // Write the parameters in the desired format
    cfg_log_f << "Namespace(";
    cfg_log_f << "eval=" << (params.eval ? "True" : "False") << ", ";
    cfg_log_f << "images='" << params.images << "', ";
    cfg_log_f << "model_path='" << params.output_path.string() << "', ";
    cfg_log_f << "resolution=" << params.resolution << ", ";
    cfg_log_f << "sh_degree=" << params.sh_degree << ", ";
    cfg_log_f << "source_path='" << params.source_path.string() << "', ";
    cfg_log_f << "white_background=" << (params.white_background ? "True" : "False") << ")";
    cfg_log_f.close();

    std::cout << "Output folder: " << params.output_path.string() << std::endl;
}

std::vector<int> get_random_indices(int max_index) {
    std::vector<int> indices(max_index);
    std::iota(indices.begin(), indices.end(), 0);
    // Shuffle the vector
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine());
    return indices;
}

int parse_cmd_line_args(const std::vector<std::string>& args,
                        ModelParameters& modelParams,
                        OptimizationParameters& optimParams) {
    if (args.empty()) {
        std::cerr << "No command line arguments provided!" << std::endl;
        return -1;
    }
    args::ArgumentParser parser("3D Gaussian Splatting CUDA Implementation\n",
                                "This program provides a lightning-fast CUDA implementation of the 3D Gaussian Splatting algorithm for real-time radiance field rendering.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<float> convergence_rate(parser, "convergence_rate", "Set convergence rate", {'c', "convergence_rate"});
    args::Flag enable_cr_monitoring(parser, "enable_cr_monitoring", "Enable convergence rate monitoring", {"enable-cr-monitoring"});
    args::Flag force_overwrite_output_path(parser, "force", "Forces to overwrite output folder", {'f', "force"});
    args::Flag empty_gpu_memory(parser, "empty_gpu_cache", "Forces to reset GPU Cache. Should be lighter on VRAM", {"empty-gpu-cache"});
    args::ValueFlag<std::string> data_path(parser, "data_path", "Path to the training data", {'d', "data-path"});
    args::ValueFlag<std::string> output_path(parser, "output_path", "Path to the training output", {'o', "output-path"});
    args::ValueFlag<uint32_t> iterations(parser, "iterations", "Number of iterations to train the model", {'i', "iter"});
    args::CompletionFlag completion(parser, {"complete"});

    try {
        parser.Prog(args.front());
        parser.ParseArgs(std::vector<std::string>(args.begin() + 1, args.end()));
    } catch (const args::Completion& e) {
        std::cout << e.what();
        return 0;
    } catch (const args::Help&) {
        std::cout << parser;
        return -1;
    } catch (const args::ParseError& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return -1;
    }

    if (data_path) {
        modelParams.source_path = args::get(data_path);
    } else {
        std::cerr << "No data path provided!" << std::endl;
        return -1;
    }
    if (output_path) {
        modelParams.output_path = args::get(output_path);
    } else {
        std::filesystem::path executablePath = std::filesystem::canonical("/proc/self/exe");
        std::filesystem::path parentDir = executablePath.parent_path().parent_path();
        std::filesystem::path outputDir = parentDir / "output";
        try {

            bool isCreated = std::filesystem::create_directory(outputDir);
            if (!isCreated) {
                if (!force_overwrite_output_path) {
                    std::cerr << "Directory already exists! Not overwriting it" << std::endl;
                    return -1;
                } else {
                    std::filesystem::create_directory(outputDir);
                    std::filesystem::remove_all(outputDir);
                }
            }
        } catch (...) {
            std::cerr << "Failed to create output directory!" << std::endl;
            return -1;
        }
        modelParams.output_path = outputDir;
    }

    if (iterations) {
        optimParams.iterations = args::get(iterations);
    }
    optimParams.early_stopping = args::get(enable_cr_monitoring);
    if (optimParams.early_stopping && convergence_rate) {
        optimParams.convergence_threshold = args::get(convergence_rate);
    }

    optimParams.empty_gpu_cache = args::get(empty_gpu_memory);
    return 0;
}

float psnr_metric(const torch::Tensor& rendered_img, const torch::Tensor& gt_img) {

    torch::Tensor squared_diff = (rendered_img - gt_img).pow(2);
    torch::Tensor mse_val = squared_diff.view({rendered_img.size(0), -1}).mean(1, true);
    return (20.f * torch::log10(1.0 / mse_val.sqrt())).mean().item<float>();
}

int main(int argc, char* argv[]) {
    std::vector<std::string> args;
    args.reserve(argc);

    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }
    // TODO: read parameters from JSON file or command line
    auto modelParams = ModelParameters();
    auto optimParams = OptimizationParameters();
    if (parse_cmd_line_args(args, modelParams, optimParams) < 0) {
        return -1;
    };
    Write_model_parameters_to_file(modelParams);

    auto gaussians = gs::GaussianModel(modelParams.sh_degree);
    auto scene = gs::Scene(gaussians, modelParams);
    gaussians.Training_setup(optimParams);

    //    auto ref_gaussians = ref::GaussianModel(modelParams.sh_degree);
    //    auto ref_scene = ref::Scene(ref_gaussians, modelParams);
    //    ref_gaussians.Training_setup(optimParams);

    if (!torch::cuda::is_available()) {
        // At the moment, I want to make sure that my GPU is utilized.
        std::cout << "CUDA is not available! Training on CPU." << std::endl;
        exit(-1);
    }
    auto pointType = torch::TensorOptions().dtype(torch::kFloat32);
    auto background = modelParams.white_background ? torch::tensor({1.f, 1.f, 1.f}) : torch::tensor({0.f, 0.f, 0.f}, pointType).to(torch::kCUDA);
    //    auto ref_background = modelParams.white_background ? torch::tensor({1.f, 1.f, 1.f}) : torch::tensor({0.f, 0.f, 0.f}, pointType).to(torch::kCUDA);

    const int window_size = 11;
    const int channel = 3;
    const auto conv_window = gs::loss::create_window(window_size, channel).to(torch::kFloat32).to(torch::kCUDA, true);
    //    const auto ref_conv_window = gs::loss::create_window(window_size, channel).to(torch::kFloat32).to(torch::kCUDA, true);
    const int camera_count = scene.Get_camera_count();

    std::vector<int> indices;
    int last_status_len = 0;
    auto start_time = std::chrono::steady_clock::now();
    float loss_add = 0.f;

    LossMonitor loss_monitor(200);
    float avg_converging_rate = 0.f;

    float psnr_value = 0.f;
    for (int iter = 1; iter < optimParams.iterations + 1; ++iter) {
        if (indices.empty()) {
            indices = get_random_indices(camera_count);
        }
        const int camera_index = indices.back();
        auto cam = scene.Get_training_camera(camera_index);
        //        auto ref_cam = scene.Get_training_camera(camera_index);
        auto gt_image = cam.Get_original_image().clone().to(torch::kCUDA);
        //        auto ref_gt_image = cam.Get_original_image().clone().to(torch::kCUDA);
        indices.pop_back(); // remove last element to iterate over all cameras randomly
        if (iter % 1000 == 0) {
            gaussians.One_up_sh_degree();
            //            ref_gaussians.One_up_sh_degree();
        }

        // Render
        //        auto [ref_image, ref_viewspace_points, ref_visibility_filter, ref_radii] = ref::render(ref_cam, ref_gaussians, ref_background);
        //        ref_image.set_requires_grad(true);
        //        ref_image.retain_grad();
        //        ref_gt_image.set_requires_grad(true);
        //        ref_gt_image.retain_grad();
        //        ref_gaussians._optimizer->zero_grad();
        //        auto [ref_L1l, ref_dL_l1_loss] = ref::loss::l1_loss(ref_image, ref_gt_image);
        //        auto [ref_ssim_loss, ref_dL_ssim_dimg1] = ref::loss::ssim(ref_image, ref_gt_image, ref_conv_window, window_size, channel);
        //        auto ref_loss = (1.f - optimParams.lambda_dssim) * ref_L1l + optimParams.lambda_dssim * (1.f - ref_ssim_loss);
        //        const auto ref_dloss_dssim = -optimParams.lambda_dssim;
        //        const auto ref_dloss_dLl1 = 1.0 - optimParams.lambda_dssim;
        //        const auto ref_dloss_dimage = ref_dloss_dLl1 * ref_dL_l1_loss + ref_dloss_dssim * ref_dL_ssim_dimg1;
        //        torch::save(ref_dloss_dimage, "ref_image_loss.pt");
        //        ref_loss.backward();

        //        auto diff = torch::abs(ref_dloss_dimage - ref_image.grad());
        //        auto max = torch::max(diff);
        //        std::cout << "Diff ref_dloss_dimage max: " << max.item<float>() <<  std::endl;

        cudaDeviceSynchronize();
        // Loss Computations
        //        torch::Tensor grad;
        //        {
        //            torch::NoGradGuard no_grad;
        //            grad = ref_image.grad().clone().set_requires_grad(false);
        //        }
        gs::SaveForBackward saveForBackwars;
        auto [image, visibility_filter, radii] = gs::render(saveForBackwars, cam, gaussians, background);
        auto [L1l, dL_l1_loss] = gs::loss::l1_loss(image, gt_image);
        auto [ssim_loss, dL_ssim_dimg1] = gs::loss::ssim(image, gt_image, conv_window, window_size, channel);
        auto loss = (1.f - optimParams.lambda_dssim) * L1l + optimParams.lambda_dssim * (1.f - ssim_loss);

        const auto dloss_dssim = -optimParams.lambda_dssim;
        const auto dloss_dLl1 = 1.0 - optimParams.lambda_dssim;
        const auto dloss_dimage = dloss_dLl1 * dL_l1_loss + dloss_dssim * dL_ssim_dimg1;

        //        if (!torch::allclose(dloss_dimage, ref_image.grad(), 1e-5, 1e-5)) {
        //            std::cout << "Diff dloss_dimage" << std::endl;
        //            auto diff = torch::abs(dloss_dimage - ref_image.grad());
        //            auto max = torch::max(diff);
        //            std::cout << "Diff dloss_dimage Max: " << max <<  std::endl;
        //        }
        auto [grad_means3D, grad_means2D, grad_sh, grad_color_precomp, grad_opacities, grad_scales, grad_rotations, grad_cov3Ds_precomp] = gs::_RasterizeGaussians::Backward(saveForBackwars, dloss_dimage);
        gaussians.Update_Grads(grad_means3D, grad_sh, grad_opacities, grad_scales, grad_rotations);
        //        {
        //            gaussians.Update_Grads(ref_gaussians._xyz.grad().clone(),
        //                                   ref_gaussians._features_dc.grad().clone(),
        //                                   ref_gaussians._features_rest.grad().clone(),
        //                                   ref_gaussians._opacity.grad().clone(),
        //                                   ref_gaussians._scaling.grad().clone(),
        //                                   ref_gaussians._rotation.grad().clone());
        //        }

        cudaDeviceSynchronize();
        // Update status line
        if (iter % 100 == 0) {
            auto cur_time = std::chrono::steady_clock::now();
            std::chrono::duration<double> time_elapsed = cur_time - start_time;
            // XXX shouldn't have to create a new stringstream, but resetting takes multiple calls
            std::stringstream status_line;
            // XXX Use thousand separators, but doesn't work for some reason
            status_line.imbue(std::locale(""));
            status_line
                << "\rIter: " << std::setw(6) << iter
                << "  Loss: " << std::fixed << std::setw(9) << std::setprecision(6) << loss.item<float>();
            if (optimParams.early_stopping) {
                status_line
                    << "  ACR: " << std::fixed << std::setw(9) << std::setprecision(6) << avg_converging_rate;
            }
            status_line
                << "  Splats: " << std::setw(10) << (int)gaussians.Get_xyz().size(0)
                << "  Time: " << std::fixed << std::setw(8) << std::setprecision(3) << time_elapsed.count() << "s"
                << "  Avg iter/s: " << std::fixed << std::setw(5) << std::setprecision(1) << 1.0 * iter / time_elapsed.count()
                << "  " // Some extra whitespace, in case a "Pruning ... points" message gets printed after
                ;
            const int curlen = status_line.str().length();
            const int ws = last_status_len - curlen;
            if (ws > 0)
                status_line << std::string(ws, ' ');
            std::cout << status_line.str() << std::flush;
            last_status_len = curlen;
        }

        if (optimParams.early_stopping) {
            avg_converging_rate = loss_monitor.Update(loss.item<float>());
        }
        loss_add += loss.item<float>();
        //        std::cout << "    Iter: " << iter << ", Splats: " << grad_means2D.size(0) << ", Loss: " << std::fixed << std::setw(9) << std::setprecision(6) << loss.item<float>() << std::endl;
        //        std::cout << "Ref Iter: " << iter << ", Splats: " << ref_visibility_filter.size(0) << ", Loss: " << std::fixed << std::setw(9) << std::setprecision(6) << ref_loss.item<float>() << std::endl;
        //        std::cout << "Diff Iter: " << iter << ", Diff Splats: " << std::abs(ref_visibility_filter.size(0) - grad_means2D.size(0)) << ", Diff Loss: " << std::fixed << std::setw(9) << std::setprecision(6) << std::abs(loss.item<float>() - ref_loss.item<float>()) << std::endl;

        {
            torch::NoGradGuard no_grad;
            //            double abs_tol = 1e-3;
            //            double rel_tol = 1e-3;
            //            ts::print_debug_info(ref_image.grad(), "ref_image.grad()");
            //            ts::print_debug_info(dloss_dimage, "dloss_dimage");
            //            if (!torch::allclose(dloss_dimage, ref_image.grad(), 1e-5, 1e-5)) {
            //                std::cout << "Diff dloss_dimage" << std::endl;
            //                auto diff = torch::abs(dloss_dimage - ref_image.grad());
            //                auto max = torch::max(diff);
            //                auto min = torch::min(diff);
            //                std::cout << "Diff dloss_dimage Max: " << max << ", Min: " << min << std::endl;
            //            }
            //
            //            if (!torch::allclose(grad_means2D, ref_viewspace_points.grad(), rel_tol, abs_tol)) {
            //                std::cout << "Diff grad_means 2D" << std::endl;
            //                auto diff = torch::abs(grad_means2D - ref_viewspace_points.grad());
            //                auto max = torch::max(diff);
            //                auto min = torch::min(diff);
            //                std::cout << "Diff grad_means 2D Max: " << max << ", Min: " << min << std::endl;
            //            }
            //
            //            if (!torch::allclose(grad_means3D, ref_gaussians._xyz.grad(), rel_tol, abs_tol)) {
            //                std::cout << "Diff grad_means 3D" << std::endl;
            //                auto diff = torch::abs(grad_means3D - ref_gaussians._xyz.grad());
            //                auto max = torch::max(diff);
            //                auto min = torch::min(diff);
            //                std::cout << "Diff grad_means 3D Max: " << max << ", Min: " << min << std::endl;
            //            }
            //
            //            const auto grad_features_dc = grad_sh.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1), torch::indexing::Slice()}).contiguous();
            //            const auto grad_features_rest = grad_sh.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()}).contiguous();
            //
            //            if (!torch::allclose(grad_features_dc, ref_gaussians._features_dc.grad(), rel_tol, abs_tol)) {
            //                std::cout << "Diff _features_dc" << std::endl;
            //                auto diff = torch::abs(grad_features_dc - ref_gaussians._features_dc.grad());
            //                auto max = torch::max(diff);
            //                auto min = torch::min(diff);
            //                std::cout << "Diff _features_dc Max: " << max << ", Min: " << min << std::endl;
            //            }
            //
            //            if (!torch::allclose(grad_features_rest, ref_gaussians._features_rest.grad(), rel_tol, abs_tol)) {
            //                std::cout << "Diff _features_rest" << std::endl;
            //                auto diff = torch::abs(grad_features_rest - ref_gaussians._features_rest.grad());
            //                auto max = torch::max(diff);
            //                auto min = torch::min(diff);
            //                std::cout << "Diff _features_rest Max: " << max << ", Min: " << min << std::endl;
            //            }
            //
            //            ts::print_debug_info(grad_opacities, "grad_opacities");
            //            ts::print_debug_info(ref_gaussians._opacity.grad(), "ref_gaussians._opacity.grad()");
            //            if (!torch::allclose(grad_opacities, ref_gaussians._opacity.grad(), rel_tol, abs_tol)) {
            //                std::cout << "Diff _opacity" << std::endl;
            //                auto diff = torch::abs(grad_opacities - ref_gaussians._opacity.grad());
            //                auto max = torch::max(diff);
            //                auto min = torch::min(diff);
            //                std::cout << "Diff _opacity Max: " << max << ", Min: " << min << std::endl;
            //            }
            //
            //            ts::print_debug_info(grad_scales, "grad_scales");
            //            ts::print_debug_info(ref_gaussians._scaling.grad(), "ref_gaussians._scaling.grad()");
            //            if (!torch::allclose(grad_scales, ref_gaussians._scaling.grad(), rel_tol, abs_tol)) {
            //                std::cout << "Diff _scaling" << std::endl;
            //                auto diff = torch::abs(grad_scales - ref_gaussians._scaling.grad());
            //                auto max = torch::max(diff);
            //                auto min = torch::min(diff);
            //                std::cout << "Diff _scaling Max: " << max << ", Min: " << min << std::endl;
            //            }
            //
            //            if (!torch::allclose(grad_rotations, ref_gaussians._rotation.grad(), rel_tol, abs_tol)) {
            //                std::cout << "Diff _rotation" << std::endl;
            //                auto diff = torch::abs(grad_rotations - ref_gaussians._rotation.grad());
            //                auto max = torch::max(diff);
            //                auto min = torch::min(diff);
            //                std::cout << "Diff _rotation Max: " << max << ", Min: " << min << std::endl;
            //            }

            auto visible_max_radii = gaussians._max_radii2D.masked_select(visibility_filter);
            auto visible_radii = radii.masked_select(visibility_filter);
            auto max_radii = torch::max(visible_max_radii, visible_radii);
            gaussians._max_radii2D.masked_scatter_(visibility_filter, max_radii);

            //            auto ref_visible_max_radii = ref_gaussians._max_radii2D.masked_select(ref_visibility_filter);
            //            auto ref_visible_radii = ref_radii.masked_select(ref_visibility_filter);
            //            auto ref_max_radii = torch::max(ref_visible_max_radii, ref_visible_radii);
            //            ref_gaussians._max_radii2D.masked_scatter_(ref_visibility_filter, ref_max_radii);

            //  Optimizer step
            cudaDeviceSynchronize();
            if (iter < optimParams.iterations) {
                gaussians._optimizer->Step(nullptr);
                gaussians.Update_Params();
                gaussians.Update_learning_rate(iter);

                //                cudaDeviceSynchronize();
                //                ref_gaussians._optimizer->step();
                //                ref_gaussians._optimizer->zero_grad(true);
                //                ref_gaussians.Update_learning_rate(iter);
            }

            if (iter == optimParams.iterations) {
                std::cout << std::endl;
                gaussians.Save_ply(modelParams.output_path, iter, true);
                psnr_value = psnr_metric(image, gt_image);
                //                ref_gaussians.Save_ply("ref" + modelParams.output_path.string(), iter, true);
                break;
            }

            if (iter % 7'000 == 0) {
                gaussians.Save_ply(modelParams.output_path, iter, false);
                //                ref_gaussians.Save_ply("ref" + modelParams.output_path.string(), iter, false);
            }

            // Densification
            if (iter < optimParams.densify_until_iter) {
                gaussians.Add_densification_stats(grad_means2D, visibility_filter);
                //                ref_gaussians.Add_densification_stats(ref_viewspace_points, ref_visibility_filter);
                if (iter > optimParams.densify_from_iter && iter % optimParams.densification_interval == 0) {
                    float size_threshold = iter > optimParams.opacity_reset_interval ? 20.f : -1.f;
                    gaussians.Densify_and_prune(optimParams.densify_grad_threshold, 0.00005f, scene.Get_cameras_extent(), size_threshold);
                    //                    cudaDeviceSynchronize();
                    //                    ref_gaussians.Densify_and_prune(optimParams.densify_grad_threshold, 0.005f, ref_scene.Get_cameras_extent(), size_threshold);
                }

                if (iter % optimParams.opacity_reset_interval == 0 || (modelParams.white_background && iter == optimParams.densify_from_iter)) {
                    gaussians.Reset_opacity();
                    //                    cudaDeviceSynchronize();
                    //                    ref_gaussians.Reset_opacity();
                }
            }

            if (iter >= optimParams.densify_until_iter && loss_monitor.IsConverging(optimParams.convergence_threshold)) {
                std::cout << "Converged after " << iter << " iterations!" << std::endl;
                gaussians.Save_ply(modelParams.output_path, iter, true);
                break;
            }

            //            if (optimParams.empty_gpu_cache && iter % 100) {
            //                c10::cuda::CUDACachingAllocator::emptyCache();
            //            }
        }
    }

    auto cur_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_elapsed = cur_time - start_time;

    std::cout << std::endl
              << "All done in "
              << std::fixed << std::setw(7) << std::setprecision(3) << time_elapsed.count() << "sec, avg "
              << std::fixed << std::setw(4) << std::setprecision(1) << 1.0 * optimParams.iterations / time_elapsed.count() << " iter/sec, "
              << gaussians.Get_xyz().size(0) << " splats, "
              << std::fixed << std::setw(7) << std::setprecision(6) << ", psrn: " << psnr_value << std::endl
              << std::endl
              << std::endl;

    return 0;
}
