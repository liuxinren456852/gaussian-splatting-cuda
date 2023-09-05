// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once

#include "debug_utils.cuh"
#include "rasterize_points.cuh"
#include "rasterizer_impl.h"
#include "serialization.h"

#define WRITE_TEST_DATA
#undef WRITE_TEST_DATA

namespace ref {

    struct GaussianRasterizationSettings {
        int image_height;
        int image_width;
        float tanfovx;
        float tanfovy;
        torch::Tensor bg;
        float scale_modifier;
        torch::Tensor viewmatrix;
        torch::Tensor projmatrix;
        int sh_degree;
        torch::Tensor camera_center;
    };

    torch::autograd::tensor_list rasterize_gaussians(torch::Tensor means3D,
                                                     torch::Tensor means2D,
                                                     torch::Tensor sh,
                                                     torch::Tensor colors_precomp,
                                                     torch::Tensor opacities,
                                                     torch::Tensor scales,
                                                     torch::Tensor rotations,
                                                     torch::Tensor cov3Ds_precomp,
                                                     GaussianRasterizationSettings raster_settings);

    class _RasterizeGaussians : public torch::autograd::Function<_RasterizeGaussians> {
    public:
        static torch::autograd::tensor_list forward(torch::autograd::AutogradContext* ctx,
                                                    torch::Tensor means3D,
                                                    torch::Tensor means2D,
                                                    torch::Tensor sh,
                                                    torch::Tensor colors_precomp,
                                                    torch::Tensor opacities,
                                                    torch::Tensor scales,
                                                    torch::Tensor rotations,
                                                    torch::Tensor cov3Ds_precomp,
                                                    torch::Tensor image_height,
                                                    torch::Tensor image_width,
                                                    torch::Tensor tanfovx,
                                                    torch::Tensor tanfovy,
                                                    torch::Tensor bg,
                                                    torch::Tensor scale_modifier,
                                                    torch::Tensor viewmatrix,
                                                    torch::Tensor projmatrix,
                                                    torch::Tensor sh_degree,
                                                    torch::Tensor camera_center) {

            int image_height_val = image_height.item<int>();
            int image_width_val = image_width.item<int>();
            float tanfovx_val = tanfovx.item<float>();
            float tanfovy_val = tanfovy.item<float>();
            float scale_modifier_val = scale_modifier.item<float>();
            int sh_degree_val = sh_degree.item<int>();

            // TODO: should it be this way? Bug?
            camera_center = camera_center.contiguous();

            auto [num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer] = RasterizeGaussiansCUDA(
                bg,
                means3D,
                colors_precomp,
                opacities,
                scales,
                rotations,
                scale_modifier_val,
                cov3Ds_precomp,
                viewmatrix,
                projmatrix,
                tanfovx_val,
                tanfovy_val,
                image_height_val,
                image_width_val,
                sh,
                sh_degree_val,
                camera_center,
                false,
                false);

            //            auto [num_rendered1, color1, radii1, geomBuffer1, binningBuffer1, imgBuffer1] = RasterizeGaussiansCUDA(
            //                bg,
            //                means3D,
            //                colors_precomp,
            //                opacities,
            //                scales,
            //                rotations,
            //                scale_modifier_val,
            //                cov3Ds_precomp,
            //                viewmatrix,
            //                projmatrix,
            //                tanfovx_val,
            //                tanfovy_val,
            //                image_height_val,
            //                image_width_val,
            //                sh,
            //                sh_degree_val,
            //                camera_center,
            //                false,
            //                false);

            //            imgBuffer = imgBuffer.to(torch::kCPU);
            //            imgBuffer1 = imgBuffer1.to(torch::kCPU);
            //            char* imgBufPtr = reinterpret_cast<char*>(imgBuffer.contiguous().data_ptr());
            //            char* imgBufPtr1 = reinterpret_cast<char*>(imgBuffer1.contiguous().data_ptr());
            //            auto imgState = CudaRasterizer::ImageState::fromChunk(imgBufPtr, image_height_val * image_width_val);
            //            auto imgState1 = CudaRasterizer::ImageState::fromChunk(imgBufPtr1, image_height_val * image_width_val);
            //
            //            int accu_alpha_diffs = 0;
            //            int n_contrib_diffs = 0;
            //            int ranges_diffs = 0;
            //
            //            int pixel_count = image_height_val * image_width_val;
            //            for (int i = 0; i < pixel_count; ++i) {
            //                if (imgState.accum_alpha[i] != imgState1.accum_alpha[i]) {
            //                    ++accu_alpha_diffs;
            //                }
            //                if (imgState.n_contrib[i] != imgState1.n_contrib[i]) {
            //                    ++n_contrib_diffs;
            //                }
            //                if (imgState.ranges[i].x != imgState1.ranges[i].x || imgState.ranges[i].y != imgState1.ranges[i].y) {
            //                    ++ranges_diffs;
            //                }
            //            }
            //
            //            imgBuffer = imgBuffer.to(torch::kCUDA);
            //            imgBuffer1 = imgBuffer1.to(torch::kCUDA);
            //
            //            std::cout << "accu_alpha_diffs: " << accu_alpha_diffs << std::endl;
            //            std::cout << "n_contrib_diffs: " << n_contrib_diffs << std::endl;
            //            std::cout << "ranges_diffs: " << ranges_diffs << std::endl;
            //
            //            std::cout << "============torch::autograd::tensor_list forward(torch::autograd::AutogradContext* ctx===========" << std::endl;
            //            if (num_rendered == num_rendered1) {
            //                std::cout << "num_rendered and num_rendered1 are equal!" << std::endl;
            //            } else {
            //                std::cout << "num_rendered and num_rendered1 are NOT equal!" << std::endl;
            //            }
            //
            //            if (torch::equal(color1, color)) {
            //                std::cout << "color1 and color are equal!" << std::endl;
            //            } else {
            //                std::cout << "color1 and color are NOT equal!" << std::endl;
            //            }
            //            if (torch::equal(radii1, radii)) {
            //                std::cout << "radii1 and radii are equal!" << std::endl;
            //            } else {
            //                std::cout << "radii1 and radii are NOT equal!" << std::endl;
            //            }
            //
            //            if (torch::equal(geomBuffer1, geomBuffer)) {
            //                std::cout << "geomBuffer1 and geomBuffer are equal!" << std::endl;
            //            } else {
            //                std::cout << "geomBuffer1 and geomBuffer are NOT equal!" << std::endl;
            //            }
            //
            //            if (torch::equal(binningBuffer1, binningBuffer)) {
            //                std::cout << "binningBuffer1 and binningBuffer are equal!" << std::endl;
            //            } else {
            //                std::cout << "binningBuffer1 and binningBuffer are NOT equal!" << std::endl;
            //            }
            //
            //            if (torch::equal(imgBuffer1, imgBuffer)) {
            //                std::cout << "imgBuffer1 and imgBuffer are equal!" << std::endl;
            //            } else {
            //                std::cout << "imgBuffer1 and imgBuffer are NOT equal!" << std::endl;
            //            }
            //            std::cout << "============torch::autograd::tensor_list forward(torch::autograd::AutogradContext* ctx===========" << std::endl;
            //
            //            torch::save(bg, "forward_bg.pt");
            //            torch::save(means3D, "forward_means3D.pt");
            //            torch::save(colors_precomp, "forward_colors_precomp.pt");
            //            torch::save(opacities, "forward_opacities.pt");
            //            torch::save(scales, "forward_scales.pt");
            //            torch::save(rotations, "forward_rotations.pt");
            //            torch::save(scale_modifier, "forward_scale_modifier_val.pt");
            //            torch::save(cov3Ds_precomp, "forward_cov3Ds_precomp.pt");
            //            torch::save(viewmatrix, "forward_viewmatrix.pt");
            //            torch::save(projmatrix, "forward_projmatrix.pt");
            //            torch::save(tanfovx, "forward_tanfovx_val.pt");
            //            torch::save(tanfovy, "forward_tanfovy_val.pt");
            //            torch::save(image_height, "forward_image_height_val.pt");
            //            torch::save(image_width, "forward_image_width_val.pt");
            //            torch::save(sh, "forward_sh.pt");
            //            torch::save(sh_degree, "forward_sh_degree_val.pt");
            //            torch::save(camera_center, "forward_camera_center.pt");
            //
            //            torch::save(torch::tensor(num_rendered, torch::dtype(torch::kInt32)), "out_forward_num_rendered.pt");
            //            torch::save(color, "out_forward_color.pt");
            //            torch::save(radii, "out_forward_radii.pt");
            //            torch::save(geomBuffer, "out_forward_geomBuffer.pt");
            //            torch::save(binningBuffer, "out_forward_binningBuffer.pt");
            //            torch::save(imgBuffer, "out_forward_imgBuffer.pt");

            ctx->save_for_backward({colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer});
            // TODO: Clean up. Too much data saved.
            ctx->saved_data["num_rendered"] = num_rendered;
            ctx->saved_data["background"] = bg;
            ctx->saved_data["scale_modifier"] = scale_modifier_val;
            ctx->saved_data["viewmatrix"] = viewmatrix;
            ctx->saved_data["projmatrix"] = projmatrix;
            ctx->saved_data["tanfovx"] = tanfovx_val;
            ctx->saved_data["tanfovy"] = tanfovy_val;
            ctx->saved_data["image_height"] = image_height_val;
            ctx->saved_data["image_width"] = image_width_val;
            ctx->saved_data["sh_degree"] = sh_degree_val;
            ctx->saved_data["camera_center"] = camera_center;
            return {color, radii};
        }

        static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs) {
            auto grad_out_color = grad_outputs[0];
            torch::save(grad_out_color, "grad_out_color.pt");

            auto num_rendered = ctx->saved_data["num_rendered"].to<int>();
            auto saved = ctx->get_saved_variables();
            auto colors_precomp = saved[0];
            auto means3D = saved[1];
            auto scales = saved[2];
            auto rotations = saved[3];
            auto cov3Ds_precomp = saved[4];
            auto radii = saved[5];
            auto sh = saved[6];
            auto geomBuffer = saved[7];
            auto binningBuffer = saved[8];
            auto imgBuffer = saved[9];

            //            torch::save(ctx->saved_data["background"].to<torch::Tensor>(), "background.pt");
            //            torch::save(torch::tensor(ctx->saved_data["scale_modifier"].to<float>(), torch::dtype(torch::kFloat32)), "scale_modifier.pt");
            //            torch::save(ctx->saved_data["viewmatrix"].to<torch::Tensor>(), "viewmatrix.pt");
            //            torch::save(ctx->saved_data["projmatrix"].to<torch::Tensor>(), "projmatrix.pt");
            //            torch::save(torch::tensor(ctx->saved_data["tanfovx"].to<float>(), torch::dtype(torch::kFloat32)), "tanfovx.pt");
            //            torch::save(torch::tensor(ctx->saved_data["tanfovy"].to<float>(), torch::dtype(torch::kFloat32)), "tanfovy.pt");
            //            torch::save(torch::tensor(ctx->saved_data["image_height"].to<int>(), torch::dtype(torch::kInt32)), "image_height.pt");
            //            torch::save(torch::tensor(ctx->saved_data["image_width"].to<int>(), torch::dtype(torch::kInt32)), "image_width.pt");
            //            torch::save(torch::tensor(ctx->saved_data["sh_degree"].to<int>(), torch::dtype(torch::kInt32)), "sh_degree.pt");
            //            torch::save(ctx->saved_data["camera_center"].to<torch::Tensor>(), "camera_center.pt");
            //            torch::save(means3D, "means3D.pt");
            //            torch::save(radii, "radii.pt");
            //            torch::save(colors_precomp, "colors_precomp.pt");
            //            torch::save(scales, "scales.pt");
            //            torch::save(rotations, "rotations.pt");
            //            torch::save(cov3Ds_precomp, "cov3Ds_precomp.pt");
            //            torch::save(sh, "sh.pt");
            //            torch::save(geomBuffer, "geomBuffer.pt");
            //            torch::save(binningBuffer, "binningBuffer.pt");
            //            torch::save(imgBuffer, "imgBuffer.pt");

#ifdef WRITE_TEST_DATA
            auto grad_out_color_copy = grad_out_color.clone();
            auto grad_out_radii_copy = grad_out_radii.clone();
            auto num_rendered_copy = num_rendered;
            auto colors_precomp_copy = colors_precomp.clone();
            auto means3D_copy = means3D.clone();
            auto scales_copy = scales.clone();
            auto rotations_copy = rotations.clone();
            auto cov3Ds_precomp_copy = cov3Ds_precomp.clone();
            auto radii_copy = radii.clone();
            auto sh_copy = sh.clone();
            auto geomBuffer_copy = geomBuffer.clone();
            auto binningBuffer_copy = binningBuffer.clone();
            auto imgBuffer_copy = imgBuffer.clone();
            auto background_copy = ctx->saved_data["background"].to<torch::Tensor>().clone();
            auto scale_modifier_copy = ctx->saved_data["scale_modifier"].to<float>();
            auto viewmatrix_copy = ctx->saved_data["viewmatrix"].to<torch::Tensor>();
            auto projmatrix_copy = ctx->saved_data["projmatrix"].to<torch::Tensor>();
            auto tanfovx_copy = ctx->saved_data["tanfovx"].to<float>();
            auto tanfovy_copy = ctx->saved_data["tanfovy"].to<float>();
            auto sh_degree_copy = ctx->saved_data["sh_degree"].to<int>();
            auto camera_center_copy = ctx->saved_data["camera_center"].to<torch::Tensor>();
#endif

            auto [grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations] = RasterizeGaussiansBackwardCUDA(
                ctx->saved_data["background"].to<torch::Tensor>(),
                means3D,
                radii,
                colors_precomp,
                scales,
                rotations,
                ctx->saved_data["scale_modifier"].to<float>(),
                cov3Ds_precomp,
                ctx->saved_data["viewmatrix"].to<torch::Tensor>(),
                ctx->saved_data["projmatrix"].to<torch::Tensor>(),
                ctx->saved_data["tanfovx"].to<float>(),
                ctx->saved_data["tanfovy"].to<float>(),
                grad_out_color,
                sh,
                ctx->saved_data["sh_degree"].to<int>(),
                ctx->saved_data["camera_center"].to<torch::Tensor>(),
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                false);

//            auto [grad_means2D1, grad_colors_precomp1, grad_opacities1, grad_means3D1, grad_cov3Ds_precomp1, grad_sh1, grad_scales1, grad_rotations1] = RasterizeGaussiansBackwardCUDA(
//                ctx->saved_data["background"].to<torch::Tensor>(),
//                means3D,
//                radii,
//                colors_precomp,
//                scales,
//                rotations,
//                ctx->saved_data["scale_modifier"].to<float>(),
//                cov3Ds_precomp,
//                ctx->saved_data["viewmatrix"].to<torch::Tensor>(),
//                ctx->saved_data["projmatrix"].to<torch::Tensor>(),
//                ctx->saved_data["tanfovx"].to<float>(),
//                ctx->saved_data["tanfovy"].to<float>(),
//                grad_out_color,
//                sh,
//                ctx->saved_data["sh_degree"].to<int>(),
//                ctx->saved_data["camera_center"].to<torch::Tensor>(),
//                geomBuffer,
//                num_rendered,
//                binningBuffer,
//                imgBuffer,
//                false);
#ifdef WRITE_TEST_DATA
            saveFunctionData("rasterize_backward_test_data.dat",
                             grad_means2D,
                             grad_colors_precomp,
                             grad_opacities,
                             grad_means3D,
                             grad_cov3Ds_precomp,
                             grad_sh,
                             grad_scales,
                             grad_rotations,
                             background_copy,
                             means3D_copy,
                             radii_copy,
                             colors_precomp_copy,
                             scales_copy,
                             rotations_copy,
                             scale_modifier_copy,
                             cov3Ds_precomp_copy,
                             viewmatrix_copy,
                             projmatrix_copy,
                             tanfovx_copy,
                             tanfovy_copy,
                             grad_out_color_copy,
                             sh_copy,
                             sh_degree_copy,
                             camera_center_copy,
                             geomBuffer_copy,
                             num_rendered_copy,
                             binningBuffer_copy,
                             imgBuffer_copy);
#endif

            //            if (torch::equal(grad_means3D, grad_means3D1)) {
            //                std::cout << "Backwardsref grad_means3D and grad_means3D1 are equal!" << std::endl;
            //            } else {
            //                std::cout << "Backwardsref grad_means3D and grad_means3D1 are NOT equal!" << std::endl;
            //            }
            //
            //            if (torch::equal(grad_means2D, grad_means2D1)) {
            //                std::cout << "Backwardsref grad_means2D and grad_means2D1 are equal!" << std::endl;
            //            } else {
            //                std::cout << "Backwardsref grad_means2D and grad_means2D1 are NOT equal!" << std::endl;
            //            }
            //
            //            if (torch::equal(grad_sh, grad_sh1)) {
            //                std::cout << "Backwardsref grad_sh and grad_sh1 are equal!" << std::endl;
            //            } else {
            //                std::cout << "Backwardsref grad_sh and grad_sh1 are NOT equal!" << std::endl;
            //            }
            //
            //            if (torch::equal(grad_colors_precomp, grad_colors_precomp1)) {
            //                std::cout << "Backwardsref grad_colors_precomp and grad_colors_precomp1 are equal!" << std::endl;
            //            } else {
            //                std::cout << "Backwardsref grad_colors_precomp and grad_colors_precomp1 are NOT equal!" << std::endl;
            //            }
            //
            //            if (torch::equal(grad_opacities, grad_opacities1)) {
            //                std::cout << "Backwardsref grad_opacities and grad_opacities1 are equal!" << std::endl;
            //            } else {
            //                std::cout << "Backwardsref grad_opacities and grad_opacities1 are NOT equal!" << std::endl;
            //            }
            //
            //            if (torch::equal(grad_scales, grad_scales1)) {
            //                std::cout << "Backwardsref grad_scales and grad_scales1 are equal!" << std::endl;
            //            } else {
            //                std::cout << "Backwardsref grad_scales and grad_scales1 are NOT equal!" << std::endl;
            //            }
            //
            //            if (torch::equal(grad_rotations, grad_rotations1)) {
            //                std::cout << "Backwardsref grad_rotations and grad_rotations1 are equal!" << std::endl;
            //            } else {
            //                std::cout << "Backwardsref grad_rotations and grad_rotations1 are NOT equal!" << std::endl;
            //            }
            //
            //            if (torch::equal(grad_cov3Ds_precomp, grad_cov3Ds_precomp1)) {
            //                std::cout << "Backwardsref grad_cov3Ds_precomp and grad_cov3Ds_precomp1 are equal!" << std::endl;
            //            } else {
            //                std::cout << "Backwardsref grad_cov3Ds_precomp and grad_cov3Ds_precomp1 are NOT equal!" << std::endl;
            //            }
            //
            //            torch::save(grad_means3D, "out_grad_means3D.pt");
            //            torch::save(grad_means2D, "out_grad_means2D.pt");
            //            torch::save(grad_sh, "out_grad_sh.pt");
            //            torch::save(grad_colors_precomp, "out_grad_colors_precomp.pt");
            //            torch::save(grad_opacities, "out_grad_opacities.pt");
            //            torch::save(grad_scales, "out_grad_scales.pt");
            //            torch::save(grad_rotations, "out_grad_rotations.pt");
            //            torch::save(grad_cov3Ds_precomp, "out_grad_cov3Ds_precomp.pt");

            // return gradients for all inputs, 19 in total. :D
            return {grad_means3D,
                    grad_means2D,
                    grad_sh,
                    grad_colors_precomp,
                    grad_opacities,
                    grad_scales,
                    grad_rotations,
                    grad_cov3Ds_precomp,
                    torch::Tensor(), // from here placeholder, not used: #forwards args = #backwards args.
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor(),
                    torch::Tensor()};
        }
    };

    class GaussianRasterizer : torch::nn::Module {
    public:
        GaussianRasterizer(GaussianRasterizationSettings raster_settings) : raster_settings_(raster_settings) {}

        torch::Tensor mark_visible(torch::Tensor positions) {
            torch::NoGradGuard no_grad;
            auto visible = markVisible(
                positions,
                raster_settings_.viewmatrix,
                raster_settings_.projmatrix);

            return visible;
        }

        std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor means3D,
                                                         torch::Tensor means2D,
                                                         torch::Tensor opacities,
                                                         torch::Tensor shs = torch::Tensor(),
                                                         torch::Tensor colors_precomp = torch::Tensor(),
                                                         torch::Tensor scales = torch::Tensor(),
                                                         torch::Tensor rotations = torch::Tensor(),
                                                         torch::Tensor cov3D_precomp = torch::Tensor()) {

            if ((shs.defined() && colors_precomp.defined()) || (!shs.defined() && !colors_precomp.defined())) {
                throw std::invalid_argument("Please provide exactly one of either SHs or precomputed colors!");
            }
            if (((scales.defined() || rotations.defined()) && cov3D_precomp.defined()) ||
                (!scales.defined() && !rotations.defined() && !cov3D_precomp.defined())) {
                throw std::invalid_argument("Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!");
            }

            // Check if tensors are undefined, and if so, initialize them
            torch::Device device = torch::kCUDA;
            if (!shs.defined()) {
                shs = torch::empty({0}, device);
            }
            if (!colors_precomp.defined()) {
                colors_precomp = torch::empty({0}, device);
            }
            if (!scales.defined()) {
                scales = torch::empty({0}, device);
            }
            if (!rotations.defined()) {
                rotations = torch::empty({0}, device);
            }
            if (!cov3D_precomp.defined()) {
                cov3D_precomp = torch::empty({0}, device);
            }

            auto result = rasterize_gaussians(
                means3D,
                means2D,
                shs,
                colors_precomp,
                opacities,
                scales,
                rotations,
                cov3D_precomp,
                raster_settings_);

            return {result[0], result[1]};
        }

    private:
        GaussianRasterizationSettings raster_settings_;
    };
} // namespace ref
