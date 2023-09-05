// Copyright (c) 2023 Janusch Patas.
#pragma once

#include "debug_utils.cuh"
#include "rasterize_points.cuh"
#include "rasterizer.cuh"
#include "serialization.h"
#include <utility>

//#define WRITE_TEST_DATA
#undef WRITE_TEST_DATA

namespace gs {
    struct RasterizerInput {
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

    struct SaveForBackward {
        torch::Tensor colors_precomp;
        torch::Tensor means3D;
        torch::Tensor scales;
        torch::Tensor rotations;
        torch::Tensor cov3Ds_precomp;
        torch::Tensor radii;
        torch::Tensor sh;
        torch::Tensor geomBuffer;
        torch::Tensor binningBuffer;
        torch::Tensor imgBuffer;
        int num_rendered;
        torch::Tensor bg;
        float scale_modifier;
        torch::Tensor viewmatrix;
        torch::Tensor projmatrix;
        float tanfovx;
        float tanfovy;
        int image_height;
        int image_width;
        int sh_degree;
        torch::Tensor camera_center;
    };

    std::tuple<torch::Tensor, torch::Tensor> rasterize_gaussians(
        SaveForBackward& saveForBackwars,
        torch::Tensor means3D,
        torch::Tensor sh,
        torch::Tensor colors_precomp,
        torch::Tensor opacities,
        torch::Tensor scales,
        torch::Tensor rotations,
        torch::Tensor cov3Ds_precomp,
        RasterizerInput& raster_settings);

    class _RasterizeGaussians {
    public:
        static std::tuple<torch::Tensor, torch::Tensor> Forward(
            SaveForBackward& saveForBackward,
            torch::Tensor means3D,
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

            //            torch::Tensor bg_fwd, means3D_fwd, colors_precomp_fwd, opacities_fwd, scales_fwd, rotations_fwd, cov3Ds_precomp_fwd, viewmatrix_fwd, projmatrix_fwd, sh_fwd, camera_center_fwd;
            //            float scale_modifier_val_fwd, tanfovx_val_fwd, tanfovy_val_fwd;
            //            int image_height_val_fwd, image_width_val_fwd, sh_degree_val_fwd;
            //
            //            // Load tensors from disk
            //            torch::load(bg_fwd, "forward_bg.pt");
            //            torch::load(means3D_fwd, "forward_means3D.pt");
            //            torch::load(colors_precomp_fwd, "forward_colors_precomp.pt");
            //            torch::load(opacities_fwd, "forward_opacities.pt");
            //            torch::load(scales_fwd, "forward_scales.pt");
            //            torch::load(rotations_fwd, "forward_rotations.pt");
            //
            //            torch::Tensor tmp;
            //            torch::load(tmp, "forward_scale_modifier_val.pt");
            //            scale_modifier_val_fwd = tmp.item<float>();
            //
            //            torch::load(cov3Ds_precomp_fwd, "forward_cov3Ds_precomp.pt");
            //            torch::load(viewmatrix_fwd, "forward_viewmatrix.pt");
            //            torch::load(projmatrix_fwd, "forward_projmatrix.pt");
            //
            //            torch::load(tmp, "forward_tanfovx_val.pt");
            //            tanfovx_val_fwd = tmp.item<float>();
            //
            //            torch::load(tmp, "forward_tanfovy_val.pt");
            //            tanfovy_val_fwd = tmp.item<float>();
            //
            //            auto height = torch::tensor({});
            //            auto width = torch::tensor({});
            //            torch::load(height, "forward_image_height_val.pt");
            //            image_height_val_fwd = height.item<int>();
            //
            //            torch::load(width, "forward_image_width_val.pt");
            //            image_width_val_fwd = width.item<int>();
            //
            //            torch::load(sh_fwd, "forward_sh.pt");
            //
            //            torch::load(tmp, "forward_sh_degree_val.pt");
            //            sh_degree_val_fwd = tmp.item<int>();
            //
            //            torch::load(camera_center_fwd, "forward_camera_center.pt");
            //
            //            if (!torch::equal(bg_fwd, bg)) {
            //                std::cout << "bg_fwd and bg are NOT equal!" << std::endl;
            //            } else {
            //                std::cout << "bg_fwd and bg are equal!" << std::endl;
            //            }
            //            if (!torch::equal(means3D_fwd, means3D)) {
            //                std::cout << "means3D_fwd and means3D are NOT equal!" << std::endl;
            //            } else {
            //                std::cout << "means3D_fwd and means3D are equal!" << std::endl;
            //            }
            //            if (!torch::equal(colors_precomp_fwd, colors_precomp)) {
            //                std::cout << "colors_precomp_fwd and colors_precomp are NOT equal!" << std::endl;
            //            } else {
            //                std::cout << "colors_precomp_fwd and colors_precomp are equal!" << std::endl;
            //            }
            //            if (!torch::equal(opacities_fwd, opacities)) {
            //                std::cout << "opacities_fwd and opacities are NOT equal!" << std::endl;
            //            } else {
            //                std::cout << "opacities_fwd and opacities are equal!" << std::endl;
            //            }
            //            if (!torch::equal(scales_fwd, scales)) {
            //                std::cout << "scales_fwd and scales are NOT equal!" << std::endl;
            //            } else {
            //                std::cout << "scales_fwd and scales are equal!" << std::endl;
            //            }
            //            if (!torch::equal(rotations_fwd, rotations)) {
            //                std::cout << "rotations_fwd and rotations are NOT equal!" << std::endl;
            //            } else {
            //                std::cout << "rotations_fwd and rotations are equal!" << std::endl;
            //            }
            //            if (scale_modifier_val_fwd != scale_modifier_val) {
            //                std::cout << "scale_modifier_val_fwd and scale_modifier_val are NOT equal!" << std::endl;
            //            } else {
            //                std::cout << "scale_modifier_val_fwd and scale_modifier_val are equal!" << std::endl;
            //            }
            //            if (!torch::equal(cov3Ds_precomp_fwd, cov3Ds_precomp)) {
            //                std::cout << "cov3Ds_precomp_fwd and cov3Ds_precomp are NOT equal!" << std::endl;
            //            } else {
            //                std::cout << "cov3Ds_precomp_fwd and cov3Ds_precomp are equal!" << std::endl;
            //            }
            //            if (!torch::equal(viewmatrix_fwd, viewmatrix)) {
            //                std::cout << "viewmatrix_fwd and viewmatrix are NOT equal!" << std::endl;
            //            } else {
            //                std::cout << "viewmatrix_fwd and viewmatrix are equal!" << std::endl;
            //            }
            //            if (!torch::equal(projmatrix_fwd, projmatrix)) {
            //                std::cout << "projmatrix_fwd and projmatrix are NOT equal!" << std::endl;
            //            } else {
            //                std::cout << "projmatrix_fwd and projmatrix are equal!" << std::endl;
            //            }
            //            if (tanfovx_val_fwd != tanfovx_val) {
            //                std::cout << "tanfovx_val_fwd and tanfovx_val are NOT equal!" << std::endl;
            //            } else {
            //                std::cout << "tanfovx_val_fwd and tanfovx_val are equal!" << std::endl;
            //            }
            //            if (tanfovy_val_fwd != tanfovy_val) {
            //                std::cout << "tanfovy_val_fwd and tanfovy_val are NOT equal!" << std::endl;
            //            } else {
            //                std::cout << "tanfovy_val_fwd and tanfovy_val are equal!" << std::endl;
            //            }
            //            if (image_height_val_fwd != image_height_val) {
            //                std::cout << "image_height_val_fwd: " << image_height_val_fwd
            //                          << " and image_height_val: " << image_height_val << " are NOT equal!" << std::endl;
            //            } else {
            //                std::cout << "image_height_val_fwd and image_height_val are equal!" << std::endl;
            //            }
            //            if (image_width_val_fwd != image_width_val) {
            //                std::cout << "image_width_val_fwd: " << image_width_val_fwd
            //                          << " and image_width_val: " << image_width_val << " are NOT equal!" << std::endl;
            //                std::cout << "image_width_val_fwd and image_width_val are NOT equal!" << std::endl;
            //            } else {
            //                std::cout << "image_width_val_fwd and image_width_val are equal!" << std::endl;
            //            }
            //            if (!torch::equal(sh_fwd, sh)) {
            //                std::cout << "sh_fwd and sh are NOT equal!" << std::endl;
            //            } else {
            //                std::cout << "sh_fwd and sh are equal!" << std::endl;
            //            }
            //            if (sh_degree_val_fwd != sh_degree_val) {
            //                std::cout << "sh_degree_val_fwd and sh_degree_val are NOT equal!" << std::endl;
            //            } else {
            //                std::cout << "sh_degree_val_fwd and sh_degree_val are equal!" << std::endl;
            //            }
            //            if (!torch::equal(camera_center_fwd, camera_center)) {
            //                std::cout << "camera_center_fwd and camera_center are NOT equal!" << std::endl;
            //            } else {
            //                std::cout << "camera_center_fwd and camera_center are equal!" << std::endl;
            //            }

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
            //
            //            std::cout << "====================================================================" << std::endl;
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
            //            std::cout << "====================================================================" << std::endl;
            //
            //            torch::Tensor n_rendered = torch::tensor({}, torch::dtype(torch::kInt32));
            //            torch::load(n_rendered, "out_forward_num_rendered.pt");
            //            if (n_rendered.item<int>() != num_rendered) {
            //                std::cout << "num_rendered: " << num_rendered << " and tmp.item<int>(): " << n_rendered.item<int>() << " are NOT equal!" << std::endl;
            //            } else {
            //                std::cout << "num_rendered and tmp.item<int> are equal!" << std::endl;
            //            }
            //            torch::Tensor col = torch::tensor({}, torch::dtype(torch::kFloat));
            //            torch::load(col, "out_forward_color.pt");
            //            if (torch::equal(col, color)) {
            //                std::cout << "col and color are equal!" << std::endl;
            //            } else {
            //                std::cout << "col and color are NOT equal!" << std::endl;
            //            }
            //            torch::Tensor r = torch::tensor({}, torch::dtype(torch::kFloat));
            //            torch::load(r, "out_forward_radii.pt");
            //            if (torch::equal(r, radii)) {
            //                std::cout << "r and radii are equal!" << std::endl;
            //            } else {
            //                std::cout << "r and radii are NOT equal!" << std::endl;
            //            }
            //            torch::Tensor geo_buf = torch::tensor({}, torch::dtype(torch::kByte));
            //            torch::load(geo_buf, "out_forward_geomBuffer.pt");
            //            if (torch::equal(geo_buf, geomBuffer)) {
            //                std::cout << "geo_buf and geomBuffer are equal!" << std::endl;
            //            } else {
            //                std::cout << "geo_buf and geomBuffer are NOT equal!" << std::endl;
            //            }
            //
            //            torch::Tensor bin_buf = torch::tensor({}, torch::dtype(torch::kByte));
            //            torch::load(bin_buf, "out_forward_binningBuffer.pt");
            //            if (torch::equal(bin_buf, binningBuffer)) {
            //                std::cout << "bin_buf and binningBuffer are equal!" << std::endl;
            //            } else {
            //                std::cout << "bin_buf and binningBuffer are NOT equal!" << std::endl;
            //            }
            //            torch::Tensor img_buf = torch::tensor({}, torch::dtype(torch::kByte));
            //            torch::load(img_buf, "out_forward_imgBuffer.pt");
            //            if (torch::equal(img_buf, imgBuffer)) {
            //                std::cout << "img_buf and imgBuffer are equal!" << std::endl;
            //            } else {
            //                std::cout << "img_buf and imgBuffer are NOT equal!" << std::endl;
            //            }
            //
            //            // Compute absolute differences
            //            torch::Tensor diff = torch::abs(img_buf - imgBuffer);
            //
            //            // Define a tolerance
            //            float tol = 1e-10;
            //
            //            // Find elements that differ more than the tolerance
            //            torch::Tensor large_diff_mask = diff > tol;
            //
            //            // Count the number of large differences
            //            int64_t num_large_diff = torch::sum(large_diff_mask).item<int64_t>();
            //
            //            // Print some of the differences
            //            if (num_large_diff > 0) {
            //                std::cout << "Number of large differences: " << num_large_diff << std::endl;
            //
            //                // Extract and print the indices and values of the first few large differences
            //                torch::Tensor large_diff_indices = torch::nonzero(large_diff_mask);
            //                torch::Tensor large_diff_values = torch::masked_select(diff, large_diff_mask);
            //
            //                int max_to_print = 10; // Change this to print more or fewer values
            //                for (int i = 0; i < std::min(max_to_print, static_cast<int>(num_large_diff)); ++i) {
            //                    int64_t index = large_diff_indices[i].item<int64_t>();
            //                    float value = large_diff_values[i].item<float>();
            //                    std::cout << "Index: " << index << ", Difference: " << value << std::endl;
            //                }
            //            } else {
            //                std::cout << "All differences are within tolerance." << std::endl;
            //            }
            cudaDeviceSynchronize();
            saveForBackward.colors_precomp = colors_precomp;
            saveForBackward.means3D = means3D;
            saveForBackward.scales = scales;
            saveForBackward.rotations = rotations;
            saveForBackward.cov3Ds_precomp = cov3Ds_precomp;
            saveForBackward.radii = radii;
            saveForBackward.sh = sh;
            saveForBackward.geomBuffer = geomBuffer;
            saveForBackward.binningBuffer = binningBuffer;
            saveForBackward.imgBuffer = imgBuffer;
            saveForBackward.num_rendered = num_rendered;
            saveForBackward.bg = bg;
            saveForBackward.scale_modifier = scale_modifier_val;
            saveForBackward.viewmatrix = viewmatrix;
            saveForBackward.projmatrix = projmatrix;
            saveForBackward.tanfovx = tanfovx_val;
            saveForBackward.tanfovy = tanfovy_val;
            saveForBackward.image_height = image_height_val;
            saveForBackward.image_width = image_width_val;
            saveForBackward.sh_degree = sh_degree_val;
            saveForBackward.camera_center = camera_center;

            //            torch::save(imgBuffer, "imgBuffer_rasterizer_cuh.pt");
            //            torch::save(binningBuffer, "binningBuffer_rasterizer_cuh.pt");

            return {color, radii};
        }

        static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Backward(SaveForBackward saveForBackward, torch::Tensor g_color) {

#ifdef WRITE_TEST_DATA
            auto grad_out_color_copy = grad_out_color.clone();
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

            //            auto dl_color = torch::zeros_like(g_color, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            //            torch::load(dl_color, "grad_out_color.pt");
            //            if (torch::equal(dl_color, g_color)) {
            //                std::cout << "\n================dl_color and g_color are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================dl_color and g_color are NOT equal!====================" << std::endl;
            //            }
            //            torch::Tensor background, scale_modifier, viewmatrix, projmatrix, tanfovx, tanfovy;
            //            torch::Tensor image_height, image_width, sh_degree, camera_center;
            //            torch::Tensor means3D, radii, colors_precomp, scales, rotations, cov3Ds_precomp, sh, geomBuffer, binningBuffer, imgBuffer;
            //
            //            // Load the saved tensors
            //            torch::load(background, "background.pt");
            //            torch::load(scale_modifier, "scale_modifier.pt");
            //            torch::load(viewmatrix, "viewmatrix.pt");
            //            torch::load(projmatrix, "projmatrix.pt");
            //            torch::load(tanfovx, "tanfovx.pt");
            //            torch::load(tanfovy, "tanfovy.pt");
            //            torch::load(image_height, "image_height.pt");
            //            torch::load(image_width, "image_width.pt");
            //            torch::load(sh_degree, "sh_degree.pt");
            //            torch::load(camera_center, "camera_center.pt");
            //            torch::load(means3D, "means3D.pt");
            //            torch::load(radii, "radii.pt");
            //            torch::load(colors_precomp, "colors_precomp.pt");
            //            torch::load(scales, "scales.pt");
            //            torch::load(rotations, "rotations.pt");
            //            torch::load(cov3Ds_precomp, "cov3Ds_precomp.pt");
            //            torch::load(sh, "sh.pt");
            //            torch::load(geomBuffer, "geomBuffer.pt");
            //            torch::load(binningBuffer, "binningBuffer.pt");
            //            torch::load(imgBuffer, "imgBuffer.pt");
            //
            //            torch::Tensor imgBuffer_rasterizer_cuh, binningBuffer_rasterizer_cuh;
            //            torch::load(imgBuffer_rasterizer_cuh, "imgBuffer_rasterizer_cuh.pt");
            //            torch::load(binningBuffer_rasterizer_cuh, "binningBuffer_rasterizer_cuh.pt");
            //            if (torch::equal(background, saveForBackward.bg)) {
            //                std::cout << "\n================background and saveForBackward.bg are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================background and saveForBackward.bg are NOT equal!====================" << std::endl;
            //            }
            //            if (scale_modifier.item<float>() == saveForBackward.scale_modifier) {
            //                std::cout << "\n================scale_modifier and saveForBackward.scale_modifier are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================scale_modifier and saveForBackward.scale_modifier are NOT equal!====================" << std::endl;
            //            }
            //            if (torch::equal(viewmatrix, saveForBackward.viewmatrix)) {
            //                std::cout << "\n================viewmatrix and saveForBackward.viewmatrix are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================viewmatrix and saveForBackward.viewmatrix are NOT equal!====================" << std::endl;
            //            }
            //            if (torch::equal(projmatrix, saveForBackward.projmatrix)) {
            //                std::cout << "\n================projmatrix and saveForBackward.projmatrix are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================projmatrix and saveForBackward.projmatrix are NOT equal!====================" << std::endl;
            //            }
            //            if (tanfovx.item<float>() == saveForBackward.tanfovx) {
            //                std::cout << "\n================tanfovx and saveForBackward.tanfovx are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================tanfovx and saveForBackward.tanfovx are NOT equal!====================" << std::endl;
            //            }
            //            if (tanfovy.item<float>() == saveForBackward.tanfovy) {
            //                std::cout << "\n================tanfovy and saveForBackward.tanfovy are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================tanfovy and saveForBackward.tanfovy are NOT equal!====================" << std::endl;
            //            }
            //            if (image_height.item<int>() == saveForBackward.image_height) {
            //                std::cout << "\n================image_height and saveForBackward.image_height are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================image_height and saveForBackward.image_height are NOT equal!====================" << std::endl;
            //            }
            //            if (image_width.item<int>() == saveForBackward.image_width) {
            //                std::cout << "\n================image_width and saveForBackward.image_width are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================image_width and saveForBackward.image_width are NOT equal!====================" << std::endl;
            //            }
            //            if (sh_degree.item<int>() == saveForBackward.sh_degree) {
            //                std::cout << "\n================sh_degree and saveForBackward.sh_degree are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================sh_degree and saveForBackward.sh_degree are NOT equal!====================" << std::endl;
            //                std::cout << "sh_degree.item<int>() = " << sh_degree.item<int>() << std::endl;
            //                std::cout << "saveForBackward.sh_degree = " << saveForBackward.sh_degree << std::endl;
            //            }
            //            if (torch::equal(camera_center, saveForBackward.camera_center)) {
            //                std::cout << "\n================camera_center and saveForBackward.camera_center are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================camera_center and saveForBackward.camera_center are NOT equal!====================" << std::endl;
            //            }
            //            if (torch::equal(means3D, saveForBackward.means3D)) {
            //                std::cout << "\n================means3D and saveForBackward.means3D are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================means3D and saveForBackward.means3D are NOT equal!====================" << std::endl;
            //            }
            //            if (torch::equal(radii, saveForBackward.radii)) {
            //                std::cout << "\n================radii and saveForBackward.radii are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================radii and saveForBackward.radii are NOT equal!====================" << std::endl;
            //            }
            //            if (torch::equal(colors_precomp, saveForBackward.colors_precomp)) {
            //                std::cout << "\n================colors_precomp and saveForBackward.colors_precomp are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================colors_precomp and saveForBackward.colors_precomp are NOT equal!====================" << std::endl;
            //            }
            //            if (torch::equal(scales, saveForBackward.scales)) {
            //                std::cout << "\n================scales and saveForBackward.scales are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================scales and saveForBackward.scales are NOT equal!====================" << std::endl;
            //            }
            //            if (torch::equal(rotations, saveForBackward.rotations)) {
            //                std::cout << "\n================rotations and saveForBackward.rotations are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================rotations and saveForBackward.rotations are NOT equal!====================" << std::endl;
            //            }
            //            if (torch::equal(cov3Ds_precomp, saveForBackward.cov3Ds_precomp)) {
            //                std::cout << "\n================cov3Ds_precomp and saveForBackward.cov3Ds_precomp are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================cov3Ds_precomp and saveForBackward.cov3Ds_precomp are NOT equal!====================" << std::endl;
            //            }
            //            if (torch::equal(sh, saveForBackward.sh)) {
            //                std::cout << "\n================sh and saveForBackward.sh are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================sh and saveForBackward.sh are NOT equal!====================" << std::endl;
            //            }
            //            if (torch::equal(geomBuffer, saveForBackward.geomBuffer)) {
            //                std::cout << "\n================geomBuffer and saveForBackward.geomBuffer are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================geomBuffer and saveForBackward.geomBuffer are NOT equal!====================" << std::endl;
            //            }
            //            if (torch::equal(binningBuffer, saveForBackward.binningBuffer)) {
            //                ts::print_debug_info(binningBuffer, "binningBuffer");
            //                ts::print_debug_info(saveForBackward.binningBuffer, "saveForBackward.binningBuffer");
            //                std::cout << "\n================binningBuffer and saveForBackward.binningBuffer are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================binningBuffer and saveForBackward.binningBuffer are NOT equal!====================" << std::endl;
            //            }
            //            if (torch::equal(imgBuffer, saveForBackward.imgBuffer)) {
            //                std::cout << "\n================imgBuffer and saveForBackward.imgBuffer are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                ts::print_debug_info(imgBuffer, "imgBuffer");
            //                ts::print_debug_info(saveForBackward.imgBuffer, "saveForBackward.imgBuffer");
            //                std::cout << "=================imgBuffer and saveForBackward.imgBuffer are NOT equal!====================" << std::endl;
            //            }

            auto [grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations] = RasterizeGaussiansBackwardCUDA(
                saveForBackward.bg,
                saveForBackward.means3D,
                saveForBackward.radii,
                saveForBackward.colors_precomp,
                saveForBackward.scales,
                saveForBackward.rotations,
                saveForBackward.scale_modifier,
                saveForBackward.cov3Ds_precomp,
                saveForBackward.viewmatrix,
                saveForBackward.projmatrix,
                saveForBackward.tanfovx,
                saveForBackward.tanfovy,
                g_color,
                saveForBackward.sh,
                saveForBackward.sh_degree,
                saveForBackward.camera_center,
                saveForBackward.geomBuffer,
                saveForBackward.num_rendered,
                saveForBackward.binningBuffer,
                saveForBackward.imgBuffer,
                false);

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

            //            torch::Tensor ref_grad_means3D = torch::tensor({});
            //            torch::Tensor ref_grad_means2D = torch::tensor({});
            //            torch::Tensor ref_grad_sh = torch::tensor({});
            //            torch::Tensor ref_grad_colors_precomp = torch::tensor({});
            //            torch::Tensor ref_grad_opacities = torch::tensor({});
            //            torch::Tensor ref_grad_scales = torch::tensor({});
            //            torch::Tensor ref_grad_rotations = torch::tensor({});
            //            torch::Tensor ref_grad_cov3Ds_precomp = torch::tensor({});
            //
            //            torch::load(ref_grad_means3D, "out_grad_means3D.pt");
            //            if (torch::equal(ref_grad_means3D, grad_means3D)) {
            //                std::cout << "\n================ref_grad_means3D and grad_means3D are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================ref_grad_means3D and grad_means3D are NOT equal!====================" << std::endl;
            //            }
            //            torch::load(ref_grad_means2D, "out_grad_means2D.pt");
            //            if (torch::equal(ref_grad_means2D, grad_means2D)) {
            //                std::cout << "\n================ref_grad_means2D and grad_means2D are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================ref_grad_means2D and grad_means2D are NOT equal!====================" << std::endl;
            //            }
            //            torch::load(ref_grad_sh, "out_grad_sh.pt");
            //            if (torch::equal(ref_grad_sh, grad_sh)) {
            //                std::cout << "\n================ref_grad_sh and grad_sh are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================ref_grad_sh and grad_sh are NOT equal!====================" << std::endl;
            //            }
            //            torch::load(ref_grad_colors_precomp, "out_grad_colors_precomp.pt");
            //            if (torch::equal(ref_grad_colors_precomp, grad_colors_precomp)) {
            //                std::cout << "\n================ref_grad_colors_precomp and grad_colors_precomp are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================ref_grad_colors_precomp and grad_colors_precomp are NOT equal!====================" << std::endl;
            //            }
            //            torch::load(ref_grad_opacities, "out_grad_opacities.pt");
            //            if (torch::equal(ref_grad_opacities, grad_opacities)) {
            //                std::cout << "\n================ref_grad_opacities and grad_opacities are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================ref_grad_opacities and grad_opacities are NOT equal!====================" << std::endl;
            //            }
            //            torch::load(ref_grad_scales, "out_grad_scales.pt");
            //            if (torch::equal(ref_grad_scales, grad_scales)) {
            //                std::cout << "\n================ref_grad_scales and grad_scales are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================ref_grad_scales and grad_scales are NOT equal!====================" << std::endl;
            //            }
            //            torch::load(ref_grad_rotations, "out_grad_rotations.pt");
            //            if (torch::equal(ref_grad_rotations, grad_rotations)) {
            //                std::cout << "\n================ref_grad_rotations and grad_rotations are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================ref_grad_rotations and grad_rotations are NOT equal!====================" << std::endl;
            //            }
            //            torch::load(ref_grad_cov3Ds_precomp, "out_grad_cov3Ds_precomp.pt");
            //            if (torch::equal(ref_grad_cov3Ds_precomp, grad_cov3Ds_precomp)) {
            //                std::cout << "\n================ref_grad_cov3Ds_precomp and grad_cov3Ds_precomp are equal!================\n"
            //                          << std::endl;
            //            } else {
            //                std::cout << "=================ref_grad_cov3Ds_precomp and grad_cov3Ds_precomp are NOT equal!====================" << std::endl;
            //            }

            // return gradients for all inputs, 19 in total. :D
            return {grad_means3D,
                    grad_means2D.clone(),
                    grad_sh,
                    grad_colors_precomp,
                    grad_opacities,
                    grad_scales,
                    grad_rotations,
                    grad_cov3Ds_precomp};
        }
    };

    class GaussianRasterizer {
    public:
        void SetRasterizerInput(RasterizerInput raster_settings) { raster_settings_ = raster_settings; }
        torch::Tensor mark_visible(torch::Tensor positions) {
            auto visible = markVisible(
                positions,
                raster_settings_.viewmatrix,
                raster_settings_.projmatrix);

            return visible;
        }

        std::tuple<torch::Tensor, torch::Tensor> Forward_RG(SaveForBackward& saveForBackwars,
                                                            torch::Tensor means3D,
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

            auto [color, radii] = gs::rasterize_gaussians(
                saveForBackwars,
                means3D,
                shs,
                colors_precomp,
                opacities,
                scales,
                rotations,
                cov3D_precomp,
                raster_settings_);

            return {color, radii};
        }

    private:
        RasterizerInput raster_settings_;
    };
} // namespace gs
