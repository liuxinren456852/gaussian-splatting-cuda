// Copyright (c) 2023 Janusch Patas.
#pragma once

#include "debug_utils.cuh"
#include "rasterize_points.cuh"
#include "rasterizer.cuh"
#include "serialization.h"
#include <utility>

#define WRITE_TEST_DATA
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
        bool prefiltered;
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
        bool prefiltered;
    };

    std::tuple<torch::Tensor, torch::Tensor> rasterize_gaussians(
        SaveForBackward& saveForBackwars,
        torch::Tensor means3D,
        torch::Tensor means2D,
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
            torch::Tensor camera_center,
            torch::Tensor prefiltered) {

            int image_height_val = image_height.item<int>();
            int image_width_val = image_width.item<int>();
            float tanfovx_val = tanfovx.item<float>();
            float tanfovy_val = tanfovy.item<float>();
            float scale_modifier_val = scale_modifier.item<float>();
            int sh_degree_val = sh_degree.item<int>();
            bool prefiltered_val = prefiltered.item<bool>();

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
                prefiltered_val,
                false);

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
            saveForBackward.prefiltered = prefiltered_val;

            return {color, radii};
        }

        static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> Backward(SaveForBackward saveForBackward, torch::Tensor g_color) {
            auto num_rendered = saveForBackward.num_rendered;
            auto colors_precomp = saveForBackward.colors_precomp;
            auto means3D = saveForBackward.means3D;
            auto scales = saveForBackward.scales;
            auto rotations = saveForBackward.rotations;
            auto cov3Ds_precomp = saveForBackward.cov3Ds_precomp;
            auto radii = saveForBackward.radii;
            auto sh = saveForBackward.sh;
            auto geomBuffer = saveForBackward.geomBuffer;
            auto binningBuffer = saveForBackward.binningBuffer;
            auto imgBuffer = saveForBackward.imgBuffer;

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

            auto [grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations] = RasterizeGaussiansBackwardCUDA(
                saveForBackward.bg,
                means3D,
                radii,
                colors_precomp,
                scales,
                rotations,
                saveForBackward.scale_modifier,
                cov3Ds_precomp,
                saveForBackward.viewmatrix,
                saveForBackward.projmatrix,
                saveForBackward.tanfovx,
                saveForBackward.tanfovy,
                g_color,
                sh,
                saveForBackward.sh_degree,
                saveForBackward.camera_center,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
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
            // return gradients for all inputs, 19 in total. :D
            return {grad_means3D,
                    grad_means2D,
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
            torch::NoGradGuard no_grad;
            auto visible = markVisible(
                positions,
                raster_settings_.viewmatrix,
                raster_settings_.projmatrix);

            return visible;
        }

        std::tuple<torch::Tensor, torch::Tensor> Forward_RG(SaveForBackward& saveForBackwars,
                                                            torch::Tensor means3D,
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

            auto [color, radii] = gs::rasterize_gaussians(
                saveForBackwars,
                means3D,
                means2D,
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
