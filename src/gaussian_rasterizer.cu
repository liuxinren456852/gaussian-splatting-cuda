// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#include "gaussian_rasterizer.cuh"

#define WRITE_TEST_DATA
#undef WRITE_TEST_DATA

namespace gs {
    std::tuple<torch::Tensor, torch::Tensor> rasterize_gaussians(GaussianRasterizer::RasterizerInput raster_settings);

    GaussianRasterizer::Forward_Output GaussianRasterizer::forward(torch::Tensor means3D,
                                                                   torch::Tensor means2D,
                                                                   torch::Tensor opacities,
                                                                   torch::Tensor shs,
                                                                   torch::Tensor colors_precomp,
                                                                   torch::Tensor scales,
                                                                   torch::Tensor rotations,
                                                                   torch::Tensor cov3D_precomp) {

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

        raster_settings_.means3D = means3D;
        raster_settings_.means2D = means2D;
        raster_settings_.sh = shs;
        raster_settings_.colors_precomp = colors_precomp;
        raster_settings_.opacities = opacities;
        raster_settings_.scales = scales;
        raster_settings_.rotations = rotations;
        raster_settings_.cov3Ds_precomp = cov3D_precomp;

        auto [color, fh] = rasterize_gaussians(raster_settings_);

        return {color, fh};
    }
    GaussianRasterizer::Backward_Output GaussianRasterizer::backward(torch::autograd::AutogradContext* ctx,
                                                                     torch::autograd::tensor_list grad_outputs) {

        auto grad_out_color = grad_outputs[0];
        auto grad_out_radii = grad_outputs[1];

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

        auto [grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations] = gs::rasterizer::RasterizeGaussiansBackwardCUDA(
            raster_settings_.bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            raster_settings_.scale_modifier,
            cov3Ds_precomp,
            raster_settings_.viewmatrix,
            raster_settings_.projmatrix,
            raster_settings_.tanfovx,
            raster_settings_.tanfovy,
            grad_out_color,
            sh,
            raster_settings_.sh_degree,
            raster_settings_.camera_center,
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
        return {grad_means3D,
                grad_means2D,
                grad_sh,
                grad_colors_precomp,
                grad_opacities,
                grad_scales,
                grad_rotations,
                grad_cov3Ds_precomp};
    }

    std::tuple<torch::Tensor, torch::Tensor> rasterize_gaussians(GaussianRasterizer::RasterizerInput raster_settings) {

        torch::Device device = torch::kCUDA;
        if (!raster_settings.bg.is_cuda()) {
            raster_settings.bg = raster_settings.bg.to(device);
        }
        auto scale_modifier = torch::tensor(raster_settings.scale_modifier, device);

        if (!raster_settings.viewmatrix.is_cuda()) {
            raster_settings.viewmatrix = raster_settings.viewmatrix.to(device);
        }
        if (!raster_settings.projmatrix.is_cuda()) {
            raster_settings.projmatrix = raster_settings.projmatrix.to(device);
        }
        auto sh_degree = torch::tensor(raster_settings.sh_degree, device);

        if (!raster_settings.camera_center.is_cuda()) {
            raster_settings.camera_center = raster_settings.camera_center.to(device);
        }
        auto prefiltered = torch::tensor(raster_settings.prefiltered, device);

        raster_settings.means2D = raster_settings.means2D.to(device);
        raster_settings.means3D = raster_settings.means3D.to(device);
        raster_settings.sh = raster_settings.sh.to(device);
        raster_settings.colors_precomp = raster_settings.colors_precomp.to(device);
        raster_settings.opacities = raster_settings.opacities.to(device);
        raster_settings.scales = raster_settings.scales.to(device);
        raster_settings.rotations = raster_settings.rotations.to(device);
        raster_settings.cov3Ds_precomp = raster_settings.cov3Ds_precomp.to(device);

        auto [num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer] = gs::rasterizer::RasterizeGaussiansCUDA(
            raster_settings.bg,
            raster_settings.means3D,
            raster_settings.colors_precomp,
            raster_settings.opacities,
            raster_settings.scales,
            raster_settings.rotations,
            raster_settings.scale_modifier,
            raster_settings.cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.sh,
            raster_settings.sh_degree,
            raster_settings.camera_center.contiguous(),
            raster_settings.prefiltered,
            false);

        return {color, radii};
    }
} // namespace gs
