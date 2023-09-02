// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#include "gaussian_rasterizer.cuh"

#define WRITE_TEST_DATA
#undef WRITE_TEST_DATA

namespace gs {
    void rasterize_gaussians(GaussianRasterizer::RasterizerInput& raster_input,
                             GaussianRasterizer::RasterizerOutput& raster_ouput);

    GaussianRasterizer::Forward_Output GaussianRasterizer::Forward(torch::Tensor means3D,
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

        _raster_input.means3D = means3D;
        _raster_input.means2D = means2D;
        _raster_input.sh = shs;
        _raster_input.colors_precomp = colors_precomp;
        _raster_input.opacities = opacities;
        _raster_input.scales = scales;
        _raster_input.rotations = rotations;
        _raster_input.cov3Ds_precomp = cov3D_precomp;

        rasterize_gaussians(_raster_input, _raster_output);

        return {_raster_output.out_color, _raster_output.radii};
    }
    GaussianRasterizer::Backward_Output GaussianRasterizer::Backward(const torch::Tensor& grad_out_color) {

#ifdef WRITE_TEST_DATA
        auto grad_out_color_copy = grad_out_color.clone();
        auto grad_out_radii_copy = grad_out_radii.clone();
        auto num_rendered_copy = _raster_output.num_rendered;
        auto colors_precomp_copy = colors_precomp.clone();
        auto means3D_copy = means3D.clone();
        auto scales_copy = scales.clone();
        auto rotations_copy = rotations.clone();
        auto cov3Ds_precomp_copy = cov3Ds_precomp.clone();
        auto radii_copy = radii.clone();
        auto sh_copy = sh.clone();
        auto geomBuffer_copy = geomBuffer.clone();
        auto binningBuffer_copy = _raster_output..clone();
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
            _raster_input.bg,
            _raster_input.means3D,
            _raster_output.radii,
            _raster_input.colors_precomp,
            _raster_input.scales,
            _raster_input.rotations,
            _raster_input.scale_modifier,
            _raster_input.cov3Ds_precomp,
            _raster_input.viewmatrix,
            _raster_input.projmatrix,
            _raster_input.tanfovx,
            _raster_input.tanfovy,
            grad_out_color,
            _raster_input.sh,
            _raster_input.sh_degree,
            _raster_input.camera_center,
            _raster_output.geomBuffer,
            _raster_output.num_rendered,
            _raster_output.binningBuffer,
            _raster_output.imgBuffer,
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
        //        ts::print_debug_info(grad_means3D, "grad_means3D");
        //        ts::print_debug_info(grad_means2D, "grad_means2D");
        //        ts::print_debug_info(grad_sh, "grad_sh");
        //        ts::print_debug_info(grad_colors_precomp, "grad_colors_precomp");
        //        ts::print_debug_info(grad_opacities, "grad_opacities");
        //        ts::print_debug_info(grad_scales, "grad_scales");
        //        ts::print_debug_info(grad_rotations, "grad_rotations");
        //        ts::print_debug_info(grad_cov3Ds_precomp, "grad_cov3Ds_precomp");

        return {grad_means3D,
                grad_means2D,
                grad_sh,
                grad_colors_precomp,
                grad_opacities,
                grad_scales,
                grad_rotations,
                grad_cov3Ds_precomp};
    }

    void rasterize_gaussians(GaussianRasterizer::RasterizerInput& raster_input, GaussianRasterizer::RasterizerOutput& raster_ouput) {

        torch::Device device = torch::kCUDA;
        if (!raster_input.bg.is_cuda()) {
            raster_input.bg = raster_input.bg.to(device);
        }
        auto scale_modifier = torch::tensor(raster_input.scale_modifier, device);

        if (!raster_input.viewmatrix.is_cuda()) {
            raster_input.viewmatrix = raster_input.viewmatrix.to(device);
        }
        if (!raster_input.projmatrix.is_cuda()) {
            raster_input.projmatrix = raster_input.projmatrix.to(device);
        }
        auto sh_degree = torch::tensor(raster_input.sh_degree, device);

        if (!raster_input.camera_center.is_cuda()) {
            raster_input.camera_center = raster_input.camera_center.to(device);
        }
        auto prefiltered = torch::tensor(raster_input.prefiltered, device);

        raster_input.means2D = raster_input.means2D.to(device);
        raster_input.means3D = raster_input.means3D.to(device);
        raster_input.sh = raster_input.sh.to(device);
        raster_input.colors_precomp = raster_input.colors_precomp.to(device);
        raster_input.opacities = raster_input.opacities.to(device);
        raster_input.scales = raster_input.scales.to(device);
        raster_input.rotations = raster_input.rotations.to(device);
        raster_input.cov3Ds_precomp = raster_input.cov3Ds_precomp.to(device);
        raster_input.camera_center = raster_input.camera_center.contiguous();

        auto [num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer] = gs::rasterizer::RasterizeGaussiansCUDA(
            raster_input.bg,
            raster_input.means3D,
            raster_input.colors_precomp,
            raster_input.opacities,
            raster_input.scales,
            raster_input.rotations,
            raster_input.scale_modifier,
            raster_input.cov3Ds_precomp,
            raster_input.viewmatrix,
            raster_input.projmatrix,
            raster_input.tanfovx,
            raster_input.tanfovy,
            raster_input.image_height,
            raster_input.image_width,
            raster_input.sh,
            raster_input.sh_degree,
            raster_input.camera_center.contiguous(),
            raster_input.prefiltered,
            false);

        raster_ouput.num_rendered = num_rendered;
        raster_ouput.out_color = color;
        raster_ouput.radii = radii;
        raster_ouput.geomBuffer = geomBuffer;
        raster_ouput.binningBuffer = binningBuffer;
        raster_ouput.imgBuffer = imgBuffer;
    }
} // namespace gs
