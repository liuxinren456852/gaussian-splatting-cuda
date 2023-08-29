// Copyright (c) 2023 Janusch Patas.

#pragma once

#include "debug_utils.cuh"
#include "rasterize_points.cuh"
#include "serialization.h"

namespace gs {
    class GaussianRasterizer {
    public:
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
            torch::Tensor means3D;
            torch::Tensor means2D;
            torch::Tensor sh;
            torch::Tensor colors_precomp;
            torch::Tensor opacities;
            torch::Tensor scales;
            torch::Tensor rotations;
            torch::Tensor cov3Ds_precomp;
        };

        struct RasterizerOutput {
            int num_rendered;
            torch::Tensor out_color;
            torch::Tensor radii;
            torch::Tensor geomBuffer;
            torch::Tensor binningBuffer;
            torch::Tensor imgBuffer;
        };

        using Forward_Output = std::tuple<torch::Tensor, torch::Tensor>;

        using Backward_Output = std::tuple<torch::Tensor,
                                           torch::Tensor,
                                           torch::Tensor,
                                           torch::Tensor,
                                           torch::Tensor,
                                           torch::Tensor,
                                           torch::Tensor,
                                           torch::Tensor>;

    public:
        GaussianRasterizer() = default;

        void SetRasterizerInput(RasterizerInput raster_settings) {
            _raster_input = raster_settings;
        }

        Forward_Output Forward(torch::Tensor means3D,
                               torch::Tensor means2D,
                               torch::Tensor opacities,
                               torch::Tensor shs = torch::Tensor(),
                               torch::Tensor colors_precomp = torch::Tensor(),
                               torch::Tensor scales = torch::Tensor(),
                               torch::Tensor rotations = torch::Tensor(),
                               torch::Tensor cov3D_precomp = torch::Tensor());

        Backward_Output Backward(const torch::Tensor& grad_out_color);

    private:
        RasterizerInput _raster_input;
        RasterizerOutput _raster_output;
    };

} // namespace gs
