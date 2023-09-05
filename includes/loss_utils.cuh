// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.
#pragma once
#include "debug_utils.cuh"
#include <cmath>
#include <torch/torch.h>

namespace gs {
    namespace loss {
        std::pair<torch::Tensor, torch::Tensor> l1_loss(const torch::Tensor& network_output, const torch::Tensor& gt) {
            auto L1l = torch::abs((network_output - gt)).mean();
            auto dL_l1_loss = torch::sign(network_output - gt) / static_cast<float>(network_output.numel());
            return {L1l, dL_l1_loss};
        }

        // 1D Gaussian kernel
        torch::Tensor gaussian(int window_size, float sigma) {
            torch::Tensor gauss = torch::empty(window_size);
            for (int x = 0; x < window_size; ++x) {
                gauss[x] = std::exp(-(std::pow(std::floor(static_cast<float>(x - window_size) / 2.f), 2)) / (2.f * sigma * sigma));
            }
            return gauss / gauss.sum();
        }

        torch::Tensor create_window(int window_size, int channel) {
            auto _1D_window = gaussian(window_size, 1.5).unsqueeze(1);
            auto _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0);
            return _2D_window.expand({channel, 1, window_size, window_size}).contiguous();
        }

        // Image Quality Assessment: From Error Visibility to
        // Structural Similarity (SSIM), Wang et al. 2004
        // The SSIM value lies between -1 and 1, where 1 means perfect similarity.
        // It's considered a better metric than mean squared error for perceptual image quality as it considers changes in structural information,
        // luminance, and contrast.
        std::pair<torch::Tensor, torch::Tensor> ssim(const torch::Tensor& img1, const torch::Tensor& img2, const torch::Tensor& window, int window_size, int channel) {

            const uint32_t N_sq = img1.numel() * img1.numel();
            static const float C1 = 0.01f * 0.01f;
            static const float C2 = 0.03f * 0.03f;

            const auto mu1 = torch::nn::functional::conv2d(img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel));
            const auto mu1_sq = mu1.pow(2);
            const auto sigma1_sq = torch::nn::functional::conv2d(img1 * img1, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel)) - mu1_sq;

            const auto mu2 = torch::nn::functional::conv2d(img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel));
            const auto mu2_sq = mu2.pow(2);
            const auto sigma2_sq = torch::nn::functional::conv2d(img2 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel)) - mu2_sq;

            const auto mu1_mu2 = mu1 * mu2;
            const auto sigma12 = torch::nn::functional::conv2d(img1 * img2, window, torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel)) - mu1_mu2;
            const auto ssim_map = ((2.f * mu1_mu2 + C1) * (2.f * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

            const auto l_p = (2.f * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1);
            const auto cs_p = (2.f * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2);

            auto lp_x = 2.f * ((mu2 - mu1 * l_p) / (mu1_sq + mu2_sq + C1));
            auto cs_x = 2.f / (sigma1_sq + sigma2_sq + C2);

            // ts::print_debug_info(window, "window");
            lp_x = torch::nn::functional::conv2d(lp_x, torch::flip(window, {3, 2}), torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel));
            cs_x = torch::nn::functional::conv2d(cs_x, torch::flip(window, {3, 2}), torch::nn::functional::Conv2dFuncOptions().padding(window_size / 2).groups(channel));
            cs_x = cs_x * ((img2 - mu2) - cs_p * (img1 - mu1));
            auto dL_ssim_dimg1 = (lp_x * cs_p + l_p * cs_x) / static_cast<float>(N_sq);

            return {ssim_map.mean(), dL_ssim_dimg1.to(dtype(torch::kFloat).device(torch::kCUDA))};
        }
    } // namespace loss
} // namespace gs