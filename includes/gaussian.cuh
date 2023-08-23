// Copyright (c) 2023 Janusch Patas.
// All rights reserved. Derived from 3D Gaussian Splatting for Real-Time Radiance Field Rendering software by Inria and MPII.

#pragma once

#include "general_utils.cuh"
#include "parameters.cuh"
#include "point_cloud.cuh"
#include "sh_utils.cuh"
#include "spatial.h"
#include <memory>
#include <string>
#include <torch/torch.h>

class GaussianModel {
public:
    explicit GaussianModel(int sh_degree);
    ~GaussianModel();
    // Copy constructor
    GaussianModel(const GaussianModel& other) = delete;
    // Copy assignment operator
    GaussianModel& operator=(const GaussianModel& other) = delete;
    // Move constructor
    GaussianModel(GaussianModel&& other) = default;
    // Move assignment operator
    GaussianModel& operator=(GaussianModel&& other) = default;

public:
    // Getters
    inline torch::Tensor Get_xyz() const { return _xyz; }
    inline torch::Tensor Get_opacity() const { return torch::sigmoid(_opacity); }
    inline torch::Tensor Get_rotation() const { return torch::nn::functional::normalize(_rotation); }
    torch::Tensor Get_features() const;
    torch::Tensor Get_covariance(float scaling_modifier = 1.0);
    int Get_active_sh_degree() const { return _active_sh_degree; }
    int Get_max_sh_degree() const { return _max_sh_degree; }
    torch::Tensor Get_scaling() { return torch::exp(_scaling); }

    // Methods
    void One_up_sh_degree();
    void Create_from_pcd(PointCloud& pcd, float spatial_lr_scale);
    void Training_setup(const OptimizationParameters& params);
    void Update_learning_rate(float iteration);
    void Reset_opacity();
    void Add_densification_stats(torch::Tensor& viewspace_point_tensor, torch::Tensor& update_filter);
    void Densify_and_prune(float max_grad, float min_opacity, float extent, float max_screen_size);
    void Save_ply(const std::filesystem::path& file_path, int iteration, bool isLastIteration);

public:
    // should not be public or it should maybe be pulled out here. Not sure yet
    // This is all public mostly for debugging purposes
    std::unique_ptr<torch::optim::Adam> _optimizer;
    torch::Tensor _max_radii2D;

private:
    void prune_points(torch::Tensor mask);
    void densification_postfix(torch::Tensor& new_xyz,
                               torch::Tensor& new_features_dc,
                               torch::Tensor& new_features_rest,
                               torch::Tensor& new_scaling,
                               torch::Tensor& new_rotation,
                               torch::Tensor& new_opacity);

    void densify_and_clone(torch::Tensor& grads, float grad_threshold, float scene_extent);
    void densify_and_split(torch::Tensor& grads, float grad_threshold, float scene_extent, float min_opacity, float max_screen_size);
    std::vector<std::string> construct_list_of_attributes();
    void select_elements_and_cat(
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
        float* new_xyz,
        const std::vector<long>& new_xyz_size,
        float* new_features_dc,
        const std::vector<long>& new_features_dc_size,
        float* new_features_rest,
        const std::vector<long>& new_features_rest_size,
        float* new_opacity,
        const std::vector<long>& new_opacity_size,
        float* new_scaling,
        const std::vector<long>& new_scaling_size,
        float* new_rotation,
        const std::vector<long>& new_rotation_size,
        long F1,
        long F2,
        long F3,
        long original_size,
        long extension_size);

private:
    int _active_sh_degree;
    int _max_sh_degree;
    float _spatial_lr_scale{};
    float _percent_dense{};

    Expon_lr_func _xyz_scheduler_args;
    cudaStream_t _stream1;
    cudaStream_t _stream2;
    cudaStream_t _stream3;
    cudaStream_t _stream4;
    cudaStream_t _stream5;
    cudaStream_t _stream6;
    torch::Tensor _denom;
    torch::Tensor _xyz;
    torch::Tensor _features_dc;
    torch::Tensor _features_rest;
    torch::Tensor _scaling;
    torch::Tensor _rotation;
    torch::Tensor _xyz_gradient_accum;
    torch::Tensor _opacity;
};
