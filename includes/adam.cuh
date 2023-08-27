// Copyright (c) 2023 Janusch Patas.
#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace optim {
    class Adam {
    public:
        Adam(float learning_rate, float beta1, float beta2, float epsilon);
        ~Adam();
        void InitializePos(std::vector<int> shape, float learning_rate);
        void InitializeScaling(std::vector<int> shape, float learning_rate);
        void InitializeRotation(std::vector<int> shape, float learning_rate);
        void InitializeOpacity(std::vector<int> shape, float learning_rate);
        void InitializeFeaturesDC(std::vector<int> shape, float learning_rate);
        void InitializeFeaturesRest(std::vector<int> shape, float learning_rate);

        void sync();
        void step();

    private:
        float* _d_ma_pos;
        float* _d_ma_scaling;
        float* _d_ma_rotation;
        float* _d_ma_opacity;
        float* _d_ma_features_dc;
        float* _d_ma_features_rest;

    private:
        float _pos_lr;
        float _scaling_lr;
        float _rotation_lr;
        float _opacity_lr;
        float _features_dc_lr;
        float _features_rest_lr;
        float _beta1;
        float _beta2;
        float _epsilon;
        float _beta1_t;
        float _beta2_t;

        cudaStream_t _stream_pos;
        cudaStream_t _stream_scaling;
        cudaStream_t _stream_rotation;
        cudaStream_t _stream_opacity;
        cudaStream_t _stream_features_dc;
        cudaStream_t _stream_features_rest;
    };
} // namespace optim