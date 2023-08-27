// Copyright (c) 2023 Janusch Patas.
#include "adam.cuh"
optim::Adam::Adam(float learning_rate, float beta1, float beta2, float epsilon) : _pos_lr(learning_rate),
                                                                                  _scaling_lr(learning_rate),
                                                                                  _rotation_lr(learning_rate),
                                                                                  _opacity_lr(learning_rate),
                                                                                  _features_dc_lr(learning_rate),
                                                                                  _features_rest_lr(learning_rate),
                                                                                  _beta1(beta1),
                                                                                  _beta2(beta2),
                                                                                  _epsilon(epsilon),
                                                                                  _beta1_t(beta1),
                                                                                  _beta2_t(beta2) {

    cudaStreamCreate(&_stream_pos);
    cudaStreamCreate(&_stream_scaling);
    cudaStreamCreate(&_stream_rotation);
    cudaStreamCreate(&_stream_opacity);
    cudaStreamCreate(&_stream_features_dc);
    cudaStreamCreate(&_stream_features_rest);
}
optim::Adam::~Adam() {
    sync();
    cudaStreamDestroy(_stream_pos);
    cudaStreamDestroy(_stream_scaling);
    cudaStreamDestroy(_stream_rotation);
    cudaStreamDestroy(_stream_opacity);
    cudaStreamDestroy(_stream_features_dc);
    cudaStreamDestroy(_stream_features_rest);

    cudaFree(_d_ma_pos);
    cudaFree(_d_ma_scaling);
    cudaFree(_d_ma_rotation);
    cudaFree(_d_ma_opacity);
    cudaFree(_d_ma_features_dc);
    cudaFree(_d_ma_features_rest);
}
void optim::Adam::InitializePos(std::vector<int> shape, float learning_rate) {
    _pos_lr = learning_rate;

    cudaMallocAsync(&_d_ma_pos, sizeof(float) * shape[0] * shape[1], _stream_pos);
    cudaMemsetAsync(_d_ma_pos, 0, sizeof(float) * shape[0] * shape[1], _stream_pos);
}
void optim::Adam::InitializeScaling(std::vector<int> shape, float learning_rate) {
    _scaling_lr = learning_rate;

    cudaMallocAsync(&_d_ma_scaling, sizeof(float) * shape[0] * shape[1], _stream_scaling);
    cudaMemsetAsync(_d_ma_scaling, 0, sizeof(float) * shape[0] * shape[1], _stream_scaling);
}
void optim::Adam::InitializeRotation(std::vector<int> shape, float learning_rate) {
    _rotation_lr = learning_rate;

    cudaMallocAsync(&_d_ma_rotation, sizeof(float) * shape[0] * shape[1], _stream_rotation);
    cudaMemsetAsync(_d_ma_rotation, 0, sizeof(float) * shape[0] * shape[1], _stream_rotation);
}
void optim::Adam::InitializeOpacity(std::vector<int> shape, float learning_rate) {
    _opacity_lr = learning_rate;

    cudaMallocAsync(&_d_ma_opacity, sizeof(float) * shape[0], _stream_opacity);
    cudaMemsetAsync(_d_ma_opacity, 0, sizeof(float) * shape[0], _stream_opacity);
}
void optim::Adam::InitializeFeaturesDC(std::vector<int> shape, float learning_rate) {
    _features_dc_lr = learning_rate;

    cudaMallocAsync(&_d_ma_features_dc, sizeof(float) * shape[0] * shape[1] * shape[2], _stream_features_dc);
    cudaMemsetAsync(_d_ma_features_dc, 0, sizeof(float) * shape[0] * shape[1] * shape[2], _stream_features_dc);
}
void optim::Adam::InitializeFeaturesRest(std::vector<int> shape, float learning_rate) {
    _features_rest_lr = learning_rate;

    cudaMallocAsync(&_d_ma_features_rest, sizeof(float) * shape[0] * shape[1] * shape[2], _stream_features_rest);
    cudaMemsetAsync(_d_ma_features_rest, 0, sizeof(float) * shape[0] * shape[1] * shape[2], _stream_features_rest);
}

void optim::Adam::step() {
}

void optim::Adam::sync() {
    cudaStreamSynchronize(_stream_pos);
    cudaStreamSynchronize(_stream_scaling);
    cudaStreamSynchronize(_stream_rotation);
    cudaStreamSynchronize(_stream_opacity);
    cudaStreamSynchronize(_stream_features_dc);
    cudaStreamSynchronize(_stream_features_rest);
}

__global__ void AdamUpdateKernel(float* params, float* d_params, float* m, float* v, int size, float lr_t, float beta1, float beta2, float epsilon) {
    // calculate the index for the weight/bias
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // only execute if the index is within the size of the weights/biases
    if (idx < size) {
        // compute the new moving average of the gradient
        m[idx] = beta1 * m[idx] + (1 - beta1) * d_params[idx];

        // compute the new moving average of the squared gradient
        v[idx] = beta2 * v[idx] + (1 - beta2) * d_params[idx] * d_params[idx];

        // update the weights/biases
        params[idx] -= lr_t * m[idx] / (sqrt(v[idx]) + epsilon);
    }
}
