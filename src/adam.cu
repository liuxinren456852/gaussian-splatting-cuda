// Copyright (c) 2023 Janusch Patas.
#include "adam.cuh"
#include <stdexcept>
#include <utility>

namespace gs {
    namespace optim {
        std::string map_param_type_to_string(ParamType param_type) {
            switch (param_type) {
            case ParamType::Pos:
                return "pos";
            case ParamType::Scaling:
                return "scaling";
            case ParamType::Rotation:
                return "rotation";
            case ParamType::Opacity:
                return "opacity";
            case ParamType::Features_dc:
                return "features_dc";
            case ParamType::Features_rest:
                return "features_rest";
            default:
                throw std::runtime_error("Unknown parameter type");
            }
        }

        __global__ void AdamUpdatePos_Scaling_Kernel(float3* params,
                                                     float3* d_params_grad,
                                                     float3* d_avg,
                                                     float3* d_avg_sq,
                                                     int size, float lr_t,
                                                     float beta1,
                                                     float beta2,
                                                     float epsilon);
        __global__ void AdamUpdateRotationKernel(float4* params,
                                                 float4* d_params_grad,
                                                 float4* d_avg,
                                                 float4* d_avg_sq,
                                                 int size,
                                                 float lr_t,
                                                 float beta1,
                                                 float beta2,
                                                 float epsilon);
        __global__ void AdamUpdateOpactiyKernel(float* params,
                                                float* d_params_grad,
                                                float* d_avg,
                                                float* d_avg_sq,
                                                int size,
                                                float lr_t,
                                                float beta1,
                                                float beta2,
                                                float epsilon);
        __global__ void AdamUpdateFeatureKernel(float3* params,
                                                float3* d_params_grad,
                                                float3* d_avg,
                                                float3* d_avg_sq,
                                                int size,
                                                int dim1,
                                                float lr_t,
                                                float beta1,
                                                float beta2,
                                                float epsilon);

        template <typename T>
        AdamParameter<T>::AdamParameter(ParamType param_type,
                                        std::vector<int> shape,
                                        float learning_rate,
                                        float beta1 /*= 0.9f */,
                                        float beta2 /*= 0.999f */,
                                        float epsilon /*= 0.999f */) : _param_type(param_type),
                                                                       _param_name(map_param_type_to_string(param_type)),
                                                                       _shape(std::move(shape)),
                                                                       _lr(learning_rate),
                                                                       _beta1(beta1),
                                                                       _beta2(beta2),
                                                                       _epsilon(epsilon) {
            cudaStreamCreate(&_stream);
            size_t numel = 1;
            // We assume the last dim is the dim of the parameter like float, float2, float3, ...
            for (int i = 0; i < _shape.size() - 1; ++i) {
                numel *= _shape[i];
            }

            cudaMallocAsync(&_d_params, sizeof(T) * numel, _stream);
            cudaMemsetAsync(_d_params, 0, sizeof(T) * numel, _stream);
            cudaMallocAsync(&_d_params_grad, sizeof(T) * numel, _stream);
            cudaMemsetAsync(_d_params_grad, 0, sizeof(T) * numel, _stream);
            cudaMallocAsync(&_d_avg, sizeof(T) * numel, _stream);
            cudaMemsetAsync(_d_avg, 0, sizeof(T) * _shape[0], _stream);
        }

        template <typename T>
        AdamParameter<T>::~AdamParameter() {
            cudaStreamDestroy(_stream);
            cudaFree(_d_params);
            cudaFree(_d_params_grad);
            cudaFree(_d_avg);
        }

        template <typename T>
        void AdamParameter<T>::Step() {
            static const int threads_per_block = 256;
            static const int blocks_per_grid = (_shape[0] + threads_per_block - 1) / threads_per_block;
            switch (_param_type) {
            case ParamType::Pos:
            case ParamType::Scaling:
                AdamUpdatePos_Scaling_Kernel<<<blocks_per_grid, threads_per_block, 0, _stream>>>(
                    reinterpret_cast<float3*>(_d_params),
                    reinterpret_cast<float3*>(_d_params_grad),
                    reinterpret_cast<float3*>(_d_avg),
                    reinterpret_cast<float3*>(_d_avg_sq),
                    _shape[0],
                    _lr,
                    _beta1,
                    _beta2,
                    _epsilon);
                break;
            case ParamType::Rotation:
                AdamUpdateRotationKernel<<<blocks_per_grid, threads_per_block, 0, _stream>>>(
                    reinterpret_cast<float4*>(_d_params),
                    reinterpret_cast<float4*>(_d_params_grad),
                    reinterpret_cast<float4*>(_d_avg),
                    reinterpret_cast<float4*>(_d_avg_sq),
                    _shape[0],
                    _lr,
                    _beta1,
                    _beta2,
                    _epsilon);
                break;
            case ParamType::Opacity:
                AdamUpdateOpactiyKernel<<<blocks_per_grid, threads_per_block, 0, _stream>>>(
                    reinterpret_cast<float*>(_d_params),
                    reinterpret_cast<float*>(_d_params_grad),
                    reinterpret_cast<float*>(_d_avg),
                    reinterpret_cast<float*>(_d_avg_sq),
                    _shape[0],
                    _lr,
                    _beta1,
                    _beta2,
                    _epsilon);
                break;
            case ParamType::Features_dc:
            case ParamType::Features_rest:
                AdamUpdateFeatureKernel<<<blocks_per_grid, threads_per_block, 0, _stream>>>(
                    reinterpret_cast<float3*>(_d_params),
                    reinterpret_cast<float3*>(_d_params_grad),
                    reinterpret_cast<float3*>(_d_avg),
                    reinterpret_cast<float3*>(_d_avg_sq),
                    _shape[0],
                    _shape[1],
                    _lr,
                    _beta1,
                    _beta2,
                    _epsilon);
                break;
            default:
                throw std::runtime_error("Unknown parameter type");
            }
        }

        template <typename T>
        void AdamParameter<T>::Sync() {
            cudaStreamSynchronize(_stream);
        }

        void Adam::Step() {
            for (auto& [key, param] : _params) {
                param->Step();
            }
        }

        template <typename T>
        void AdamParameter<T>::Set_Exp_Avg_Sq(T* d_avg_sq, std::vector<int> size) {
        }
        template <typename T>
        void AdamParameter<T>::Set_Exp_Avg(T* d_avg, std::vector<int> size) {
        }

        void Adam::Sync() {
            for (auto& [key, param] : _params) {
                param->Sync();
            }
        }
        void Adam::AddParameter(std::shared_ptr<AdamParameterBase> param) {
            _params[param->GetType()] = param;
        }

        __global__ void AdamUpdatePos_Scaling_Kernel(float3* __restrict__ params,
                                                     float3* __restrict__ d_params_grad,
                                                     float3* __restrict__ d_avg,
                                                     float3* __restrict__ d_avg_sq,
                                                     int size,
                                                     float lr_t,
                                                     float beta1,
                                                     float beta2,
                                                     float epsilon) {
            // calculate the index for the weight/bias
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            // only execute if the index is within the size of the weights/biases
            if (idx < size) {
                // compute the new moving average of the gradient
                float3 avg = d_avg[idx];
                float3 avg_sq = d_avg_sq[idx];
                float3 param = params[idx];
                const float3 param_grad = d_params_grad[idx];

                avg.x = beta1 * avg.x + (1 - beta1) * param_grad.x;
                avg.y = beta1 * avg.y + (1 - beta1) * param_grad.y;
                avg.z = beta1 * avg.z + (1 - beta1) * param_grad.z;

                // compute the new moving average of the squared gradient
                avg_sq.x = beta2 * avg_sq.x + (1 - beta2) * param_grad.x * param_grad.x;
                avg_sq.y = beta2 * avg_sq.y + (1 - beta2) * param_grad.y * param_grad.y;
                avg_sq.z = beta2 * avg_sq.z + (1 - beta2) * param_grad.z * param_grad.z;

                // update the weights/biases
                param.x -= lr_t * avg.x / (sqrt(avg_sq.x) + epsilon);
                param.y -= lr_t * avg.y / (sqrt(avg_sq.y) + epsilon);
                param.z -= lr_t * avg.z / (sqrt(avg_sq.z) + epsilon);
                params[idx] = param;
            }
        }

        __global__ void AdamUpdateRotationKernel(float4* __restrict__ params,
                                                 float4* __restrict__ d_params_grad,
                                                 float4* __restrict__ d_avg,
                                                 float4* __restrict__ d_avg_sq,
                                                 int size,
                                                 float lr_t,
                                                 float beta1,
                                                 float beta2,
                                                 float epsilon) {

            // calculate the index for the weight/bias
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            // only execute if the index is within the size of the weights/biases
            if (idx < size) {
                float4 avg = d_avg[idx];
                float4 avg_sq = d_avg_sq[idx];
                float4 param = params[idx];
                const float4 param_grad = d_params_grad[idx];

                avg.x = beta1 * avg.x + (1 - beta1) * param_grad.x;
                avg.y = beta1 * avg.y + (1 - beta1) * param_grad.y;
                avg.z = beta1 * avg.z + (1 - beta1) * param_grad.z;
                avg.w = beta1 * avg.w + (1 - beta1) * param_grad.w;

                // compute the new moving average of the squared gradient
                avg_sq.x = beta2 * avg_sq.x + (1 - beta2) * param_grad.x * param_grad.x;
                avg_sq.y = beta2 * avg_sq.y + (1 - beta2) * param_grad.y * param_grad.y;
                avg_sq.z = beta2 * avg_sq.z + (1 - beta2) * param_grad.z * param_grad.z;
                avg_sq.w = beta2 * avg_sq.w + (1 - beta2) * param_grad.w * param_grad.w;

                // update the weights/biases
                param.x -= lr_t * avg.x / (sqrt(avg_sq.x) + epsilon);
                param.y -= lr_t * avg.y / (sqrt(avg_sq.y) + epsilon);
                param.z -= lr_t * avg.z / (sqrt(avg_sq.z) + epsilon);
                param.w -= lr_t * avg.w / (sqrt(avg_sq.w) + epsilon);
                params[idx] = param;
            }
        }

        __global__ void AdamUpdateOpactiyKernel(float* params,
                                                float* d_params_grad,
                                                float* d_avg,
                                                float* d_avg_sq,
                                                int size,
                                                float lr_t,
                                                float beta1,
                                                float beta2,
                                                float epsilon) {
            // calculate the index for the weight/bias
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            // only execute if the index is within the size of the weights/biases
            if (idx < size) {
                // compute the new moving average of the gradient
                float avg = d_avg[idx];
                float avg_sq = d_avg_sq[idx];
                float param = params[idx];
                const float param_grad = d_params_grad[idx];

                avg = beta1 * avg + (1 - beta1) * param_grad;

                // compute the new moving average of the squared gradient
                avg_sq = beta2 * avg_sq + (1 - beta2) * param_grad * param_grad;

                // update the weights/biases
                param -= lr_t * avg / (sqrt(avg_sq) + epsilon);
                params[idx] = param;
            }
        }

        __global__ void AdamUpdateFeatureKernel(float3* params,
                                                float3* d_params_grad,
                                                float3* d_avg,
                                                float3* d_avg_sq,
                                                int size,
                                                int dim1,
                                                float lr_t,
                                                float beta1,
                                                float beta2,
                                                float epsilon) {
            // calculate the index for the weight/bias
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            // only execute if the index is within the size of the weights/biases
            if (idx < size) {
                // compute the new moving average of the gradient

                for (int j = 0; j < dim1; j++) {
                    const int current_index = idx * dim1 + j;
                    float3 avg = d_avg[current_index];
                    float3 avg_sq = d_avg_sq[current_index];
                    float3 param = params[current_index];
                    const float3 param_grad = d_params_grad[current_index];

                    avg.x = beta1 * avg.x + (1 - beta1) * param_grad.x;
                    avg.y = beta1 * avg.y + (1 - beta1) * param_grad.y;
                    avg.z = beta1 * avg.z + (1 - beta1) * param_grad.z;

                    // compute the new moving average of the squared gradient
                    avg_sq.x = beta2 * avg_sq.x + (1 - beta2) * param_grad.x * param_grad.x;
                    avg_sq.y = beta2 * avg_sq.y + (1 - beta2) * param_grad.y * param_grad.y;
                    avg_sq.z = beta2 * avg_sq.z + (1 - beta2) * param_grad.z * param_grad.z;

                    // update the weights/biases
                    param.x -= lr_t * avg.x / (sqrt(avg_sq.x) + epsilon);
                    param.y -= lr_t * avg.y / (sqrt(avg_sq.y) + epsilon);
                    param.z -= lr_t * avg.z / (sqrt(avg_sq.z) + epsilon);
                    params[current_index] = param;
                }
            }
        }
    } // namespace optim
} // namespace gs