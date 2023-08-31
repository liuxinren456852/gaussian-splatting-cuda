// Copyright (c) 2023 Janusch Patas.
#include "adam.cuh"
#include "debug_utils.cuh"
#include <utility>

namespace gs {
    namespace optim {

        __global__ void AdamUpdatePos_Scaling_Kernel(float3* params,
                                                     const float3* d_params_grad,
                                                     float3* d_avg,
                                                     float3* d_avg_sq,
                                                     int size, float lr_t,
                                                     float beta1,
                                                     float beta2,
                                                     float epsilon);
        __global__ void AdamUpdatePos_Kernel1(float3* params,
                                              const float3* d_params_grad,
                                              float3* d_avg,
                                              float3* d_avg_sq,
                                              int size, float lr_t,
                                              float beta1,
                                              float beta2,
                                              float epsilon);
        __global__ void AdamUpdatePos_Kernel2(float3* params,
                                              const float3* d_params_grad,
                                              float3* d_avg,
                                              float3* d_avg_sq,
                                              int size, float lr_t,
                                              float beta1,
                                              float beta2,
                                              float epsilon);
        __global__ void AdamUpdatePos_Kernel3(float3* params,
                                              const float3* d_params_grad,
                                              float3* d_avg,
                                              float3* d_avg_sq,
                                              int size, float lr_t,
                                              float beta1,
                                              float beta2,
                                              float epsilon);
        __global__ void AdamUpdateRotationKernel(float4* params,
                                                 const float4* d_params_grad,
                                                 float4* d_avg,
                                                 float4* d_avg_sq,
                                                 int size,
                                                 float lr_t,
                                                 float beta1,
                                                 float beta2,
                                                 float epsilon);
        __global__ void AdamUpdateOpactiyKernel(float* params,
                                                const float* d_params_grad,
                                                float* d_avg,
                                                float* d_avg_sq,
                                                int size,
                                                float lr_t,
                                                float beta1,
                                                float beta2,
                                                float epsilon);
        __global__ void AdamUpdateFeatureKernel(float3* params,
                                                const float3* d_params_grad,
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
                                        cudaStream_t stream,
                                        float beta1 /*= 0.9f */,
                                        float beta2 /*= 0.999f */,
                                        float epsilon /*= 1e-8 */) : _param_type(param_type),
                                                                     _param_name(Map_param_type_to_string(param_type)),
                                                                     _param_shape(std::move(shape)),
                                                                     _lr(learning_rate),
                                                                     _beta1(beta1),
                                                                     _beta2(beta2),
                                                                     _epsilon(epsilon) {
            size_t numel = 1;
            // We assume the last dim is the dim of the parameter like float, float2, float3, ...
            int total = std::max(1, static_cast<int>(_param_shape.size()) - 1);
            for (int i = 0; i < total; ++i) {
                numel *= _param_shape[i];
            }

            // no memory allocatioon for _d_param. It just a pointer to the gausian paramter
            // At this point I believe it should be part of adam optimizer. Not sure yet
            //          CHECK_CUDA_ERROR(cudaMallocAsync(&_d_params, sizeof(T) * numel, _stream));
            //          CHECK_CUDA_ERROR(cudaMemsetAsync(_d_params, 0, sizeof(T) * numel, _stream));

            CHECK_CUDA_ERROR(cudaMalloc(&_d_params_grad, sizeof(T) * numel));
            CHECK_CUDA_ERROR(cudaMemset(_d_params_grad, 0, sizeof(T) * numel));
            CHECK_CUDA_ERROR(cudaMalloc(&_d_avg, sizeof(T) * numel));
            CHECK_CUDA_ERROR(cudaMemset(_d_avg, 0, sizeof(T) * numel));
            CHECK_CUDA_ERROR(cudaMalloc(&_d_avg_sq, sizeof(T) * numel));
            CHECK_CUDA_ERROR(cudaMemset(_d_avg_sq, 0, sizeof(T) * numel));
        }

        template <typename T>
        AdamParameter<T>::~AdamParameter() {
            CHECK_CUDA_ERROR(cudaFree(_d_params_grad));
            CHECK_CUDA_ERROR(cudaFree(_d_avg));
            CHECK_CUDA_ERROR(cudaFree(_d_avg_sq));
        }

        template <typename T>
        void AdamParameter<T>::Step(cudaStream_t stream) {
            static const int threads_per_block = 256;
            static const int blocks_per_grid = (_param_shape[0] + threads_per_block - 1) / threads_per_block;
            if (_gradient_shape != _param_shape) {
                throw std::runtime_error("Gradient shape does not match parameter shape for " + Map_param_type_to_string(GetType()));
            }
            switch (_param_type) {
            case ParamType::Pos: {
                std::cout << "\nAdamUpdatePos_Scaling_Kernel " + Map_param_type_to_string(GetType()) + " shape: " << _param_shape[0] << ", " << _param_shape[1] << std::endl;
                std::cout << "lr " << _lr << ", beta1 " << _beta1 << ", beta2 " << _beta2 << std::endl;
                if (_d_params == nullptr) {
                    throw std::runtime_error("Parameter pointer is not set");
                } else if (_d_params_grad == nullptr) {
                    throw std::runtime_error("Gradient pointer is not set");
                } else if (_d_avg == nullptr) {
                    throw std::runtime_error("Average pointer is not set");
                } else if (_d_avg_sq == nullptr) {
                    throw std::runtime_error("Average squared pointer is not set");
                }
                AdamUpdatePos_Kernel2<<<blocks_per_grid, threads_per_block>>>(
                    reinterpret_cast<float3*>(_d_params),
                    reinterpret_cast<float3*>(_d_params_grad),
                    reinterpret_cast<float3*>(_d_avg),
                    reinterpret_cast<float3*>(_d_avg_sq),
                    _param_shape[0],
                    _lr,
                    _beta1,
                    _beta2,
                    _epsilon);
                CHECK_LAST_CUDA_ERROR();
                AdamUpdatePos_Kernel1<<<blocks_per_grid, threads_per_block>>>(
                    reinterpret_cast<float3*>(_d_params),
                    reinterpret_cast<float3*>(_d_params_grad),
                    reinterpret_cast<float3*>(_d_avg),
                    reinterpret_cast<float3*>(_d_avg_sq),
                    _param_shape[0],
                    _lr,
                    _beta1,
                    _beta2,
                    _epsilon);
                CHECK_LAST_CUDA_ERROR();
                AdamUpdatePos_Kernel3<<<blocks_per_grid, threads_per_block>>>(
                    reinterpret_cast<float3*>(_d_params),
                    reinterpret_cast<float3*>(_d_params_grad),
                    reinterpret_cast<float3*>(_d_avg),
                    reinterpret_cast<float3*>(_d_avg_sq),
                    _param_shape[0],
                    _lr,
                    _beta1,
                    _beta2,
                    _epsilon);
                CHECK_LAST_CUDA_ERROR();
                cudaDeviceSynchronize();
            } break;
            case ParamType::Scaling: {
                std::cout << "\nAdamUpdatePos_Scaling_Kernel " + Map_param_type_to_string(GetType()) + " shape: " << _param_shape[0] << ", " << _param_shape[1] << std::endl;
                std::cout << "lr " << _lr << ", beta1 " << _beta1 << ", beta2 " << _beta2 << std::endl;
                AdamUpdatePos_Scaling_Kernel<<<blocks_per_grid, threads_per_block>>>(
                    reinterpret_cast<float3*>(_d_params),
                    reinterpret_cast<float3*>(_d_params_grad),
                    reinterpret_cast<float3*>(_d_avg),
                    reinterpret_cast<float3*>(_d_avg_sq),
                    _param_shape[0],
                    _lr,
                    _beta1,
                    _beta2,
                    _epsilon);
                CHECK_LAST_CUDA_ERROR();
                cudaDeviceSynchronize();
            } break;
            case ParamType::Rotation: {
                std::cout << "AdamUpdateRotationKernel " + Map_param_type_to_string(GetType()) + " shape: " << _param_shape[0] << ", " << _param_shape[1] << std::endl;
                AdamUpdateRotationKernel<<<blocks_per_grid, threads_per_block>>>(
                    reinterpret_cast<float4*>(_d_params),
                    reinterpret_cast<float4*>(_d_params_grad),
                    reinterpret_cast<float4*>(_d_avg),
                    reinterpret_cast<float4*>(_d_avg_sq),
                    _param_shape[0],
                    _lr,
                    _beta1,
                    _beta2,
                    _epsilon);
                CHECK_LAST_CUDA_ERROR();
                cudaDeviceSynchronize();
            } break;
            case ParamType::Opacity: {
                std::cout << "AdamUpdateOpactiyKernel " + Map_param_type_to_string(GetType()) + " shape: " << _param_shape[0] << ", " << _param_shape[1] << std::endl;
                AdamUpdateOpactiyKernel<<<blocks_per_grid, threads_per_block>>>(
                    reinterpret_cast<float*>(_d_params),
                    reinterpret_cast<float*>(_d_params_grad),
                    reinterpret_cast<float*>(_d_avg),
                    reinterpret_cast<float*>(_d_avg_sq),
                    _param_shape[0],
                    _lr,
                    _beta1,
                    _beta2,
                    _epsilon);
                CHECK_LAST_CUDA_ERROR();
                cudaDeviceSynchronize();
            } break;
            case ParamType::Features_dc: {
                std::cout << "AdamUpdateFeatureKernel " + Map_param_type_to_string(GetType()) + " shape: " << _param_shape[0] << ", " << _param_shape[1] << std::endl;
                AdamUpdateFeatureKernel<<<blocks_per_grid, threads_per_block>>>(
                    reinterpret_cast<float3*>(_d_params),
                    reinterpret_cast<float3*>(_d_params_grad),
                    reinterpret_cast<float3*>(_d_avg),
                    reinterpret_cast<float3*>(_d_avg_sq),
                    _param_shape[0],
                    _param_shape[1],
                    _lr,
                    _beta1,
                    _beta2,
                    _epsilon);
                CHECK_LAST_CUDA_ERROR();
                cudaDeviceSynchronize();
            } break;
            case ParamType::Features_rest: {
                std::cout << "AdamUpdateFeatureKernel " + Map_param_type_to_string(GetType()) + " shape: " << _param_shape[0] << ", " << _param_shape[1] << std::endl;
                AdamUpdateFeatureKernel<<<blocks_per_grid, threads_per_block>>>(
                    reinterpret_cast<float3*>(_d_params),
                    reinterpret_cast<float3*>(_d_params_grad),
                    reinterpret_cast<float3*>(_d_avg),
                    reinterpret_cast<float3*>(_d_avg_sq),
                    _param_shape[0],
                    _param_shape[1],
                    _lr,
                    _beta1,
                    _beta2,
                    _epsilon);
                CHECK_LAST_CUDA_ERROR();
                cudaDeviceSynchronize();
            } break;
            default:
                throw std::runtime_error("Unknown parameter type");
            }
        }

        void Adam::Step(cudaStream_t stream) {
            //            for (auto& [key, param] : _params) {
            //                param->Step(stream);
            //                cudaDeviceSynchronize();
            //            }
            _params[ParamType::Scaling]->Step(stream);
            _params[ParamType::Rotation]->Step(stream);
            _params[ParamType::Opacity]->Step(stream);
            _params[ParamType::Pos]->Step(stream);
            _params[ParamType::Features_dc]->Step(stream);
            _params[ParamType::Features_rest]->Step(stream);
        }

        template <typename T>
        void AdamParameter<T>::Set_Exp_Avg_Sq(T* d_avg_sq, std::vector<int> size, cudaStream_t stream) {
            int numel = 1;
            int total = std::max(1, static_cast<int>(size.size() - 1));
            for (int i = 0; i < total; ++i) {
                numel *= size[i];
            }
            CHECK_CUDA_ERROR(cudaFree(_d_avg_sq));
            CHECK_CUDA_ERROR(cudaMalloc(&_d_avg_sq, sizeof(T) * numel));
            CHECK_CUDA_ERROR(cudaMemcpy(_d_avg_sq, d_avg_sq, sizeof(T) * numel, cudaMemcpyDeviceToDevice));
        }
        template <typename T>
        void AdamParameter<T>::Set_Exp_Avg(T* d_avg, std::vector<int> size, cudaStream_t stream) {
            // for now super inefficient
            int numel = 1;
            int total = std::max(1, static_cast<int>(size.size() - 1));
            for (int i = 0; i < total; ++i) {
                numel *= size[i];
            }
            CHECK_CUDA_ERROR(cudaFree(_d_avg));
            CHECK_CUDA_ERROR(cudaMalloc(&_d_avg, sizeof(T) * numel));
            CHECK_CUDA_ERROR(cudaMemcpy(_d_avg, d_avg, sizeof(T) * numel, cudaMemcpyDeviceToDevice));
        }
        template <typename T>
        void AdamParameter<T>::Set_Gradient(T* d_param_grad, std::vector<int> gradient_shape, cudaStream_t stream) {
            int numel = 1;
            int total = std::max(1, static_cast<int>(gradient_shape.size() - 1));
            for (int i = 0; i < total; ++i) {
                numel *= gradient_shape[i];
            }
            _gradient_shape = gradient_shape;
            CHECK_CUDA_ERROR(cudaFree(_d_params_grad));
            CHECK_CUDA_ERROR(cudaMalloc(&_d_params_grad, sizeof(T) * numel));
            CHECK_CUDA_ERROR(cudaMemcpy(_d_params_grad, d_param_grad, sizeof(T) * numel, cudaMemcpyDeviceToDevice));
        }

        void Adam::AddParameter(std::shared_ptr<AdamParameterBase> param) {
            _params[param->GetType()] = param;
        }
        __global__ void AdamUpdatePos_Kernel1(float3* params,
                                              const float3* d_params_grad,
                                              float3* d_avg,
                                              float3* d_avg_sq,
                                              int size, float lr_t,
                                              float beta1,
                                              float beta2,
                                              float epsilon) {
            // calculate the index for the weight/bias
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            // only execute if the index is within the size of the weights/biases
            if (idx < size) {
                // compute the new moving average of the gradient
                d_avg[idx].x = beta1 * d_avg[idx].x + (1 - beta1) * d_params_grad[idx].x;
                d_avg[idx].y = beta1 * d_avg[idx].y + (1 - beta1) * d_params_grad[idx].y;
                d_avg[idx].z = beta1 * d_avg[idx].z + (1 - beta1) * d_params_grad[idx].z;
            }
        }

        __global__ void AdamUpdatePos_Kernel2(float3* params,
                                              const float3* d_params_grad,
                                              float3* d_avg,
                                              float3* d_avg_sq,
                                              int size, float lr_t,
                                              float beta1,
                                              float beta2,
                                              float epsilon) {
            // calculate the index for the weight/bias
            int idx = blockIdx.x * blockDim.x + threadIdx.x;

            // only execute if the index is within the size of the weights/biases
            if (idx < size) {
                // compute the new moving average of the gradient
                const float3 param_grad = d_params_grad[idx];
                // compute the new moving average of the squared gradient
                d_avg_sq[idx].x = beta2 * d_avg_sq[idx].x + (1 - beta2) * param_grad.x * param_grad.x;
                d_avg_sq[idx].y = beta2 * d_avg_sq[idx].y + (1 - beta2) * param_grad.y * param_grad.y;
                d_avg_sq[idx].z = beta2 * d_avg_sq[idx].z + (1 - beta2) * param_grad.z * param_grad.z;
                //                d_avg_sq[idx].x = 1.f;
                //                d_avg_sq[idx].y = 1.f;
                //                d_avg_sq[idx].z = 1.f;
            }
        }

        __global__ void AdamUpdatePos_Kernel3(float3* params,
                                              const float3* d_params_grad,
                                              float3* d_avg,
                                              float3* d_avg_sq,
                                              int size, float lr_t,
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

                // update the weights/biases
                param.x -= lr_t * avg.x / (sqrt(avg_sq.x + epsilon));
                param.y -= lr_t * avg.y / (sqrt(avg_sq.y + epsilon));
                param.z -= lr_t * avg.z / (sqrt(avg_sq.z + epsilon));
                params[idx] = param;
            }
        }
        __global__ void AdamUpdatePos_Scaling_Kernel(float3* __restrict__ params,
                                                     const float3* __restrict__ d_params_grad,
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
                param.x -= lr_t * avg.x / (sqrt(avg_sq.x + epsilon));
                param.y -= lr_t * avg.y / (sqrt(avg_sq.y + epsilon));
                param.z -= lr_t * avg.z / (sqrt(avg_sq.z + epsilon));
                params[idx] = param;
                d_avg[idx] = avg;
                d_avg_sq[idx] = avg_sq;
            }
        }

        __global__ void AdamUpdateRotationKernel(float4* __restrict__ params,
                                                 const float4* __restrict__ d_params_grad,
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
                d_avg[idx] = avg;
                d_avg_sq[idx] = avg_sq;
            }
        }

        __global__ void AdamUpdateOpactiyKernel(float* params,
                                                const float* d_params_grad,
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
                d_avg[idx] = avg;
                d_avg_sq[idx] = avg_sq;
            }
        }

        __global__ void AdamUpdateFeatureKernel(float3* params,
                                                const float3* d_params_grad,
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
                    d_avg[idx] = avg;
                    d_avg_sq[idx] = avg_sq;
                }
            }
        }
    } // namespace optim
} // namespace gs