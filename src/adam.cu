// Copyright (c) 2023 Janusch Patas.
#include "adam.cuh"
#include "debug_utils.cuh"
#include <cuda_runtime.h>
#include <utility>

namespace gs {
    namespace optim {

        __global__ void AdamUpdatePos_Scaling_Kernel(float* params,
                                                     const float* d_params_grad,
                                                     float* d_avg,
                                                     float* d_avg_sq,
                                                     int32_t step,
                                                     long size,
                                                     float lr_t,
                                                     float beta1,
                                                     float beta2,
                                                     float epsilon);
        __global__ void AdamUpdateRotationKernel(float4* params,
                                                 const float4* d_params_grad,
                                                 float4* d_avg,
                                                 float4* d_avg_sq,
                                                 int32_t* __restrict__ d_steps,
                                                 int size,
                                                 float lr_t,
                                                 float beta1,
                                                 float beta2,
                                                 float epsilon);
        __global__ void AdamUpdateOpactiyKernel(float* params,
                                                const float* d_params_grad,
                                                float* d_avg,
                                                float* d_avg_sq,
                                                int32_t* __restrict__ d_steps,
                                                int size,
                                                float lr_t,
                                                float beta1,
                                                float beta2,
                                                float epsilon);
        __global__ void AdamUpdateFeatureKernel(float3* params,
                                                const float3* d_params_grad,
                                                float3* d_avg,
                                                float3* d_avg_sq,
                                                int32_t* __restrict__ d_steps,
                                                int size,
                                                int dim1,
                                                float lr_t,
                                                float beta1,
                                                float beta2,
                                                float epsilon);

        AdamParameter::AdamParameter(ParamType param_type,
                                     torch::Tensor param,
                                     float learning_rate,
                                     cudaStream_t stream,
                                     float beta1 /*= 0.9f */,
                                     float beta2 /*= 0.999f */,
                                     float epsilon /*= 1e-8 */) : _param_type(param_type),
                                                                  _d_params(param),
                                                                  _param_name(Map_param_type_to_string(param_type)),
                                                                  _lr(learning_rate),
                                                                  _beta1(beta1),
                                                                  _beta2(beta2),
                                                                  _epsilon(epsilon) {
            _d_params_grad = torch::zeros_like(_d_params);
            _d_avg = torch::zeros_like(_d_params);
            _d_avg_sq = torch::zeros_like(_d_params);
            _d_steps = torch::zeros({_d_params.size(0), 1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        }

        AdamParameter::~AdamParameter() {
        }

        void AdamParameter::Step(cudaStream_t stream, torch::Tensor step) {
            static const int threads_per_block = 256;
            static const int blocks_per_grid = (static_cast<int>(_d_params.numel()) + threads_per_block - 1) / threads_per_block;
            //            const bool avg_equal = _d_avg.sizes() == _d_params.sizes();
            //            const bool param_equal = _d_params.sizes() == _d_params_grad.sizes();

            //            float3 avg = d_avg[idx];
            //            float3 avg_sq = d_avg_sq[idx];
            //            float3 param = params[idx];
            //            const float3 param_grad = d_params_grad[idx];
            //            const int32_t current_step = ++d_steps[idx];
            //
            //            // Bias correction terms
            //            const float bias_correction1 = 1.f - powf(beta1, (float)current_step);
            //            const float bias_correction2 = 1.f - powf(beta2, (float)current_step);
            //            const float bias_correction2_sqrt = sqrtf(bias_correction2);
            //
            //            avg.x = beta1 * avg.x + (1.f - beta1) * param_grad.x;
            //            avg.y = beta1 * avg.y + (1.f - beta1) * param_grad.y;
            //            avg.z = beta1 * avg.z + (1.f - beta1) * param_grad.z;
            //
            //            // compute the new moving average of the squared gradient
            //            avg_sq.x = beta2 * avg_sq.x + (1.f - beta2) * param_grad.x * param_grad.x;
            //            avg_sq.y = beta2 * avg_sq.y + (1.f - beta2) * param_grad.y * param_grad.y;
            //            avg_sq.z = beta2 * avg_sq.z + (1.f - beta2) * param_grad.z * param_grad.z;
            //
            //            // Compute step size considering bias correction
            //            const float step_size = -lr_t / bias_correction1;
            //
            //            const float denom_x = sqrtf(avg_sq.x) / bias_correction2_sqrt + epsilon;
            //            const float denom_y = sqrtf(avg_sq.y) / bias_correction2_sqrt + epsilon;
            //            const float denom_z = sqrtf(avg_sq.z) / bias_correction2_sqrt + epsilon;
            //            // update the weights/biases
            //            param.x = param.x + step_size * avg.x / denom_x;
            //            param.y = param.y + step_size * avg.y / denom_y;
            //            param.z = param.z + step_size * avg.z / denom_z;
            // This has all be satified to update the paramaters successfully
            //            if (!avg_equal || !param_equal) {
            //                throw std::runtime_error("Gradient shape does not match parameter shape for " + Map_param_type_to_string(GetType()));
            //            }
            switch (_param_type) {
                //            case ParamType::Scaling:
                //            case ParamType::Rotation:
                //            case ParamType::Opacity:
                //            case ParamType::Pos: {
                //                // std::cout << "\nAdamUpdatePos_Scaling_Kernel " + Map_param_type_to_string(GetType()) + " shape: " << _d_params.size(0) << ", " << _d_params.size(1) << std::endl;
                //                // std::cout << std::setprecision(6) << "lr " << _lr << ", beta1 " << _beta1 << ", beta2 " << _beta2 << std::endl;
                //                AdamUpdatePos_Scaling_Kernel<<<blocks_per_grid, threads_per_block>>>(
                //                    _d_params.contiguous().data_ptr<float>(),
                //                    _d_params_grad.contiguous().data_ptr<float>(),
                //                    _d_avg.contiguous().data_ptr<float>(),
                //                    _d_avg_sq.contiguous().data_ptr<float>(),
                //                    step.item<int>(),
                //                    _d_params.numel(),
                //                    _lr,
                //                    _beta1,
                //                    _beta2,
                //                    _epsilon);
                //                CHECK_LAST_CUDA_ERROR();
                //            } break;
                //            case ParamType::Scaling: {
                //                //                std::cout << "\nAdamUpdatePos_Scaling_Kernel " + Map_param_type_to_string(GetType()) + " shape: " << _d_params.size(0) << ", " << _d_params.size(1) << std::endl;
                //                //                std::cout <<  std::setprecision(6) <<"lr " << _lr << ", beta1 " << _beta1 << ", beta2 " << _beta2 << std::endl;
                //                AdamUpdatePos_Scaling_Kernel<<<blocks_per_grid, threads_per_block>>>(
                //                    reinterpret_cast<scaling_param_t*>(_d_params.data_ptr<float>()),
                //                    reinterpret_cast<scaling_param_t*>(_d_params_grad.data_ptr<float>()),
                //                    reinterpret_cast<scaling_param_t*>(_d_avg.data_ptr<float>()),
                //                    reinterpret_cast<scaling_param_t*>(_d_avg_sq.data_ptr<float>()),
                //                    _d_steps.data_ptr<int32_t>(),
                //                    _d_params.size(0),
                //                    _lr,
                //                    _beta1,
                //                    _beta2,
                //                    _epsilon);
                //                CHECK_LAST_CUDA_ERROR();
                //                cudaDeviceSynchronize();
                //            } break;
                //            case ParamType::Rotation: {
                //                //                std::cout << "AdamUpdateRotationKernel " + Map_param_type_to_string(GetType()) + " shape: " << _d_params.size(0) << ", " << _d_params.size(1) << std::endl;
                //                AdamUpdateRotationKernel<<<blocks_per_grid, threads_per_block>>>(
                //                    reinterpret_cast<rotation_param_t*>(_d_params.data_ptr<float>()),
                //                    reinterpret_cast<rotation_param_t*>(_d_params_grad.data_ptr<float>()),
                //                    reinterpret_cast<rotation_param_t*>(_d_avg.data_ptr<float>()),
                //                    reinterpret_cast<rotation_param_t*>(_d_avg_sq.data_ptr<float>()),
                //                    _d_steps.data_ptr<int32_t>(),
                //                    _d_params.size(0),
                //                    _lr,
                //                    _beta1,
                //                    _beta2,
                //                    _epsilon);
                //                CHECK_LAST_CUDA_ERROR();
                //                cudaDeviceSynchronize();
                //            } break;
                //            case ParamType::Opacity: {
                //                //                std::cout << "AdamUpdateOpactiyKernel " + Map_param_type_to_string(GetType()) + " shape: " << _d_params.size(0) << ", " << _d_params.size(1) << std::endl;
                //                AdamUpdateOpactiyKernel<<<blocks_per_grid, threads_per_block>>>(
                //                    reinterpret_cast<opacity_param_t*>(_d_params.data_ptr<float>()),
                //                    reinterpret_cast<opacity_param_t*>(_d_params_grad.data_ptr<float>()),
                //                    reinterpret_cast<opacity_param_t*>(_d_avg.data_ptr<float>()),
                //                    reinterpret_cast<opacity_param_t*>(_d_avg_sq.data_ptr<float>()),
                //                    _d_steps.data_ptr<int32_t>(),
                //                    _d_params.size(0),
                //                    _lr,
                //                    _beta1,
                //                    _beta2,
                //                    _epsilon);
                //                CHECK_LAST_CUDA_ERROR();
                //                cudaDeviceSynchronize();
                //            } break;
                //            case ParamType::Features_dc: {
                //                //                std::cout << "AdamUpdateFeatureKernel " + Map_param_type_to_string(GetType()) + " shape: " << _d_params.size(0) << ", " << _d_params.size(1) << std::endl;
                //                AdamUpdateFeatureKernel<<<blocks_per_grid, threads_per_block>>>(
                //                    reinterpret_cast<feature_dc_param_t*>(_d_params.data_ptr<float>()),
                //                    reinterpret_cast<feature_dc_param_t*>(_d_params_grad.data_ptr<float>()),
                //                    reinterpret_cast<feature_dc_param_t*>(_d_avg.data_ptr<float>()),
                //                    reinterpret_cast<feature_dc_param_t*>(_d_avg_sq.data_ptr<float>()),
                //                    _d_steps.data_ptr<int32_t>(),
                //                    _d_params.size(0),
                //                    _d_params.size(1),
                //                    _lr,
                //                    _beta1,
                //                    _beta2,
                //                    _epsilon);
                //                CHECK_LAST_CUDA_ERROR();
                //                cudaDeviceSynchronize();
                //            } break;
                //            case ParamType::Features_rest: {
                //                //                std::cout << "AdamUpdateFeatureKernel " + Map_param_type_to_string(GetType()) + " shape: " << _d_params.size(0) << ", " << _d_params.size(1) << std::endl;
                //                AdamUpdateFeatureKernel<<<blocks_per_grid, threads_per_block>>>(
                //                    reinterpret_cast<feature_rest_param_t*>(_d_params.data_ptr<float>()),
                //                    reinterpret_cast<feature_rest_param_t*>(_d_params_grad.data_ptr<float>()),
                //                    reinterpret_cast<feature_rest_param_t*>(_d_avg.data_ptr<float>()),
                //                    reinterpret_cast<feature_rest_param_t*>(_d_avg_sq.data_ptr<float>()),
                //                    _d_steps.data_ptr<int32_t>(),
                //                    _d_params.size(0),
                //                    _d_params.size(1),
                //                    _lr,
                //                    _beta1,
                //                    _beta2,
                //                    _epsilon);
                //                CHECK_LAST_CUDA_ERROR();
                //                cudaDeviceSynchronize();
                //            } break;
            default:
                _d_avg = _beta1 * _d_avg + (1.f - _beta1) * _d_params_grad;
                _d_avg_sq = _beta2 * _d_avg_sq + (1.f - _beta2) * _d_params_grad * _d_params_grad;

                float bias_correction1 = 1.f - std::pow(_beta1, static_cast<float>(step.item<int>()));
                float bias_correction2 = 1.f - std::pow(_beta2, static_cast<float>(step.item<int>()));
                float bias_correction2_sqrt_inv = 1.f / std::sqrt(bias_correction2);
                const auto denom = (_d_avg_sq.sqrt() * bias_correction2_sqrt_inv) + _epsilon;
                auto step_size = _lr / bias_correction1;
                _d_params -= step_size * (_d_avg / denom);
            }
        }

        void Adam::Step(cudaStream_t stream) {
            //            for (auto& [key, param] : _params) {
            //                param->Step(stream);
            //                cudaDeviceSynchronize();
            //            }
            _global_step = 1 + _global_step;
            _params[ParamType::Pos]->Step(stream, _global_step);
            _params[ParamType::Scaling]->Step(stream, _global_step);
            _params[ParamType::Rotation]->Step(stream, _global_step);
            _params[ParamType::Opacity]->Step(stream, _global_step);
            _params[ParamType::Features_dc]->Step(stream, _global_step);
            _params[ParamType::Features_rest]->Step(stream, _global_step);
        }

        void AdamParameter::Set_Exp_Avg_Sq(torch::Tensor d_avg_sq) {
            _d_avg_sq = d_avg_sq;
        }
        void AdamParameter::Set_Exp_Avg(torch::Tensor d_avg) {
            _d_avg = d_avg;
        }

        void AdamParameter::Set_Gradient(torch::Tensor d_param_grad) {
            if (_d_params_grad.dim() == 2) {
                if (d_param_grad.size(1) != _d_params_grad.size(1)) {
                    throw std::runtime_error("Gradient shape does not match parameter shape for " + Map_param_type_to_string(GetType()));
                }
            } else if (_d_params_grad.dim() == 3) {
                if (d_param_grad.size(1) != _d_params_grad.size(1) || d_param_grad.size(2) != _d_params_grad.size(2)) {
                    throw std::runtime_error("Gradient shape does not match parameter shape for " + Map_param_type_to_string(GetType()));
                }
            } else {
                throw std::runtime_error("Gradient shape does not match parameter shape for " + Map_param_type_to_string(GetType()));
            }
            _d_params_grad = d_param_grad;
        }
        void AdamParameter::Set_Param(torch::Tensor d_param) {
            if (d_param.dim() == 2) {
                if (d_param.size(1) != _d_params.size(1)) {
                    throw std::runtime_error("Gradient shape does not match parameter shape for " + Map_param_type_to_string(GetType()));
                }
            } else if (d_param.dim() == 3) {
                if (d_param.size(1) != _d_params.size(1) || d_param.size(2) != _d_params.size(2)) {
                    throw std::runtime_error("Params shape does not match parameter shape for " + Map_param_type_to_string(GetType()));
                }
            } else {
                throw std::runtime_error("Params shape does not match parameter shape for " + Map_param_type_to_string(GetType()));
            }
            _d_params = d_param;
        }

        void Adam::AddParameter(std::shared_ptr<AdamParameterBase> param) {
            _params[param->GetType()] = param;
        }

        __global__ void AdamUpdatePos_Scaling_Kernel(float* __restrict__ params,
                                                     const float* __restrict__ d_params_grad,
                                                     float* __restrict__ d_avg,
                                                     float* __restrict__ d_avg_sq,
                                                     int32_t step,
                                                     long size,
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
                const float param_grad = d_params_grad[idx];
                const int32_t current_step = step;

                avg = beta1 * avg + (1.f - beta1) * param_grad;
                avg_sq = beta2 * avg_sq + (1.f - beta2) * (param_grad * param_grad);

                // Bias correction terms
                const float bias_correction1 = 1.f - pow(beta1, (float)current_step);
                const float bias_correction2 = 1.f - pow(beta2, (float)current_step);
                const float bias_correction2_sqrt_inv = 1.f / sqrtf(bias_correction2);

                const float denom = sqrtf(avg_sq) * bias_correction2_sqrt_inv + epsilon;
                const float step_size = lr_t / bias_correction1;
                // update the weights/biases
                params[idx] -= step_size * (avg / denom);
                d_avg[idx] = avg;
                d_avg_sq[idx] = avg_sq;
            }
        }

        __global__ void AdamUpdateRotationKernel(float4* __restrict__ params,
                                                 const float4* __restrict__ d_params_grad,
                                                 float4* __restrict__ d_avg,
                                                 float4* __restrict__ d_avg_sq,
                                                 int32_t* __restrict__ d_steps,
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
                const int32_t current_step = ++d_steps[idx];

                // Bias correction terms
                float bias_correction1 = 1.0f - powf(beta1, current_step);
                float bias_correction2 = 1.0f - powf(beta2, current_step);
                float bias_correction2_sqrt = sqrtf(bias_correction2);

                avg.x = beta1 * avg.x + (1.f - beta1) * param_grad.x;
                avg.y = beta1 * avg.y + (1.f - beta1) * param_grad.y;
                avg.z = beta1 * avg.z + (1.f - beta1) * param_grad.z;
                avg.w = beta1 * avg.w + (1.f - beta1) * param_grad.w;

                // compute the new moving average of the squared gradient
                avg_sq.x = beta2 * avg_sq.x + (1.f - beta2) * param_grad.x * param_grad.x;
                avg_sq.y = beta2 * avg_sq.y + (1.f - beta2) * param_grad.y * param_grad.y;
                avg_sq.z = beta2 * avg_sq.z + (1.f - beta2) * param_grad.z * param_grad.z;
                avg_sq.w = beta2 * avg_sq.w + (1.f - beta2) * param_grad.w * param_grad.w;

                // Compute step size considering bias correction
                float step_size = -lr_t / bias_correction1;

                const float denom_x = sqrtf(avg_sq.x) / bias_correction2_sqrt + epsilon;
                const float denom_y = sqrtf(avg_sq.y) / bias_correction2_sqrt + epsilon;
                const float denom_z = sqrtf(avg_sq.z) / bias_correction2_sqrt + epsilon;
                const float denom_w = sqrtf(avg_sq.w) / bias_correction2_sqrt + epsilon;
                // update the weights/biases
                param.x = param.x + step_size * avg.x / denom_x;
                param.y = param.y + step_size * avg.y / denom_y;
                param.z = param.z + step_size * avg.z / denom_z;
                param.w = param.w + step_size * avg.w / denom_w;
                params[idx] = param;
                d_avg[idx] = avg;
                d_avg_sq[idx] = avg_sq;
            }
        }

        __global__ void AdamUpdateOpactiyKernel(float* params,
                                                const float* d_params_grad,
                                                float* d_avg,
                                                float* d_avg_sq,
                                                int32_t* __restrict__ d_steps,
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
                const int32_t current_step = ++d_steps[idx];

                // Bias correction terms
                float bias_correction1 = 1.0f - powf(beta1, current_step);
                float bias_correction2 = 1.0f - powf(beta2, current_step);
                float bias_correction2_sqrt = sqrtf(bias_correction2);

                avg = beta1 * avg + (1.f - beta1) * param_grad;

                // compute the new moving average of the squared gradient
                avg_sq = beta2 * avg_sq + (1.f - beta2) * param_grad * param_grad;

                const float step_size = -lr_t / bias_correction1;

                // update the weights/biases
                const float denom = sqrtf(avg_sq) / bias_correction2_sqrt + epsilon;
                param = param + step_size * avg / denom;

                params[idx] = param;
                d_avg[idx] = avg;
                d_avg_sq[idx] = avg_sq;
            }
        }

        __global__ void AdamUpdateFeatureKernel(float3* params,
                                                const float3* d_params_grad,
                                                float3* d_avg,
                                                float3* d_avg_sq,
                                                int32_t* __restrict__ d_steps,
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
                const int32_t current_step = ++d_steps[idx];
                // Bias correction terms
                float bias_correction1 = 1.0f - powf(beta1, current_step);
                float bias_correction2 = 1.0f - powf(beta2, current_step);
                float bias_correction2_sqrt = sqrtf(bias_correction2);

                for (int j = 0; j < dim1; j++) {
                    const int current_index = idx * dim1 + j;
                    float3 avg = d_avg[current_index];
                    float3 avg_sq = d_avg_sq[current_index];
                    float3 param = params[current_index];
                    const float3 param_grad = d_params_grad[current_index];

                    avg.x = beta1 * avg.x + (1.f - beta1) * param_grad.x;
                    avg.y = beta1 * avg.y + (1.f - beta1) * param_grad.y;
                    avg.z = beta1 * avg.z + (1.f - beta1) * param_grad.z;

                    // compute the new moving average of the squared gradient
                    avg_sq.x = beta2 * avg_sq.x + (1.f - beta2) * param_grad.x * param_grad.x;
                    avg_sq.y = beta2 * avg_sq.y + (1.f - beta2) * param_grad.y * param_grad.y;
                    avg_sq.z = beta2 * avg_sq.z + (1.f - beta2) * param_grad.z * param_grad.z;

                    // Compute step size considering bias correction
                    const float step_size = -lr_t / bias_correction1;
                    const float denom_x = sqrtf(avg_sq.x) / bias_correction2_sqrt + epsilon;
                    const float denom_y = sqrtf(avg_sq.y) / bias_correction2_sqrt + epsilon;
                    const float denom_z = sqrtf(avg_sq.z) / bias_correction2_sqrt + epsilon;

                    // update the weights/biases
                    param.x = param.x + step_size * avg.x / denom_x;
                    param.y = param.y + step_size * avg.y / denom_y;
                    param.z = param.z + step_size * avg.z / denom_z;
                    params[current_index] = param;
                    d_avg[current_index] = avg;
                    d_avg_sq[current_index] = avg_sq;
                }
            }
        }
    } // namespace optim
} // namespace gs