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
                                                     int32_t* __restrict__ d_steps,
                                                     int size, float lr_t,
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

        void AdamParameter::Step(cudaStream_t stream) {
            static const int threads_per_block = 256;
            static const int blocks_per_grid = (_d_params.size(0) + threads_per_block - 1) / threads_per_block;
            const bool avg_equal = _d_avg.sizes() == _d_params.sizes();
            const bool param_equal = _d_params.sizes() == _d_params_grad.sizes();
            // This has all be satified to update the paramaters successfully
            if (!avg_equal || !param_equal) {
                throw std::runtime_error("Gradient shape does not match parameter shape for " + Map_param_type_to_string(GetType()));
            }
            switch (_param_type) {
            case ParamType::Pos: {
                //                std::cout << "\nAdamUpdatePos_Scaling_Kernel " + Map_param_type_to_string(GetType()) + " shape: " << _d_params.size(0) << ", " << _d_params.size(1) << std::endl;
                //                std::cout << std::setprecision(6) << "lr " << _lr << ", beta1 " << _beta1 << ", beta2 " << _beta2 << std::endl;
                AdamUpdatePos_Scaling_Kernel<<<blocks_per_grid, threads_per_block>>>(
                    reinterpret_cast<pos_param_t*>(_d_params.data_ptr<float>()),
                    reinterpret_cast<pos_param_t*>(_d_params_grad.data_ptr<float>()),
                    reinterpret_cast<pos_param_t*>(_d_avg.data_ptr<float>()),
                    reinterpret_cast<pos_param_t*>(_d_avg_sq.data_ptr<float>()),
                    _d_steps.data_ptr<int32_t>(),
                    _d_params.size(0),
                    _lr,
                    _beta1,
                    _beta2,
                    _epsilon);
                CHECK_LAST_CUDA_ERROR();
                cudaDeviceSynchronize();
            } break;
            case ParamType::Scaling: {
                //                std::cout << "\nAdamUpdatePos_Scaling_Kernel " + Map_param_type_to_string(GetType()) + " shape: " << _d_params.size(0) << ", " << _d_params.size(1) << std::endl;
                //                std::cout <<  std::setprecision(6) <<"lr " << _lr << ", beta1 " << _beta1 << ", beta2 " << _beta2 << std::endl;
                AdamUpdatePos_Scaling_Kernel<<<blocks_per_grid, threads_per_block>>>(
                    reinterpret_cast<scaling_param_t*>(_d_params.data_ptr<float>()),
                    reinterpret_cast<scaling_param_t*>(_d_params_grad.data_ptr<float>()),
                    reinterpret_cast<scaling_param_t*>(_d_avg.data_ptr<float>()),
                    reinterpret_cast<scaling_param_t*>(_d_avg_sq.data_ptr<float>()),
                    _d_steps.data_ptr<int32_t>(),
                    _d_params.size(0),
                    _lr,
                    _beta1,
                    _beta2,
                    _epsilon);
                CHECK_LAST_CUDA_ERROR();
                cudaDeviceSynchronize();
            } break;
            case ParamType::Rotation: {
                //                std::cout << "AdamUpdateRotationKernel " + Map_param_type_to_string(GetType()) + " shape: " << _d_params.size(0) << ", " << _d_params.size(1) << std::endl;
                AdamUpdateRotationKernel<<<blocks_per_grid, threads_per_block>>>(
                    reinterpret_cast<rotation_param_t*>(_d_params.data_ptr<float>()),
                    reinterpret_cast<rotation_param_t*>(_d_params_grad.data_ptr<float>()),
                    reinterpret_cast<rotation_param_t*>(_d_avg.data_ptr<float>()),
                    reinterpret_cast<rotation_param_t*>(_d_avg_sq.data_ptr<float>()),
                    _d_steps.data_ptr<int32_t>(),
                    _d_params.size(0),
                    _lr,
                    _beta1,
                    _beta2,
                    _epsilon);
                CHECK_LAST_CUDA_ERROR();
                cudaDeviceSynchronize();
            } break;
            case ParamType::Opacity: {
                //                std::cout << "AdamUpdateOpactiyKernel " + Map_param_type_to_string(GetType()) + " shape: " << _d_params.size(0) << ", " << _d_params.size(1) << std::endl;
                AdamUpdateOpactiyKernel<<<blocks_per_grid, threads_per_block>>>(
                    reinterpret_cast<opacity_param_t*>(_d_params.data_ptr<float>()),
                    reinterpret_cast<opacity_param_t*>(_d_params_grad.data_ptr<float>()),
                    reinterpret_cast<opacity_param_t*>(_d_avg.data_ptr<float>()),
                    reinterpret_cast<opacity_param_t*>(_d_avg_sq.data_ptr<float>()),
                    _d_steps.data_ptr<int32_t>(),
                    _d_params.size(0),
                    _lr,
                    _beta1,
                    _beta2,
                    _epsilon);
                CHECK_LAST_CUDA_ERROR();
                cudaDeviceSynchronize();
            } break;
            case ParamType::Features_dc: {
                //                std::cout << "AdamUpdateFeatureKernel " + Map_param_type_to_string(GetType()) + " shape: " << _d_params.size(0) << ", " << _d_params.size(1) << std::endl;
                AdamUpdateFeatureKernel<<<blocks_per_grid, threads_per_block>>>(
                    reinterpret_cast<feature_dc_param_t*>(_d_params.data_ptr<float>()),
                    reinterpret_cast<feature_dc_param_t*>(_d_params_grad.data_ptr<float>()),
                    reinterpret_cast<feature_dc_param_t*>(_d_avg.data_ptr<float>()),
                    reinterpret_cast<feature_dc_param_t*>(_d_avg_sq.data_ptr<float>()),
                    _d_steps.data_ptr<int32_t>(),
                    _d_params.size(0),
                    _d_params.size(1),
                    _lr,
                    _beta1,
                    _beta2,
                    _epsilon);
                CHECK_LAST_CUDA_ERROR();
                cudaDeviceSynchronize();
            } break;
            case ParamType::Features_rest: {
                //                std::cout << "AdamUpdateFeatureKernel " + Map_param_type_to_string(GetType()) + " shape: " << _d_params.size(0) << ", " << _d_params.size(1) << std::endl;
                AdamUpdateFeatureKernel<<<blocks_per_grid, threads_per_block>>>(
                    reinterpret_cast<feature_rest_param_t*>(_d_params.data_ptr<float>()),
                    reinterpret_cast<feature_rest_param_t*>(_d_params_grad.data_ptr<float>()),
                    reinterpret_cast<feature_rest_param_t*>(_d_avg.data_ptr<float>()),
                    reinterpret_cast<feature_rest_param_t*>(_d_avg_sq.data_ptr<float>()),
                    _d_steps.data_ptr<int32_t>(),
                    _d_params.size(0),
                    _d_params.size(1),
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

        void AdamParameter::Set_Exp_Avg_Sq(torch::Tensor d_avg_sq) {
            _d_avg_sq = d_avg_sq;
        }
        void AdamParameter::Set_Exp_Avg(torch::Tensor d_avg) {
            _d_avg = d_avg;
        }

        void AdamParameter::Set_Gradient(torch::Tensor d_param_grad) {
            _d_params_grad = d_param_grad;
        }

        void Adam::AddParameter(std::shared_ptr<AdamParameterBase> param) {
            _params[param->GetType()] = param;
        }

        __global__ void AdamUpdatePos_Scaling_Kernel(float3* __restrict__ params,
                                                     const float3* __restrict__ d_params_grad,
                                                     float3* __restrict__ d_avg,
                                                     float3* __restrict__ d_avg_sq,
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
                float3 avg = d_avg[idx];
                float3 avg_sq = d_avg_sq[idx];
                float3 param = params[idx];
                const float3 param_grad = d_params_grad[idx];
                const int32_t current_step = ++d_steps[idx];

                // Bias correction terms
                float bias_correction1 = 1.0f - powf(beta1, current_step);
                float bias_correction2 = 1.0f - powf(beta2, current_step);
                float bias_correction2_sqrt = sqrtf(bias_correction2);

                avg.x = beta1 * avg.x + (1.f - beta1) * param_grad.x;
                avg.y = beta1 * avg.y + (1.f - beta1) * param_grad.y;
                avg.z = beta1 * avg.z + (1.f - beta1) * param_grad.z;

                // compute the new moving average of the squared gradient
                avg_sq.x = beta2 * avg_sq.x + (1.f - beta2) * param_grad.x * param_grad.x;
                avg_sq.y = beta2 * avg_sq.y + (1.f - beta2) * param_grad.y * param_grad.y;
                avg_sq.z = beta2 * avg_sq.z + (1.f - beta2) * param_grad.z * param_grad.z;

                // Compute step size considering bias correction
                float step_size = lr_t / bias_correction1;

                // update the weights/biases
                param.x -= step_size * avg.x / (sqrtf(avg_sq.x / bias_correction2_sqrt) + epsilon);
                param.y -= step_size * avg.y / (sqrtf(avg_sq.y / bias_correction2_sqrt) + epsilon);
                param.z -= step_size * avg.z / (sqrtf(avg_sq.z / bias_correction2_sqrt) + epsilon);

                params[idx] = param;
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
                float step_size = lr_t / bias_correction1;

                // update the weights/biases
                param.x -= step_size * avg.x / (sqrtf(avg_sq.x / bias_correction2_sqrt) + epsilon);
                param.y -= step_size * avg.y / (sqrtf(avg_sq.y / bias_correction2_sqrt) + epsilon);
                param.z -= step_size * avg.z / (sqrtf(avg_sq.z / bias_correction2_sqrt) + epsilon);
                param.w -= step_size * avg.w / (sqrtf(avg_sq.w / bias_correction2_sqrt) + epsilon);
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

                float step_size = lr_t / bias_correction1;

                // update the weights/biases
                param -= step_size * avg / (sqrtf(avg_sq / bias_correction2_sqrt) + epsilon);

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
                    float step_size = lr_t / bias_correction1;

                    // update the weights/biases
                    param.x -= step_size * avg.x / (sqrtf(avg_sq.x / bias_correction2_sqrt) + epsilon);
                    param.y -= step_size * avg.y / (sqrtf(avg_sq.y / bias_correction2_sqrt) + epsilon);
                    param.z -= step_size * avg.z / (sqrtf(avg_sq.z / bias_correction2_sqrt) + epsilon);
                    params[current_index] = param;
                    d_avg[current_index] = avg;
                    d_avg_sq[current_index] = avg_sq;
                }
            }
        }
    } // namespace optim
} // namespace gs