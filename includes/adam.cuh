// Copyright (c) 2023 Janusch Patas.
#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

namespace gs {
    namespace optim {
        enum class ParamType {
            Pos,
            Scaling,
            Rotation,
            Opacity,
            Features_dc,
            Features_rest,
        };

        inline std::string Map_param_type_to_string(ParamType param_type) {
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

        class AdamParameterBase {
        public:
            virtual ~AdamParameterBase() = default;
            virtual void Step(cudaStream_t stream) = 0;
            virtual ParamType GetType() = 0;
            virtual void UpdateLearningRate(float lr) = 0;
        };

        class AdamParameter final : public AdamParameterBase {
        public:
            AdamParameter(ParamType param_type,
                          torch::Tensor param,
                          float learning_rate,
                          cudaStream_t stream,
                          float beta1 = 0.9f,
                          float beta2 = 0.999f,
                          float epsilon = 1e-8);
            ~AdamParameter() override;
            void Step(cudaStream_t stream) override;
            inline ParamType GetType() override { return _param_type; }
            inline void UpdateLearningRate(float lr) override { _lr = lr; }
            torch::Tensor Get_Exp_Avg() { return _d_avg; }
            torch::Tensor Get_Exp_Avg_Sq() { return _d_avg_sq; }
            void Set_Exp_Avg(torch::Tensor d_avg);
            void Set_Exp_Avg_Sq(torch::Tensor d_avg_sq);
            void Set_Gradient(torch::Tensor d_param_grad);
            void Set_Param(torch::Tensor d_param) { _d_params = d_param; }

        private:
            torch::Tensor _d_params;
            torch::Tensor _d_params_grad;
            torch::Tensor _d_avg;
            torch::Tensor _d_avg_sq;

            ParamType _param_type;
            float _lr;
            float _beta1;
            float _beta2;
            float _epsilon;
            std::string _param_name;
        };

        // define types for convenience
        using opacity_param_t = float;
        using scaling_param_t = float3;
        using rotation_param_t = float4;
        using pos_param_t = float3;
        using feature_dc_param_t = float3;
        using feature_rest_param_t = float3;

        class Adam {
        public:
            void Step(cudaStream_t stream);
            void AddParameter(std::shared_ptr<AdamParameterBase> param);

            inline std::shared_ptr<AdamParameter> GetAdamParameter(ParamType param_type) {
                return std::dynamic_pointer_cast<AdamParameter>(_params[param_type]);
            }

        private:
            std::unordered_map<ParamType, std::shared_ptr<AdamParameterBase>> _params;
        };

    } // namespace optim
} // namespace gs