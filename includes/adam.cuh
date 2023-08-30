// Copyright (c) 2023 Janusch Patas.
#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
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

        class AdamParameterBase {
        public:
            virtual ~AdamParameterBase() = default;
            virtual void Sync() = 0;
            virtual void Step() = 0;
            virtual ParamType GetType() = 0;
            virtual void UpdateLearningRate(float lr) = 0;
        };

        template <typename T>
        class AdamParameter final : public AdamParameterBase {
        public:
            AdamParameter(ParamType param_type,
                          std::vector<int> shape,
                          float learning_rate,
                          float beta1 = 0.9f,
                          float beta2 = 0.999f,
                          float epsilon = 1e-15);
            ~AdamParameter() override;
            void Sync() override;
            void Step() override;
            inline ParamType GetType() override { return _param_type; }
            inline void UpdateLearningRate(float lr) override { _lr = lr; }
            T* Get_Exp_Avg() { return _d_avg; }
            T* Get_Exp_Avg_Sq() { return _d_avg_sq; }
            void Set_Exp_Avg(T* d_avg, std::vector<int> size);
            void Set_Exp_Avg_Sq(T* d_avg_sq, std::vector<int> size);
            void Set_Gradient(T* d_param_grad, std::vector<int> size);
            void Update_Parameter_Pointer(T* d_param) { _d_params = d_param; };

        private:
            T* _d_params{};
            T* _d_params_grad{};
            T* _d_avg{};
            T* _d_avg_sq{};

            ParamType _param_type;
            std::vector<int> _shape;
            float _lr;
            float _beta1;
            float _beta2;
            float _epsilon;
            std::string _param_name;
            cudaStream_t _stream{};
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
            void Sync();
            void Step();
            void AddParameter(std::shared_ptr<AdamParameterBase> param);
            inline std::shared_ptr<AdamParameterBase> GetParameters(ParamType paramType) { return _params[paramType]; };

            template <typename T>
            inline std::shared_ptr<AdamParameter<T>> GetAdamParameter(ParamType param_type) {
                return std::dynamic_pointer_cast<AdamParameter<T>>(_params[param_type]);
            }

        private:
            std::unordered_map<ParamType, std::shared_ptr<AdamParameterBase>> _params;
        };

        // preinstantiate templates -> faster compile time
        template class AdamParameter<float>;
        template class AdamParameter<float3>;
        template class AdamParameter<float4>;

    } // namespace optim
} // namespace gs