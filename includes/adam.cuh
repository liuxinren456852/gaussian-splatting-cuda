// Copyright (c) 2023 Janusch Patas.
#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace optim {
    class AdamParameterBase {
    public:
        virtual ~AdamParameterBase() = default;
        virtual void Sync() = 0;
        virtual void Step() = 0;
    };
    enum class ParamType {
        Pos,
        Scaling,
        Rotation,
        Opacity,
        Features_dc,
        Features_rest,
    };
    template <typename T>
    class AdamParameter final : public AdamParameterBase {
    public:
        AdamParameter(ParamType param_type,
                      std::vector<int> shape,
                      float learning_rate,
                      float beta1,
                      float beta2,
                      float epsilon);
        ~AdamParameter() override;
        void Sync() override;
        void Step() override;

    protected:
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

    class Adam {
    public:
        void Sync();
        void Step();
        void AddParameter(std::shared_ptr<AdamParameterBase>& param);

    private:
        std::vector<std::shared_ptr<AdamParameterBase>> _params;
    };
} // namespace optim