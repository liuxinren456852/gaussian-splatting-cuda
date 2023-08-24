// Copyright (c) 2023 Janusch Patas.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace gpu {
    using Size = std::vector<uint32_t>;

    template <typename T>
    class Tensor {
    public:
        explicit Tensor(Size size);
        explicit Tensor(Size size, cudaStream_t stream);

        [[nodiscard]] uint32_t Number_Elements() const;
        [[nodiscard]] size_t Number_Bytes() const;

    private:
        Size _size;
        cudaStream_t _stream;
    };
} // namespace gpu
