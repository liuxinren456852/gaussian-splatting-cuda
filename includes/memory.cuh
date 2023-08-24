// Copyright (c) 2023 Janusch Patas.
#pragma once
#include <cuda_runtime.h>

namespace gpu {
    template <typename T>
    class Memory {
    public:
        Memory(size_t size) {}
        Memory(size_t size, cudaStream_t stream) {}
        ~Memory() {}

    private:
        size_t _size;
        cudaStream_t _stream;
    };
} // namespace gpu