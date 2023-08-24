// Copyright (c) 2023 Janusch Patas.
#include "tensor.cuh"

namespace gpu {

    template <typename T>
    Tensor<T>::Tensor(std::vector<uint32_t> size) : _size{size},
                                                    _stream{nullptr} {}

    template <typename T>
    Tensor<T>::Tensor(std::vector<uint32_t> size, cudaStream_t stream) : _size{size},
                                                                         _stream{stream} {}

    template <typename T>
    uint32_t Tensor<T>::Number_Elements() const {
        uint32_t num_elements = 1;
        for (const auto& dim : _size) {
            num_elements *= dim;
        }
        return num_elements;
    }

    template <typename T>
    size_t Tensor<T>::Number_Bytes() const {
        return Number_Elements() * sizeof(T);
    }
} // namespace gpu
