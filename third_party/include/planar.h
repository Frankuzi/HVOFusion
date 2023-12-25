#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CHECK_CUDA(x)                                             \
  do                                                              \
  {                                                               \
    TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor"); \
  } while (0)

#define CHECK_CONTIGUOUS(x)                                            \
  do                                                                   \
  {                                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor"); \
  } while (0)

#define CHECK_IS_INT(x)                                 \
  do                                                    \
  {                                                     \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, \
                #x " must be an int tensor");           \
  } while (0)

#define CHECK_IS_LONG(x)                                 \
  do                                                    \
  {                                                     \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Long, \
                #x " must be an int tensor");           \
  } while (0)

#define CHECK_IS_FLOAT(x)                                 \
  do                                                      \
  {                                                       \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, \
                #x " must be a float tensor");            \
  } while (0)

void find_planar_kernel_wrapper(
    const long faceNum, const float thv, 
    at::Tensor indices, 
    at::Tensor vertexNormals,
    at::Tensor faceNormals, 
    at::Tensor isPlanar, 
    at::Tensor planarNormals
);