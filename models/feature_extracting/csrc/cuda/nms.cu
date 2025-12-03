
// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <cstdint>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cfloat>
#include <cmath>
#include <vector>
#include <iostream>
#include <cstring>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

__host__ __device__ inline int CeilDiv(const int a, const int b) {
  return (a + b - 1) / b;
}

template <typename T>
__device__ inline float devIoU(const T* const a, const T* const b) {
  T left   = max(a[0], b[0]);
  T right  = min(a[2], b[2]);
  T top    = max(a[1], b[1]);
  T bottom = min(a[3], b[3]);
  T width  = max(right - left + static_cast<T>(1), static_cast<T>(0));
  T height = max(bottom - top + static_cast<T>(1), static_cast<T>(0));
  T interS = width * height;
  T Sa = (a[2] - a[0] + static_cast<T>(1)) * (a[3] - a[1] + static_cast<T>(1));
  T Sb = (b[2] - b[0] + static_cast<T>(1)) * (b[3] - b[1] + static_cast<T>(1));
  return interS / (Sa + Sb - interS);
}

template <typename T>
__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const T* dev_boxes, unsigned long long* dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  const int threadsPerBlock = 64;
  __shared__ T block_boxes[threadsPerBlock * 5];

  if (row_start > col_start)
    return;

  const int row_size =
      min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
      min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  if (threadIdx.x < col_size) {
    for (int i = 0; i < 5; ++i) {
      block_boxes[threadIdx.x * 5 + i] =
          dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + i];
    }
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const T* cur_box = dev_boxes + cur_box_idx * 5;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start)
      start = threadIdx.x + 1;
    for (int i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = CeilDiv(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

at::Tensor nms_cuda(const at::Tensor& dets, const float threshold) {
  TORCH_CHECK(dets.is_cuda(), "dets must be a CUDA tensor");

  const int threadsPerBlock = 64;
  const int boxes_num = dets.size(0);
  const int col_blocks = CeilDiv(boxes_num, threadsPerBlock);

  at::Tensor mask =
      at::zeros({boxes_num * col_blocks}, dets.options().dtype(at::kLong));

  if (boxes_num == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong));
  }

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);

  nms_kernel<float><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      boxes_num,
      threshold,
      dets.contiguous().data_ptr<float>(),
      (unsigned long long*)mask.data_ptr<int64_t>());

  C10_CUDA_CHECK(cudaGetLastError());

  at::Tensor mask_cpu = mask.to(at::kCPU);
  auto* mask_cpu_ptr = mask_cpu.data_ptr<int64_t>();
  std::vector<unsigned long long> mask_host(
      mask_cpu_ptr,
      mask_cpu_ptr + mask_cpu.numel());

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  std::vector<int> keep;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;
    if (!(remv[nblock] & (1ULL << inblock))) {
      keep.push_back(i);
      unsigned long long* p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  return at::from_blob(
           keep.data(),
           {static_cast<int64_t>(keep.size())},
           dets.options().dtype(at::kLong)
       ).clone();
}

