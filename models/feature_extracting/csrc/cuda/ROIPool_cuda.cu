
// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cmath>
#include <cfloat>
#include <algorithm>
#include <cstdint>

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

__host__ __device__ inline int CeilDiv(const int a, const int b) {
  return (a + b - 1) / b;
}

template <typename T>
__global__ void RoIPoolFForward(const int nthreads,
                                const T* bottom_data,
                                const T spatial_scale,
                                const int channels,
                                const int height,
                                const int width,
                                const int pooled_height,
                                const int pooled_width,
                                const T* bottom_rois,
                                T* top_data,
                                int64_t* argmax_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);

    int roi_start_w = static_cast<int>(::round(offset_bottom_rois[1] * spatial_scale));
    int roi_start_h = static_cast<int>(::round(offset_bottom_rois[2] * spatial_scale));
    int roi_end_w   = static_cast<int>(::round(offset_bottom_rois[3] * spatial_scale));
    int roi_end_h   = static_cast<int>(::round(offset_bottom_rois[4] * spatial_scale));

    // Force malformed ROIs to be 1x1
    int roi_width  = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width)  / static_cast<T>(pooled_width);

    int hstart = static_cast<int>(::floor(static_cast<T>(ph) * bin_size_h));
    int wstart = static_cast<int>(::floor(static_cast<T>(pw) * bin_size_w));
    int hend   = static_cast<int>(::ceil (static_cast<T>(ph + 1) * bin_size_h));
    int wend   = static_cast<int>(::ceil (static_cast<T>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend   = min(max(hend   + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend   = min(max(wend   + roi_start_w, 0), width);

    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    T maxval = is_empty ? static_cast<T>(0) : -static_cast<T>(FLT_MAX);
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;

    const T* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        T val = offset_bottom_data[bottom_index];
        if (val > maxval) {
          maxval = val;
          maxidx = bottom_index;
        }
      }
    }

    top_data[index]    = maxval;
    argmax_data[index] = maxidx;
  }
}

template <typename T>
__global__ void RoIPoolFBackward(const int nthreads,
                                 const T* top_diff,
                                 const int64_t* argmax_data,
                                 const int num_rois,
                                 const T spatial_scale,
                                 const int channels,
                                 const int height,
                                 const int width,
                                 const int pooled_height,
                                 const int pooled_width,
                                 T* bottom_diff,
                                 const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c  = (index / pooled_width / pooled_height) % channels;
    int n  = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);

    int bottom_offset = (roi_batch_ind * channels + c) * height * width;
    int top_offset    = (n * channels + c) * pooled_height * pooled_width;

    const T*   offset_top_diff     = top_diff + top_offset;
    T*         offset_bottom_diff  = bottom_diff + bottom_offset;
    const int64_t* offset_argmax_data  = argmax_data + top_offset;


    int argmax = offset_argmax_data[ph * pooled_width + pw];
    if (argmax != -1) {
      atomicAdd(offset_bottom_diff + argmax,
                static_cast<T>(offset_top_diff[ph * pooled_width + pw]));
    }
  }
}

std::tuple<at::Tensor, at::Tensor>
ROIPool_forward_cuda(const at::Tensor& input,
                     const at::Tensor& rois,
                     const float spatial_scale,
                     const int pooled_height,
                     const int pooled_width) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(rois.is_cuda(),  "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height   = input.size(2);
  auto width    = input.size(3);

  auto output = at::empty(
      {num_rois, channels, pooled_height, pooled_width},
      input.options());

  auto argmax = at::zeros(
    {num_rois, channels, pooled_height, pooled_width},
    input.options().dtype(at::kLong));

  int output_size = static_cast<int>(output.numel());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 block(512);
  dim3 grid(std::min(CeilDiv(output_size, static_cast<int>(block.x)), 4096));

  if (output_size == 0) {
    C10_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(output, argmax);
  }

  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "ROIPool_forward_cuda", [&] {
        RoIPoolFForward<scalar_t><<<grid, block, 0, stream>>>(
            output_size,
            input.contiguous().data_ptr<scalar_t>(),
            static_cast<scalar_t>(spatial_scale),
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            rois.contiguous().data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            argmax.data_ptr<int64_t>());
      });

  C10_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(output, argmax);
}

// TODO remove the dependency on input and use instead its sizes -> save memory
at::Tensor ROIPool_backward_cuda(const at::Tensor& grad,
                                 const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const at::Tensor& argmax,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int batch_size,
                                 const int channels,
                                 const int height,
                                 const int width) {
  TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");

  auto grad_input = at::zeros(
      {batch_size, channels, height, width},
      grad.options());

  int nthreads = static_cast<int>(grad.numel());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (nthreads == 0) {
    C10_CUDA_CHECK(cudaGetLastError());
    return grad_input;
  }

  dim3 block(512);
  dim3 grid(std::min(CeilDiv(nthreads, static_cast<int>(block.x)), 4096));

  AT_DISPATCH_FLOATING_TYPES(
      grad.scalar_type(), "ROIPool_backward_cuda", [&] {
        RoIPoolFBackward<scalar_t><<<grid, block, 0, stream>>>(
            nthreads,
            grad.contiguous().data_ptr<scalar_t>(),
            argmax.contiguous().data_ptr<int64_t>(),
            rois.size(0),
            static_cast<scalar_t>(spatial_scale),
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            grad_input.data_ptr<scalar_t>(),
            rois.contiguous().data_ptr<scalar_t>());
      });

  C10_CUDA_CHECK(cudaGetLastError());
  return grad_input;
}
