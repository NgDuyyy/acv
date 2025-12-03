// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
at::Tensor nms_cuda(const at::Tensor& dets, const float threshold);
#endif

at::Tensor nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold);

