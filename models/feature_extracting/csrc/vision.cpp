// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "nms.h"
#ifdef WITH_CUDA
#include "nms.h"
#endif
#include "ROIAlign.h"
#include "ROIPool.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("nms", &nms, "non-maximum suppression (both CPU/CUDA)");
  m.def("nms_cuda", &nms_cuda, "non-maximum suppression (CUDA)"); // thêm dòng này
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");
}
