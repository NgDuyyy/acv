# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torchvision.ops import nms as _nms

def nms(boxes, scores_or_thresh, thresh=None):
    """
    Dispatch to torchvision.ops.nms
    Supports two signatures:
    1. nms(dets, thresh): dets=(N,5), thresh=float
    2. nms(boxes, scores, thresh): boxes=(N,4), scores=(N,), thresh=float
    """
    if thresh is None:
        # Case 1: nms(dets, thresh)
        dets = boxes
        actual_thresh = scores_or_thresh
        if dets.shape[0] == 0:
            return torch.empty(0, dtype=torch.long, device=dets.device)
        return _nms(dets[:, :4], dets[:, 4], actual_thresh)
    else:
        # Case 2: nms(boxes, scores, thresh)
        scores = scores_or_thresh
        actual_thresh = thresh
        if boxes.shape[0] == 0:
            return torch.empty(0, dtype=torch.long, device=boxes.device)
        return _nms(boxes, scores, actual_thresh)
