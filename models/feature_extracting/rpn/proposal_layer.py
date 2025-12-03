from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
import yaml
from utils.feature_extracting.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
# from model.nms.nms_wrapper import nms
from models.feature_extracting.roi_layers import nms
import pdb

DEBUG = False

class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales, ratios):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales),
            ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # top[0].reshape(1, 5)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

    def forward(self, input):

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)


        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        scores = input[0][:, self._num_anchors:, :, :]
        bbox_deltas = input[1]
        im_info = input[2]
        cfg_key = input[3]

        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
        min_size      = cfg[cfg_key].RPN_MIN_SIZE

        batch_size = bbox_deltas.size(0)

        feat_height, feat_width = scores.size(2), scores.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()

        A = self._num_anchors
        K = shifts.size(0)

        # Ensure anchors and shifts are on the same device as scores
        device = scores.device
        self._anchors = self._anchors.type_as(scores).to(device)
        shifts = shifts.to(device)
        # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:

        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)

        # Same story for the scores:
        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)

        # Ensure device consistency before bbox operations
        device = scores.device
        anchors = anchors.to(device)
        bbox_deltas = bbox_deltas.to(device)
        im_info = im_info.to(device)

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info, batch_size)
        
        # Ensure proposals are on correct device
        proposals = proposals.to(device)
        # proposals = clip_boxes_batch(proposals, im_info, batch_size)

        # assign the score to 0 if it's non keep.
        # keep = self._filter_boxes(proposals, min_size * im_info[:, 2])

        # trim keep index to make it euqal over batch
        # keep_idx = torch.cat(tuple(keep_idx), 0)

        # scores_keep = scores.view(-1)[keep_idx].view(batch_size, trim_size)
        # proposals_keep = proposals.view(-1, 4)[keep_idx, :].contiguous().view(batch_size, trim_size, 4)

        # _, order = torch.sort(scores_keep, 1, True)

        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)

        # Ensure output tensor is on the correct device
        device = scores.device
        output = scores.new(batch_size, post_nms_topN, 5).zero_().to(device)
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            # Check if we have any proposals
            if proposals_single.numel() == 0:
                # No proposals, create empty result
                keep_idx_i = torch.tensor([], dtype=torch.long, device=proposals_single.device)
            else:
                # Ensure all tensors are on the same device before NMS
                device = proposals_single.device
                proposals_single = proposals_single.contiguous().to(device)
                scores_single = scores_single.contiguous().to(device)
                
                # 6. apply nms (e.g. threshold = 0.7)
                # 7. take after_nms_topN (e.g. 300)
                # 8. return the top proposals (-> RoIs top)
                try:
                    # Convert to CPU for NMS if needed, then back to GPU
                    if device.type == 'cuda':
                        proposals_cpu = proposals_single.cpu()
                        scores_cpu = scores_single.squeeze(1).cpu()
                        keep_idx_cpu = nms(proposals_cpu, scores_cpu, nms_thresh)
                        keep_idx_i = keep_idx_cpu.to(device)
                    else:
                        keep_idx_i = nms(proposals_single, scores_single.squeeze(1), nms_thresh)
                except Exception as nms_error:
                    print(f"[WARNING] NMS failed: {nms_error}, using fallback")
                    # Fallback: take top proposals by score without NMS
                    _, top_indices = torch.topk(scores_single.squeeze(1), 
                                              min(scores_single.size(0), post_nms_topN))
                    keep_idx_i = top_indices
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            
            if keep_idx_i.numel() > 0:
                proposals_single = proposals_single[keep_idx_i, :]
                scores_single = scores_single[keep_idx_i, :]
            else:
                # No proposals after NMS, create empty tensors
                device = proposals_single.device
                proposals_single = torch.zeros(0, 4, device=device, dtype=proposals_single.dtype)
                scores_single = torch.zeros(0, 1, device=device, dtype=scores_single.dtype)

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single

        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1,1).expand_as(ws)) & (hs >= min_size.view(-1,1).expand_as(hs)))
        return keep
