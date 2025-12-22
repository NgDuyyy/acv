
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import cv2
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports for Faster R-CNN (Feature Extraction)
from utils.feature_extracting import config as cfg_frcnn
from models.feature_extracting.faster_rcnn import resnet as faster_rcnn
from utils.feature_extracting.blob import im_list_to_blob

# Imports for LSTM (Captioning)
import models.LSTM as models
import utils.LSTM.utils.misc as utils

def parse_args():
    parser = argparse.ArgumentParser(description="Image Captioning Inference")
    parser.add_argument("--image", type=str, default="test_image/2hotman.jpg", help="Path to input image")
    parser.add_argument("--frcnn_model", type=str, default="models/feature_extracting/pretrained_model/faster_rcnn_res101_vg.pth", help="Path to Faster R-CNN checkpoint")
    parser.add_argument("--caption_model", type=str, default="result/log_lstm/model-best.pth", help="Path to Captioning Model checkpoint")
    parser.add_argument("--infos_path", type=str, default="result/log_lstm/infos_-best.pkl", help="Path to infos.pkl file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    return parser.parse_args()

# ==========================================
# 1. Feature Extraction (Faster R-CNN)
# ==========================================

def get_image_blob(im):
    """Converts an image into a network input."""
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg_frcnn.cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg_frcnn.cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > cfg_frcnn.cfg.TEST.MAX_SIZE:
            im_scale = float(cfg_frcnn.cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)

def load_frcnn(checkpoint_path, use_gpu=True):
    if not os.path.exists(checkpoint_path):
        # Falback check
        alt_path = os.path.join("FRCNN", "models", os.path.basename(checkpoint_path))
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            print(f"Error: Faster R-CNN checkpoint not found at {checkpoint_path}")
            sys.exit(1)
            
    cfg_frcnn.cfg.USE_GPU_NMS = use_gpu
    cfg_frcnn.cfg.CUDA = use_gpu
    cfg_frcnn.cfg.POOLING_MODE = 'align'
    # Update config to match checkpoint (12 anchors: 4 scales * 3 ratios)
    cfg_frcnn.cfg.ANCHOR_SCALES = [4, 8, 16, 32]
    
    classes = ['class_' + str(i) for i in range(1601)]
    model = faster_rcnn.resnet(classes, 101, pretrained=False, class_agnostic=False)
    model.create_architecture()
    
    print(f"Loading Faster R-CNN from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu' if not use_gpu else None)
    model.load_state_dict(checkpoint['model'])
    
    if use_gpu:
        model.cuda()
    model.eval()
    return model

def get_features(model, image_path, use_gpu=True):
    im_in = cv2.imread(image_path)
    if im_in is None:
        print(f"Error: Could not read image {image_path}")
        sys.exit(1)
        
    if len(im_in.shape) == 2:
        im_in = im_in[:,:,np.newaxis]
        im_in = np.concatenate((im_in,im_in,im_in), axis=2)
        
    blobs, im_scales = get_image_blob(im_in)
    im_info_np = np.array([[blobs.shape[1], blobs.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy(blobs).permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    with torch.no_grad():
        if use_gpu:
            im_data_pt = im_data_pt.cuda()
            im_info_pt = im_info_pt.cuda()

        gt_boxes = torch.zeros((1, 1, 5))
        num_boxes = torch.zeros(1)
        if use_gpu:
            gt_boxes = gt_boxes.cuda()
            num_boxes = num_boxes.cuda()

        # Forward pass
        # Request pool_feat=True to get valid pooled features
        res = model(im_data_pt, im_info_pt, gt_boxes, num_boxes, pool_feat=True)
        # res tuple: rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label, pooled_feat
        pooled_feat = res[-1] 

    return pooled_feat

# ==========================================
# 2. Caption Generation (LSTM)
# ==========================================

def load_captioner(model_path, infos_path, use_gpu=True):
    if not os.path.exists(infos_path):
        print(f"Error: Infos file not found at {infos_path}")
        sys.exit(1)
        
    with open(infos_path, 'rb') as f:
        infos = utils.pickle_load(f)
        
    opt = infos['opt']
    # Force evaluation mode settings
    opt.use_gpu = use_gpu
    
    # Load vocabulary
    vocab = infos['vocab'] # ix_to_word
    opt.vocab = vocab
    
    print(f"Loading Caption Model from {model_path}...")
    model = models.setup(opt)
    model.load_state_dict(torch.load(model_path, map_location='cpu' if not use_gpu else None))
    
    if use_gpu:
        model.cuda()
    model.eval()
    
    return model, vocab, opt

def generate_caption(model, vocab, feature_tensor, use_gpu=True):
    # feature_tensor: (1, 2048) or (1, N, 2048)
    
    # Prepare inputs for 'sample' method
    # AttModel typically expects: fc_feats, att_feats
    
    # If feature is (1, 2048), we treat it as fc_feats. 
    # But for Attention models, we need att_feats.
    # In this dataset/project, it seems 'pooled_feat' is treated as 'att_feats' 
    # and we take the mean for 'fc_feats'.
    
    att_feats = feature_tensor
    if len(att_feats.shape) == 2:
        # Standardize to (Batch, Num_Regions, Feat_Dim) ??
        # Or if it came from ROI pooling, it might be (Num_ROIs, 2048).
        # If batch=1, and we have multiple ROIs, it should be (1, N, 2048).
        
        # If the input was just 1 image -> 1 global feature? 
        # The Faster R-CNN here seems to output features for BOXES (ROIs).
        # Let's inspect shape at runtime if needed.
        # For now, let's assume it matches what the DataLoader delivers.
        
        # In dataloader.py, it loads .npz 'feat'. 
        # Then: fc_feat = feat.mean(0)
        #       att_feat = feat
        # So we do the same.
        pass
        
    # Sample
    att_feats = feature_tensor
    if att_feats.dim() == 2:
        # (N, 2048) -> (1, N, 2048) for batch dimension
        att_feats = att_feats.unsqueeze(0)
        
    fc_feats = att_feats.mean(1) # Average over regions to get global FC feat

    if use_gpu:
        fc_feats = fc_feats.cuda()
        att_feats = att_feats.cuda()
        
    with torch.no_grad():
        # Using beam search by default or greedy
        # Based on eval_utils.py: model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        eval_kwargs = {'beam_size': 5, 'sample_n': 1}
        # We pass att_masks=None since we have a single image or no masking needed for 1 element
        seq, seq_logprobs = model(fc_feats, att_feats, att_masks=None, opt=eval_kwargs, mode='sample')
        
    # Decode
    sents = utils.decode_sequence(vocab, seq)
    return sents[0]


def main():
    args = parse_args()
    
    use_gpu = args.gpu and torch.cuda.is_available()
    if use_gpu:
        print("Using GPU...")
    else:
        print("Using CPU...")
        
    # 1. Load models
    frcnn = load_frcnn(args.frcnn_model, use_gpu)
    caption_model, vocab, opt = load_captioner(args.caption_model, args.infos_path, use_gpu)
    
    # 2. Process Image
    print(f"Processing image: {args.image}")
    features = get_features(frcnn, args.image, use_gpu)
    
    # 3. Generate Caption
    caption = generate_caption(caption_model, vocab, features, use_gpu)
    
    print("-" * 30)
    print(f"Caption: {caption}")
    print("-" * 30)

if __name__ == "__main__":
    main()
