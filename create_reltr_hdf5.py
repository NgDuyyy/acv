
import argparse
import json
import os
import sys
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import h5py
import numpy as np

# Add RelTR to path (No longer needed as we use package structure)
# sys.path.append(os.getcwd())

# Patch pathlib for Windows checkpoint loading (moved to main)
import pathlib

from models.RelTR import build_model

def get_args_parser():
    parser = argparse.ArgumentParser('RelTR Feature Extraction', add_help=False)
    # Basic RelTR args
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_entities', default=100, type=int)
    parser.add_argument('--num_triplets', default=200, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--aux_loss', action='store_false') # default True -> dest=aux_loss
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--resume', default='data/RelTR_ckpt/checkpoint0149.pth', help='checkpoint')
    
    # Loss args (needed for build_model but unused here)
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--set_iou_threshold', default=0.7, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float)
    parser.add_argument('--return_interm_layers', action='store_true')
    
    # Extraction specific args
    parser.add_argument('--json_path', default=r"E:\hoc_voi_cha_hanh\advanced_cv\project\decoder_lstm\data\LSTM\data_merged.json")
    parser.add_argument('--data_root', default=r"E:\hoc_voi_cha_hanh\advanced_cv\project\decoder_lstm\data")
    parser.add_argument('--output_h5', default=r"E:\hoc_voi_cha_hanh\advanced_cv\project\decoder_lstm\data\features\reltr_features.h5")
    parser.add_argument('--batch_size', default=1, type=int, help="Currently only supports 1")
    parser.add_argument('--limit', default=-1, type=int, help="Limit number of images for testing")
    
    return parser

def main(args):
    print(f"Loading model from {args.resume}...")
    model, _, _ = build_model(args)
    model.to(args.device)
    
    # Safe checkpoint loading with OS detection
    import platform
    temp = None
    if platform.system() == 'Windows':
        # On Windows, map PosixPath to WindowsPath to load Linux checkpoints
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
    
    try:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return
    finally:
        # Restore pathlib
        if platform.system() == 'Windows' and temp:
            pathlib.PosixPath = temp

    model.eval()

    # Restore pathlib (good practice) - Removed as handled in finally block
    # pathlib.PosixPath = temp

    # Image Transforms
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"Loading data from {args.json_path}...")
    with open(args.json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Found {len(data['images'])} images.")

    images = data['images']
    if args.limit > 0:
        images = images[:args.limit]
        print(f"limiting to first {args.limit} images.")

    h5_mode = 'w'
    if os.path.exists(args.output_h5):
        # We might want to append? For now let's just write fresh or 'r+' if we were clever
        # but let's stick to 'w' for simplicity of this script
        print(f"Output file {args.output_h5} exists. Overwriting.")
    
    f_h5 = h5py.File(args.output_h5, h5_mode)
    
    # Create dataset for keys if needed? 
    # Usually h5py dictionary style is fine: f[key] = data
    
    # We will map split name to folder name
    split_map = {
        'train': os.path.join('train', 'train_images'),
        'val': os.path.join('val', 'val_images'),
        'test': os.path.join('test', 'test_images'),
        'restval': os.path.join('train', 'train_images') # Assuming restval is in train_images?
    }

    print("Starting extraction...")
    count = 0
    with torch.no_grad():
        for img_info in tqdm(images):
            img_id = str(img_info['id'])
            
            # Check if exists
            if img_id in f_h5:
                continue

            split = img_info.get('split', 'train')
            if split not in split_map:
                # heuristic fallback
                folder = 'train/train_images' 
            else:
                folder = split_map[split]
            
            rel_path = os.path.join(folder, img_info.get('file_path', ''))
            abs_path = os.path.join(args.data_root, rel_path)

            if not os.path.exists(abs_path):
                # Debug paths
                print(f"Warning: {abs_path} not found. (Split: {split}, File: {img_info.get('file_path', '')})")
                continue

            try:
                # Load Image
                im = Image.open(abs_path).convert("RGB")
                img_tensor = transform(im).unsqueeze(0).to(args.device)

                # Inference
                outputs = model(img_tensor)
                
                # Extract features
                # The corrected reltr.py returns 'rel_features'
                if 'rel_features' in outputs:
                    rel_feats_tensor = outputs['rel_features']
                    # print(f"DEBUG: Output shape: {rel_feats_tensor.shape}")
                    rel_feats = rel_feats_tensor.cpu().numpy() 
                    
                    # Handle batch dimension safely
                    if len(rel_feats.shape) == 3 and rel_feats.shape[0] == 1:
                         rel_feats = rel_feats[0]
                    
                    # Save to h5
                    f_h5.create_dataset(img_id, data=rel_feats, dtype='float32')
                    count += 1
                else:
                    print(f"Error: 'rel_features' key missing in model output. Did you patch reltr.py?")
                    break
                    
            except Exception as e:
                print(f"Error processing {img_id}: {e}")
                # continue

            if count % 100 == 0:
                f_h5.flush()
                
    f_h5.close()
    print(f"Finished. Processed {count} images.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR Batch Extraction', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
