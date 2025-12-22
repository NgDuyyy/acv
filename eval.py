from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import utils.LSTM.utils.opts as opts
import models.LSTM as models
from utils.LSTM.dataloader import *
from utils.LSTM.dataloaderraw import *
import utils.LSTM.utils.eval_utils as eval_utils
import argparse
import utils.LSTM.utils.misc as utils
import models.LSTM.modules.losses as losses
import torch

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='result/log_lstm/model-best.pth',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='result/log_lstm/infos_-best.pkl',
                help='path to infos to evaluate')
parser.add_argument('--only_lang_eval', type=int, default=0,
                help='lang eval on saved results')
parser.add_argument('--force', type=int, default=0,
                help='force to evaluate no matter if there are results available')
parser.add_argument('--device', type=str, default='cuda',
                help='cpu or cuda')
# Dataloader workers (set 0 on Windows to avoid h5py pickling issues)
parser.add_argument('--num_workers', type=int, default=4,
                help='number of DataLoader workers; use 0 on Windows')
opts.add_eval_options(parser)
opts.add_diversity_opts(parser)
# CSV Export Options
parser.add_argument('--save_csv_results', type=int, default=1, help='dump results to csv files')
parser.add_argument('--predictions_csv', type=str, default='', help='custom path for predictions csv')
parser.add_argument('--metrics_csv', type=str, default='', help='custom path for metrics csv')

opt = parser.parse_args()

# Load infos
with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

# override and collect parameters
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model
        elif getattr(opt, k) is None:
            # If explicitly passed as None, fallback to checkpoint value
            setattr(opt, k, getattr(infos['opt'], k, None))

vocab = infos['vocab'] # ix -> word mapping

pred_fn = os.path.join('eval_results/', '.saved_pred_'+ opt.id + '_' + opt.split + '.pth')
result_fn = os.path.join('eval_results/', opt.id + '_' + opt.split + '.json')

if opt.only_lang_eval == 1 or (not opt.force and os.path.isfile(pred_fn)): 
    # if results existed, then skip, unless force is on
    if not opt.force:
        try:
            if os.path.isfile(result_fn):
                print(result_fn)
                json.load(open(result_fn, 'r'))
                print('already evaluated')
                os._exit(0)
        except:
            pass

    predictions, n_predictions = torch.load(pred_fn)
    lang_stats = eval_utils.language_eval(opt.input_json, predictions, n_predictions, vars(opt), opt.split)
    print(lang_stats)
    os._exit(0)

# At this point only_lang_eval if 0
if not opt.force:
    # Check out if 
    try:
        # if no pred exists, then continue
        tmp = torch.load(pred_fn)
        # if language_eval == 1, and no pred exists, then continue
        if opt.language_eval == 1:
            json.load(open(result_fn, 'r'))
        print('Result is already there')
        os._exit(0)
    except:
        pass

# Setup the model
opt.vocab = vocab
model = models.setup(opt)
del opt.vocab
model.load_state_dict(torch.load(opt.model, map_location='cpu'))
model.to(opt.device)
model.eval()
crit = losses.LanguageModelCriterion()

# Create the Data Loader instance
if len(opt.image_folder) == 0:
    loader = DataLoader(opt)
else:
    loader = DataLoaderRaw({'folder_path': opt.image_folder, 
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.dataset.ix_to_word = infos['vocab']


# Set sample options
opt.dataset = opt.input_json
loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, 
        vars(opt))

print('loss: ', loss)
print('lang_stats type:', type(lang_stats))
print('lang_stats value:', lang_stats)
if lang_stats:
    print(lang_stats)



# -----------------------------------------------------------------------------
# CSV Export Logic (Extracted and Adapted from pipeline)
# -----------------------------------------------------------------------------
import csv
from pathlib import Path

if opt.save_csv_results == 1:
    print(f"Exporting results to CSV...")
    
    # 1. Define Output Paths (Defaults or Arguments)
    # If no paths provided, save in eval_results with auto-generated names
    base_name = f"{opt.id}_{opt.split}"
    pred_csv_path = Path(opt.predictions_csv) if opt.predictions_csv else Path(f"eval_results/{base_name}_predictions.csv")
    metrics_csv_path = Path(opt.metrics_csv) if opt.metrics_csv else Path(f"eval_results/{base_name}_metrics.csv")

    pred_csv_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Prepare Data for Predictions CSV
    # Need to map image_id to filename.
    # We can get this from the 'loader' or the 'dataset' (coco_json) or the 'infos'.
    # loader.dataset.ix_to_word is vocab. 
    # loader.dataset.coco_annotation has images info if available.
    
    id_to_filename = {}
    id_to_captions = {}
    
    # Attempt to load ground truth annotations for filename and GT captions
    try:
        if opt.input_json:
            with open(opt.input_json, 'r') as f:
                ds = json.load(f)
                
            # Map id -> filename
            for img in ds.get('images', []):
                # Try various keys for filename
                fname = img.get('file_name') or img.get('filename') or img.get('file_path') or str(img.get('id'))
                id_to_filename[img['id']] = fname
            
            # Map id -> GT captions
            for ann in ds.get('annotations', []):
                img_id = ann.get('image_id')
                if img_id is not None:
                    id_to_captions.setdefault(img_id, []).append(ann.get('caption', ''))
    except Exception as e:
        print(f"[WARN] Could not load input_json for CSV metadata: {e}")

    # 3. Write Predictions CSV
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "filename", "caption_gt", "caption_pred"])
        
        for entry in split_predictions:
            img_id = entry['image_id']
            caption_pred = entry['caption']
            
            # Retrieve metadata
            filename = id_to_filename.get(img_id, str(img_id))
            # Get first GT caption if available
            gt_caps = id_to_captions.get(img_id, [])
            caption_gt = gt_caps[0] if gt_caps else ""
            
            writer.writerow([img_id, filename, caption_gt, caption_pred])
    
    print(f"Saved predictions to: {pred_csv_path}")

    # 4. Write Metrics CSV
    if lang_stats:
        write_header = not metrics_csv_path.exists()
        fieldnames = ["checkpoint", "split", "bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rouge", "cider", "spice"]
        
        row = {
            "checkpoint": opt.model,
            "split": opt.split,
            "bleu1": lang_stats.get("Bleu_1"),
            "bleu2": lang_stats.get("Bleu_2"),
            "bleu3": lang_stats.get("Bleu_3"),
            "bleu4": lang_stats.get("Bleu_4"),
            "meteor": lang_stats.get("METEOR"),
            "rouge": lang_stats.get("ROUGE_L"),
            "cider": lang_stats.get("CIDEr"),
            "spice": lang_stats.get("SPICE"),
        }
        
        with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            
        print(f"Saved metrics to: {metrics_csv_path}")

