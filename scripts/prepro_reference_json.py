# coding: utf-8
"""
Create a reference json file used for evaluation with `coco-caption` repo.
Used when reference json is not provided, (e.g., flickr30k, or you have your own split of train/val/test)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
import sys
import hashlib
from random import shuffle, seed


def main(params):

    with open(params['input_json'][0], 'r', encoding='utf-8') as f:
        data = json.load(f)

    # create output json file
    out = {'info': {'description': 'This is stable 1.0 version of the 2014 MS COCO dataset.', 'url': 'http://mscoco.org', 'version': '1.0', 'year': 2014, 'contributor': 'Microsoft COCO group', 'date_created': '2015-01-27 09:11:52.357475'}, 'licenses': [{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'}, {'url': 'http://creativecommons.org/licenses/by-nc/2.0/', 'id': 2, 'name': 'Attribution-NonCommercial License'}, {'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/', 'id': 3, 'name': 'Attribution-NonCommercial-NoDerivs License'}, {'url': 'http://creativecommons.org/licenses/by/2.0/', 'id': 4, 'name': 'Attribution License'}, {'url': 'http://creativecommons.org/licenses/by-sa/2.0/', 'id': 5, 'name': 'Attribution-ShareAlike License'}, {'url': 'http://creativecommons.org/licenses/by-nd/2.0/', 'id': 6, 'name': 'Attribution-NoDerivs License'}, {'url': 'http://flickr.com/commons/usage/', 'id': 7, 'name': 'No known copyright restrictions'}, {'url': 'http://www.usa.gov/copyright.shtml', 'id': 8, 'name': 'United States Government Work'}], 'type': 'captions'}
    out.update({'images': [], 'annotations': []})

    # Check if input is COCO-style (has 'annotations') or Karpathy-style
    if 'annotations' in data:
        # COCO-style
        print("Detected COCO-style input format.")
        imgs = data['images']
        anns = data['annotations']
        
        # Create a map of image_id to split if available, otherwise assume all are valid
        # In this specific test_data.json, images don't have split, so we take all.
        
        for img in imgs:
            if img.get('split', '') == 'train':
                continue
            # -- FIX: Add offset for Val (COCO style) --
            # Assuming we are processing val set
            new_id = int(img['id']) + 1000000
            out['images'].append({'id': new_id})
            
        # Filter annotations for the selected images
        # We need to map original ID to new ID for annotations
        # But wait, annotations have 'image_id' pointing to the original ID.
        # We must update annotation 'image_id' as well!
        
        valid_img_ids_map = {img['id']: (int(img['id']) + 1000000) for img in imgs if img.get('split', '') != 'train'}
        
        for ann in anns:
            if ann['image_id'] in valid_img_ids_map:
                out['annotations'].append({
                    'image_id': valid_img_ids_map[ann['image_id']],
                    'caption': ann['caption'],
                    'id': ann['id']
                })
    else:
        # Karpathy-style
        print("Detected Karpathy-style input format.")
        imgs = data.get('images', data) # Handle if root is list or dict with images
        cnt = 0
        for i, img in enumerate(imgs):
            if img.get('split', '') == 'train':
                continue
            img_id = img.get('cocoid', img.get('imgid', img.get('id')))
            
            # --- FIX: ADD OFFSET FOR VALIDATION ---
            # Assuming this script is run on val_data.json
            img_id = int(img_id) + 1000000

            if img_id is None:
                raise KeyError(f"Image entry missing id fields: {img}")
            out['images'].append({'id': img_id})
            for j, s in enumerate(img.get('sentences', [])):
                if len(s) == 0:
                    continue
                s = ' '.join(s['tokens'])
                out['annotations'].append(
                    {'image_id': out['images'][-1]['id'], 'caption': s, 'id': cnt})
                cnt += 1

    with open(params['output_json'], 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)
    print('wrote ', params['output_json'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', nargs='+', required=True,
                        help='input json file to process into hdf5')
    parser.add_argument('--output_json', default='data.json',
                        help='output json file')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
