from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import argparse

parser = argparse.ArgumentParser()

# output_dir
parser.add_argument('--downloaded_feats', default=r'data/features/features_tsv', help='downloaded feature directory')
parser.add_argument('--output_dir', default=r'data/features/features_extracted', help='output feature files')

args = parser.parse_args()

# Fix for OverflowError: Python int too large to convert to C long
# On Windows, sys.maxsize can be larger than what C long supports
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int/10)


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
infiles = [
    'train.tsv', 'val.tsv', 'test.tsv'
]

if not os.path.exists(args.output_dir + '_att'):
    os.makedirs(args.output_dir + '_att')
if not os.path.exists(args.output_dir + '_fc'):
    os.makedirs(args.output_dir + '_fc')
if not os.path.exists(args.output_dir + '_box'):
    os.makedirs(args.output_dir + '_box')

for infile in infiles:
    print('Reading ' + infile)
    with open(os.path.join(args.downloaded_feats, infile), "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        
        # Determine offset based on filename
        offset = 0
        if 'val' in infile:
            offset = 1000000
        elif 'test' in infile:
            offset = 2000000

        for item in reader:
            item['image_id'] = int(item['image_id']) + offset
            item['num_boxes'] = int(item['num_boxes'])
            # for field in ['boxes', 'features']:
            #     item[field] = np.frombuffer(base64.decodebytes(item[field].encode('ascii')), 
            #             dtype=np.float32).reshape((item['num_boxes'],-1))
            for field in ['boxes', 'features']:
                s = str(item[field]).strip()
                if s.startswith("b'") or s.startswith('b"'):
                    s = s[2:-1]  # Cắt bỏ 2 ký tự đầu và 1 ký tự cuối
                item[field] = s
                item[field] = np.frombuffer(base64.decodebytes(item[field].encode('ascii')),
                                            dtype=np.float32).reshape((item['num_boxes'],-1))

            np.savez_compressed(os.path.join(args.output_dir+'_att', str(item['image_id'])), feat=item['features'])
            np.save(os.path.join(args.output_dir+'_fc', str(item['image_id'])), item['features'].mean(0))
            np.save(os.path.join(args.output_dir+'_box', str(item['image_id'])), item['boxes'])