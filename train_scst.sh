# Install dependencies
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip

python3 -m pip install -r requirements.txt

#!/bin/bash

# Move to the directory where run.sh is located
cd "$(dirname "$0")"

echo "[INFO] Running from: $PWD"

# Helper script to run SCST training
python train.py --cfg configs/lstm_scst.yml
