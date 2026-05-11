#!/usr/bin/env bash
set -euo pipefail

# Run this script from inside the acpl/ folder.
# It assumes conda is already initialized in the current shell:
#   source /usr/local/conda/etc/profile.d/conda.sh

conda env create -f environment.yml
conda run -n LAMP_acpl python -m pip install torch==2.9.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
conda run -n LAMP_acpl python -m pip install -r requirements.txt
