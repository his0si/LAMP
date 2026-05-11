#!/usr/bin/env bash
set -euo pipefail

conda env create -f environment.yml
conda run -n LAMP python -m pip install torch==2.9.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
conda run -n LAMP python -m pip install -r requirements.txt
