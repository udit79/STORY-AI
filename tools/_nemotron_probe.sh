#!/usr/bin/env bash
# Probe Nemotron import in its venv
set -e
source /mnt/d/STORY-AI/.venv-nemotron/bin/activate
export PYTHONPATH=/mnt/d/STORY-AI/nemotron-ocr-v2/nemotron-ocr/src
export CUDA_HOME=/mnt/d/STORY-AI/.cuda/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_HOME=/mnt/d/STORY-AI/.torch
python -c "
import nemotron_ocr
print('import OK:', nemotron_ocr.__file__)
from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
print('NemotronOCRV2 OK')
"
