#!/bin/bash

# Copy of ci_test_xpu.sh, adapted for the self-hosted 2x Intel Data Center GPU
# Max 1550 (Ponte Vecchio) runner. Intended to be narrowed down to tests that
# specifically exercise both GPU cards (multi-device/distributed paths) rather
# than the full single-GPU XPU test suite.
# TODO: restrict the pytest selection below to 2-GPU-only test cases.

conda create -yn xpu_ao_ci python=3.10 pip
source activate xpu_ao_ci

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export SCCACHE_DISABLE=1

python -m pip install --upgrade pip setuptools wheel

python -m pip install torch torchvision torchaudio pytorch-triton-xpu --index-url https://download.pytorch.org/whl/nightly/xpu --force-reinstall --no-cache-dir 
cd torchao && python -m pip install . --no-build-isolation && cd ..

python -c "import torch; import torchao; print(f'Torch version: {torch.__version__}')"

python -m pip install pytest expecttest parameterized accelerate hf_transfer 'modelscope!=1.15.0' transformers tabulate fire

pytest -v -s --ignore=torchao/test/quantization/pt2e/test_x86inductor_fusion.py \
        torchao/test/quantization/pt2e/ \
        torchao/test/quantization/*.py \
        torchao/test/dtypes/ \
        torchao/test/float8/ \
        torchao/test/integration/test_integration.py \
        torchao/test/prototype/ \
        torchao/test/quantization/quantize_/workflows/
