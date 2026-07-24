#!/bin/bash

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

# 2-card distributed tests.
#
# Delegate to the repo's own multi-GPU test scripts instead of reimplementing their
# launch commands (torchrun vs mp.spawn, flags, etc.) here, so this CI script doesn't
# need to change whenever those tests' internals change.
#
# NOTE: not using test/float8/test_everything_multi_gpu.sh here - it also chains
# test_dtensor.sh, whose second half launches test_fsdp2_tp.py with
# `torchrun --nproc_per_node 4`, requiring 4 GPUs. That's incompatible with a 2-card
# runner, so only the two scripts below (each self-contained at world_size=2 via
# CUDA_VISIBLE_DEVICES=0,1) are run for now.
#
# NOTE: these scripts currently hardcode CUDA (torch.cuda.is_available() skip-guards,
# "nccl" backend, CUDA_VISIBLE_DEVICES), so they will silently no-op skip on XPU until
# made device-agnostic. Tracked as a prerequisite before this job can be enabled in CI -
# see _zapiski.priv/xpu_2card_workflow_analysis.md.
#
# NOT YET APPLIED (intentionally deferred): the fix is `dist.get_default_backend_for_device`
# for the "nccl" backend + torch.accelerator.* for device selection - already implemented
# upstream in open/draft PRs pytorch/ao#4511 (test_fsdp.py/.sh) and pytorch/ao#4512
# (test_fsdp_compile.py/.sh). Rebase/cherry-pick from those rather than reimplementing.
cd torchao
./test/float8/test_fsdp.sh
./test/float8/test_fsdp_compile.sh
