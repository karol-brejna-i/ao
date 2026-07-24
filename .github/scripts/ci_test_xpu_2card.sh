#!/bin/bash

# Copy of ci_test_xpu.sh, adapted for the self-hosted 2x Intel Data Center GPU
# Max 1550 (Ponte Vecchio) runner, running the repo's own 2-GPU distributed
# test scripts instead of the full single-GPU XPU test suite.
#
# Naming/approach aligned with a more mature parallel analysis of this same
# problem - see runner-setup-analysis.md §6 for the full writeup and links to
# upstream PRs #4510/#4511/#4512/#4532.

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
# see runner-setup-analysis.md §6.
#
# NOT YET APPLIED (intentionally deferred): the fix is `dist.get_default_backend_for_device`
# for the "nccl" backend + torch.accelerator.* for device selection - already implemented
# upstream in open/draft PRs pytorch/ao#4511 (test_fsdp.py/.sh) and pytorch/ao#4512
# (test_fsdp_compile.py/.sh). Rebase/cherry-pick from those rather than reimplementing.
cd torchao
./test/float8/test_fsdp.sh
./test/float8/test_fsdp_compile.sh
#./test/float8/test_everything_multi_gpu.sh  # not run due to 4-GPU requirement
