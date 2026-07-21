# Self-hosted GitHub Actions Runner Setup Analysis

**Date:** 2026-07-21
**Node:** `DUT1043PVC`
**Goal:** Configure this node as a self-hosted GitHub Actions runner with its 2 Intel
Ponte Vecchio (PVC) GPUs, for the `karol-brejna-i/ao` fork of `pytorch/ao`.

This document is a snapshot of findings so a future AI agent (or human) can continue
the work without re-discovering the same facts.

## 1. Machine assessment

| Item | Finding |
|---|---|
| Hostname | `DUT1043PVC` — the `PVC` suffix confirms **Ponte Vecchio** (Intel Data Center GPU Max), not Kubernetes "PersistentVolumeClaim" |
| GPUs | **2× Intel Data Center GPU Max 1550** (Ponte Vecchio XT, 2-tile each), confirmed via `xpu-smi discovery` and `lspci`. PCI addrs `0000:4d:00.0` and `0000:9a:00.0`; DRM devices `/dev/dri/card1`, `/dev/dri/card2` |
| CPU | 2× Intel Xeon Platinum 8352Y @ 2.20GHz, 128 threads total |
| RAM | 247 GiB total, ~240 GiB available |
| Disk | `/dev/sda1` on `/`: 1.2 TB total, 657 GB free |
| OS | Ubuntu 24.04.3 LTS, kernel `6.8.0-71-generic` |
| Docker | 28.2.2 installed |
| Kubernetes | Not installed (`kubectl` absent) — irrelevant, this is a bare-metal/VM runner, not an ARC/k8s runner |
| Intel software stack | oneAPI apt repo configured at `/etc/apt/sources.list.d/oneAPI.list` (`https://repositories.intel.com/gpu/ubuntu noble/lts/2523 unified`); `xpu-smi` v1.2.42.20251106 installed; Level-Zero present; `/opt/intel/oneapi` present |
| User | `kbrejna` (uid 1004), groups: `docker`, `video`, `render`, `sudo` |
| Sudo | Passwordless (`sudo -n true` succeeds) |
| GitHub CLI | `gh` v2.45.0 installed |
| Existing runner | **None** — no `actions-runner` directory found, no systemd runner service registered |
| Local `ao` checkout | `/home/kbrejna/workspace/ao`, `origin` = `https://github.com/pytorch/ao.git` (points at upstream, **not** the fork `karol-brejna-i/ao`) |

Conclusion: this node is already provisioned almost identically to PyTorch's official
Intel XPU CI runner image (same oneAPI repo, same `xpu-smi` tooling, GPU device
groups already set up) — it just isn't registered as a GitHub Actions runner yet.

## 2. External research: what `linux.idc.xpu` means in torchao/pytorch CI

Searched `pytorch/pytorch`, `pytorch/ci-infra`, `pytorch/ao`, `pytorch/kineto`,
`pytorch/helion` via GitHub code search:

- `linux.idc.xpu` is an **org-wide, statically-registered ("passthrough") runner
  pool** hosted by Intel for the `pytorch` GitHub org. It is *not* autoscaled via
  the ARC/Kubernetes system used for other runner types — see
  `pytorch/pytorch/.github/arc.yaml` (`linux.idc.xpu: linux.idc.xpu # passthrough`)
  and `pytorch/pytorch/.github/actionlint.yaml`
  (`# Organization-wide Intel hosted XPU runners`).
- Used across many workflows: `pytorch/ao/.github/workflows/xpu_test.yml`,
  `pytorch/pytorch/.github/workflows/xpu.yml`, `pull.yml`,
  `inductor-perf-test-nightly-xpu.yml`, `_binary-test-xpu-linux.yml`,
  `pytorch/kineto/.github/workflows/linux_xpu_*.yml`,
  `pytorch/helion/.github/matrix.json`.
- `pytorch/ci-infra/osdc/docs/current_runner_load_distribution.md` shows
  `linux.idc.xpu` had 5,721 jobs / 30 days, peak 70 concurrent — a heavily used
  shared pool, only usable by repos inside the `pytorch` org's runner group.

**Implication:** a personal fork like `karol-brejna-i/ao` cannot use the
`linux.idc.xpu` label — it must register its own self-hosted runner with a
different (custom) label.

### torchao's `xpu_test.yml` requirements (`.github/workflows/xpu_test.yml`)

- Triggers: `push: tags: ciflow/xpu/*`, `workflow_dispatch`, weekly `schedule`
  (**no** `pull_request` trigger — good, avoids running untrusted fork PR code on
  the self-hosted runner).
- `runs-on: linux.idc.xpu`, checks out `pytorch/pytorch@nightly` + the `ao` repo.
- Health checks expect: `/etc/os-release`, `/etc/apt/sources.list.d/oneAPI.list`,
  `/etc/apt/sources.list.d/intel-gpu-jammy.list`, `xpu-smi discovery` reporting
  ≥1 GPU.
- GPU passthrough for the docker test container:
  `--device=/dev/mem --device=/dev/dri --group-add video --group-add <render_gid>`.
- Docker image `ci-image:pytorch-linux-noble-xpu-n-py3` is pulled from a
  **private ECR mirror** via `aws-actions/configure-aws-credentials` (OIDC role
  `arn:aws:iam::308535385114:role/gha_workflow_s3_and_ecr_read_only`) — this OIDC
  trust relationship is scoped to the `pytorch` org and **will not work** from a
  personal fork. A public mirror exists at `ghcr.io/pytorch/ci-image:<tag>` and
  should be substituted when adapting the workflow for the fork.

## 3. Dedicated technical user for running the runner

**Decision: yes, create a dedicated non-`kbrejna` user for the runner.**

Rationale:
- **Blast radius isolation.** The runner executes arbitrary job code (checkout
  scripts, docker builds, etc.). Running it as `kbrejna` — the primary
  interactive account with **passwordless `sudo`** — would expose that `sudo`
  access, SSH keys, dotfiles, and shell history to anything a workflow does.
- **Cleanup/state hygiene.** A dedicated user gets its own `$HOME`, so cached
  pip/conda/docker-buildx state and `_work` dirs don't mix with personal files.
- **Least privilege.** The runner only needs `docker` group (build/run
  containers) and `video` + `render` groups (GPU device access) — no `sudo` at
  all. GPU passthrough is done via docker device flags, not root.
- **Auditability.** Distinguishes CI-caused activity (different UID) from
  interactive activity.

**Chosen username:** `gha-runner`

### Creation steps

```bash
# Create the user with a real home dir (needed for tool caches: pip, npm, docker buildx, etc.)
sudo useradd -m -s /bin/bash gha-runner

# Grant only the groups needed for docker builds + Intel GPU device access
sudo usermod -aG docker,video,render gha-runner

# Verify group membership
id gha-runner

# Do NOT add gha-runner to the sudo group, and do NOT configure passwordless sudo for it.
```

Notes:
- Use `-m` (create home) and a real shell (`/bin/bash`) — do **not** use
  `--no-create-home` or `nologin`; the runner needs a writable home for caches
  and job steps may need an interactive-capable shell.
- The runner install directory (e.g. `/opt/actions-runner`) should be owned by
  `gha-runner:gha-runner` rather than `kbrejna`.
- The systemd service created by `svc.sh install` should be installed with
  `gha-runner` as the run-as user (`sudo ./svc.sh install gha-runner`), so job
  steps execute as that user, not `kbrejna` and not root.

## 4. Proposed plan to register this node as a runner for `karol-brejna-i/ao`

Not yet executed — requires the user to supply a GitHub registration token (a
credential that should never be pasted into chat).

1. **Create the dedicated `gha-runner` user** (see Section 3) and install the
   runner agent under a directory it owns (e.g. `/opt/actions-runner`).
2. **Obtain a registration token** for `karol-brejna-i/ao`, e.g.:
   `gh api -X POST repos/karol-brejna-i/ao/actions/runners/registration-token -q .token`
   — run this by the user directly, not shared in chat.
3. **Configure the runner** (as `gha-runner`) with a custom label set instead of
   `linux.idc.xpu`, e.g. `--labels self-hosted,xpu,pvc-2,linux`, name
   `dut1043pvc-xpu`.
4. **Install as a systemd service** (`sudo ./svc.sh install gha-runner && sudo
   ./svc.sh start`) so it survives reboots and doesn't run as root or as
   `kbrejna`.
5. **Adapt a workflow** to target `runs-on: [self-hosted, xpu, pvc-2]` and swap
   the ECR-based image pull for the public `ghcr.io/pytorch/ci-image:...` mirror
   (drop the AWS OIDC / ECR login steps, which won't work outside the `pytorch`
   org). Keep the existing `GPU_FLAG` step (`--device=/dev/mem --device=/dev/dri
   --group-add video --group-add render`) — it already matches the groups
   granted to `gha-runner`.
6. **Keep triggers restricted** to `push`/`workflow_dispatch`/`schedule` only —
   do not add `pull_request` for a self-hosted runner on a repo that could
   receive external PRs, to avoid arbitrary code execution on this box.

### Security considerations
- Run the runner service as the dedicated unprivileged `gha-runner` user, not
  `kbrejna` and not root.
- `gha-runner` must **not** have `sudo` access.
- Registration should be per-repo (fork-scoped token), not an org-wide token, to
  limit blast radius.
- Do not trigger the runner from `pull_request` events from forks.
- Periodically prune stopped containers/images (the existing workflow already
  does `docker container prune`).

## 5. Open items / not yet done
- [x] `gha-runner` user created (uid 1005, gid 1006), groups: `gha-runner`,
      `video`, `render`, `docker`. No `sudo` group membership. Home dir
      `/home/gha-runner` created with `/bin/bash` shell.
- [x] Runner agent installed: `/opt/actions-runner` (owned by
      `gha-runner:gha-runner`), GitHub Actions runner **v2.336.0** downloaded
      and extracted. Tarball retained at
      `/opt/actions-runner/actions-runner-linux-x64.tar.gz` (safe to delete
      once configured).
- [x] Runner registered with `karol-brejna-i/ao` via `config.sh --unattended`,
      name `dut1043pvc-xpu`, labels `self-hosted,xpu,pvc-2,linux`, work folder
      `_work`. Run as `gha-runner` (`sudo -u gha-runner -H ./config.sh ...`).
      Registration token used was single-use/short-lived and is now consumed.
- [x] Installed and started as a systemd service:
      `actions.runner.karol-brejna-i-ao.dut1043pvc-xpu.service`
      (`/etc/systemd/system/...`), enabled at boot, runs as `gha-runner`
      (uid 1005). Confirmed `active (running)` via `sudo ./svc.sh status`.
- [x] Added `fork` git remote in local checkout:
      `fork https://github.com/karol-brejna-i/ao.git` (alongside existing
      `origin` = `pytorch/ao`, left unchanged).
- [x] Created [.github/workflows/xpu_test_2gpu.yml](.github/workflows/xpu_test_2gpu.yml) —
      copy of `xpu_test.yml`, adapted to run on this self-hosted runner:
      - `runs-on: [self-hosted, xpu, pvc-2]` instead of `linux.idc.xpu`.
      - Dropped the AWS OIDC (`configure-aws-credentials`) + private ECR login
        steps (won't work outside the `pytorch` org). Kept the
        `calculate-docker-image` action (local hash calc, no AWS needed) and
        added a plain `docker pull` from the public `ghcr.io/pytorch/ci-image`
        mirror instead of the ECR-specific `pull-docker-image` action.
      - Trigger restricted to `workflow_dispatch` only for now (no
        push-tag/schedule) while validating.
      - Points `TEST_COMMAND` at the new `ci_test_xpu_2gpu.sh` script.
- [x] Created [.github/scripts/ci_test_xpu_2gpu.sh](.github/scripts/ci_test_xpu_2gpu.sh) —
      copy of `ci_test_xpu.sh` (same conda env + pytest invocation as upstream
      for now). **TODO left in place:** narrow the pytest selection down to
      2-GPU/distributed-only test cases.

## 6. Open items / not yet done
- [ ] Narrow `ci_test_xpu_2gpu.sh` pytest selection to 2-GPU/multi-device test
      cases only (currently identical to the full upstream single-GPU suite).
      Candidate test files use `world_size`/`skip_if_lt_x_gpu`
      (e.g. `test/float8/test_fsdp2/test_fsdp2.py`,
      `test/quantization/quantize_/workflows/nf4/test_nf4_tensor.py`,
      `test/test_low_bit_optim.py`) but are currently written against CUDA
      (`nccl`/`device_mesh("cuda", ...)`) — need to check XPU/`xccl` support
      before wiring them in.
- [ ] `xpu_test_2gpu.yml` not yet exercised end-to-end (no `workflow_dispatch`
      run triggered yet) — validate the ghcr.io image pull and container run
      actually work without AWS credentials.
- [ ] Not yet pushed to the `fork` remote — changes exist only in the local
      checkout so far.
