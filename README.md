# TPU v7x Kernel Benchmarks & Standalone TorchTPU Kernels

Self-contained benchmark suite and standalone kernel examples for reproducing
four core LLM operator families on **Google Cloud TPU v7x (Ironwood)** using
native **TorchTPU (`torch_tpu`)** without a serving-engine (`vLLM`) dependency:

1. **Flash / Splash Causal MHA (`temu_table` & `attention`)**
   * Standalone `fwd` and standalone `bwd` (forward residuals materialize once
     outside the timed loop so `bwd` never fuses or re-executes `fwd`).
   * Includes unfused 2-kernel `splash_sep` (`use_fused_bwd_kernel=False`,
     separate `dQ` and `dKV` Pallas kernels), fused `splash_fused`, native C++
     Kiln Mosaic `torchtpu_flash`, `splash_opt`, `tokamax_splash`,
     `flash_attention`, `fa2`, `ref`, and `dpa`.
2. **Paged Attention (`paged`)**
   * Standalone `torch.tpu.pallas.jax_op` wrapper around Ragged Paged Attention
     v3 (`RPAv3`, `head_dim=64` and `128`, prefill & decode) plus `(bkvp, bq)`
     tile tuning (`(8, 256)`) and GQA-4 packing sweeps.
3. **RMSNorm (`rmsnorm`)**
   * `F.rms_norm`, `float32`-accumulated, and `bfloat16` RMSNorm with
     standalone `fwd` and fused analytical standalone `bwd`.
4. **Grouped-Matmul SwiGLU MoE (`moe`)**
   * End-to-end (`e2e`, `e2e_opt`) and 6-stage (`route`, `gather`, `gmm_up`,
     `act`, `gmm_down`, `reduce`) benchmarks across `torch._grouped_mm`,
     `torch.tpu.ops.ragged_dot`, and `tokamax.ragged_dot` Mosaic TPU kernels.

All verified hardware `results.jsonl` records from a `4`-chip TPU v7x `2x2x1`
host are checked into [`results/torchtpu_tpu7x_2x2x1/`][results-dir].

## 1. Hardware Topology & Official Spec Comparison

Every physical **Cloud TPU v7x** chip has a dual-chiplet architecture exposing
**2 independent TensorCore devices** (`devices_per_chip=2`, no MegaCore).

* Running `1` process (`TPU_VISIBLE_DEVICES=0`) uses **only `0.5` physical
  chip** (`1` TensorCore).
* Launching `2` processes (`torchrun --nproc_per_node=2`, sharding
  `global_batch=128` as `batch=64` per TensorCore) uses **1 full physical
  chip** with identical mathematical semantics.
* A single `tpu7x` `2x2x1` host (`4` physical chips = `8` TensorCores) is the
  optimal node topology for single-operator benchmarks: it tests `1-Core`
  (`scope=core`), `1-Chip` (`scope=chip`), and `8-Core Host` (`scope=host`) on
  one VM without multi-host gang-scheduling overhead.

| Official Spec (Dense) | H100 SXM5 (1 GPU) | TPU v7x 1-Core | TPU v7x 1-Chip |
| --- | --- | --- | --- |
| **BF16 Peak** | `989.4 TFLOPS` | `1,153.5 TFLOPS` | `2,307 TFLOPS` |
| **FP8 Peak** | `1,978.9 TFLOPS` | `2,307 TFLOPS` | `4,614 TFLOPS` |
| **HBM Capacity** | `80 GB` HBM3 | `96 GiB` HBM3e | `192 GiB` HBM3e |
| **HBM Bandwidth** | `3,350 GB/s` | `3,690 GB/s` | `7,380 GB/s` |
| **Matrix Geometry** | `wgmma n=64..256` | `2x 256x256 MXU` | `4x 256x256 MXU` |
| **Logical Devices** | `1` (`cuda:0`) | `1` (`0.5` chip) | `2` (`1.0` chip) |

## 2. Pinned Software Stack & Commit Hashes

| Component | Version | Upstream Repo & Commit / Tag |
| --- | --- | --- |
| `torch-tpu` | `0.3.0` | [google-pytorch/torch_tpu][tt] (`68fdc954ae`) |
| `openxla/xla` | `XLA_COMMIT` | [openxla/xla][xla] (`ce107528b1`) |
| `libtpu` | `0.0.53` | Cloud TPU PJRT plugin (`PJRT C API 0.117`) |
| `tokamax` | `0.0.18.dev` | [openxla/tokamax][tk] (`0d3e82b41d`) |
| `vllm-torchtpu-core` | `0.33.0` | [vllm-project/vllm-torchtpu][vt] (`46ae4b7a62`) |
| `torch` | `2.15.0+cpu` | [pytorch/pytorch][pt] (`31a77fc9c0`) |
| `jax` / `jaxlib` | `0.12.0.1` | [jax-ml/jax][jx] (`dc4af68444`) |
| `jax-flash-attn2` | `0.0.3` | [erfanzar/jax-flash-attn2][fa2] (`af278984a4`) |
| `xprof` | `2.24.2` | PyPI `xprof==2.24.2` |

## 3. Step-by-Step Reproduction Guide

### Step 1 — Clone & Build the Unified Container Image

The [`Dockerfile`](Dockerfile) installs all dependencies directly from public
PyPI and GitHub and runs CPU Mosaic TPU MLIR export checks + unit tests at
build time:

```bash
git clone https://github.com/Alina-PANG/temu_tpu_kernels.git
cd temu_tpu_kernels
DOCKER_BUILDKIT=1 docker build -t temu-kbench-torchtpu:v1 .
```

*(If using a pre-exported `.tar.gz` image archive instead of building from
source, load it via `docker load -i temu-kbench-torchtpu_v1.tar.gz`.)*

### Step 2 — Verify MLIR Export & Unit Tests on Any CPU Machine

You can verify package versions, Mosaic TPU MLIR lowering, standalone Pallas op
registration, and mathematical parity (`fp32` reference) without a TPU:

```bash
docker run --rm temu-kbench-torchtpu:v1 bash -lc \
  "python3 /workspace/temu_kbench/verify_build.py && \
   python3 /workspace/temu_kbench/examples/standalone_custom_pallas_op.py && \
   python3 /workspace/temu_kbench/test_torchtpu_bench.py -v"
```

### Step 3 — One-Command Full Reproduction on TPU v7x (`~11 Minutes`)

`SUITE=all` runs all 6 stages (`env_check`, `temu_table_1core`,
`temu_table_1chip`, `kernels_1core`, `kernels_1chip`, `kernels_host8`, and
`sweeps_1core`) and writes `results.jsonl` + XProf traces under `/results`.

**Option A — Direct Docker Run on a Cloud TPU v7x `2x2x1` VM:**

```bash
mkdir -p ./out
docker run --rm --privileged --net=host --shm-size=32g \
  -v "$(pwd)/out:/results" \
  -e SUITE=all \
  -e OUT_DIR=/results \
  -e ITERS=10 \
  -e WARMUP=2 \
  -e TRACE_MODE=key \
  temu-kbench-torchtpu:v1 /usr/local/bin/run_all.sh
```

**Option B — Kubernetes Job on GKE (`tpu7x` `2x2x1`, `google.com/tpu: 4`):**

```bash
export IMAGE_URI="<YOUR_REGISTRY>/temu-kbench-torchtpu:v1"
docker tag temu-kbench-torchtpu:v1 "${IMAGE_URI}"
docker push "${IMAGE_URI}"

sed "s|<IMAGE_URI>|${IMAGE_URI}|g" k8s/job_tpu7x_2x2x1.yaml \
  | kubectl apply -f -
kubectl logs -f job/tpu7x-torchtpu-kbench
```

### Step 4 — Retrigger Individual Suites or Standalone Examples

```bash
# 1. Customer attention tables only (1-Core + 1-Chip, ~4 min):
docker run --rm --privileged --net=host --shm-size=32g \
  -v "$(pwd)/out:/results" -e SUITE=temu_table \
  temu-kbench-torchtpu:v1 /usr/local/bin/run_all.sh

# 2. Single-core specific kernel family (attention | paged | rmsnorm | moe):
docker run --rm --privileged --net=host --shm-size=32g \
  -v "$(pwd)/out:/results" temu-kbench-torchtpu:v1 bash -lc \
  "TORCH_TPU_TOPOLOGY=1,1,1,1 TPU_VISIBLE_DEVICES=0 \
   torchtpu_kernel_bench.py --kernel moe --preset temu --scope core \
     --out /results/moe_1core"

# 3. Full 1-Chip (2 TensorCores) run via torchrun:
docker run --rm --privileged --net=host --shm-size=32g \
  -v "$(pwd)/out:/results" temu-kbench-torchtpu:v1 bash -lc \
  'eval "$(python3 -c "import torch,torch_tpu; \
     torch.tpu.distributed.set_tpu_launch_env(nproc_per_node=2); \
     import os; print(\"export TORCH_TPU_SLICEBUILDER_ADDRESSES=\" + \
     os.environ[\"TORCH_TPU_SLICEBUILDER_ADDRESSES\"])")" && \
   TORCH_TPU_TOPOLOGY=1,1,1,2 torchrun --standalone --nnodes=1 \
     --nproc_per_node=2 /usr/local/bin/torchtpu_kernel_bench.py \
     --kernel temu_table --preset temu --scope chip --out /results/chip2'

# 4. Standalone custom Pallas kernel in PyTorch (zero vLLM):
docker run --rm --privileged --net=host --shm-size=32g \
  temu-kbench-torchtpu:v1 \
  python3 /workspace/temu_kbench/examples/standalone_custom_pallas_op.py
```

*(Pure JAX baseline scripts from the initial JAX comparison are also provided
under [`jax_baseline/`](jax_baseline/).)*

### How Timing & Precision Are Measured

* **Primary Metric (`event_ms_p50`):** After 1 compilation call and `warmup=2`
  synchronized steps, each of `iters=10` steps is bracketed by
  `torch.tpu.Event(enable_timing=True)` (`ev_start.elapsed_time(ev_end)`),
  reporting the median (`p50`) in milliseconds.
* **Hardware Kernel Duration (`hw`):** `TRACE_MODE=key` captures `2` profiled
  steps via `torch.profiler` (`ProfilerActivity.TPU`), writing `.xplane.pb` and
  `.pt.trace.json` traces (viewable in `ui.perfetto.dev` or `xprof`) to isolate
  pure TensorCore `tpu_custom_call` kernel time without tensor-transpose glue.
* **Standalone Backward (`standalone_bwd=true`):** Forward runs **once** before
  timing to materialize concrete residuals (`(out, logsumexp)` or `(o, l, m)`),
  so timed `bwd` executes **only** the backward kernel (`dQ`, `dK`, `dV`).

## 4. Verified TPU v7x Benchmark Results (Default Clocks)

All numbers below were measured on `1x` Cloud TPU v7x `2x2x1` host at default
clocks with `bfloat16` inputs and verified against an exact `float32` reference
(`fwd_cos_sim >= 0.99998`, `bwd_dq_cos_sim >= 0.99997`). Raw JSONL outputs are
in [`results/torchtpu_tpu7x_2x2x1/`][results-dir].

### 4.1 Causal MHA Implementations (`batch=128, heads=4, dim=128`)

Parenthetical `(hw)` values on `seq=2048` show pure TensorCore kernel time from
XProf (`splash_sep` sums `5.825 ms dkv + 4.845 ms dq = 10.670 ms` across its 2
unfused Pallas backward kernels):

| seq | Implementation | Blocks (`q/kv/kvc`) | 1-Core `fwd / bwd` (ms) | 1-Chip `fwd / bwd` (ms) |
| --- | --- | --- | --- | --- |
| 1024 | `flash_attention` | `256/1024/512` | `2.504 / 6.323` | `1.391 / 3.359` |
| 1024 | `splash_sep` | `512/1024/512` | `1.942 / 3.909` | `1.227 / 2.164` |
| 1024 | `splash_fused` | `512/1024/512` | `1.939 / 2.718` | `1.193 / 1.585` |
| 1024 | `fa2` | `256/256` | `2.956 / 5.695` | `1.721 / 3.013` |
| 1024 | `ref` | — | `1.932 / 10.764` | `1.121 / 5.553` |
| 1024 | `dpa` | — | `1.642 / 5.682` | `0.978 / 3.126` |
| 1024 | **`torchtpu_flash`** | Kiln C++ Mosaic | **`1.088`** / `3.200` | **`0.760`** / `1.806` |
| 1024 | **`splash_opt`** | `1024^3` / `512..` | `1.494 / 2.634` | `0.896 / 1.540` |
| 1024 | **`tokamax_splash`** | `1024^3` / `512..` | `1.525` / **`2.560`** | `0.905` / **`1.480`** |
| 2048 | `flash_attention` | `256/2048/512` | `8.005 / 19.678` | `4.169 / 9.797` |
| 2048 | `splash_sep` | `512/2048/1024` | `5.390 (4.58) / 12.191 (10.67)` | `2.898 / 6.064` |
| 2048 | `splash_fused` | `512/2048/1024` | `5.404 (4.57) / 8.419 (7.44)` | `2.940 / 4.464` |
| 2048 | `fa2` | `256/256` | `10.149 / 19.760` | `5.792 / 9.809` |
| 2048 | `ref` | — | `6.889 / 40.377` | `3.612 / 20.652` |
| 2048 | `dpa` | — | `5.799 / 26.468` | `3.071 / 12.340` |
| 2048 | **`torchtpu_flash`** | Kiln C++ Mosaic | **`2.514 (2.26)`** / `11.564` | **`1.456 (1.13)`** / `5.690` |
| 2048 | **`splash_opt`** | `1024^3` / `512..` | `4.560` / **`7.826 (6.65)`** | `2.447` / **`4.158 (3.33)`** |
| 2048 | **`tokamax_splash`** | `1024^3` / `512..` | `3.744 (3.31) / 8.005 (6.82)` | `2.025 / 4.230` |

### 4.2 Sequence Scaling (`1024..10240`, `batch=128, h=4, d=128`) vs. H100 FA3

| seq | H100 FA3 `fwd/bwd` | TPU 1-Core Opt | TPU 1-Chip Opt | 1-Chip / H100 (`<1` = Faster) |
| --- | --- | --- | --- | --- |
| 1024 | `0.402 / 1.294` | `1.088 / 2.560` | **`0.760 / 1.480`** | `1.89x / 1.14x` |
| 2048 | `1.352 / 4.292` | `2.514 / 7.826` | **`1.456 / 4.158`** | `1.08x` (`0.84x` hw) / **`0.97x`** |
| 4096 | `4.886 / 15.281` | `9.776 / 25.864` | **`5.069 / 13.021`** | `1.04x` / **`0.85x`** |
| 8192 | `18.652 / 57.735` | `34.033 / 87.201` | **`17.701 / 44.380`** | **`0.95x / 0.77x`** |
| 10240 | `28.920 / 89.512` | `51.518 / 131.652` | **`26.073 / 66.215`** | **`0.90x / 0.74x`** |

### 4.3 Target Shape `[64, 2048, 8, 64]` vs. Iso-FLOP `d=128` & `d=256`

TPU v7x MXU columns are `256`-wide (`128` VPU lanes), so `d=128` and `d=256`
avoid `d=64` lane padding (`2.48x` speedup at identical `275 GFLOPs`) and scale
at **`94.4%–99.5%` efficiency** across all `8` TensorCores (`4` chips):

| Shape `[B, S, H, D]` | Kernel & Pass | 1-Core `[hw]` (ms) | 1-Chip (ms) | 8-Core Host (ms) | Host TFLOPS |
| --- | --- | --- | --- | --- | --- |
| `[64, 2048, 8, 64]` | `torchtpu_flash fwd` | **`3.587 [2.302]`** | `3.632` | `3.720` (`96.4%`) | `591.4` |
| `[64, 2048, 8, 64]` | `tokamax_splash fwd` | `4.536` | — | — | — |
| `[64, 2048, 8, 64]` | `splash_opt bwd` | **`9.474 [6.714]`** | `9.482` | `9.518` (`99.5%`) | `577.9` |
| `[64, 2048, 4, 128]` | `torchtpu_flash fwd` | **`1.447 [1.130]`** | `1.472` | `1.533` (`94.4%`) | **`1,435.4`** |
| `[64, 2048, 4, 128]` | `splash_opt bwd` | **`4.171 [3.329]`** | `4.117` | `4.216` (`98.9%`) | **`1,304.7`** |
| `[64, 2048, 2, 256]` | `tokamax / splash fwd` | **`1.539` / `1.743`** | `1.767` | `1.827` (`95.4%`) | **`1,204.4`** |

### 4.4 Paged Attention (`RPAv3`, `batch=64, ctx=2048, page_size=256`)

Overriding the `(bkvp=8, bq=32)` fallback with `(8, 256)` cuts `d=64` MHA
kernel time **`1.89x`** (`14.027 -> 7.408 ms`) and GQA-4 **`3.44x`**
(`4.076 ms`):

| Phase & Heads (`b=64, ctx=2048`) | Block `(bkvp, bq)` | 1-Core `[hw]` (ms) | 1-Chip (ms) | 8-Core (ms) |
| --- | --- | --- | --- | --- |
| Prefill `hq=8, hkv=8, d=64` (MHA) | `auto -> (8, 32)` | `17.610 [14.027]` | `17.874` | `17.664` |
| Prefill `hq=8, hkv=8, d=64` (MHA) | **Tuned `(8, 256)`** | **`10.916 [7.408]`** | `11.057` | `10.860` |
| Prefill `hq=8, hkv=2, d=64` (GQA-4) | **Tuned `(8, 256)`** | **`5.382 [4.076]`** | `5.394` | `5.415` |
| Prefill `hq=4, hkv=4, d=128` (MHA) | `auto -> bq=2048` | **`5.217 [2.696]`** | `5.250` | `5.263` |
| Decode `hq=8, hkv=8, d=64` (`s=1`) | `auto -> (8, 32)` | `0.509` | `0.594` | `0.580` |
| Decode `hq=4, hkv=4, d=128` (`s=1`) | `auto -> bq=1` | **`0.468`** | `0.522` | `0.528` |

### 4.5 RMSNorm (Standalone `fwd` & Fused Analytical `bwd`)

| Shape `[Rows, Dim]` | Pass | 1-Core `[hw]` (ms) | 1-Chip (ms) | 8-Core (ms) | HBM Bandwidth |
| --- | --- | --- | --- | --- | --- |
| `[131072, 512]` | `fwd` | `0.359` | `0.352` | `0.391` | `5,486.7 GB/s` host Event |
| `[131072, 512]` | `bwd` | `0.576` | `0.595` | `0.606` | `5,319.1 GB/s` host Event |
| `[32768, 4096]` | `fwd` | `0.474` **`[0.247]`** | `0.464` | `0.526` | **`4,349 GB/s/chip` hw** |
| `[32768, 4096]` | `bwd` | `0.788` **`[0.444]`** | `0.764` | `0.825` | **`3,626 GB/s/chip` hw** |
| `[8192, 8192]` | `fwd` | `0.337` | `0.332` | `0.411` | `5,225.3 GB/s` host Event |
| `[8192, 8192]` | `bwd` | `0.565` | `0.578` | `0.630` | `5,116.7 GB/s` host Event |

### 4.6 Grouped-Matmul SwiGLU MoE (`[32768, 4096, 1024, 128, top-8]`)

Replacing `torch.bincount` (`45.68 ms` scatter) with a fused comparison
histogram `(flat[:, None] == exp_ids).sum(0)` cuts `route` **`37x`** (`1.30 ms`)
and end-to-end MoE **`3.2x`** (`21.203 ms`, matching/beating pure JAX `21.48 ms`
on XLA `ragged_dot`):

| Stage (`32768, 4096, 1024, 128, k=8`) | `_grouped_mm` (ms) | `tpu.ops.ragged_dot` | `tokamax` `[hw]` (ms) | Throughput / Notes |
| --- | --- | --- | --- | --- |
| **End-to-End (`e2e`)** | **`21.203 [20.261]`** | `21.375 [20.265]` | `22.590` | **`622.3 TFLOPS/chip`** |
| `1. route` (GEMM + top-8 + hist) | `1.330` | **`1.300`** | `1.314` | **`37.1x`** vs `bincount` |
| `2. gather` (`x[tok_ids]`) | `3.080` | `3.058` | `3.070` | SparseCore offload |
| `3. gmm_up` (`4.40 TFLOPs`) | `7.641` | `7.655` | **`7.555 [7.061]`** | **`1,246.3 TFLOPS/chip`** |
| `4. act` (`SwiGLU`) | `0.730` | `0.724` | **`0.711`** | VPU elementwise fusion |
| `5. gmm_down` (`2.20 TFLOPs`) | `4.824` | `4.819` | **`3.638 [3.166]`** | **`1,389.8 TFLOPS/chip`** |
| `6. reduce` (unsort + top-8 sum) | `4.724 [4.463]` | `4.740 [4.480]` | **`4.708 [4.461]`** | SparseCore + reduce |
| **Small `[4096, 1024, 256, 32, k=4]`** | **`0.670`** | `0.676` | `1.400` (`e2e_opt`) | `6.0x` vs `bincount` |

## 5. Standalone Custom Kernel Integration References (Zero vLLM)

* Minimal standalone example in this repo:
  [`examples/standalone_custom_pallas_op.py`][demo-op]
* TorchTPU Custom Kernels Guide (`Tokamax`, `torch.tpu.pallas.jax_op`,
  `Helion`): [google-pytorch/torch_tpu `docs/custom_kernels.md`][tt-ck]
* OpenXLA Tokamax TPU Kernels (`splash_attention`, `ragged_dot`, `fused_moe`):
  [openxla/tokamax][tk]
* JAX TPU Pallas & SparseCore Authoring Guides:
  [Pallas TPU][pl-tpu] | [SparseCore Pallas][pl-sc] | [StableHLO AOT][jx-aot]

[results-dir]: results/torchtpu_tpu7x_2x2x1/
[demo-op]: examples/standalone_custom_pallas_op.py
[tt]: https://github.com/google-pytorch/torch_tpu
[tt-ck]: https://github.com/google-pytorch/torch_tpu/blob/main/docs/custom_kernels.md
[xla]: https://github.com/openxla/xla
[tk]: https://github.com/openxla/tokamax
[vt]: https://github.com/vllm-project/vllm-torchtpu
[pt]: https://github.com/pytorch/pytorch
[jx]: https://github.com/jax-ml/jax
[fa2]: https://github.com/erfanzar/jax-flash-attn2
[pl-tpu]: https://docs.jax.dev/en/latest/pallas/tpu/index.html
[pl-sc]: https://docs.jax.dev/en/latest/pallas/tpu/sparsecore.html
[jx-aot]: https://docs.jax.dev/en/latest/aot.html
