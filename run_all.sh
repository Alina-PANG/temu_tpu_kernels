#!/bin/bash
# Runs the unified TorchTPU TPU v7x kernel benchmark suite.
#
# Usage:
#   SUITE=all OUT_DIR=/results run_all.sh
#
# Environment variables (all optional):
#   SUITE       : smoke | temu_table | kernels | sweeps | all (default: all)
#   OUT_DIR     : Output directory for JSONL logs & XProf traces (/results)
#   ITERS       : Timed iterations per fast case (default: 10)
#   WARMUP      : Warmup iterations per fast case (default: 2)
#   TRACE_MODE  : none | key | all (default: key)
#   TRACE_ITERS : Profiled steps when tracing (default: 2)
#   GCS_DEST    : Optional gs://<bucket>/<prefix> to upload OUT_DIR on finish

set -uo pipefail

SUITE="${SUITE:-all}"
OUT_DIR="${OUT_DIR:-/results}"
ITERS="${ITERS:-10}"
WARMUP="${WARMUP:-2}"
TRACE_MODE="${TRACE_MODE:-key}"
TRACE_ITERS="${TRACE_ITERS:-2}"
GCS_DEST="${GCS_DEST:-}"
BENCH_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
KB="${BENCH_DIR}/torchtpu_kernel_bench.py"

mkdir -p "${OUT_DIR}"
: > "${OUT_DIR}/steps.txt"
FAIL_COUNT=0

# Route compilation cache to /dev/shm when available to avoid disk I/O jitter.
if [[ -d /dev/shm ]]; then
  mkdir -p /dev/shm/torchtpu_cache
  export TMPDIR="${TMPDIR:-/dev/shm/torchtpu_cache}"
fi
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1

log() {
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) $* ==="
}

run_step() {
  local name="$1"
  local limit_s="$2"
  shift 2
  log "START ${name}"
  timeout "${limit_s}" "$@" > >(tee "${OUT_DIR}/${name}.log") 2>&1
  local rc=$?
  log "END ${name} rc=${rc}"
  echo "${name} ${rc}" >> "${OUT_DIR}/steps.txt"
  if [[ "${rc}" -ne 0 ]]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
}

# Bootstraps TORCH_TPU_TOPOLOGY and TORCH_TPU_SLICEBUILDER_ADDRESSES in a
# short-lived helper process, then launches torchrun across `nproc` TensorCores.
run_torchrun() {
  local name="$1"
  local limit_s="$2"
  local nproc="$3"
  shift 3
  local exports
  exports="$(
    env -u TPU_VISIBLE_DEVICES -u TPU_VISIBLE_CHIPS \
        -u TORCH_TPU_TOPOLOGY -u TORCH_TPU_SLICEBUILDER_ADDRESSES \
      python3 -c '
import os, sys, torch, torch_tpu
torch.tpu.distributed.set_tpu_launch_env(nproc_per_node=int(sys.argv[1]))
for k in ("TORCH_TPU_TOPOLOGY", "TORCH_TPU_SLICEBUILDER_ADDRESSES",
          "TORCH_TPU_XPROF_SESSION_ID"):
  print(f"export {k}=\"{os.environ[k]}\"")
' "${nproc}" 2>/dev/null
  )"
  run_step "${name}" "${limit_s}" bash -c "
    unset TPU_VISIBLE_DEVICES TPU_VISIBLE_CHIPS
    ${exports}
    exec torchrun --standalone --nnodes=1 --nproc_per_node=${nproc} \"\$@\"
  " _ "$@"
}

COMMON_FLAGS=(
  --iters "${ITERS}"
  --warmup "${WARMUP}"
  --trace_mode "${TRACE_MODE}"
  --trace_iters "${TRACE_ITERS}"
)

# 0. Record software and TPU device environment.
run_step env_check 180 env TORCH_TPU_TOPOLOGY="1,1,1,1" TPU_VISIBLE_DEVICES=0 \
  python3 -c '
import importlib.metadata as md, warnings, torch, torch_tpu
warnings.filterwarnings("ignore", message=".*ExperimentalApi.*")
dev = torch.device("tpu")
x = torch.ones((4, 4), device=dev, dtype=torch.bfloat16)
y = (x @ x).cpu()
print("torch:", torch.__version__, "sum:", float(y.sum()))
for p in ("torch_tpu", "libtpu", "jax", "jaxlib", "tokamax",
          "vllm_torchtpu_core", "jax_flash_attn2", "xprof"):
  try:
    print(f"{p}: {md.version(p)}")
  except Exception:
    print(f"{p}: not_found")
print("props:", torch.tpu.get_device_properties(0))
'

if [[ "${SUITE}" == "smoke" ]]; then
  run_step smoke_core 600 env TORCH_TPU_TOPOLOGY="1,1,1,1" \
    TPU_VISIBLE_DEVICES=0 python3 "${KB}" --preset smoke \
    --kernel temu_table,attention,paged,rmsnorm,moe --scope core \
    --iters 2 --warmup 1 --trace_mode none --out "${OUT_DIR}/smoke_core"

  run_torchrun smoke_chip 600 2 "${KB}" --preset smoke --kernel temu_table \
    --scope chip --iters 2 --warmup 1 --trace_mode none \
    --out "${OUT_DIR}/smoke_chip"
fi

if [[ "${SUITE}" == "temu_table" || "${SUITE}" == "all" ]]; then
  # 1-TensorCore (0.5 physical chip, batch=128) — reproduces client methodology.
  run_step temu_table_1core 1800 env TORCH_TPU_TOPOLOGY="1,1,1,1" \
    TPU_VISIBLE_DEVICES=0 python3 "${KB}" --preset temu \
    --kernel temu_table --scope core "${COMMON_FLAGS[@]}" \
    --out "${OUT_DIR}/temu_table_1core"

  # Full 1-Physical-Chip (2 TensorCores, batch=64x2=128) — exact math, 2x speed.
  run_torchrun temu_table_1chip 1500 2 "${KB}" --preset temu \
    --kernel temu_table --scope chip --iters "${ITERS}" --warmup "${WARMUP}" \
    --trace_mode none --verify_precision --out "${OUT_DIR}/temu_table_1chip"
fi

if [[ "${SUITE}" == "kernels" || "${SUITE}" == "all" ]]; then
  # All four customer kernel suites on 1 TensorCore.
  run_step kernels_1core 1800 env TORCH_TPU_TOPOLOGY="1,1,1,1" \
    TPU_VISIBLE_DEVICES=0 python3 "${KB}" --preset temu \
    --kernel attention,paged,rmsnorm,moe --scope core "${COMMON_FLAGS[@]}" \
    --out "${OUT_DIR}/kernels_1core"

  # Dual-core (1 chip) and full-host (8 TensorCores / 4 chips) scaling checks.
  run_torchrun kernels_1chip 900 2 "${KB}" --preset temu \
    --kernel attention,paged,rmsnorm \
    --impls splash_opt,torchtpu_flash,rpa_v3_hd64,rpa_v3,torch_rms_norm \
    --scope chip --iters "${ITERS}" --warmup "${WARMUP}" --trace_mode none \
    --no-verify_precision --out "${OUT_DIR}/kernels_1chip"

  run_torchrun kernels_host8 900 8 "${KB}" --preset temu \
    --kernel attention,paged,rmsnorm \
    --impls splash_opt,torchtpu_flash,rpa_v3_hd64,rpa_v3,torch_rms_norm \
    --scope host --iters "${ITERS}" --warmup "${WARMUP}" --trace_mode none \
    --no-verify_precision --out "${OUT_DIR}/kernels_host8"
fi

if [[ "${SUITE}" == "sweeps" || "${SUITE}" == "all" ]]; then
  # Attention 9-point block sweep + RPAv3 HD64 (bkvp, bq) & GQA-4 tuning sweep.
  run_step sweeps_1core 1200 env TORCH_TPU_TOPOLOGY="1,1,1,1" \
    TPU_VISIBLE_DEVICES=0 python3 "${KB}" --preset sweep \
    --kernel attention,paged --scope core --iters "${ITERS}" \
    --warmup "${WARMUP}" --trace_mode none --no-verify_precision \
    --out "${OUT_DIR}/sweeps_1core"
fi

if [[ -n "${GCS_DEST}" ]]; then
  log "START gcs_upload"
  timeout 600 python3 "${BENCH_DIR}/gcs_upload.py" \
    "${OUT_DIR}" "${GCS_DEST}" > >(tee "${OUT_DIR}/gcs_upload.log") 2>&1 || \
    log "WARN: gcs_upload returned non-zero (results remain in ${OUT_DIR})."
  log "END gcs_upload"
fi

log "SUMMARY OF STEPS (failed=${FAIL_COUNT}):"
cat "${OUT_DIR}/steps.txt"

if [[ "${FAIL_COUNT}" -ne 0 ]]; then
  log "ERROR: ${FAIL_COUNT} step(s) failed."
  exit 1
fi
log "ALL STEPS PASSED."
exit 0
