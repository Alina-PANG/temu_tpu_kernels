#!/usr/bin/env python3
"""Unified TorchTPU kernel benchmark harness for TPU v7x reproduction.

Benchmarks all four customer kernel families plus the complete 12-row and
5-sequence-length comparison tables on Cloud TPU v7x using native TorchTPU
(`torch_tpu` + PyTorch `device="tpu"`):

  temu_table  All implementations & sequence lengths (`1024..10240`) from the
              customer's comparison tables (`batch=128, heads=4, dim=128`),
              with standalone `fwd` and standalone `bwd` (never fusing `fwd`
              into `bwd`), unfused `splash_sep` (`use_fused_bwd_kernel=False`),
              and precision-verified TorchTPU optimizations.
  attention   Target `[64, 2048, 8, 64]`, iso-FLOP `d=128` & `d=256` shapes,
              and 9-point block-size sweep.
  paged       `vllm_torchtpu` Ragged Paged Attention v3 (`d=64` & `d=128`,
              prefill & decode) plus the 14-point `(bkvp, bq)` & GQA-4 sweep.
  rmsnorm     Native `F.rms_norm`, `f32`-accumulated, and `bf16` RMSNorm
              (standalone `fwd` and standalone `bwd`).
  moe         SwiGLU MoE on `torch.tpu.ops.ragged_dot`, `torch._grouped_mm`,
              and `tokamax.ragged_dot` (6 stages + `e2e` + `e2e_opt`).
"""

from __future__ import annotations

import os

# Keep JAX in trace-only CPU mode so TorchTPU exclusively owns the TPU PJRT
# client and `torch.tpu.pallas.jax_op` exports Mosaic TPU MLIR without locks.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# pylint: disable=g-import-not-at-top
import argparse
from collections.abc import Callable, Sequence
import dataclasses
import glob
import gzip
import json
import pathlib
import statistics
import time
import traceback
from typing import Any
import warnings

import torch
import torch.distributed as dist
import torch.nn.functional as F

import attn_ops
import other_ops
# pylint: enable=g-import-not-at-top

warnings.filterwarnings("ignore", message=".*ExperimentalApi.*")
warnings.filterwarnings("ignore", message=".*barrier.*current context.*")

_PREC_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}

_RESULT_PREFIX = "RESULT_JSON "


@dataclasses.dataclass
class BenchCtx:
  """Run-wide benchmark settings and collected result records."""

  out_dir: str
  device: torch.device
  dtype: torch.dtype
  scope: str
  rank: int
  world_size: int
  iters: int
  warmup: int
  trace_mode: str
  trace_iters: int
  verify_precision: bool
  impls: set[str] | None
  results: list[dict[str, Any]] = dataclasses.field(default_factory=list)


def init_runtime(device_pref: str, scope: str) -> tuple[torch.device, int, int]:
  """Initializes TorchTPU device and optional `torchrun` process group.

  Args:
    device_pref: `"auto"`, `"tpu"`, or `"cpu"`.
    scope: Execution scope label (`"core"`, `"chip"`, or `"host"`).

  Returns:
    Tuple `(device, rank, world_size)`.
  """
  del scope
  rank = int(os.environ.get("RANK", "0"))
  world_size = int(os.environ.get("WORLD_SIZE", "1"))
  if device_pref == "cpu":
    dev = torch.device("cpu")
  elif device_pref == "tpu":
    import torch_tpu  # pylint: disable=g-import-not-at-top,unused-import

    dev = torch.device("tpu")
  else:
    try:
      import torch_tpu  # pylint: disable=g-import-not-at-top,unused-import

      dev = (
          torch.device("tpu")
          if torch.accelerator.is_available()
          else torch.device("cpu")
      )
    except ImportError:
      dev = torch.device("cpu")

  if world_size > 1 and not dist.is_initialized():
    backend = "tpu_dist" if dev.type == "tpu" else "gloo"
    dist.init_process_group(backend=backend)
  return dev, rank, world_size


def sync_device(dev: torch.device) -> None:
  """Synchronizes the active accelerator when running on TPU."""
  if dev.type == "tpu":
    torch.accelerator.synchronize()


def barrier_if_dist(world_size: int) -> None:
  """Synchronizes all `torchrun` ranks before starting a timed loop."""
  if world_size > 1 and dist.is_initialized():
    dist.barrier()


def summarize_trace(trace_dir: str, top: int = 15) -> dict[str, Any]:
  """Extracts per-op TPU hardware durations from `.trace.json.gz`.

  Args:
    trace_dir: Directory written by `torch.profiler`.
    top: Maximum number of top hardware ops to keep.

  Returns:
    Dictionary listing trace files and top ops sorted by `avg_ms`.
  """
  files = sorted(
      glob.glob(os.path.join(trace_dir, "**", "*"), recursive=True),
      key=os.path.getmtime,
  )
  files = [f for f in files if os.path.isfile(f)]
  summary: dict[str, Any] = {
      "files": [os.path.relpath(f, trace_dir) for f in files]
  }
  gz_files = [f for f in files if f.endswith(".trace.json.gz")]
  if not gz_files:
    summary["note"] = "No .trace.json.gz emitted; inspect .xplane.pb in xprof."
    return summary

  with gzip.open(gz_files[-1], "rb") as f:
    events = json.load(f).get("traceEvents", [])

  totals: dict[tuple[Any, str], tuple[int, float]] = {}
  for ev in events:
    ev_args = ev.get("args")
    if not isinstance(ev_args, dict) or "name" not in ev:
      continue
    dur_ps = ev_args.get("device_duration_ps")
    if dur_ps is None:
      continue
    key = (ev.get("pid", -1), ev["name"])
    cnt, tot = totals.get(key, (0, 0.0))
    totals[key] = (cnt + 1, tot + float(dur_ps) / 1e9)

  if not totals:
    summary["note"] = "No device_duration_ps events found in trace."
    return summary

  first_pid = min(pid for pid, _ in totals)
  rows = [
      dict(op=name, count=c, total_ms=round(t, 4), avg_ms=round(t / c, 4))
      for (pid, name), (c, t) in totals.items()
      if pid == first_pid
  ]
  rows.sort(key=lambda r: -r["total_ms"])
  summary.update(pid=first_pid, ops=rows[:top])
  return summary


def maybe_trace_step(
    ctx: BenchCtx, case_name: str, step_fn: Callable[[], Any], key_case: bool
) -> dict[str, Any] | None:
  """Captures an XProf TPU trace when enabled by `--trace_mode`.

  Args:
    ctx: Benchmark context.
    case_name: Unique case name.
    step_fn: Zero-arg callable under test.
    key_case: Whether this case is marked as a representative trace target.

  Returns:
    Parsed trace summary dictionary, or `None` if tracing was skipped.
  """
  if ctx.trace_iters <= 0 or ctx.trace_mode == "none":
    return None
  if ctx.trace_mode == "key" and not key_case:
    return None
  if ctx.rank != 0:
    for _ in range(ctx.trace_iters):
      step_fn()
      sync_device(ctx.device)
    return None

  trace_dir = os.path.join(ctx.out_dir, "xprof", case_name, ctx.scope)
  os.makedirs(trace_dir, exist_ok=True)
  activities = [torch.profiler.ProfilerActivity.CPU]
  exp_cfg = None
  if ctx.device.type == "tpu":
    tpu_act = getattr(torch.profiler.ProfilerActivity, "TPU", None)
    if tpu_act is not None:
      activities.append(tpu_act)
    tpu_prof = getattr(getattr(torch, "tpu", None), "profiler", None)
    if tpu_prof is not None and hasattr(tpu_prof, "TpuProfilerConfig"):
      exp_cfg = tpu_prof.TpuProfilerConfig(
          host_tracer_level=2,
          device_tracer_level=1,
          python_tracer_level=0,
          run_dir=pathlib.Path(trace_dir),
      )

  prof_kw: dict[str, Any] = dict(
      activities=activities,
      on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
  )
  if exp_cfg is not None:
    prof_kw["experimental_config"] = exp_cfg

  with torch.profiler.profile(**prof_kw) as prof:
    for _ in range(ctx.trace_iters):
      step_fn()
      sync_device(ctx.device)
      prof.step()
  return summarize_trace(trace_dir)


def emit(ctx: BenchCtx, record: dict[str, Any]) -> None:
  """Records and prints one result line on rank 0."""
  if ctx.rank != 0:
    return
  ctx.results.append(record)
  line = json.dumps(record, sort_keys=True, default=str)
  print(_RESULT_PREFIX + line, flush=True)
  with open(os.path.join(ctx.out_dir, "results.jsonl"), "a") as f:
    f.write(line + "\n")


def time_callable(
    ctx: BenchCtx,
    record: dict[str, Any],
    step_fn: Callable[[], Any],
    *,
    flops_per_device: float | None = None,
    bytes_per_device: float | None = None,
    iters_override: int | None = None,
    warmup_override: int | None = None,
    key_trace: bool = False,
) -> None:
  """Compiles, warms up, times, and optionally traces a zero-arg TPU step.

  Records both `torch.tpu.Event` hardware elapsed time (when available) and
  host-synchronized `perf_counter()` median (`wall_ms_p50`).

  Args:
    ctx: Benchmark context.
    record: Metadata dictionary to populate and emit.
    step_fn: Zero-argument callable executing ONLY the target pass.
    flops_per_device: Optional FLOPs per device per call.
    bytes_per_device: Optional HBM bytes per device per call.
    iters_override: Optional iteration count override for slow cases.
    warmup_override: Optional warmup count override.
    key_trace: Whether to capture an XProf trace under `--trace_mode=key`.
  """
  rec = dict(record, scope=ctx.scope, num_devices=ctx.world_size)
  n_iters = iters_override if iters_override is not None else ctx.iters
  n_warm = warmup_override if warmup_override is not None else ctx.warmup
  try:
    t0 = time.perf_counter()
    step_fn()
    sync_device(ctx.device)
    rec["compile_s"] = round(time.perf_counter() - t0, 3)

    for _ in range(n_warm):
      step_fn()
      sync_device(ctx.device)

    barrier_if_dist(ctx.world_size)
    has_events = (
        ctx.device.type == "tpu"
        and hasattr(getattr(torch, "tpu", None), "Event")
    )
    wall_ms: list[float] = []
    event_ms: list[float] = []

    for _ in range(n_iters):
      if has_events:
        ev_start = torch.tpu.Event(enable_timing=True)
        ev_end = torch.tpu.Event(enable_timing=True)
        t_s = time.perf_counter()
        ev_start.record()
        step_fn()
        ev_end.record()
        ev_end.synchronize()
        wall_ms.append((time.perf_counter() - t_s) * 1e3)
        event_ms.append(float(ev_start.elapsed_time(ev_end)))
      else:
        t_s = time.perf_counter()
        step_fn()
        sync_device(ctx.device)
        wall_ms.append((time.perf_counter() - t_s) * 1e3)

    wall_ms.sort()
    rec["wall_ms_p50"] = round(statistics.median(wall_ms), 4)
    rec["wall_ms_min"] = round(wall_ms[0], 4)
    rec["wall_ms_max"] = round(wall_ms[-1], 4)
    best_ms = rec["wall_ms_p50"]
    if event_ms:
      event_ms.sort()
      rec["event_ms_p50"] = round(statistics.median(event_ms), 4)
      rec["event_ms_min"] = round(event_ms[0], 4)
      if rec["event_ms_p50"] > 0:
        best_ms = rec["event_ms_p50"]
    rec["primary_ms_p50"] = best_ms

    if flops_per_device:
      rec["flops_per_device"] = flops_per_device
      tflops = flops_per_device / (best_ms / 1e3) / 1e12
      rec["tflops_per_device"] = round(tflops, 2)
      rec["tflops_total_scope"] = round(tflops * ctx.world_size, 2)
    if bytes_per_device:
      rec["bytes_per_device"] = bytes_per_device
      gbps = bytes_per_device / (best_ms / 1e3) / 1e9
      rec["hbm_gbps_per_device"] = round(gbps, 1)
      rec["hbm_gbps_total_scope"] = round(gbps * ctx.world_size, 1)

    try:
      trace_summary = maybe_trace_step(ctx, rec["case"], step_fn, key_trace)
      if trace_summary is not None:
        rec["device_ops"] = trace_summary
    except Exception as trace_err:  # pylint: disable=broad-exception-caught
      rec["trace_error"] = f"{type(trace_err).__name__}: {trace_err}"[:400]
  except Exception as err:  # pylint: disable=broad-exception-caught
    rec["error"] = f"{type(err).__name__}: {err}"[:1500]
    rec["traceback"] = traceback.format_exc()[-2500:]
  emit(ctx, rec)


def selected(ctx: BenchCtx, impl: str) -> bool:
  """Returns whether `--impls` includes `impl`."""
  return ctx.impls is None or impl in ctx.impls


def attention_flops(
    batch: int, seq: int, heads: int, dim: int, mode: str
) -> float:
  """Returns standalone FLOPs for causal attention (`fwd` or standalone `bwd`).

  Args:
    batch: Batch size per device.
    seq: Sequence length.
    heads: Number of query heads.
    dim: Head dimension.
    mode: `"fwd"` (`1x`) or standalone `"bwd"` (`2.5x` forward FLOPs).

  Returns:
    Floating-point operations executed in one pass on one device.
  """
  fwd = 4.0 * batch * heads * dim * (seq * (seq + 1) // 2)
  return fwd if mode == "fwd" else 2.5 * fwd


def check_attn_precision(
    ctx: BenchCtx,
    impl: str,
    heads: int,
    kv_heads: int,
    dim: int,
    blocks: Sequence[int] | None,
    bwd_blocks: Sequence[int] | None = None,
) -> dict[str, float]:
  """Verifies `fwd` and standalone `bwd` numerical accuracy vs `fp32` SDPA.

  Runs on a small `[2, heads, 512, dim]` tensor so verification takes `<50 ms`
  while confirming cosine similarity and max absolute error against `float32`.

  Args:
    ctx: Benchmark context.
    impl: Attention implementation name.
    heads: Query head count.
    kv_heads: KV head count.
    dim: Head dimension.
    blocks: Forward block sizes.
    bwd_blocks: Optional backward block sizes.

  Returns:
    Dictionary with `fwd_cos_sim`, `fwd_max_abs`, `bwd_dq_cos_sim`, and
    `bwd_dq_max_abs`.
  """
  seq_chk = 512
  blk_chk = tuple(min(int(b), seq_chk) for b in blocks) if blocks else None
  bwd_chk = (
      tuple(min(int(b), seq_chk) for b in bwd_blocks) if bwd_blocks else None
  )
  torch.manual_seed(123)
  q = torch.randn(2, heads, seq_chk, dim, dtype=ctx.dtype, device=ctx.device)
  k = torch.randn(
      2, kv_heads, seq_chk, dim, dtype=ctx.dtype, device=ctx.device
  )
  v = torch.randn(
      2, kv_heads, seq_chk, dim, dtype=ctx.dtype, device=ctx.device
  )
  do = torch.randn(2, heads, seq_chk, dim, dtype=ctx.dtype, device=ctx.device)

  ref_out, ref_grads = attn_ops.reference_attention_fp32(
      q.cpu(), k.cpu(), v.cpu(), do.cpu()
  )
  assert ref_grads is not None
  fwd_fn, bwd_setup = attn_ops.build_attention_ops(
      impl, seq_chk, heads, kv_heads, dim, blk_chk, ctx.device.type, bwd_chk
  )
  out = fwd_fn(q, k, v).detach().cpu().float()
  bwd_step = bwd_setup(q, k, v, do)
  dq, _, _ = bwd_step()
  dq_f = dq.detach().cpu().float()

  fwd_cos = float(
      F.cosine_similarity(out.flatten(), ref_out.flatten(), dim=0).item()
  )
  bwd_cos = float(
      F.cosine_similarity(dq_f.flatten(), ref_grads[0].flatten(), dim=0).item()
  )
  return {
      "fwd_cos_sim": round(fwd_cos, 6),
      "fwd_max_abs": round(float((out - ref_out).abs().max().item()), 5),
      "bwd_dq_cos_sim": round(bwd_cos, 6),
      "bwd_dq_max_abs": round(
          float((dq_f - ref_grads[0]).abs().max().item()), 5
      ),
  }


def _run_attn_pair(
    ctx: BenchCtx,
    *,
    suite_name: str,
    impl: str,
    batch: int,
    seq: int,
    heads: int,
    kv_heads: int,
    dim: int,
    blocks: Sequence[int] | None,
    bwd_blocks: Sequence[int] | None = None,
    modes: Sequence[str] = ("fwd", "bwd"),
    key_trace: bool = False,
    extra_meta: dict[str, Any] | None = None,
) -> None:
  """Allocates inputs once and runs standalone `fwd` and standalone `bwd`."""
  if not selected(ctx, impl):
    return
  tag = "x".join(map(str, blocks)) if blocks else "na"
  if bwd_blocks:
    tag += "_bwd" + "x".join(map(str, bwd_blocks))

  prec_stats: dict[str, Any] = {}
  if ctx.verify_precision:
    pkey = (
        impl,
        heads,
        kv_heads,
        dim,
        tuple(blocks) if blocks else None,
        tuple(bwd_blocks) if bwd_blocks else None,
    )
    if pkey not in _PREC_CACHE:
      try:
        _PREC_CACHE[pkey] = check_attn_precision(
            ctx, impl, heads, kv_heads, dim, blocks, bwd_blocks
        )
      except Exception as p_err:  # pylint: disable=broad-exception-caught
        _PREC_CACHE[pkey] = {"precision_error": str(p_err)[:200]}
    prec_stats = _PREC_CACHE[pkey]

  slow = impl in ("fa2", "ref") or seq >= 4096
  iters_ov = min(ctx.iters, 5) if slow else None
  warm_ov = min(ctx.warmup, 1) if slow else None

  base_rec: dict[str, Any] = dict(
      kernel=suite_name,
      impl=impl,
      batch=batch,
      global_batch=batch * ctx.world_size,
      seq=seq,
      heads=heads,
      kv_heads=kv_heads,
      head_dim=dim,
      blocks=blocks,
      bwd_blocks=bwd_blocks,
      standalone_bwd=True,
      **prec_stats,
      **(extra_meta or {}),
  )

  try:
    torch.manual_seed(42)
    q = torch.randn(batch, heads, seq, dim, dtype=ctx.dtype, device=ctx.device)
    k = torch.randn(
        batch, kv_heads, seq, dim, dtype=ctx.dtype, device=ctx.device
    )
    v = torch.randn(
        batch, kv_heads, seq, dim, dtype=ctx.dtype, device=ctx.device
    )
    do = torch.randn(batch, heads, seq, dim, dtype=ctx.dtype, device=ctx.device)
    fwd_fn, bwd_setup = attn_ops.build_attention_ops(
        impl, seq, heads, kv_heads, dim, blocks, ctx.device.type, bwd_blocks
    )
  except Exception as err:  # pylint: disable=broad-exception-caught
    for mode in modes:
      cname = f"attn_{impl}_{mode}_b{batch}_s{seq}_h{heads}_d{dim}_blk{tag}"
      emit(
          ctx,
          dict(
              base_rec,
              case=cname,
              mode=mode,
              scope=ctx.scope,
              error=f"{type(err).__name__}: {err}"[:1500],
          ),
      )
    return

  if "fwd" in modes:
    cname = f"attn_{impl}_fwd_b{batch}_s{seq}_h{heads}_d{dim}_blk{tag}"
    time_callable(
        ctx,
        dict(base_rec, case=cname, mode="fwd"),
        lambda: fwd_fn(q, k, v),
        flops_per_device=attention_flops(batch, seq, heads, dim, "fwd"),
        iters_override=iters_ov,
        warmup_override=warm_ov,
        key_trace=key_trace,
    )

  if "bwd" in modes:
    cname = f"attn_{impl}_bwd_b{batch}_s{seq}_h{heads}_d{dim}_blk{tag}"
    try:
      bwd_step = bwd_setup(q, k, v, do)
    except Exception as err:  # pylint: disable=broad-exception-caught
      emit(
          ctx,
          dict(
              base_rec,
              case=cname,
              mode="bwd",
              scope=ctx.scope,
              error=f"{type(err).__name__}: {err}"[:1500],
          ),
      )
    else:
      time_callable(
          ctx,
          dict(base_rec, case=cname, mode="bwd"),
          bwd_step,
          flops_per_device=attention_flops(batch, seq, heads, dim, "bwd"),
          iters_override=iters_ov,
          warmup_override=warm_ov,
          key_trace=key_trace,
      )


def run_temu_table(ctx: BenchCtx, preset: str) -> None:
  """Runs every row of the customer's Table 1.1 and Table 1.2 (`fwd` & `bwd`).

  When launched with `WORLD_SIZE == 2` (`scope=chip`, 1 full TPU v7x physical
  chip = 2 TensorCores), automatically shards `global_batch=128` as `batch=64`
  per TensorCore so every row reports exact full-chip latency alongside 1-core
  (`batch=128`) parity.

  Args:
    ctx: Benchmark context.
    preset: `"smoke"` or `"temu"`.
  """
  if preset == "smoke":
    b_smoke = max(1, 4 // ctx.world_size)
    for impl, blk in (
        ("torchtpu_flash", None),
        ("splash_sep", (256, 512, 256)),
        ("splash_fused", (256, 512, 256)),
        ("splash_opt", (512, 512, 512)),
        ("tokamax_splash", (256, 512, 256)),
        ("flash_attention", (256, 512, 256)),
        ("fa2", (256, 256)),
        ("ref", None),
        ("dpa", None),
    ):
      _run_attn_pair(
          ctx,
          suite_name="temu_table",
          impl=impl,
          batch=b_smoke,
          seq=512,
          heads=4,
          kv_heads=4,
          dim=128,
          blocks=blk,
      )
    return

  global_batch = 128
  per_dev_batch = global_batch // ctx.world_size
  heads, dim = 4, 128

  # Part 1: All 12 customer rows (`seq_len = 1024` and `2048`, `fwd` &
  # standalone `bwd`) plus precision-preserving optimized TorchTPU variants.
  for seq in (1024, 2048):
    cust_splash_blk = (512, 1024, 512) if seq == 1024 else (512, 2048, 1024)
    fa_blk = (256, seq, 512)
    table_rows: list[
        tuple[str, Sequence[int] | None, Sequence[int] | None, bool]
    ] = [
        ("flash_attention", fa_blk, None, False),
        ("splash_sep", cust_splash_blk, None, True),
        ("splash_fused", cust_splash_blk, None, True),
        ("fa2", (256, 256), None, False),
        ("ref", None, None, False),
        ("dpa", None, None, False),
        ("torchtpu_flash", None, None, True),
        ("splash_opt", (1024, 1024, 1024), (512, 1024, 512), True),
        ("tokamax_splash", (1024, 1024, 1024), (512, 1024, 512), True),
        ("tokamax_splash_sep", cust_splash_blk, None, False),
    ]
    for impl, f_blk, b_blk, trace_flag in table_rows:
      _run_attn_pair(
          ctx,
          suite_name="temu_table",
          impl=impl,
          batch=per_dev_batch,
          seq=seq,
          heads=heads,
          kv_heads=heads,
          dim=dim,
          blocks=f_blk,
          bwd_blocks=bwd_blk_or_none(b_blk),
          key_trace=(trace_flag and seq == 2048),
          extra_meta={"table": "table_1_1"},
      )

  # Part 2: Sequence-length scaling vs H100 FA3 (`seq_len = 4096, 8192, 10240`,
  # since `1024` and `2048` are already measured above).
  seq_configs = [
      (4096, (512, 2048, 1024)),
      (8192, (512, 4096, 1024)),
      (10240, (512, 5120, 1024)),
  ]
  for seq, cust_blk in seq_configs:
    for impl, f_blk, b_blk in (
        ("splash_fused", cust_blk, None),
        ("splash_sep", cust_blk, None),
        ("splash_opt", (1024, 2048, 1024), (512, 1024, 512)),
        ("tokamax_splash", (1024, 2048, 1024), (512, 1024, 512)),
        ("torchtpu_flash", None, None),
    ):
      _run_attn_pair(
          ctx,
          suite_name="temu_table",
          impl=impl,
          batch=per_dev_batch,
          seq=seq,
          heads=heads,
          kv_heads=heads,
          dim=dim,
          blocks=f_blk,
          bwd_blocks=b_blk,
          key_trace=False,
          extra_meta={"table": "table_1_2"},
      )


def bwd_blk_or_none(
    blk: Sequence[int] | None,
) -> Sequence[int] | None:
  """Normalizes optional backward block tuples."""
  return tuple(blk) if blk is not None else None


_SWEEP_GRID = (
    (512, 2048, 1024),
    (512, 2048, 2048),
    (1024, 2048, 1024),
    (1024, 2048, 2048),
    (2048, 2048, 1024),
    (2048, 2048, 2048),
    (1024, 1024, 1024),
    (512, 1024, 512),
    (256, 2048, 1024),
)


def run_attention(ctx: BenchCtx, preset: str) -> None:
  """Runs target `[64, 2048, 8, 64]`, iso-FLOP `d=128`/`256`, and sweep."""
  if preset == "smoke":
    for dim in (64, 128):
      _run_attn_pair(
          ctx,
          suite_name="attention",
          impl="torchtpu_flash",
          batch=2,
          seq=512,
          heads=4,
          kv_heads=4,
          dim=dim,
          blocks=None,
      )
    return

  if preset == "sweep":
    for batch, seq, heads, dim in ((64, 2048, 8, 64), (64, 2048, 4, 128)):
      for impl in ("splash_opt", "tokamax_splash"):
        for blk in _SWEEP_GRID:
          _run_attn_pair(
              ctx,
              suite_name="attention",
              impl=impl,
              batch=batch,
              seq=seq,
              heads=heads,
              kv_heads=heads,
              dim=dim,
              blocks=blk,
              modes=("fwd",),
          )
    return

  # Customer target shape `[64, 2048, 8, 64]` and iso-FLOP `d=128`, `d=256`.
  for batch, seq, heads, dim in (
      (64, 2048, 8, 64),
      (64, 2048, 4, 128),
      (64, 2048, 2, 256),
  ):
    modes = ("fwd",) if dim == 256 else ("fwd", "bwd")
    for impl, f_blk, b_blk in (
        ("splash_fused", (512, 2048, 1024), None),
        ("splash_sep", (512, 2048, 1024), None),
        ("splash_opt", (1024, 1024, 1024), (512, 1024, 512)),
        ("tokamax_splash", (1024, 1024, 1024), (512, 1024, 512)),
        ("torchtpu_flash", None, None),
        ("dpa", None, None),
    ):
      if dim == 256 and impl in ("torchtpu_flash", "splash_sep"):
        continue
      _run_attn_pair(
          ctx,
          suite_name="attention",
          impl=impl,
          batch=batch,
          seq=seq,
          heads=heads,
          kv_heads=heads,
          dim=dim,
          blocks=f_blk,
          bwd_blocks=b_blk,
          modes=modes,
          key_trace=(impl in ("splash_opt", "torchtpu_flash") and dim <= 128),
      )


def run_paged(ctx: BenchCtx, preset: str) -> None:
  """Runs PageAttention RPAv3 (`d=64`, `d=128`) and HD64 `(bkvp, bq)` tuning."""
  if preset == "smoke":
    cases = [
        (4, 512, 2, 2, 64, "prefill", None, None),
        (4, 512, 2, 2, 128, "decode", None, None),
    ]
  elif preset == "sweep":
    # 14-point HD64 `(bkvp, bq)` and MHA (`hq=8,hkv=8`) vs GQA-4 (`hq=8,hkv=2`).
    cases = [
        (64, 2048, 8, 8, 64, "prefill", None, None),
        (64, 2048, 8, 8, 64, "prefill", 8, 64),
        (64, 2048, 8, 8, 64, "prefill", 8, 128),
        (64, 2048, 8, 8, 64, "prefill", 8, 256),
        (64, 2048, 8, 8, 64, "prefill", 4, 256),
        (64, 2048, 8, 8, 64, "prefill", 8, 512),
        (64, 2048, 8, 8, 64, "prefill", 4, 512),
        (64, 2048, 8, 2, 64, "prefill", None, None),
        (64, 2048, 8, 2, 64, "prefill", 8, 128),
        (64, 2048, 8, 2, 64, "prefill", 8, 256),
        (64, 2048, 8, 2, 64, "prefill", 8, 512),
    ]
  else:
    cases = [
        (64, 2048, 8, 8, 64, "prefill", None, None),
        (64, 2048, 8, 8, 64, "prefill", 8, 256),
        (64, 2048, 8, 2, 64, "prefill", 8, 256),
        (64, 2048, 4, 4, 128, "prefill", None, None),
        (64, 2048, 8, 8, 64, "decode", None, None),
        (64, 2048, 4, 4, 128, "decode", None, None),
    ]

  page_size = 256
  for batch, ctx_len, hq, hkv, dim, phase, bkvp, bq in cases:
    impl = "rpa_v3_hd64" if dim == 64 else "rpa_v3"
    if not (selected(ctx, impl) or selected(ctx, "rpa_v3")):
      continue
    btag = "auto" if bq is None else f"bkvp{bkvp}_bq{bq}"
    cname = (
        f"rpa3_{phase}_b{batch}_ctx{ctx_len}_hq{hq}_hkv{hkv}"
        f"_d{dim}_p{page_size}_{btag}"
    )
    rec = dict(
        kernel="paged",
        case=cname,
        impl=impl,
        mode=phase,
        batch=batch,
        seq=ctx_len,
        heads=hq,
        kv_heads=hkv,
        head_dim=dim,
        page_size=page_size,
        bkvp=bkvp,
        bq=bq,
    )
    try:
      step_fn, meta = other_ops.build_paged_attention(
          batch, ctx_len, hq, hkv, dim, page_size, phase, ctx.dtype,
          ctx.device, bkvp=bkvp, bq=bq,
      )
    except Exception as err:  # pylint: disable=broad-exception-caught
      emit(
          ctx,
          dict(
              rec,
              scope=ctx.scope,
              error=f"{type(err).__name__}: {err}"[:1500],
          ),
      )
      continue
    time_callable(
        ctx,
        rec,
        step_fn,
        flops_per_device=meta["flops"],
        key_trace=(preset == "temu" and phase == "prefill"),
    )


def run_rmsnorm(ctx: BenchCtx, preset: str) -> None:
  """Runs RMSNorm standalone `fwd` and standalone `bwd` on TorchTPU."""
  shapes = (
      [(1024, 512)]
      if preset == "smoke"
      else [(131072, 512), (32768, 4096), (8192, 8192)]
  )
  elem_bytes = 2 if ctx.dtype == torch.bfloat16 else 4
  for rows, dim in shapes:
    torch.manual_seed(0)
    x = torch.randn(rows, dim, dtype=ctx.dtype, device=ctx.device)
    w = torch.ones(dim, dtype=ctx.dtype, device=ctx.device)
    dy = torch.randn(rows, dim, dtype=ctx.dtype, device=ctx.device)
    for impl in ("torch_rms_norm", "rmsnorm_f32acc", "rmsnorm_bf16"):
      if not selected(ctx, impl):
        continue
      fwd_fn, bwd_setup = other_ops.build_rmsnorm_ops(
          impl, dim, ctx.device.type
      )
      fwd_bytes = float((2 * rows * dim + dim) * elem_bytes)
      bwd_bytes = float((3 * rows * dim + 2 * dim) * elem_bytes)

      cname_f = f"rmsnorm_{impl}_fwd_m{rows}_n{dim}"
      time_callable(
          ctx,
          dict(
              kernel="rmsnorm",
              case=cname_f,
              impl=impl,
              mode="fwd",
              rows=rows,
              dim=dim,
          ),
          lambda f=fwd_fn, x=x, w=w: f(x, w),
          bytes_per_device=fwd_bytes,
          key_trace=(rows == 32768 and impl == "torch_rms_norm"),
      )

      cname_b = f"rmsnorm_{impl}_bwd_m{rows}_n{dim}"
      bwd_step = bwd_setup(x, w, dy)
      time_callable(
          ctx,
          dict(
              kernel="rmsnorm",
              case=cname_b,
              impl=impl,
              mode="bwd",
              rows=rows,
              dim=dim,
              standalone_bwd=True,
          ),
          bwd_step,
          bytes_per_device=bwd_bytes,
          key_trace=(rows == 32768 and impl == "torch_rms_norm"),
      )


def run_moe(ctx: BenchCtx, preset: str) -> None:
  """Runs SwiGLU MoE (`e2e`, precision-identical `e2e_opt`, and 6 stages)."""
  shapes = (
      [(512, 256, 128, 8, 2)]
      if preset == "smoke"
      else [(4096, 1024, 256, 32, 4), (32768, 4096, 1024, 128, 8)]
  )
  impl_list = (
      ("torchtpu_ragged_dot",)
      if ctx.device.type == "cpu"
      else ("tokamax", "torchtpu_ragged_dot", "torch_grouped_mm")
  )
  for impl in impl_list:
    if not selected(ctx, impl):
      continue
    for tokens, hidden, inter, experts, top_k in shapes:
      stag = f"t{tokens}_h{hidden}_i{inter}_e{experts}_k{top_k}"
      base = dict(
          kernel="moe",
          impl=impl,
          tokens=tokens,
          hidden=hidden,
          intermediate=inter,
          experts=experts,
          top_k=top_k,
      )
      try:
        stages, chk = other_ops.build_moe_suite(
            impl, tokens, hidden, inter, experts, top_k, ctx.dtype, ctx.device
        )
      except Exception as err:  # pylint: disable=broad-exception-caught
        emit(
            ctx,
            dict(
                base,
                case=f"moe_{impl}_e2e_{stag}",
                mode="e2e",
                scope=ctx.scope,
                error=f"{type(err).__name__}: {err}"[:1500],
            ),
        )
        continue
      for stage_name, fn, flops in stages:
        cname = f"moe_{impl}_{stage_name}_{stag}"
        time_callable(
            ctx,
            dict(base, case=cname, mode=stage_name, **chk),
            fn,
            flops_per_device=flops,
            key_trace=(
                tokens == 32768 and stage_name in ("e2e", "e2e_opt", "reduce")
            ),
        )


_RUNNERS: dict[str, Callable[[BenchCtx, str], None]] = {
    "temu_table": run_temu_table,
    "attention": run_attention,
    "paged": run_paged,
    "rmsnorm": run_rmsnorm,
    "moe": run_moe,
}


def print_summary(results: Sequence[dict[str, Any]]) -> None:
  """Prints a compact table of median latencies and achieved TFLOPS."""
  print("\n=== SUMMARY (primary_ms_p50 = TPU Event p50 or synced wall p50) ===")
  hdr = ("case", "scope", "p50 ms", "wall ms", "TFLOPS/dev", "TFLOPS/scope")
  print("{:<60} {:<5} {:>9} {:>9} {:>10} {:>12}".format(*hdr))
  for r in results:
    if "error" in r:
      print(
          f"{r.get('case', '?'):<60} {r.get('scope', ''):<5}"
          f" ERROR {r['error'][:80]}"
      )
      continue
    tf_d = r.get("tflops_per_device", "")
    tf_s = r.get("tflops_total_scope", "")
    print(
        f"{r['case']:<60} {r['scope']:<5} {r['primary_ms_p50']:>9.4f}"
        f" {r['wall_ms_p50']:>9.4f} {str(tf_d):>10} {str(tf_s):>12}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses CLI flags for `torchtpu_kernel_bench.py`."""
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument(
      "--kernel",
      default="temu_table,attention,paged,rmsnorm,moe",
      help="Comma-separated suites: temu_table,attention,paged,rmsnorm,moe.",
  )
  parser.add_argument(
      "--preset",
      default="temu",
      choices=("smoke", "temu", "sweep"),
      help="smoke: fast CI check; temu: full customer shapes; sweep: grids.",
  )
  parser.add_argument("--out", default="/tmp/torchtpu_kbench")
  parser.add_argument(
      "--device", default="auto", choices=("auto", "tpu", "cpu")
  )
  parser.add_argument(
      "--scope", default="core", choices=("core", "chip", "host")
  )
  parser.add_argument("--iters", type=int, default=10)
  parser.add_argument("--warmup", type=int, default=2)
  parser.add_argument(
      "--trace_mode",
      default="key",
      choices=("none", "key", "all"),
      help="XProf tracing policy: none, key representative cases, or all.",
  )
  parser.add_argument("--trace_iters", type=int, default=2)
  parser.add_argument(
      "--dtype", default="bfloat16", choices=("bfloat16", "float32")
  )
  parser.add_argument("--impls", default="", help="Optional comma filter.")
  parser.add_argument(
      "--verify_precision",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Verify numerical cosine similarity vs fp32 reference.",
  )
  args = parser.parse_args(argv)
  for k in args.kernel.split(","):
    if k not in _RUNNERS:
      parser.error(f"Unknown --kernel '{k}'. Choose from {tuple(_RUNNERS)}.")
  return args


def collect_env_info(
    dev: torch.device, scope: str, world_size: int, args: argparse.Namespace
) -> dict[str, Any]:
  """Collects package versions and TPU hardware properties."""
  info: dict[str, Any] = {
      "torch": torch.__version__,
      "device": str(dev),
      "scope": scope,
      "world_size": world_size,
      "torch_tpu_topology": os.environ.get("TORCH_TPU_TOPOLOGY", ""),
      "tpu_visible_devices": os.environ.get("TPU_VISIBLE_DEVICES", ""),
      "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
      "argv": vars(args),
  }
  for pkg in ("torch_tpu", "libtpu", "jax", "tokamax", "vllm_torchtpu_core"):
    try:
      import importlib.metadata  # pylint: disable=g-import-not-at-top

      info[pkg] = importlib.metadata.version(pkg)
    except Exception:  # pylint: disable=broad-exception-caught
      info[pkg] = "not_installed"
  tpu_mod = getattr(torch, "tpu", None)
  if dev.type == "tpu" and tpu_mod is not None and hasattr(
      tpu_mod, "get_device_properties"
  ):
    try:
      props = tpu_mod.get_device_properties(0)
      info["tpu_props"] = {
          "name": getattr(props, "name", ""),
          "version": str(getattr(props, "version", "")),
          "devices_per_chip": getattr(props, "devices_per_chip", None),
          "bf16_flops_per_device": getattr(
              props, "bf16_flops_per_device", None
          ),
          "bf16_flops_per_chip": getattr(props, "bf16_flops_per_chip", None),
          "hbm_bytes_per_device": getattr(props, "hbm_bytes_per_device", None),
      }
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  return info


def main(argv: Sequence[str] | None = None) -> None:
  """Entrypoint for `torchtpu_kernel_bench.py`."""
  args = parse_args(argv)
  dev, rank, world_size = init_runtime(args.device, args.scope)
  if rank == 0:
    os.makedirs(args.out, exist_ok=True)
  barrier_if_dist(world_size)

  dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
  ctx = BenchCtx(
      out_dir=args.out,
      device=dev,
      dtype=dtype,
      scope=args.scope,
      rank=rank,
      world_size=world_size,
      iters=args.iters,
      warmup=args.warmup,
      trace_mode=args.trace_mode,
      trace_iters=args.trace_iters,
      verify_precision=args.verify_precision,
      impls=set(args.impls.split(",")) if args.impls else None,
  )
  if rank == 0:
    env_info = collect_env_info(dev, args.scope, world_size, args)
    print("ENV_JSON " + json.dumps(env_info, default=str), flush=True)
    with open(os.path.join(args.out, "env.json"), "w") as f:
      json.dump(env_info, f, default=str, indent=2)

  for kernel_name in args.kernel.split(","):
    if rank == 0:
      print(
          f"\n=== kernel: {kernel_name} (preset={args.preset},"
          f" scope={args.scope}, world={world_size}) ===",
          flush=True,
      )
    _RUNNERS[kernel_name](ctx, args.preset)

  if rank == 0:
    print_summary(ctx.results)
  if world_size > 1 and dist.is_initialized():
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
