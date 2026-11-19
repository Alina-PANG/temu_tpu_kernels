#!/usr/bin/env python3
"""TPU kernel microbenchmarks for the Temu kernel porting request.

Benchmarks four kernel families on Cloud TPU (tpu7x) with one harness:

  attention  Splash attention (JAX Pallas and Tokamax) and XLA attention.
  paged      Ragged paged attention v3 from vLLM tpu-inference.
  rmsnorm    RMSNorm forward and backward, compiled by XLA.
  moe        MoE feed-forward layer built on grouped matmul (ragged_dot).

Every case is timed with wall-clock time around block_until_ready and is
optionally traced with jax.profiler, so that device-side op times can be
read from the trace. A failing case is recorded and never stops the sweep.

Usage:
  python3 kernel_bench.py --kernel attention,rmsnorm,moe --preset temu \
      --out /results/main
"""

import argparse
import dataclasses
import glob
import gzip
import json
import os
import statistics
import time
import traceback
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

P = jax.sharding.PartitionSpec
_RESULT_PREFIX = "RESULT_JSON "
_SPLASH = "jax.experimental.pallas.ops.tpu.splash_attention"
_TOKAMAX_SPLASH = (
    "tokamax._src.ops.experimental.tpu.splash_attention",
    "tokamax._src.ops.splash_attention",
)
_RPA = "tpu_inference.kernels.ragged_paged_attention.v3"


@dataclasses.dataclass
class Ctx:
  """Run-wide settings and the collected results."""

  out_dir: str
  iters: int
  warmup: int
  trace_iters: int
  dtype: Any
  impls: set[str] | None
  scopes: list[str] | None
  results: list[dict[str, Any]] = dataclasses.field(default_factory=list)


def _import(name: str) -> Any:
  """Imports a module by dotted name."""
  return __import__(name, fromlist=["_"])


def rand(shape: Sequence[int], dtype: Any, seed: int) -> jax.Array:
  """Returns a normal random array on the default device."""
  return jax.random.normal(jax.random.key(seed), tuple(shape), dtype)


def pick_devices(scope: str) -> list[Any]:
  """Returns the devices of a scope: core (1), chip (2) or host (all)."""
  devices = jax.devices()
  if scope == "core":
    return devices[:1]
  if scope == "host":
    return devices
  coords = getattr(devices[0], "coords", None)
  chip = [d for d in devices if getattr(d, "coords", None) == coords]
  if coords is None or len(chip) < 2:
    chip = devices[:2]
  return chip[:2]


def place(arrays: Sequence[Any], devices: Sequence[Any]):
  """Puts one copy of every array on each device.

  Args:
    arrays: Host or device arrays without a device axis.
    devices: Target devices.

  Returns:
    (mesh, placed). For one device the mesh is None. Otherwise every placed
    array has a leading device axis that is sharded over the mesh.
  """
  if len(devices) == 1:
    return None, [jax.device_put(a, devices[0]) for a in arrays]
  mesh = jax.sharding.Mesh(np.array(devices), ("d",))
  sharding = jax.sharding.NamedSharding(mesh, P("d"))
  placed = []
  for a in arrays:
    host = np.asarray(a)
    stacked = np.broadcast_to(host, (len(devices),) + host.shape)
    placed.append(jax.device_put(stacked, sharding))
  return mesh, placed


def replicate(fn: Callable[..., Any], mesh: Any, args: Sequence[Any]):
  """Wraps fn so that every device of the mesh runs an identical copy."""
  shapes = [jax.ShapeDtypeStruct(a.shape[1:], a.dtype) for a in args]
  out_specs = jax.tree.map(lambda _: P("d"), jax.eval_shape(fn, *shapes))
  in_specs = tuple(P("d") for _ in args)

  def body(*shards):
    out = fn(*[s[0] for s in shards])
    return jax.tree.map(lambda x: x[None], out)

  shard_map = getattr(jax, "shard_map", None)
  if shard_map is None:
    shard_map = _import("jax.experimental.shard_map").shard_map
  for extra in ({"check_vma": False}, {"check_rep": False}, {}):
    try:
      return shard_map(
          body, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **extra
      )
    except TypeError:
      continue
  raise RuntimeError("shard_map is not usable in this JAX version.")


def summarize_trace(trace_dir: str, top: int = 15) -> dict[str, Any]:
  """Sums device time per op from the newest trace.json.gz, if there is one."""
  pattern = os.path.join(trace_dir, "plugins", "profile", "*", "*")
  files = sorted(glob.glob(pattern), key=os.path.getmtime)
  summary = {"files": [os.path.relpath(f, trace_dir) for f in files]}
  traces = [f for f in files if f.endswith(".trace.json.gz")]
  if not traces:
    summary["note"] = "No trace.json.gz; read the xplane.pb offline."
    return summary
  with gzip.open(traces[-1], "rb") as f:
    events = json.load(f).get("traceEvents", [])
  totals = {}
  for event in events:
    event_args = event.get("args")
    if not isinstance(event_args, dict) or "name" not in event:
      continue
    duration_ps = event_args.get("device_duration_ps")
    if duration_ps is None:
      continue
    key = (event.get("pid", -1), event["name"])
    count, total = totals.get(key, (0, 0.0))
    totals[key] = (count + 1, total + float(duration_ps) / 1e9)
  if not totals:
    summary["note"] = "No device_duration_ps events in the trace."
    return summary
  first_pid = min(pid for pid, _ in totals)
  rows = [
      dict(op=name, count=c, total_ms=round(t, 4), avg_ms=round(t / c, 4))
      for (pid, name), (c, t) in totals.items()
      if pid == first_pid
  ]
  rows.sort(key=lambda row: -row["total_ms"])
  summary.update(pid=first_pid, ops=rows[:top])
  return summary


def emit(ctx: Ctx, record: dict[str, Any]) -> None:
  """Stores a result and prints it as one greppable JSON line."""
  ctx.results.append(record)
  line = json.dumps(record, sort_keys=True, default=str)
  print(_RESULT_PREFIX + line, flush=True)
  with open(os.path.join(ctx.out_dir, "results.jsonl"), "a") as f:
    f.write(line + "\n")


def time_case(
    ctx: Ctx,
    record: dict[str, Any],
    fn: Callable[..., Any],
    arrays: Sequence[Any],
    scope: str = "core",
    donate: Sequence[int] = (),
    feedback: Callable[[list[Any], Any], list[Any]] | None = None,
    flops: float | None = None,
) -> None:
  """Compiles, warms up, times and traces one case, then emits its record.

  Args:
    ctx: Run-wide settings.
    record: Case description. Timing fields are added to it.
    fn: Single-device function under test.
    arrays: Inputs of fn, without a device axis.
    scope: Devices that run a copy of fn at once: core, chip or host.
    donate: Indices of the inputs that fn may overwrite.
    feedback: Maps (inputs, output) to the inputs of the next call. Needed
      when an input is donated.
    flops: Floating point operations of one call on one device.
  """
  record = dict(record, scope=scope)
  try:
    devices = pick_devices(scope)
    record["num_devices"] = len(devices)
    mesh, args = place(arrays, devices)
    step = fn if mesh is None else replicate(fn, mesh, args)
    step = jax.jit(step, donate_argnums=tuple(donate))

    def run(call_args):
      out = step(*call_args)
      jax.block_until_ready(out)
      return call_args if feedback is None else feedback(call_args, out)

    start = time.perf_counter()
    args = run(list(args))
    record["compile_s"] = round(time.perf_counter() - start, 3)
    for _ in range(ctx.warmup):
      args = run(args)
    times_ms = []
    for _ in range(ctx.iters):
      start = time.perf_counter()
      args = run(args)
      times_ms.append((time.perf_counter() - start) * 1e3)
    times_ms.sort()
    record["wall_ms_p50"] = round(statistics.median(times_ms), 4)
    record["wall_ms_min"] = round(times_ms[0], 4)
    record["wall_ms_max"] = round(times_ms[-1], 4)
    if flops:
      record["flops_per_device"] = flops
      seconds = record["wall_ms_p50"] / 1e3
      record["wall_tflops_per_device"] = round(flops / seconds / 1e12, 2)
    if ctx.trace_iters:
      trace_dir = os.path.join(ctx.out_dir, "xprof", record["case"], scope)
      try:
        with jax.profiler.trace(trace_dir):
          for _ in range(ctx.trace_iters):
            args = run(args)
        record["trace_dir"] = trace_dir
        record["device_ops"] = summarize_trace(trace_dir)
      except Exception as e:  # pylint: disable=broad-exception-caught
        record["trace_error"] = f"{type(e).__name__}: {e}"[:500]
  except Exception as e:  # pylint: disable=broad-exception-caught
    record["error"] = f"{type(e).__name__}: {e}"[:2000]
    record["traceback"] = traceback.format_exc()[-3000:]
  emit(ctx, record)


def selected(ctx: Ctx, impl: str) -> bool:
  """Returns whether the --impls filter keeps this implementation."""
  return ctx.impls is None or impl in ctx.impls


# ----------------------------- attention -------------------------------------


def splash_libs(impl: str) -> tuple[Any, Any]:
  """Returns the (kernel, mask) modules of a Splash implementation."""
  bases = (_SPLASH,) if impl == "jax_splash" else _TOKAMAX_SPLASH
  error = None
  for base in bases:
    try:
      kernel = _import(base + ".splash_attention_kernel")
      return kernel, _import(base + ".splash_attention_mask")
    except ImportError as e:
      error = e
  raise ImportError(f"No Splash attention module for {impl}: {error}")


def build_attention(
    impl: str,
    mode: str,
    seq: int,
    heads: int,
    blocks: Sequence[int],
    bwd_kernel: str = "fused",
) -> Callable[..., Any]:
  """Returns causal attention on [batch, heads, seq, head_dim] inputs.

  Args:
    impl: jax_splash, tokamax_splash or xla_dpa.
    mode: fwd, or bwd for the gradient with respect to q, k and v (this
      also runs the forward pass).
    seq: Sequence length of q, k and v.
    heads: Number of heads.
    blocks: Splash block sizes (block_q, block_kv, block_kv_compute).
    bwd_kernel: fused, or sep for separate dq and dkv kernels.
  """
  if impl == "xla_dpa":

    def fwd(q, k, v):
      q, k, v = (jnp.swapaxes(x, 1, 2) for x in (q, k, v))
      out = jax.nn.dot_product_attention(q, k, v, is_causal=True)
      return jnp.swapaxes(out, 1, 2)

  else:
    kernel_lib, mask_lib = splash_libs(impl)
    block_q, block_kv, block_kvc = (min(b, seq) for b in blocks)
    sizes = dict(block_q=block_q, block_kv=block_kv, block_kv_compute=block_kvc)
    if mode == "bwd":
      sizes.update(
          block_q_dkv=block_q,
          block_kv_dkv=block_kv,
          block_kv_dkv_compute=block_kvc,
      )
      if bwd_kernel == "fused":
        sizes["use_fused_bwd_kernel"] = True
      else:
        sizes.update(block_q_dq=block_q, block_kv_dq=block_kv)
    causal = mask_lib.CausalMask(shape=(seq, seq))
    if impl == "jax_splash":
      mask = mask_lib.MultiHeadMask(masks=[causal] * heads)
      kernel = kernel_lib.make_splash_mha_single_device(
          mask, block_sizes=kernel_lib.BlockSizes(**sizes)
      )
    else:
      names = {f.name for f in dataclasses.fields(kernel_lib.SplashConfig)}
      sizes["use_base2_exp"] = True
      config = kernel_lib.SplashConfig(
          **{k: v for k, v in sizes.items() if k in names}
      )
      kernel = kernel_lib.make_splash_mha_single_device(causal, config=config)
    fwd = jax.vmap(kernel, in_axes=(0, 0, 0))

  if mode == "fwd":
    return fwd
  return jax.grad(
      lambda q, k, v: fwd(q, k, v).astype(jnp.float32).mean(),
      argnums=(0, 1, 2),
  )


def attention_flops(batch: int, seq: int, heads: int, dim: int, mode: str):
  """FLOPs of causal attention: Q @ K^T and P @ V over attended pairs."""
  fwd = 4.0 * batch * heads * dim * (seq * (seq + 1) // 2)
  # The backward costs 2.5 times the forward; bwd mode runs both passes.
  return fwd if mode == "fwd" else 3.5 * fwd


_CUSTOMER_BLOCKS = (512, 2048, 1024)
_DEFAULT_BLOCKS = (512, 1024, 512)
_BLOCK_GRID = (
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


def attention_cases(preset: str) -> list[dict[str, Any]]:
  """Returns the attention cases of a preset.

  Shapes are (batch, seq, heads, head_dim). (64, 2048, 8, 64) is the target
  shape sent on 2026-11-13, (64, 2048, 4, 128) has the same FLOPs with full
  128-wide heads, (64, 2048, 2, 256) has the same FLOPs with 256-wide heads,
  and the batch 128 shapes are the ones in the customer's own table.
  """
  if preset == "smoke":
    tiny = (256, 512, 256)
    return [
        dict(shape=(2, 512, 2, dim), impl=impl, mode=mode, blocks=tiny)
        for dim in (64, 128)
        for impl in ("jax_splash", "tokamax_splash", "xla_dpa")
        for mode in ("fwd", "bwd")
    ]
  target, wide = (64, 2048, 8, 64), (64, 2048, 4, 128)
  cases = []
  for shape in (target, wide, (128, 2048, 4, 128), (128, 1024, 4, 128)):
    modes = ("fwd",) if shape == (128, 1024, 4, 128) else ("fwd", "bwd")
    for mode in modes:
      for impl in ("jax_splash", "tokamax_splash"):
        for blocks in (_CUSTOMER_BLOCKS, _DEFAULT_BLOCKS):
          cases.append(dict(shape=shape, impl=impl, mode=mode, blocks=blocks))
      cases.append(dict(shape=shape, impl="xla_dpa", mode=mode, blocks=None))
  if preset == "temu":
    # Same logits per head with 256-wide heads. Together with the target and
    # wide shapes this separates cost per FLOP from cost per logit.
    for impl in ("jax_splash", "tokamax_splash"):
      cases.append(
          dict(
              shape=(64, 2048, 2, 256),
              impl=impl,
              mode="fwd",
              blocks=_CUSTOMER_BLOCKS,
          )
      )
    # One chip (2 cores) and the whole host (8 cores) on the target shapes.
    for shape in (target, wide):
      cases.append(
          dict(
              shape=shape,
              impl="jax_splash",
              mode="fwd",
              blocks=_CUSTOMER_BLOCKS,
              scopes=("chip", "host"),
          )
      )
  if preset == "sweep":
    cases = [
        dict(shape=shape, impl=impl, mode="fwd", blocks=blocks)
        for shape in (target, wide)
        for impl in ("jax_splash", "tokamax_splash")
        for blocks in _BLOCK_GRID
    ]
  return cases


def run_attention(ctx: Ctx, preset: str) -> None:
  """Runs the attention cases."""
  for case in attention_cases(preset):
    impl, mode, blocks = case["impl"], case["mode"], case["blocks"]
    if not selected(ctx, impl):
      continue
    batch, seq, heads, dim = case["shape"]
    tag = "x".join(map(str, blocks)) if blocks else "na"
    name = f"attn_{impl}_{mode}_b{batch}_s{seq}_h{heads}_d{dim}_blk{tag}"
    record = dict(
        kernel="attention",
        case=name,
        impl=impl,
        mode=mode,
        batch=batch,
        seq=seq,
        heads=heads,
        head_dim=dim,
        blocks=blocks,
    )
    shape = (batch, heads, seq, dim)
    for scope in ctx.scopes or case.get("scopes", ("core",)):
      try:
        fn = build_attention(impl, mode, seq, heads, blocks or ())
        arrays = [rand(shape, ctx.dtype, seed) for seed in range(3)]
      except Exception as e:  # pylint: disable=broad-exception-caught
        error = f"{type(e).__name__}: {e}"[:2000]
        emit(ctx, dict(record, scope=scope, error=error))
        continue
      flops = attention_flops(batch, seq, heads, dim, mode)
      time_case(ctx, record, fn, arrays, scope, flops=flops)


# --------------------------- paged attention ---------------------------------


def run_paged(ctx: Ctx, preset: str) -> None:
  """Runs ragged paged attention v3: unchunked prefill and decode.

  Shapes are (batch, context_len, q_heads, kv_heads, head_dim). Prefill
  processes context_len new tokens per sequence with causal attention.
  Decode adds one token per sequence on top of context_len cached tokens.
  New K and V are written to the paged KV cache, which is donated.
  """
  if preset == "smoke":
    shapes, page_size = [(4, 512, 2, 2, 64), (4, 512, 2, 2, 128)], 256
  else:
    shapes, page_size = [(64, 2048, 8, 8, 64), (64, 2048, 4, 4, 128)], 256
  for batch, ctx_len, q_heads, kv_heads, dim in shapes:
    for phase in ("prefill", "decode"):
      name = (
          f"rpa3_{phase}_b{batch}_ctx{ctx_len}_hq{q_heads}_hkv{kv_heads}"
          f"_d{dim}_p{page_size}"
      )
      record = dict(
          kernel="paged",
          case=name,
          impl="rpa_v3_hd64" if dim == 64 else "rpa_v3",
          mode=phase,
          batch=batch,
          seq=ctx_len,
          heads=q_heads,
          kv_heads=kv_heads,
          head_dim=dim,
          page_size=page_size,
      )
      if not selected(ctx, "rpa_v3"):
        continue
      try:
        if dim == 64:
          lib = _import(_RPA + ".kernel_hd64")
          kernel = lib.ragged_paged_attention_hd64
        else:
          lib = _import(_RPA + ".kernel")
          kernel = lib.ragged_paged_attention
        decode = phase == "decode"
        pages_per_seq = -(-ctx_len // page_size)
        num_pages = batch * pages_per_seq
        cache_shape = lib.get_kv_cache_shape(
            num_pages, page_size, kv_heads, dim, ctx.dtype
        )
        tokens = batch if decode else batch * ctx_len
        q_len = 1 if decode else ctx_len
        # (decode end, chunked prefill end, mixed end) sequence indices.
        split = [batch] * 3 if decode else [0, 0, batch]
        arrays = [
            rand((tokens, q_heads, dim), ctx.dtype, 0),
            rand((tokens, kv_heads, dim), ctx.dtype, 1),
            rand((tokens, kv_heads, dim), ctx.dtype, 2),
            rand(cache_shape, ctx.dtype, 3),
            np.full((batch,), ctx_len, np.int32),
            np.arange(num_pages, dtype=np.int32),
            (np.arange(batch + 1) * q_len).astype(np.int32),
            np.array(split, np.int32),
        ]
        pairs = batch * (ctx_len if decode else ctx_len * (ctx_len + 1) // 2)
        flops = 4.0 * pairs * q_heads * dim
      except Exception as e:  # pylint: disable=broad-exception-caught
        emit(ctx, dict(record, error=f"{type(e).__name__}: {e}"[:2000]))
        continue

      def fn(q, k, v, cache, kv_lens, pages, cu_q_lens, dist, kernel=kernel):
        return kernel(q, k, v, cache, kv_lens, pages, cu_q_lens, dist)

      def feedback(call_args, out):
        return call_args[:3] + [out[1]] + call_args[4:]

      for scope in ctx.scopes or ("core",):
        time_case(
            ctx, record, fn, arrays, scope, (3,), feedback, flops=flops
        )


# ------------------------------- rmsnorm -------------------------------------


def rmsnorm(x: jax.Array, weight: jax.Array, accumulate_f32: bool):
  """Y = X / rms(X) * weight over the last axis."""
  xf = x.astype(jnp.float32) if accumulate_f32 else x
  mean_square = jnp.mean(jnp.square(xf), axis=-1, keepdims=True)
  scale = jax.lax.rsqrt(mean_square + 1e-6)
  return (xf * scale).astype(x.dtype) * weight


def run_rmsnorm(ctx: Ctx, preset: str) -> None:
  """Runs RMSNorm on [rows, dim] activations, forward and backward.

  (131072, 512) is the token count and hidden size of the target attention
  shape (64 * 2048 tokens, 8 * 64 wide). (32768, 4096) matches the SonicMoE
  benchmark input.
  """
  shapes = [(1024, 512)] if preset == "smoke" else [
      (131072, 512),
      (32768, 4096),
      (8192, 8192),
  ]
  for rows, dim in shapes:
    for impl, accumulate_f32 in (("xla_bf16", False), ("xla_f32acc", True)):
      if not selected(ctx, impl):
        continue
      for mode in ("fwd", "bwd"):
        name = f"rmsnorm_{impl}_{mode}_m{rows}_n{dim}"
        record = dict(
            kernel="rmsnorm",
            case=name,
            impl=impl,
            mode=mode,
            rows=rows,
            dim=dim,
        )
        x = rand((rows, dim), ctx.dtype, 0)
        weight = jnp.ones((dim,), ctx.dtype)

        def fwd(x, weight, accumulate_f32=accumulate_f32):
          return rmsnorm(x, weight, accumulate_f32)

        def bwd(x, weight, dy, fwd=fwd):
          return jax.vjp(fwd, x, weight)[1](dy)

        arrays = [x, weight]
        if mode == "bwd":
          arrays.append(rand((rows, dim), ctx.dtype, 1))
        for scope in ctx.scopes or ("core",):
          time_case(ctx, record, fwd if mode == "fwd" else bwd, arrays, scope)


# --------------------------------- moe ---------------------------------------


def ragged_dot_fn(impl: str) -> Callable[..., jax.Array]:
  """Returns grouped matmul: [M, K] x [G, K, N] with group sizes -> [M, N]."""
  if impl == "xla":
    return jax.lax.ragged_dot
  tokamax = _import("tokamax")
  return lambda lhs, rhs, sizes: tokamax.ragged_dot(lhs, rhs, sizes)


def run_moe(ctx: Ctx, preset: str) -> None:
  """Runs a SwiGLU MoE feed-forward layer, end to end and stage by stage.

  Shapes are (tokens, hidden, intermediate, experts, top_k).
  (32768, 4096, 1024, 128, 8) is the SonicMoE README benchmark
  (`moe-cute.py --thiek 32768,4096,1024,128,8`). The stages are: route
  (router matmul, top-k, softmax, sort by expert), gather (rows in expert
  order), gmm_up (grouped matmul to 2 * intermediate), act (SwiGLU),
  gmm_down (grouped matmul back to hidden) and reduce (weight, unsort, sum
  over top-k).
  """
  tiny, small = (2048, 256, 128, 8, 2), (4096, 1024, 256, 32, 4)
  full = (32768, 4096, 1024, 128, 8)
  if preset == "smoke":
    runs = [("tokamax", tiny), ("xla", tiny)]
  else:
    # XLA ragged_dot runs last: it may be slow or run out of memory.
    runs = [(impl, s) for impl in ("tokamax", "xla") for s in (small, full)]
  for impl, (tokens, hidden, inter, experts, top_k) in runs:
    if not selected(ctx, impl):
      continue
    shape_tag = f"t{tokens}_h{hidden}_i{inter}_e{experts}_k{top_k}"
    base = dict(
        kernel="moe",
        impl=impl,
        tokens=tokens,
        hidden=hidden,
        intermediate=inter,
        experts=experts,
        top_k=top_k,
    )
    pairs = tokens * top_k
    try:
      rdot = ragged_dot_fn(impl)
      x = rand((tokens, hidden), ctx.dtype, 0)
      w_router = rand((hidden, experts), ctx.dtype, 1)
      w_up = rand((experts, hidden, 2 * inter), ctx.dtype, 2)
      w_down = rand((experts, inter, hidden), ctx.dtype, 3)
    except Exception as e:  # pylint: disable=broad-exception-caught
      error = f"{type(e).__name__}: {e}"[:2000]
      emit(ctx, dict(base, case=f"moe_{impl}_e2e_{shape_tag}", error=error))
      continue

    def route(x, w_router, experts=experts, top_k=top_k, tokens=tokens):
      logits = (x @ w_router).astype(jnp.float32)
      top_logits, top_experts = jax.lax.top_k(logits, top_k)
      probs = jax.nn.softmax(top_logits, axis=-1).reshape(-1)
      flat = top_experts.reshape(-1)
      order = jnp.argsort(flat)
      token_ids = jnp.repeat(jnp.arange(tokens), top_k)[order]
      sizes = jnp.bincount(flat, length=experts).astype(jnp.int32)
      return token_ids, sizes, probs[order], jnp.argsort(order)

    def gather(x, token_ids):
      return x[token_ids]

    def gmm_up(rows, w_up, sizes, rdot=rdot):
      return rdot(rows, w_up, sizes)

    def act(up):
      gate, value = jnp.split(up, 2, axis=-1)
      return jax.nn.silu(gate) * value

    def gmm_down(hidden_rows, w_down, sizes, rdot=rdot):
      return rdot(hidden_rows, w_down, sizes)

    def reduce(rows, probs, unsort, tokens=tokens, top_k=top_k):
      rows = rows * probs[:, None].astype(rows.dtype)
      return rows[unsort].reshape(tokens, top_k, -1).sum(axis=1)

    def layer(x, w_router, w_up, w_down):
      token_ids, sizes, probs, unsort = route(x, w_router)
      up = gmm_up(gather(x, token_ids), w_up, sizes)
      down = gmm_down(act(up), w_down, sizes)
      return reduce(down, probs, unsort)

    gmm_flops = {
        "gmm_up": 2.0 * pairs * hidden * 2 * inter,
        "gmm_down": 2.0 * pairs * inter * hidden,
    }
    record = dict(base, case=f"moe_{impl}_e2e_{shape_tag}", mode="e2e")
    time_case(
        ctx,
        record,
        layer,
        [x, w_router, w_up, w_down],
        flops=sum(gmm_flops.values()),
    )
    try:
      token_ids, sizes, probs, unsort = jax.jit(route)(x, w_router)
      rows = jax.jit(gather)(x, token_ids)
      up = jax.jit(gmm_up)(rows, w_up, sizes)
      hidden_rows = jax.jit(act)(up)
      down = jax.jit(gmm_down)(hidden_rows, w_down, sizes)
    except Exception as e:  # pylint: disable=broad-exception-caught
      error = f"{type(e).__name__}: {e}"[:2000]
      emit(ctx, dict(base, case=f"moe_{impl}_stages_{shape_tag}", error=error))
      continue
    stages = [
        ("route", route, [x, w_router]),
        ("gather", gather, [x, token_ids]),
        ("gmm_up", gmm_up, [rows, w_up, sizes]),
        ("act", act, [up]),
        ("gmm_down", gmm_down, [hidden_rows, w_down, sizes]),
        ("reduce", reduce, [down, probs, unsort]),
    ]
    for stage, fn, arrays in stages:
      record = dict(base, case=f"moe_{impl}_{stage}_{shape_tag}", mode=stage)
      time_case(ctx, record, fn, arrays, flops=gmm_flops.get(stage))


# -------------------------------- main ---------------------------------------

_RUNNERS = {
    "attention": run_attention,
    "paged": run_paged,
    "rmsnorm": run_rmsnorm,
    "moe": run_moe,
}


def print_summary(results: Sequence[dict[str, Any]]) -> None:
  """Prints one line per case."""
  print("\n=== SUMMARY (wall-clock, includes host dispatch) ===")
  columns = ("case", "scope", "p50 ms", "min ms", "TFLOPS")
  print("{:<64} {:<5} {:>10} {:>10} {:>8}".format(*columns))
  for r in results:
    if "error" in r:
      case, scope = r.get("case", "?"), r.get("scope", "")
      print(f"{case:<64} {scope:<5} ERROR {r['error'][:90]}")
      continue
    tflops = r.get("wall_tflops_per_device", "")
    print(
        f"{r['case']:<64} {r['scope']:<5} {r['wall_ms_p50']:>10.4f}"
        f" {r['wall_ms_min']:>10.4f} {tflops:>8}"
    )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  parser.add_argument("--kernel", default="attention,paged,rmsnorm,moe")
  parser.add_argument(
      "--preset",
      default="temu",
      choices=("smoke", "temu", "sweep"),
      help="smoke: tiny shapes; temu: customer shapes; sweep: block grid.",
  )
  parser.add_argument("--out", default="/tmp/kernel_bench")
  parser.add_argument("--iters", type=int, default=20)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--trace_iters", type=int, default=5)
  parser.add_argument("--dtype", default="bfloat16")
  parser.add_argument("--impls", default="", help="Comma list; empty = all.")
  parser.add_argument(
      "--scopes", default="", help="core,chip,host; empty = per-case default."
  )
  args = parser.parse_args()
  os.makedirs(args.out, exist_ok=True)
  ctx = Ctx(
      out_dir=args.out,
      iters=args.iters,
      warmup=args.warmup,
      trace_iters=args.trace_iters,
      dtype=jnp.dtype(args.dtype),
      impls=set(args.impls.split(",")) if args.impls else None,
      scopes=args.scopes.split(",") if args.scopes else None,
  )
  devices = jax.devices()
  env = dict(
      jax=jax.__version__,
      num_devices=len(devices),
      devices=[
          dict(
              id=d.id,
              kind=getattr(d, "device_kind", "?"),
              coords=getattr(d, "coords", None),
              core=getattr(d, "core_on_chip", None),
          )
          for d in devices
      ],
      libtpu_init_args=os.environ.get("LIBTPU_INIT_ARGS", ""),
      xla_flags=os.environ.get("XLA_FLAGS", ""),
      preset=args.preset,
      argv=vars(args),
  )
  print("ENV_JSON " + json.dumps(env, default=str), flush=True)
  with open(os.path.join(args.out, "env.json"), "w") as f:
    json.dump(env, f, default=str, indent=1)
  for kernel in args.kernel.split(","):
    print(f"\n=== kernel: {kernel} (preset {args.preset}) ===", flush=True)
    _RUNNERS[kernel](ctx, args.preset)
  print_summary(ctx.results)


if __name__ == "__main__":
  main()
