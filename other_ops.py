"""PageAttention (RPAv3), RMSNorm, and Grouped-GMM MoE builders for TorchTPU.

Implements the remaining three customer kernel families on TorchTPU:
  1. `paged`: `vllm_torchtpu` Ragged Paged Attention v3 (`d=64` & `d=128`)
     imported via `torch.tpu.pallas.jax_op` with KV-cache buffer donation and
     `(num_kv_pages_per_block, num_queries_per_block)` tuning overrides.
  2. `rmsnorm`: Native `torch.nn.functional.rms_norm` plus explicit `bf16` and
     `float32`-accumulated RMSNorm with standalone `fwd` and standalone `bwd`
     (`torch.func.vjp` — no forward recomputation inside backward timing).
  3. `moe`: SwiGLU MoE (`[32768, 4096, 1024, 128, top-8]` and small shape)
     built on `torch.tpu.ops.ragged_dot`, `torch._grouped_mm`, and
     `tokamax.ragged_dot` (via `jax_op`), including the precision-identical
     weighted-before-unsort reduction optimization (`e2e_opt`).
"""

from collections.abc import Callable
import hashlib
import logging
import sys
import types
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_OP_CACHE: dict[str, Any] = {}


def _cache_id(tag: str, *parts: Any) -> str:
  """Builds a deterministic custom-op suffix.

  Args:
    tag: Prefix identifying the operator family.
    *parts: Configuration values hashed into the suffix.

  Returns:
    Valid custom-op identifier string.
  """
  digest = hashlib.md5(repr(parts).encode("utf-8")).hexdigest()[:12]
  return f"{tag}_{digest}"


def _get_jax_op() -> Callable[..., Any]:
  """Returns `torch.tpu.pallas.jax_op`."""
  tpu_mod = getattr(torch, "tpu", None)
  if tpu_mod is not None and hasattr(tpu_mod, "pallas"):
    return tpu_mod.pallas.jax_op
  from torch_tpu._internal import pallas  # pylint: disable=g-import-not-at-top

  return pallas.jax_op


def _compile_if_tpu(
    fn: Callable[..., Any], device_type: str
) -> Callable[..., Any]:
  """Compiles `fn` with the TorchTPU backend when running on TPU.

  Args:
    fn: Function to compile.
    device_type: `"tpu"` or `"cpu"`.

  Returns:
    Compiled function on TPU, or eager `fn` on CPU.
  """
  if device_type == "tpu":
    return torch.compile(fn, backend="tpu", dynamic=False)
  return fn


def _detect_tpu_version() -> int:
  """Reads the numeric TPU generation from TorchTPU without `jax.devices()`.

  Returns:
    Integer TPU version (e.g., `7` on TPU v7x), or `7` if unavailable.
  """
  if not torch.accelerator.is_available():
    return 7
  tpu_mod = getattr(torch, "tpu", None)
  if tpu_mod is not None and hasattr(tpu_mod, "get_device_properties"):
    try:
      ver = str(tpu_mod.get_device_properties(0).version)
      for digit in ("8", "7", "6", "5", "4"):
        if digit in ver:
          return int(digit)
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  return 7


_SEEN_LOG_MSGS: set[str] = set()


def _logger_warn_once(
    self: logging.Logger, msg: str, *args: Any, **kwargs: Any
) -> None:
  """Logs a warning message at most once per process."""
  if msg not in _SEEN_LOG_MSGS:
    _SEEN_LOG_MSGS.add(msg)
    self.warning(msg, *args, **kwargs)


def _logger_info_once(
    self: logging.Logger, msg: str, *args: Any, **kwargs: Any
) -> None:
  """Logs an info message at most once per process."""
  if msg not in _SEEN_LOG_MSGS:
    _SEEN_LOG_MSGS.add(msg)
    self.info(msg, *args, **kwargs)


def _tpu_abstract_mesh(jax_mod: Any) -> Any:
  """Builds a 0-axis TPU AbstractMesh so `pltpu.get_tpu_info()` resolves on CPU.

  Args:
    jax_mod: Imported `jax` module.

  Returns:
    `jax.sharding.AbstractMesh` carrying `AbstractDevice` metadata.
  """
  ver = _detect_tpu_version()
  kind = {7: "TPU7x", 6: "TPU v6 lite", 5: "TPU v5p", 4: "TPU v4"}.get(
      ver, "TPU7x"
  )
  abs_dev = jax_mod.sharding.AbstractDevice(
      device_kind=kind, num_cores=1, platform="tpu"
  )
  return jax_mod.sharding.AbstractMesh((), (), abstract_device=abs_dev)


def _install_vllm_logger_and_tpu_shims() -> None:
  """Shims `vllm.logger`/`vllm_torchtpu.utils` and patches `get_tpu_version`.

  `vllm-torchtpu-core` imports `vllm.logger` and `vllm_torchtpu.utils` (which
  imports `vllm`), and its RPAv3 `get_tpu_version()` calls `jax.devices()`,
  which would otherwise contend with TorchTPU's C++ PJRT client or return `-1`
  when `JAX_PLATFORMS=cpu` is set.
  """
  if not hasattr(logging.Logger, "warning_once"):
    setattr(logging.Logger, "warning_once", _logger_warn_once)
  if not hasattr(logging.Logger, "info_once"):
    setattr(logging.Logger, "info_once", _logger_info_once)
  tpu_ver = _detect_tpu_version()
  if "vllm.logger" not in sys.modules:
    vllm_pkg = sys.modules.setdefault("vllm", types.ModuleType("vllm"))
    log_mod = types.ModuleType("vllm.logger")
    log_mod.init_logger = logging.getLogger  # type: ignore[attr-defined]
    vllm_pkg.logger = log_mod  # type: ignore[attr-defined]
    sys.modules["vllm.logger"] = log_mod

  if "vllm_torchtpu.utils" not in sys.modules:
    # Ensure top-level `vllm_torchtpu` package is loaded first without `vllm`.
    import importlib.util  # pylint: disable=g-import-not-at-top

    if "vllm_torchtpu" not in sys.modules:
      spec = importlib.util.find_spec("vllm_torchtpu")
      if spec is not None and spec.submodule_search_locations:
        vt_pkg = types.ModuleType("vllm_torchtpu")
        vt_pkg.__path__ = list(spec.submodule_search_locations)
        sys.modules["vllm_torchtpu"] = vt_pkg
    utils_mod = types.ModuleType("vllm_torchtpu.utils")
    utils_mod.get_device_name = (  # type: ignore[attr-defined]
        lambda num_devices=None, v=tpu_ver: f"TPU v{v}"
    )
    sys.modules["vllm_torchtpu.utils"] = utils_mod
  for mod_name in (
      "vllm_torchtpu.kernels.ragged_paged_attention.v3.util",
      "vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel",
      "vllm_torchtpu.kernels.ragged_paged_attention.v3.tuned_block_sizes_hd64",
  ):
    try:
      mod = __import__(mod_name, fromlist=["_"])
      if hasattr(mod, "get_tpu_version"):
        mod.get_tpu_version = lambda v=tpu_ver: v
    except ImportError:
      continue


# ---------------------------- PageAttention (RPAv3) --------------------------


def build_paged_attention(
    batch: int,
    ctx_len: int,
    q_heads: int,
    kv_heads: int,
    dim: int,
    page_size: int,
    phase: str,
    dtype: torch.dtype,
    device: torch.device,
    bkvp: int | None = None,
    bq: int | None = None,
) -> tuple[Callable[[], torch.Tensor], dict[str, Any]]:
  """Builds a ready-to-time PageAttention RPAv3 callable on TorchTPU.

  Donates the paged KV-cache buffer (`donate_argnums=(3,)`) and repoints the
  PyTorch storage in-place via `cache.set_(updated_cache)` after each call, as
  required by `torch.tpu.pallas.jax_op` buffer donation semantics.

  Args:
    batch: Number of sequences.
    ctx_len: Total context length per sequence.
    q_heads: Number of query heads.
    kv_heads: Number of KV heads.
    dim: Head dimension (`64` or `128`).
    page_size: KV-cache page size in tokens (e.g., `256`).
    phase: `"prefill"` (unchunked causal prefill) or `"decode"` (`q_len=1`).
    dtype: Activation and KV-cache dtype (`torch.bfloat16`).
    device: Target PyTorch device.
    bkvp: Optional explicit `num_kv_pages_per_block` override.
    bq: Optional explicit `num_queries_per_block` override.

  Returns:
    Tuple `(step_fn, meta)` where `step_fn()` executes one RPAv3 call and `meta`
    records FLOPs and KV-cache shape.
  """
  is_decode = phase == "decode"
  pages_per_seq = (ctx_len + page_size - 1) // page_size
  num_pages = batch * pages_per_seq
  tokens = batch if is_decode else batch * ctx_len
  q_len = 1 if is_decode else ctx_len
  pairs = batch * (ctx_len if is_decode else ctx_len * (ctx_len + 1) // 2)
  flops = 4.0 * pairs * q_heads * dim

  if device.type == "cpu":
    q = torch.randn(batch, q_heads, q_len, dim, dtype=dtype, device=device)
    k = torch.randn(batch, kv_heads, ctx_len, dim, dtype=dtype, device=device)
    v = torch.randn(batch, kv_heads, ctx_len, dim, dtype=dtype, device=device)

    def cpu_step() -> torch.Tensor:
      return F.scaled_dot_product_attention(
          q, k, v, is_causal=(not is_decode), enable_gqa=(q_heads != kv_heads)
      )

    return cpu_step, {"flops": flops, "cache_shape": (num_pages, page_size)}

  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

  _install_vllm_logger_and_tpu_shims()
  mod_suffix = "kernel_hd64" if dim == 64 else "kernel"
  rpa_mod = None
  for pkg in (
      "vllm_torchtpu.kernels.ragged_paged_attention.v3",
      "tpu_inference.kernels.ragged_paged_attention.v3",
  ):
    try:
      rpa_mod = __import__(f"{pkg}.{mod_suffix}", fromlist=["_"])
      break
    except ImportError:
      continue
  if rpa_mod is None:
    raise ImportError("No RPAv3 kernel module found in vllm_torchtpu.")

  kern_fn = (
      rpa_mod.ragged_paged_attention_hd64
      if dim == 64
      else rpa_mod.ragged_paged_attention
  )
  j_dtype = jnp.bfloat16 if dtype == torch.bfloat16 else jnp.float32
  cache_shape = tuple(
      int(x)
      for x in rpa_mod.get_kv_cache_shape(
          num_pages, page_size, kv_heads, dim, j_dtype
      )
  )

  key = _cache_id("rpa3", dim, bkvp, bq)
  if key not in _OP_CACHE:
    jax_op = _get_jax_op()
    abs_mesh = _tpu_abstract_mesh(jax)

    def _rpa_jax(
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        cache: jax.Array,
        kv_lens: jax.Array,
        pages: jax.Array,
        cu_q_lens: jax.Array,
        dist: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
      kw: dict[str, Any] = {}
      if bkvp is not None:
        kw["num_kv_pages_per_block"] = bkvp
      if bq is not None:
        kw["num_queries_per_block"] = bq
      with jax.sharding.use_abstract_mesh(abs_mesh):
        return kern_fn(q, k, v, cache, kv_lens, pages, cu_q_lens, dist, **kw)

    _OP_CACHE[key] = jax_op(
        f"pallas::{key}", _rpa_jax, donate_argnums=(3,)
    )

  rpa_op = _OP_CACHE[key]
  torch.manual_seed(0)
  q_t = torch.randn(tokens, q_heads, dim, dtype=dtype, device=device)
  k_t = torch.randn(tokens, kv_heads, dim, dtype=dtype, device=device)
  v_t = torch.randn(tokens, kv_heads, dim, dtype=dtype, device=device)
  cache_t = torch.randn(cache_shape, dtype=dtype, device=device)
  kv_lens_t = torch.full((batch,), ctx_len, dtype=torch.int32, device=device)
  pages_t = torch.arange(num_pages, dtype=torch.int32, device=device)
  cu_q_t = (
      torch.arange(batch + 1, dtype=torch.int32, device=device) * q_len
  )
  dist_vals = [batch, batch, batch] if is_decode else [0, 0, batch]
  dist_t = (
      torch.empty((3,), dtype=torch.int32, device=device)
      if device.type == "meta"
      else torch.tensor(dist_vals, dtype=torch.int32, device=device)
  )

  def step_fn() -> torch.Tensor:
    out, new_cache = rpa_op(
        q_t, k_t, v_t, cache_t, kv_lens_t, pages_t, cu_q_t, dist_t
    )
    cache_t.set_(new_cache)
    return out

  return step_fn, {"flops": flops, "cache_shape": cache_shape}


# --------------------------------- RMSNorm -----------------------------------


def build_rmsnorm_ops(
    impl: str, dim: int, device_type: str = "tpu"
) -> tuple[
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        Callable[[], tuple[torch.Tensor, torch.Tensor]],
    ],
]:
  """Builds RMSNorm standalone forward and standalone VJP backward callables.

  Args:
    impl: `"torch_rms_norm"` (`F.rms_norm`), `"rmsnorm_f32acc"`, or
      `"rmsnorm_bf16"`.
    dim: Normalization hidden dimension.
    device_type: `"tpu"` or `"cpu"`.

  Returns:
    `(fwd_fn, bwd_setup_fn)` pair where `bwd_setup_fn(x, weight, dy)` runs
    forward once outside timing and returns a standalone zero-arg `bwd_step`.

  Raises:
    ValueError: If `impl` is unknown.
  """
  if impl == "torch_rms_norm":

    def raw_fwd(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
      return F.rms_norm(x, (dim,), weight, eps=1e-6)

  elif impl in ("rmsnorm_f32acc", "rmsnorm_bf16"):
    acc_f32 = impl == "rmsnorm_f32acc"

    def raw_fwd(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
      xf = x.float() if acc_f32 else x
      msq = torch.mean(xf * xf, dim=-1, keepdim=True)
      inv = torch.rsqrt(msq + 1e-6)
      return (xf * inv).to(x.dtype) * weight

  else:
    raise ValueError(f"Unknown RMSNorm implementation: {impl}")

  use_f32 = impl != "rmsnorm_bf16"

  def raw_rstd(x: torch.Tensor) -> torch.Tensor:
    xf = x.float() if use_f32 else x
    return torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + 1e-6)

  def raw_bwd(
      x: torch.Tensor,
      rstd: torch.Tensor,
      weight: torch.Tensor,
      dy: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    xf = x.float() if use_f32 else x
    dyf = dy.float() if use_f32 else dy
    wf = weight.float() if use_f32 else weight
    x_hat = xf * rstd
    dyw = dyf * wf
    mean_dot = torch.mean(dyw * x_hat, dim=-1, keepdim=True)
    dx = ((dyw - x_hat * mean_dot) * rstd).to(x.dtype)
    dw = torch.sum(dyf * x_hat, dim=0).to(weight.dtype)
    return dx, dw

  fwd_fn = _compile_if_tpu(raw_fwd, device_type)
  rstd_fn = _compile_if_tpu(raw_rstd, device_type)
  bwd_fn = _compile_if_tpu(raw_bwd, device_type)

  def bwd_setup(
      x: torch.Tensor, weight: torch.Tensor, dy: torch.Tensor
  ) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    xd = x.detach()
    wd = weight.detach()
    rstd = rstd_fn(xd)
    if device_type == "tpu" and xd.device.type == "tpu":
      torch.accelerator.synchronize()

    def step() -> tuple[torch.Tensor, torch.Tensor]:
      return bwd_fn(xd, rstd, wd, dy)

    return step

  return fwd_fn, bwd_setup


# --------------------------- SonicMoE / Grouped-GMM --------------------------


def build_ragged_dot(
    impl: str, dtype: torch.dtype, device_type: str = "tpu"
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
  """Builds grouped matmul `[M, K] x [G, K, N] -> [M, N]` for TorchTPU.

  Args:
    impl: `"torchtpu_ragged_dot"` (`torch.tpu.ops.ragged_dot`),
      `"torch_grouped_mm"` (`torch._grouped_mm`), or `"tokamax"`
      (`tokamax.ragged_dot` via `torch.tpu.pallas.jax_op`).
    dtype: Activation/weight dtype (`torch.bfloat16`).
    device_type: `"tpu"` or `"cpu"`.

  Returns:
    Callable `(lhs, rhs, group_sizes) -> out`.

  Raises:
    ValueError: If `impl` is unknown.
  """
  if device_type == "cpu":

    def cpu_rdot(
        lhs: torch.Tensor, rhs: torch.Tensor, sizes: torch.Tensor
    ) -> torch.Tensor:
      outs = []
      offset = 0
      for g, count in enumerate(sizes.tolist()):
        c = int(count)
        outs.append(lhs[offset : offset + c] @ rhs[g])
        offset += c
      return torch.cat(outs, dim=0)

    return cpu_rdot

  if impl == "torchtpu_ragged_dot":
    return torch.tpu.ops.ragged_dot

  if impl == "torch_grouped_mm":

    def gmm_fn(
        lhs: torch.Tensor, rhs: torch.Tensor, sizes: torch.Tensor
    ) -> torch.Tensor:
      offs = torch.cumsum(sizes, dim=0, dtype=torch.int32)
      return torch._grouped_mm(lhs, rhs, offs=offs)

    return gmm_fn

  if impl == "tokamax":
    key = _cache_id("tokamax_rdot", str(dtype))
    if key not in _OP_CACHE:
      import jax  # pylint: disable=g-import-not-at-top
      import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
      import tokamax  # pylint: disable=g-import-not-at-top

      jax_op = _get_jax_op()
      j_dt = jnp.bfloat16 if dtype == torch.bfloat16 else jnp.float32
      abs_mesh = _tpu_abstract_mesh(jax)

      def _tk_rdot(
          lhs: jax.Array, rhs: jax.Array, sizes: jax.Array
      ) -> jax.Array:
        with jax.sharding.use_abstract_mesh(abs_mesh):
          return tokamax.ragged_dot(
              lhs,
              rhs,
              sizes,
              preferred_element_type=j_dt,
              implementation="mosaic_tpu",
          )

      _OP_CACHE[key] = jax_op(f"pallas::{key}", _tk_rdot)
    return _OP_CACHE[key]

  raise ValueError(f"Unknown MoE ragged_dot implementation: {impl}")


def build_moe_suite(
    impl: str,
    tokens: int,
    hidden: int,
    inter: int,
    experts: int,
    top_k: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[list[tuple[str, Callable[[], Any], float | None]], dict[str, Any]]:
  """Builds end-to-end, precision-optimized e2e, and 6-stage SwiGLU MoE ops.

  Includes `e2e_opt`, which multiplies expert outputs by router weights *before*
  unsorting (`(down * probs[:, None])[unsort]`), eliminating a standalone
  `4.30 GB` HBM pass while producing bit-identical mathematics.

  Args:
    impl: Grouped-matmul backend (`torchtpu_ragged_dot`, `torch_grouped_mm`, or
      `tokamax`).
    tokens: Number of input tokens.
    hidden: Model hidden size.
    inter: Per-expert intermediate size.
    experts: Total routed experts.
    top_k: Activated experts per token.
    dtype: Tensor dtype (`torch.bfloat16`).
    device: Target PyTorch device.

  Returns:
    Tuple `(cases, check_stats)` where `cases` is a list of
    `(stage_name, zero_arg_callable, flops_or_none)` and `check_stats` records
    the numerical max-diff between `e2e` and `e2e_opt`.
  """
  dev_type = device.type
  rdot = build_ragged_dot(impl, dtype, dev_type)
  pairs = tokens * top_k
  flops_up = 2.0 * pairs * hidden * (2 * inter)
  flops_down = 2.0 * pairs * inter * hidden
  flops_total = flops_up + flops_down

  torch.manual_seed(0)
  x = torch.randn(tokens, hidden, dtype=dtype, device=device)
  w_router = torch.randn(hidden, experts, dtype=dtype, device=device) * 0.02
  w_up = (
      torch.randn(experts, hidden, 2 * inter, dtype=dtype, device=device)
      * 0.02
  )
  w_down = (
      torch.randn(experts, inter, hidden, dtype=dtype, device=device) * 0.02
  )

  def route_raw(x_in: torch.Tensor, wr: torch.Tensor):
    logits = (x_in @ wr).float()
    top_vals, top_idx = torch.topk(logits, top_k, dim=-1)
    probs = torch.softmax(top_vals, dim=-1).reshape(-1)
    flat = top_idx.reshape(-1)
    order = torch.argsort(flat)
    tok_ids = torch.arange(tokens, device=x_in.device).repeat_interleave(top_k)[
        order
    ]
    exp_ids = torch.arange(experts, device=x_in.device, dtype=flat.dtype)
    sizes = (flat[:, None] == exp_ids[None, :]).sum(dim=0, dtype=torch.int32)
    unsort = torch.argsort(order)
    return tok_ids, sizes, probs[order], unsort

  def gather_raw(x_in: torch.Tensor, tok_ids: torch.Tensor) -> torch.Tensor:
    return x_in[tok_ids]

  def gmm_up_raw(
      rows: torch.Tensor, wu: torch.Tensor, sizes: torch.Tensor
  ) -> torch.Tensor:
    return rdot(rows, wu, sizes)

  def act_raw(up_out: torch.Tensor) -> torch.Tensor:
    gate, val = torch.chunk(up_out, 2, dim=-1)
    return F.silu(gate) * val

  def gmm_down_raw(
      act_rows: torch.Tensor, wd: torch.Tensor, sizes: torch.Tensor
  ) -> torch.Tensor:
    return rdot(act_rows, wd, sizes)

  def reduce_baseline_raw(
      down_rows: torch.Tensor, probs: torch.Tensor, unsort: torch.Tensor
  ) -> torch.Tensor:
    unpermuted = down_rows[unsort]
    probs_unpermuted = probs[unsort].to(down_rows.dtype)[:, None]
    return (unpermuted * probs_unpermuted).reshape(tokens, top_k, hidden).sum(1)

  def reduce_opt_raw(
      down_rows: torch.Tensor, probs: torch.Tensor, unsort: torch.Tensor
  ) -> torch.Tensor:
    weighted = down_rows * probs[:, None].to(down_rows.dtype)
    return weighted[unsort].reshape(tokens, top_k, hidden).sum(1)

  def e2e_raw(x_in, wr, wu, wd) -> torch.Tensor:
    tok_ids, sizes, probs, unsort = route_raw(x_in, wr)
    up_out = gmm_up_raw(gather_raw(x_in, tok_ids), wu, sizes)
    down_out = gmm_down_raw(act_raw(up_out), wd, sizes)
    return reduce_baseline_raw(down_out, probs, unsort)

  def e2e_opt_raw(x_in, wr, wu, wd) -> torch.Tensor:
    tok_ids, sizes, probs, unsort = route_raw(x_in, wr)
    up_out = gmm_up_raw(gather_raw(x_in, tok_ids), wu, sizes)
    down_out = gmm_down_raw(act_raw(up_out), wd, sizes)
    return reduce_opt_raw(down_out, probs, unsort)

  # Compile pure-PyTorch stages; `tokamax` custom_op stages work both eager and
  # compiled, while compiling `route`, `act`, and `reduce` fuses elementwise.
  use_compile = dev_type == "tpu" and impl != "tokamax"
  wrap = (
      (lambda f: _compile_if_tpu(f, dev_type))
      if use_compile
      else (lambda f: f)
  )
  route_fn = _compile_if_tpu(route_raw, dev_type)
  gather_fn = _compile_if_tpu(gather_raw, dev_type)
  gmm_up_fn = wrap(gmm_up_raw)
  act_fn = _compile_if_tpu(act_raw, dev_type)
  gmm_down_fn = wrap(gmm_down_raw)
  reduce_fn = _compile_if_tpu(reduce_baseline_raw, dev_type)
  reduce_opt_fn = _compile_if_tpu(reduce_opt_raw, dev_type)
  e2e_fn = wrap(e2e_raw)
  e2e_opt_fn = wrap(e2e_opt_raw)

  tok_ids, sizes, probs, unsort = route_fn(x, w_router)
  rows = gather_fn(x, tok_ids)
  up_out = gmm_up_fn(rows, w_up, sizes)
  act_rows = act_fn(up_out)
  down_rows = gmm_down_fn(act_rows, w_down, sizes)

  # Verify `reduce_opt` is numerically identical to `reduce_baseline`.
  ref_red = reduce_fn(down_rows, probs, unsort)
  opt_red = reduce_opt_fn(down_rows, probs, unsort)
  max_diff = float((ref_red.float() - opt_red.float()).abs().max().cpu().item())

  cases: list[tuple[str, Callable[[], Any], float | None]] = [
      ("e2e", lambda: e2e_fn(x, w_router, w_up, w_down), flops_total),
      ("e2e_opt", lambda: e2e_opt_fn(x, w_router, w_up, w_down), flops_total),
      ("route", lambda: route_fn(x, w_router), None),
      ("gather", lambda: gather_fn(x, tok_ids), None),
      ("gmm_up", lambda: gmm_up_fn(rows, w_up, sizes), flops_up),
      ("act", lambda: act_fn(up_out), None),
      ("gmm_down", lambda: gmm_down_fn(act_rows, w_down, sizes), flops_down),
      ("reduce", lambda: reduce_fn(down_rows, probs, unsort), None),
      ("reduce_opt", lambda: reduce_opt_fn(down_rows, probs, unsort), None),
  ]
  return cases, {"reduce_opt_max_abs_diff": max_diff}
