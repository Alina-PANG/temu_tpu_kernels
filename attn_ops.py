"""Attention forward and standalone backward builders for TorchTPU.

Provides standalone forward (`fwd_fn`) and standalone backward (`bwd_setup_fn`)
implementations across all kernels in the Temu comparison table and TPU v7x
optimization suite:

  * `torchtpu_flash`: Native TorchTPU C++ Kiln Mosaic FlashAttention via
    `F.scaled_dot_product_attention(..., SDPBackend.FLASH_ATTENTION)`.
  * `dpa`: Native TorchTPU StableHLO / XLA dot-product attention (`MATH` /
    overrideable SDPA).
  * `ref`: Pure attention reference (`softmax(Q @ K^T / sqrt(d) + mask) @ V`),
    compiled with `torch.compile` and standalone `torch.func.vjp` backward.
  * `splash_fused`: Splash Attention with `use_fused_bwd_kernel=True` (`dQ` and
    `dKV` fused into one backward kernel).
  * `splash_sep`: Splash Attention with `use_fused_bwd_kernel=False` (`dQ` and
    `dKV` executed as two separate Pallas kernels, no fusion).
  * `splash_opt`: Precision-preserving optimized Splash Attention removing
    redundant `optimization_barrier`s and HBM scale passes, with independent
    forward (`save_residuals=False`) and backward tile tuning.
  * `tokamax_splash`: OpenXLA Tokamax Splash (`use_base2_exp=True`) bridged to
    PyTorch via `torch.tpu.pallas.jax_op` with standalone backward.
  * `flash_attention`: Legacy Pallas TPU `flash_attention.py` bridged via
    `torch.tpu.pallas.jax_op` with standalone `_flash_attention_bwd`.
  * `fa2`: Open-source `jax_flash_attn2` (`blocksize_q=256, blocksize_k=256`)
    bridged via `torch.tpu.pallas.jax_op` with standalone `_bwd_flash_attn`.
"""

from collections.abc import Callable, Sequence
import dataclasses
import hashlib
import math
import sys
import types
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ForwardFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
BackwardStepFn = Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
BackwardSetupFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], BackwardStepFn
]

_OP_CACHE: dict[str, Any] = {}


def _cache_id(tag: str, *parts: Any) -> str:
  """Builds a deterministic, valid custom-op identifier suffix.

  Args:
    tag: Short prefix describing the kernel family.
    *parts: Hashable configuration values.

  Returns:
    Deterministic alphanumeric op name string.
  """
  digest = hashlib.md5(repr(parts).encode("utf-8")).hexdigest()[:12]
  return f"{tag}_{digest}"


def _get_jax_op() -> Callable[..., Any]:
  """Returns `torch.tpu.pallas.jax_op` for registering trace-only JAX ops.

  Returns:
    The `jax_op` decorator/wrapper from `torch.tpu.pallas`.
  """
  tpu_mod = getattr(torch, "tpu", None)
  if tpu_mod is not None and hasattr(tpu_mod, "pallas"):
    return tpu_mod.pallas.jax_op
  from torch_tpu._internal import pallas  # pylint: disable=g-import-not-at-top

  return pallas.jax_op


def _compile_if_tpu(
    fn: Callable[..., Any], device_type: str
) -> Callable[..., Any]:
  """Wraps `fn` in `torch.compile` when targeting a TPU device.

  Args:
    fn: Callable to compile.
    device_type: PyTorch device type string (e.g., `"tpu"` or `"cpu"`).

  Returns:
    Compiled callable on TPU, or eager `fn` on CPU for unit testing.
  """
  if device_type == "tpu":
    return torch.compile(fn, backend="tpu", dynamic=False)
  return fn


def reference_attention_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
]:
  """Computes exact float32 causal attention (and gradients) for verification.

  Args:
    q: Query tensor `[B, H, S, D]`.
    k: Key tensor `[B, H_kv, S, D]`.
    v: Value tensor `[B, H_kv, S, D]`.
    do: Optional upstream gradient `[B, H, S, D]` for backward verification.

  Returns:
    Tuple `(out_fp32, grads_or_none)` where `grads_or_none` is `(dq, dk, dv)`
    in `float32` when `do` is provided.
  """
  qf = q.detach().float().requires_grad_(do is not None)
  kf = k.detach().float().requires_grad_(do is not None)
  vf = v.detach().float().requires_grad_(do is not None)
  enable_gqa = qf.shape[1] != kf.shape[1]
  out = F.scaled_dot_product_attention(
      qf, kf, vf, is_causal=True, enable_gqa=enable_gqa
  )
  if do is None:
    return out.detach(), None
  grads = torch.autograd.grad(out, (qf, kf, vf), do.detach().float())
  return out.detach(), (grads[0].detach(), grads[1].detach(), grads[2].detach())


def _build_torchtpu_sdpa(
    backend_kind: str, device_type: str
) -> tuple[ForwardFn, BackwardSetupFn]:
  """Builds native TorchTPU `scaled_dot_product_attention` fwd and bwd.

  Args:
    backend_kind: `"flash"` for TorchTPU C++ Kiln Mosaic FlashAttention, or
      `"dpa"` for StableHLO / XLA native SDPA.
    device_type: Target device type (`"tpu"` or `"cpu"`).

  Returns:
    `(fwd_fn, bwd_setup_fn)` pair.
  """
  sdp_Enum = torch.nn.attention.SDPBackend
  if device_type == "cpu":
    backends = [sdp_Enum.MATH]
  elif backend_kind == "flash":
    backends = [sdp_Enum.FLASH_ATTENTION]
  else:
    backends = [sdp_Enum.EFFICIENT_ATTENTION, sdp_Enum.MATH]

  def raw_fwd(
      q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
  ) -> torch.Tensor:
    enable_gqa = q.shape[1] != k.shape[1]
    with torch.nn.attention.sdpa_kernel(backends):
      return F.scaled_dot_product_attention(
          q, k, v, is_causal=True, enable_gqa=enable_gqa
      )

  # For `flash`, TorchTPU dispatches directly to C++ Kiln Mosaic ATen ops
  # (`AtenScaledDotProductFlashAttention` and `...Backward`) without needing
  # Dynamo wrapping; `dpa` compiles through StableHLO.
  fwd_fn = (
      raw_fwd
      if backend_kind == "flash"
      else _compile_if_tpu(raw_fwd, device_type)
  )

  def bwd_setup(
      q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, do: torch.Tensor
  ) -> BackwardStepFn:
    if backend_kind == "flash" or device_type == "cpu":
      qr = q.detach().clone().requires_grad_(True)
      kr = k.detach().clone().requires_grad_(True)
      vr = v.detach().clone().requires_grad_(True)
      with torch.nn.attention.sdpa_kernel(backends):
        out = F.scaled_dot_product_attention(
            qr, kr, vr, is_causal=True, enable_gqa=(qr.shape[1] != kr.shape[1])
        )
      if q.device.type == "tpu":
        torch.accelerator.synchronize()

      def step() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dq, dk, dv = torch.autograd.grad(
            out, (qr, kr, vr), do, retain_graph=True
        )
        return dq, dk, dv

      return step

    # Standalone VJP for `dpa`: runs forward once to materialize saved
    # residuals, then compiles ONLY `vjp_fn(do)` with zero forward pass.
    _, vjp_fn = torch.func.vjp(raw_fwd, q.detach(), k.detach(), v.detach())
    compiled_vjp = _compile_if_tpu(vjp_fn, device_type)
    if q.device.type == "tpu":
      torch.accelerator.synchronize()

    def vjp_step() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      return compiled_vjp(do)

    return vjp_step

  return fwd_fn, bwd_setup


def _build_ref_attention(
    seq: int, dim: int, device_type: str
) -> tuple[ForwardFn, BackwardSetupFn]:
  """Builds pure causal reference attention matching `mha_reference` semantics.

  Computes `Q @ K^T * (1/sqrt(d))` with a causal mask, `float32` online-safe
  softmax (`exp(logits - max)`), and `@ V`, plus standalone `torch.func.vjp`
  backward without re-running the forward matmul.

  Args:
    seq: Sequence length.
    dim: Head dimension.
    device_type: Target device type (`"tpu"` or `"cpu"`).

  Returns:
    `(fwd_fn, bwd_setup_fn)` pair.
  """
  scale = 1.0 / math.sqrt(float(dim))
  mask_val = -0.7 * float(torch.finfo(torch.float32).max)

  def raw_ref(
      q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
  ) -> torch.Tensor:
    if k.shape[1] != q.shape[1]:
      reps = q.shape[1] // k.shape[1]
      k = k.repeat_interleave(reps, dim=1)
      v = v.repeat_interleave(reps, dim=1)
    logits = torch.matmul(q, k.transpose(-2, -1)).float() * scale
    idx = torch.arange(seq, device=q.device)
    causal = idx[:, None] >= idx[None, :]
    logits = torch.where(causal, logits, mask_val)
    probs = torch.softmax(logits, dim=-1).to(v.dtype)
    return torch.matmul(probs, v)

  fwd_fn = _compile_if_tpu(raw_ref, device_type)

  def bwd_setup(
      q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, do: torch.Tensor
  ) -> BackwardStepFn:
    _, vjp_fn = torch.func.vjp(raw_ref, q.detach(), k.detach(), v.detach())
    compiled_vjp = _compile_if_tpu(vjp_fn, device_type)
    if q.device.type == "tpu":
      torch.accelerator.synchronize()

    def step() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      return compiled_vjp(do)

    return step

  return fwd_fn, bwd_setup


def _build_splash_builtin(
    seq: int,
    heads: int,
    kv_heads: int,
    blocks: Sequence[int],
    use_fused_bwd: bool,
) -> tuple[ForwardFn, BackwardSetupFn]:
  """Builds `splash_fused` or `splash_sep` using TorchTPU's Splash custom ops.

  Directly resolves the forward (`splash_attn_fwd_*`) and standalone backward
  (`splash_attn_bwd_*`) `torch.ops.pallas` custom ops so backward timing runs
  purely on precomputed `(out, logsumexp)` without re-running forward and
  without fusing `dQ` + `dKV` when `use_fused_bwd=False` (`splash_sep`).

  Args:
    seq: Sequence length.
    heads: Number of query heads.
    kv_heads: Number of KV heads.
    blocks: Tuple `(block_q, block_kv, block_kv_compute)`.
    use_fused_bwd: `True` for `splash_fused`, `False` for `splash_sep`.

  Returns:
    `(fwd_fn, bwd_setup_fn)` pair.
  """
  import importlib  # pylint: disable=g-import-not-at-top

  tt_splash = importlib.import_module(
      "torch_tpu._internal.ops.splash_attention"
  )
  bq, bkv, bkvc = (min(int(b), seq) for b in blocks)
  kwargs: dict[str, Any] = dict(
      seq_len=seq,
      n_heads=heads,
      n_kv_heads=kv_heads,
      scale=None,
      is_causal=True,
      local_window_size=None,
      block_q=bq,
      block_kv=bkv,
      block_kv_compute=bkvc,
      block_q_dkv=bq,
      block_kv_dkv=bkv,
      block_kv_dkv_compute=bkvc,
      block_q_dq=None if use_fused_bwd else bq,
      block_kv_dq=None if use_fused_bwd else bkv,
      use_fused_bwd_kernel=use_fused_bwd,
      q_layout="HEAD_DIM_MINOR",
      k_layout="HEAD_DIM_MINOR",
      v_layout="HEAD_DIM_MINOR",
      use_vmap_bwd=True,
  )
  get_names = getattr(tt_splash, "_get_splash_op_names")
  fwd_name, bwd_name = get_names(**kwargs)
  fwd_op = getattr(torch.ops.pallas, fwd_name)
  bwd_op = getattr(torch.ops.pallas, bwd_name)

  def fwd_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out, _ = fwd_op(q, k, v, None)
    return out

  def bwd_setup(
      q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, do: torch.Tensor
  ) -> BackwardStepFn:
    out, lse = fwd_op(q, k, v, None)
    if q.device.type == "tpu":
      torch.accelerator.synchronize()

    def step() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      return bwd_op(q, k, v, out, lse, do, None)

    return step

  return fwd_fn, bwd_setup


def _build_splash_opt(
    seq: int,
    heads: int,
    kv_heads: int,
    dim: int,
    fwd_blocks: Sequence[int],
    bwd_blocks: Sequence[int],
    use_fused_bwd: bool = True,
) -> tuple[ForwardFn, BackwardSetupFn]:
  """Builds precision-preserving optimized Splash Attention via `jax_op`.

  Optimizations (strictly preserving `bf16` input / `fp32` softmax & matmul
  accumulation precision):
    1. Pure forward (`save_residuals=False`) omits writing `logsumexp` to HBM.
    2. Removes the three `jax.lax.optimization_barrier` ops in backward so XLA
       fuses `dq_scaled * scale` into the epilogue.
    3. Decouples forward tiles (e.g., `1024x1024x1024`) from backward `dkv`
       tiles (e.g., `512x1024x512`) so each pass runs at peak VMEM occupancy.

  Args:
    seq: Sequence length.
    heads: Number of query heads.
    kv_heads: Number of KV heads.
    dim: Head dimension.
    fwd_blocks: Forward `(block_q, block_kv, block_kv_compute)`.
    bwd_blocks: Backward `(block_q_dkv, block_kv_dkv, block_kv_dkv_compute)`.
    use_fused_bwd: Whether to fuse `dQ` into the `dKV` kernel.

  Returns:
    `(fwd_fn, bwd_setup_fn)` pair.
  """
  key = _cache_id(
      "splash_opt", seq, heads, kv_heads, dim, fwd_blocks, bwd_blocks,
      use_fused_bwd,
  )
  if key not in _OP_CACHE:
    import importlib  # pylint: disable=g-import-not-at-top
    import jax  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

    sp_pkg = "jax.experimental.pallas.ops.tpu.splash_attention"
    sk = importlib.import_module(f"{sp_pkg}.splash_attention_kernel")
    sm = importlib.import_module(f"{sp_pkg}.splash_attention_mask")
    tt_splash = importlib.import_module(
        "torch_tpu._internal.ops.splash_attention"
    )

    jax_op = _get_jax_op()
    bq, bkv, bkvc = (min(int(b), seq) for b in fwd_blocks)
    bqd, bkvd, bkvcd = (min(int(b), seq) for b in bwd_blocks)
    bs_no_res = sk.BlockSizes(
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkvc,
        q_layout=sk.QKVLayout.HEAD_DIM_MINOR,
        k_layout=sk.QKVLayout.HEAD_DIM_MINOR,
        v_layout=sk.QKVLayout.HEAD_DIM_MINOR,
    )
    bs_with_bwd = sk.BlockSizes(
        block_q=bq,
        block_kv=bkv,
        block_kv_compute=bkvc,
        block_q_dkv=bqd,
        block_kv_dkv=bkvd,
        block_kv_dkv_compute=bkvcd,
        block_q_dq=None if use_fused_bwd else bqd,
        block_kv_dq=None if use_fused_bwd else bkvd,
        use_fused_bwd_kernel=use_fused_bwd,
        q_layout=sk.QKVLayout.HEAD_DIM_MINOR,
        k_layout=sk.QKVLayout.HEAD_DIM_MINOR,
        v_layout=sk.QKVLayout.HEAD_DIM_MINOR,
    )
    causal = sm.CausalMask(shape=(seq, seq))
    mh_mask = sm.MultiHeadMask(masks=(causal,) * heads)
    make_mi = getattr(tt_splash, "_make_numpy_splash_mask_infos")
    fwd_mi_pure, _, _, mask_fn = make_mi(mh_mask, bs_no_res)
    fwd_mi_res, dq_mi, dkv_mi, _ = make_mi(mh_mask, bs_with_bwd)
    is_mqa = kv_heads == 1
    sm_scale = 1.0 / math.sqrt(float(dim))
    mask_val = sk.DEFAULT_MASK_VALUE
    raw_fwd = getattr(sk, "_splash_attention_forward")
    raw_bwd = getattr(sk, "_splash_attention_bwd")

    def _pure_fwd_jax(
        q: jax.Array, k: jax.Array, v: jax.Array
    ) -> jax.Array:
      qs = q * jnp.asarray(sm_scale, dtype=q.dtype)

      def one(qb, kb, vb):
        return raw_fwd(
            fwd_mi_pure, qb, kb, vb, None, sinks=None, mask_value=mask_val,
            is_mqa=is_mqa, block_sizes=bs_no_res, residual_checkpoint_name=None,
            save_residuals=False, mask_function=mask_fn,
            attn_logits_soft_cap=None, interpret=False,
        )

      return jax.vmap(one)(qs, k, v)

    def _res_fwd_jax(
        q: jax.Array, k: jax.Array, v: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      qs = q * jnp.asarray(sm_scale, dtype=q.dtype)

      def one(qb, kb, vb):
        out_b, (lse_b,) = raw_fwd(
            fwd_mi_res, qb, kb, vb, None, sinks=None, mask_value=mask_val,
            is_mqa=is_mqa, block_sizes=bs_with_bwd,
            residual_checkpoint_name=None, save_residuals=True,
            mask_function=mask_fn, attn_logits_soft_cap=None, interpret=False,
        )
        return qs_and_res(qb, out_b, lse_b)

      def qs_and_res(qb, out_b, lse_b):
        return qb, out_b, lse_b

      return jax.vmap(one)(qs, k, v)

    def _pure_bwd_jax(
        q_scaled: jax.Array,
        k: jax.Array,
        v: jax.Array,
        out: jax.Array,
        lse: jax.Array,
        do: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      s_val = jnp.asarray(sm_scale, dtype=q_scaled.dtype)

      def one(qsb, kb, vb, ob, lseb, dob):
        res = (qsb, kb, vb, None, None, ob, lseb, dq_mi, dkv_mi)
        _, _, _, dq_s, dk_b, dv_b, _, _ = raw_bwd(
            False, mask_val, is_mqa, bs_with_bwd, None, mask_fn, None, False,
            res, dob,
        )
        return dq_s * s_val, dk_b, dv_b

      return jax.vmap(one)(q_scaled, k, v, out, lse, do)

    _OP_CACHE[key] = (
        jax_op(f"pallas::{key}_fwd", _pure_fwd_jax),
        jax_op(f"pallas::{key}_rfwd", _res_fwd_jax),
        jax_op(f"pallas::{key}_bwd", _pure_bwd_jax),
    )

  pure_fwd_op, res_fwd_op, pure_bwd_op = _OP_CACHE[key]

  def fwd_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return pure_fwd_op(q, k, v)

  def bwd_setup(
      q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, do: torch.Tensor
  ) -> BackwardStepFn:
    q_scaled, out, lse = res_fwd_op(q, k, v)
    if q.device.type == "tpu":
      torch.accelerator.synchronize()

    def step() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      return pure_bwd_op(q_scaled, k, v, out, lse, do)

    return step

  return fwd_fn, bwd_setup


def _build_tokamax_splash(
    seq: int,
    heads: int,
    kv_heads: int,
    dim: int,
    blocks: Sequence[int],
    bwd_blocks: Sequence[int] | None = None,
    use_fused_bwd: bool = True,
) -> tuple[ForwardFn, BackwardSetupFn]:
  """Builds OpenXLA Tokamax Splash (`use_base2_exp=True`) via `jax_op`.

  Constructs the Tokamax kernel inside the `jax_op` trace so mask metadata is
  embedded as MLIR constants without initializing a separate JAX TPU client,
  and exposes standalone backward via `jax.vjp` residuals `(out, logsumexp)`.

  Args:
    seq: Sequence length.
    heads: Number of query heads.
    kv_heads: Number of KV heads.
    dim: Head dimension.
    blocks: Forward `(block_q, block_kv, block_kv_compute)`.
    bwd_blocks: Optional backward `(block_q_dkv, block_kv_dkv, block_kvc)`.
    use_fused_bwd: Whether to use fused backward kernel (`True`) or separate
      `dQ` + `dKV` (`False`).

  Returns:
    `(fwd_fn, bwd_setup_fn)` pair.
  """
  del kv_heads  # Inferred from tensor shapes inside Tokamax Splash.
  bwd_blk = bwd_blocks or blocks
  key = _cache_id(
      "tokamax_splash", seq, heads, dim, blocks, bwd_blk, use_fused_bwd
  )
  if key not in _OP_CACHE:
    import jax  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

    tk_kernel = None
    tk_mask = None
    for mod_base in (
        "tokamax._src.ops.experimental.tpu.splash_attention",
        "tokamax._src.ops.splash_attention",
    ):
      try:
        tk_kernel = __import__(
            mod_base + ".splash_attention_kernel", fromlist=["_"]
        )
        tk_mask = __import__(
            mod_base + ".splash_attention_mask", fromlist=["_"]
        )
        break
      except ImportError:
        continue
    if tk_kernel is None or tk_mask is None:
      raise ImportError("tokamax splash_attention is not installed.")

    jax_op = _get_jax_op()
    bq, bkv, bkvc = (min(int(b), seq) for b in blocks)
    bqd, bkvd, bkvcd = (min(int(b), seq) for b in bwd_blk)
    sm_scale = 1.0 / math.sqrt(float(dim))
    field_names = {f.name for f in dataclasses.fields(tk_kernel.SplashConfig)}

    def _make_cfg(with_bwd: bool):
      kw: dict[str, Any] = dict(
          block_q=bq,
          block_kv=bkv,
          block_kv_compute=bkvc,
          use_base2_exp=True,
      )
      if with_bwd:
        kw.update(
            block_q_dkv=bqd,
            block_kv_dkv=bkvd,
            block_kv_dkv_compute=bkvcd,
            use_fused_bwd_kernel=use_fused_bwd,
        )
        if not use_fused_bwd:
          kw.update(block_q_dq=bqd, block_kv_dq=bkvd)
      return tk_kernel.SplashConfig(
          **{k: v for k, v in kw.items() if k in field_names}
      )

    abs_dev = jax.sharding.AbstractDevice(
        device_kind="TPU7x", num_cores=1, platform="tpu"
    )
    abs_mesh = jax.sharding.AbstractMesh((), (), abstract_device=abs_dev)

    def _fwd_jax(
        q: jax.Array, k: jax.Array, v: jax.Array
    ) -> jax.Array:
      with jax.sharding.use_abstract_mesh(abs_mesh):
        causal = tk_mask.CausalMask(shape=(seq, seq))
        kern = tk_kernel.make_splash_mha_single_device(
            causal, config=_make_cfg(with_bwd=False)
        )
        qs = q * jnp.asarray(sm_scale, dtype=q.dtype)
        return jax.vmap(kern, in_axes=(0, 0, 0))(qs, k, v)

    def _rfwd_jax(
        q: jax.Array, k: jax.Array, v: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      with jax.sharding.use_abstract_mesh(abs_mesh):
        causal = tk_mask.CausalMask(shape=(seq, seq))
        kern = tk_kernel.make_splash_mha_single_device(
            causal, config=_make_cfg(with_bwd=True), save_residuals=True
        )
        qs = q * jnp.asarray(sm_scale, dtype=q.dtype)

        log2e = math.log2(math.e)

        def one(qb, kb, vb):
          out_b, stats = kern(qb, kb, vb)
          lse_b = stats["logsumexp"] if isinstance(stats, dict) else stats[0]
          # Tokamax divides user-facing `stats["logsumexp"]` by LOG2E when
          # `use_base2_exp=True`, while `_splash_attention_bwd` expects base-2.
          lse_b = lse_b * jnp.asarray(log2e, dtype=lse_b.dtype)
          return qb, out_b, lse_b

        return jax.vmap(one)(qs, k, v)

    def _bwd_jax(
        q_scaled: jax.Array,
        k: jax.Array,
        v: jax.Array,
        out: jax.Array,
        lse: jax.Array,
        do: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      with jax.sharding.use_abstract_mesh(abs_mesh):
        causal = tk_mask.CausalMask(shape=(seq, seq))
        cfg = _make_cfg(with_bwd=True)
        kern = tk_kernel.make_splash_mha_single_device(causal, config=cfg)
        kw = getattr(kern, "kwargs", {})
        s_val = jnp.asarray(sm_scale, dtype=q_scaled.dtype)

        def one(qsb, kb, vb, ob, lseb, dob):
          res = (
              qsb, kb, vb, None, None, ob, lseb, kern.dkv_mask_info, None
          )
          bwd_fn = getattr(tk_kernel, "_splash_attention_bwd")
          out_bwd = bwd_fn(
              False,
              kw.get("mask_value", getattr(kern, "mask_value", -2.38e38)),
              kw.get("is_mqa", getattr(kern, "is_mqa", False)),
              kw.get("config", getattr(kern, "config", cfg)),
              kw.get("mask_function", getattr(kern, "mask_function", None)),
              kw.get("fwd_mask_sparsity", 1.0),
              kw.get("dkv_mask_sparsity", 1.0),
              res,
              dob,
          )
          return out_bwd[2] * s_val, out_bwd[3], out_bwd[4]

        return jax.vmap(one)(q_scaled, k, v, out, lse, do)

    _OP_CACHE[key] = (
        jax_op(f"pallas::{key}_fwd", _fwd_jax),
        jax_op(f"pallas::{key}_rfwd", _rfwd_jax),
        jax_op(f"pallas::{key}_bwd", _bwd_jax),
    )

  fwd_op, rfwd_op, bwd_op = _OP_CACHE[key]

  def fwd_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return fwd_op(q, k, v)

  def bwd_setup(
      q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, do: torch.Tensor
  ) -> BackwardStepFn:
    q_scaled, out, lse = rfwd_op(q, k, v)
    if q.device.type == "tpu":
      torch.accelerator.synchronize()

    def step() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      return bwd_op(q_scaled, k, v, out, lse, do)

    return step

  return fwd_fn, bwd_setup


def _build_pallas_flash_attention(
    seq: int, dim: int, blocks: Sequence[int] | None = None
) -> tuple[ForwardFn, BackwardSetupFn]:
  """Builds legacy `pallas/ops/tpu/flash_attention.py` (`flash attention` row).

  Uses Temu's exact optimal block sizes (`block_q=256, block_k_major=min(seq,
  2048), block_k=512, block_b=1`) and calls `_flash_attention_bwd` directly on
  saved `(o, l, m)` residuals so backward timing has zero forward pass.

  Args:
    seq: Sequence length.
    dim: Head dimension.
    blocks: Optional `(block_q, block_k_major, block_k)`.

  Returns:
    `(fwd_fn, bwd_setup_fn)` pair.
  """
  bq, bkm, bk = blocks if blocks is not None else (256, min(seq, 2048), 512)
  bq, bkm, bk = min(bq, seq), min(bkm, seq), min(bk, seq)
  key = _cache_id("pallas_fa", seq, dim, bq, bkm, bk)
  if key not in _OP_CACHE:
    import importlib  # pylint: disable=g-import-not-at-top
    import jax  # pylint: disable=g-import-not-at-top

    pfa = importlib.import_module(
        "jax.experimental.pallas.ops.tpu.flash_attention"
    )
    jax_op = _get_jax_op()
    sm_scale = 1.0 / math.sqrt(float(dim))
    bs = pfa.BlockSizes(
        block_q=bq,
        block_k_major=bkm,
        block_k=bk,
        block_b=1,
        block_q_major_dkv=bq,
        block_k_major_dkv=bkm,
        block_k_dkv=bk,
        block_q_dkv=bq,
        block_k_major_dq=bkm,
        block_k_dq=bk,
        block_q_dq=bq,
    )
    fa_impl = getattr(pfa, "_flash_attention_impl")
    fa_bwd = getattr(pfa, "_flash_attention_bwd")

    def _fwd_jax(
        q: jax.Array, k: jax.Array, v: jax.Array
    ) -> jax.Array:
      return fa_impl(
          q, k, v, None, None, False, True, sm_scale, 1, bq, bkm, bk, False
      )

    def _rfwd_jax(
        q: jax.Array, k: jax.Array, v: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      return fa_impl(
          q, k, v, None, None, True, True, sm_scale, 1, bq, bkm, bk, False
      )

    def _bwd_jax(
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        o: jax.Array,
        l: jax.Array,
        m: jax.Array,
        do: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      residuals = (q, k, v, None, None, o, l, m)
      dq, dk, dv, _, _ = fa_bwd(
          False, True, sm_scale, bs, False, residuals, do
      )
      return dq, dk, dv

    _OP_CACHE[key] = (
        jax_op(f"pallas::{key}_fwd", _fwd_jax),
        jax_op(f"pallas::{key}_rfwd", _rfwd_jax),
        jax_op(f"pallas::{key}_bwd", _bwd_jax),
    )

  fwd_op, rfwd_op, bwd_op = _OP_CACHE[key]

  def fwd_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return fwd_op(q, k, v)

  def bwd_setup(
      q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, do: torch.Tensor
  ) -> BackwardStepFn:
    o, l, m = rfwd_op(q, k, v)
    if q.device.type == "tpu":
      torch.accelerator.synchronize()

    def step() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      return bwd_op(q, k, v, o, l, m, do)

    return step

  return fwd_fn, bwd_setup


def _ensure_eformer_shim() -> None:
  """Installs a lightweight `eformer.escale` shim if `eformer` is absent."""
  if "eformer" in sys.modules:
    return
  try:
    __import__("eformer.escale")
  except ImportError:
    import jax  # pylint: disable=g-import-not-at-top

    eformer_mod = types.ModuleType("eformer")
    escale_mod = types.ModuleType("eformer.escale")
    escale_mod.with_sharding_constraint = (  # type: ignore[attr-defined]
        lambda arr, sharding: jax.lax.with_sharding_constraint(arr, sharding)
    )
    eformer_mod.escale = escale_mod  # type: ignore[attr-defined]
    sys.modules["eformer"] = eformer_mod
    sys.modules["eformer.escale"] = escale_mod


def _build_fa2(
    seq: int, dim: int, block_q: int = 256, block_k: int = 256
) -> tuple[ForwardFn, BackwardSetupFn]:
  """Builds `erfanzar/jax-flash-attn2` (`fa2` row) via `jax_op`.

  Imports `jax_flash_attn2.flash_attention_jax` directly (bypassing Triton GPU
  imports) and exposes standalone `_fwd_flash_attn` and `_bwd_flash_attn`.

  Args:
    seq: Sequence length.
    dim: Head dimension.
    block_q: Query tile size (customer table: `256`).
    block_k: Key/Value tile size (customer table: `256`).

  Returns:
    `(fwd_fn, bwd_setup_fn)` pair.
  """
  bq, bk = min(block_q, seq), min(block_k, seq)
  key = _cache_id("fa2", seq, dim, bq, bk)
  if key not in _OP_CACHE:
    import jax  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

    _ensure_eformer_shim()
    # Prevent `jax_flash_attn2/__init__.py` from importing Triton GPU modules.
    if "jax_flash_attn2" not in sys.modules:
      import importlib.util  # pylint: disable=g-import-not-at-top

      spec = importlib.util.find_spec("jax_flash_attn2")
      if spec is None or not spec.submodule_search_locations:
        raise ImportError("jax-flash-attn2 is not installed.")
      pkg = types.ModuleType("jax_flash_attn2")
      pkg.__path__ = list(spec.submodule_search_locations)
      sys.modules["jax_flash_attn2"] = pkg

    import importlib  # pylint: disable=g-import-not-at-top

    fa2_bwd_mod = importlib.import_module(
        "jax_flash_attn2.flash_attention_jax._backward_jax"
    )
    fa2_fwd_mod = importlib.import_module(
        "jax_flash_attn2.flash_attention_jax._forward_jax"
    )
    _bwd_flash_attn = getattr(fa2_bwd_mod, "_bwd_flash_attn")
    _fwd_flash_attn = getattr(fa2_fwd_mod, "_fwd_flash_attn")

    jax_op = _get_jax_op()
    sm_scale = 1.0 / math.sqrt(float(dim))

    def _causal_mask(batch_sz: int):
      rows = jax.lax.broadcasted_iota(jnp.int32, (seq, seq), 0)
      cols = jax.lax.broadcasted_iota(jnp.int32, (seq, seq), 1)
      return jnp.broadcast_to((cols <= rows)[None, None, :, :],
                              (batch_sz, 1, seq, seq))

    def _fwd_jax(
        q: jax.Array, k: jax.Array, v: jax.Array
    ) -> jax.Array:
      qs = q * jnp.asarray(sm_scale, dtype=q.dtype)
      mask = _causal_mask(q.shape[0])
      out, _ = _fwd_flash_attn(
          qs, k, v, mask, None, 0.0, True, None, bq, bk, q.dtype, None
      )
      return out

    def _rfwd_jax(
        q: jax.Array, k: jax.Array, v: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      qs = q * jnp.asarray(sm_scale, dtype=q.dtype)
      mask = _causal_mask(q.shape[0])
      out, res = _fwd_flash_attn(
          qs, k, v, mask, None, 0.0, True, None, bq, bk, q.dtype, None
      )
      return qs, out, res[1]

    def _bwd_jax(
        q_scaled: jax.Array,
        k: jax.Array,
        v: jax.Array,
        out: jax.Array,
        lse: jax.Array,
        do: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      s_val = jnp.asarray(sm_scale, dtype=q_scaled.dtype)
      mask = _causal_mask(q_scaled.shape[0])
      res = (out, lse, q_scaled, k, v, mask, None)
      dq_s, dk, dv, _, _ = _bwd_flash_attn(
          0.0, True, None, bq, bk, q_scaled.dtype, None, res, do
      )
      return dq_s * s_val, dk, dv

    _OP_CACHE[key] = (
        jax_op(f"pallas::{key}_fwd", _fwd_jax),
        jax_op(f"pallas::{key}_rfwd", _rfwd_jax),
        jax_op(f"pallas::{key}_bwd", _bwd_jax),
    )

  fwd_op, rfwd_op, bwd_op = _OP_CACHE[key]

  def fwd_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return fwd_op(q, k, v)

  def bwd_setup(
      q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, do: torch.Tensor
  ) -> BackwardStepFn:
    q_scaled, out, lse = rfwd_op(q, k, v)
    if q.device.type == "tpu":
      torch.accelerator.synchronize()

    def step() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      return bwd_op(q_scaled, k, v, out, lse, do)

    return step

  return fwd_fn, bwd_setup


def build_attention_ops(
    impl: str,
    seq: int,
    heads: int,
    kv_heads: int,
    dim: int,
    blocks: Sequence[int] | None,
    device_type: str = "tpu",
    bwd_blocks: Sequence[int] | None = None,
) -> tuple[ForwardFn, BackwardSetupFn]:
  """Dispatches to the requested attention forward and standalone backward ops.

  Args:
    impl: Implementation identifier (`torchtpu_flash`, `dpa`, `ref`,
      `splash_fused`, `splash_sep`, `splash_opt`, `tokamax_splash`,
      `tokamax_splash_sep`, `flash_attention`, `fa2`).
    seq: Sequence length.
    heads: Number of query heads.
    kv_heads: Number of KV heads.
    dim: Head dimension.
    blocks: Optional forward block tuple.
    device_type: `"tpu"` or `"cpu"`.
    bwd_blocks: Optional backward block tuple for `splash_opt` / `tokamax`.

  Returns:
    `(fwd_fn, bwd_setup_fn)` pair where `bwd_setup_fn` prepares forward
    residuals once outside timing and returns a standalone zero-arg `bwd_step`.

  Raises:
    ValueError: If `impl` is unknown.
  """
  default_blk = blocks or (512, min(seq, 2048), min(seq, 1024))
  if impl == "torchtpu_flash":
    return _build_torchtpu_sdpa("flash", device_type)
  if impl == "dpa":
    return _build_torchtpu_sdpa("dpa", device_type)
  if impl == "ref":
    return _build_ref_attention(seq, dim, device_type)
  if device_type == "cpu":
    # On CPU (unit tests), fall back to `dpa` for TPU-only Pallas kernels.
    return _build_torchtpu_sdpa("dpa", "cpu")
  if impl == "splash_fused":
    return _build_splash_builtin(
        seq, heads, kv_heads, default_blk, use_fused_bwd=True
    )
  if impl == "splash_sep":
    return _build_splash_builtin(
        seq, heads, kv_heads, default_blk, use_fused_bwd=False
    )
  if impl == "splash_opt":
    opt_fwd = blocks or (min(seq, 1024), min(seq, 1024), min(seq, 1024))
    opt_bwd = bwd_blocks or (min(seq, 512), min(seq, 1024), min(seq, 512))
    return _build_splash_opt(
        seq, heads, kv_heads, dim, opt_fwd, opt_bwd, use_fused_bwd=True
    )
  if impl == "tokamax_splash":
    return _build_tokamax_splash(
        seq, heads, kv_heads, dim, default_blk, bwd_blocks=bwd_blocks,
        use_fused_bwd=True,
    )
  if impl in ("splash_opt_sep", "tokamax_splash_sep"):
    opt_fwd = blocks or (min(seq, 1024), min(seq, 1024), min(seq, 1024))
    opt_bwd = bwd_blocks or (min(seq, 512), min(seq, 1024), min(seq, 512))
    return _build_splash_opt(
        seq, heads, kv_heads, dim, opt_fwd, opt_bwd, use_fused_bwd=False
    )
  if impl == "flash_attention":
    return _build_pallas_flash_attention(seq, dim, blocks=blocks)
  if impl == "fa2":
    bq = blocks[0] if blocks else 256
    bk = blocks[1] if blocks else 256
    return _build_fa2(seq, dim, block_q=bq, block_k=bk)
  raise ValueError(f"Unknown attention implementation: {impl}")
