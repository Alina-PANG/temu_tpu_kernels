#!/usr/bin/env python3
"""Build-time package and Pallas Mosaic TPU MLIR lowering check (CPU-only)."""

from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore", message=".*ExperimentalApi.*")

# pylint: disable=g-import-not-at-top
import attn_ops
from examples import standalone_custom_pallas_op
import other_ops
import torch
import torch_tpu  # pylint: disable=unused-import
import tokamax  # pylint: disable=unused-import
# pylint: enable=g-import-not-at-top


def _verify_attention_mlir_export() -> None:
  """Triggers real Mosaic TPU MLIR export via `meta` tensors for fwd & bwd."""
  q = torch.empty(2, 4, 512, 128, dtype=torch.bfloat16, device="meta")
  k = torch.empty(2, 4, 512, 128, dtype=torch.bfloat16, device="meta")
  v = torch.empty(2, 4, 512, 128, dtype=torch.bfloat16, device="meta")
  do = torch.empty(2, 4, 512, 128, dtype=torch.bfloat16, device="meta")
  for impl, blk in (
      ("splash_fused", (256, 512, 256)),
      ("splash_sep", (256, 512, 256)),
      ("splash_opt", (512, 512, 512)),
      ("tokamax_splash", (256, 512, 256)),
      ("tokamax_splash_sep", (256, 512, 256)),
      ("flash_attention", (256, 512, 256)),
      ("fa2", (256, 256)),
  ):
    fwd_fn, bwd_setup_fn = attn_ops.build_attention_ops(
        impl,
        seq=512,
        heads=4,
        kv_heads=4,
        dim=128,
        blocks=blk,
        device_type="tpu",
    )
    out = fwd_fn(q, k, v)
    dq, dk, dv = bwd_setup_fn(q, k, v, do)()
    assert out.shape == q.shape, f"Unexpected fwd shape for {impl}: {out.shape}"
    assert dq.shape == q.shape and dk.shape == k.shape and dv.shape == v.shape
    print("Verified TPU MLIR fwd & standalone bwd export:", impl)


def _verify_moe_and_paged_mlir_export() -> None:
  """Verifies Mosaic TPU MLIR export for Tokamax ragged_dot and RPAv3."""
  tk_rdot = other_ops.build_ragged_dot("tokamax", torch.bfloat16, "tpu")
  lhs = torch.empty(1024, 256, dtype=torch.bfloat16, device="meta")
  rhs = torch.empty(8, 256, 128, dtype=torch.bfloat16, device="meta")
  sizes = torch.empty(8, dtype=torch.int32, device="meta")
  out = tk_rdot(lhs, rhs, sizes)
  assert out.shape == (1024, 128)
  print("Verified Tokamax Mosaic TPU ragged_dot export:", out.shape)

  getattr(other_ops, "_install_vllm_logger_and_tpu_shims")()
  for dim in (64, 128):
    mod_name = (
        "vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel_hd64"
        if dim == 64
        else "vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel"
    )
    rpa_mod = importlib.import_module(mod_name)
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

    c_shape = tuple(
        int(x) for x in rpa_mod.get_kv_cache_shape(8, 256, 2, dim, jnp.bfloat16)
    )
    step_fn, _ = other_ops.build_paged_attention(
        batch=4,
        ctx_len=512,
        q_heads=2,
        kv_heads=2,
        dim=dim,
        page_size=256,
        phase="prefill" if dim == 64 else "decode",
        dtype=torch.bfloat16,
        device=torch.device("meta"),
    )
    del step_fn
    key = getattr(other_ops, "_cache_id")("rpa3", dim, None, None)
    rpa_op = getattr(other_ops, "_OP_CACHE")[key]
    toks = 2048 if dim == 64 else 4
    o_m, c_m = rpa_op(
        torch.empty(toks, 2, dim, dtype=torch.bfloat16, device="meta"),
        torch.empty(toks, 2, dim, dtype=torch.bfloat16, device="meta"),
        torch.empty(toks, 2, dim, dtype=torch.bfloat16, device="meta"),
        torch.empty(c_shape, dtype=torch.bfloat16, device="meta"),
        torch.empty(4, dtype=torch.int32, device="meta"),
        torch.empty(8, dtype=torch.int32, device="meta"),
        torch.empty(5, dtype=torch.int32, device="meta"),
        torch.empty(3, dtype=torch.int32, device="meta"),
    )
    assert o_m.shape == (toks, 2, dim) and c_m.shape == c_shape
    print(f"Verified RPAv3 dim={dim} TPU MLIR export:", o_m.shape, c_m.shape)


def main() -> None:
  """Verifies installed versions and Pallas Mosaic TPU MLIR lowering."""
  print("torch:", torch.__version__)
  for pkg in (
      "torch_tpu",
      "libtpu",
      "jax",
      "jaxlib",
      "tokamax",
      "vllm_torchtpu_core",
      "jax_flash_attn2",
      "xprof",
  ):
    print(f"{pkg}=={md.version(pkg)}")
  _verify_attention_mlir_export()
  _verify_moe_and_paged_mlir_export()
  standalone_elems = standalone_custom_pallas_op.export_tpu_mlir_smoke()
  print("Verified standalone custom Pallas op export:", standalone_elems)


if __name__ == "__main__":
  main()
