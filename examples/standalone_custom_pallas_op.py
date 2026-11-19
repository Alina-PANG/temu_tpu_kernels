#!/usr/bin/env python3
"""Minimal standalone Pallas TPU kernel integration in PyTorch (zero vLLM).

Demonstrates how a custom serving or training engine can author a VMEM-tiled
TPU kernel in JAX Pallas and register it as a native PyTorch operator via
`torch.tpu.pallas.jax_op` (compatible with both eager mode and `torch.compile`).

When no TPU hardware is attached, `--device auto` runs the Pallas `interpret`
mode check and validates numerical parity against a pure `float32` PyTorch
reference on CPU.
"""

from __future__ import annotations

import argparse
import os

# Keep JAX in CPU trace-only mode so TorchTPU exclusively owns the TPU runtime.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# pylint: disable=g-import-not-at-top
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax._src import mesh as mesh_lib
import torch
# pylint: enable=g-import-not-at-top


def _fused_rms_scale_kernel(x_ref, w_ref, o_ref) -> None:
  """VMEM-resident fused RMSNorm + weight scale Pallas kernel."""
  x = x_ref[...].astype(jax.numpy.float32)
  rms = jax.lax.rsqrt(
      jax.numpy.mean(x * x, axis=-1, keepdims=True) + 1e-6
  )
  w = w_ref[0:1, :].astype(jax.numpy.float32)
  o_ref[...] = (x * rms * w).astype(o_ref.dtype)


def build_rms_scale_jax(interpret: bool = False):
  """Builds a JAX callable wrapping `_fused_rms_scale_kernel`."""

  def _fn(x: jax.Array, w: jax.Array) -> jax.Array:
    rows, dim = x.shape
    block_m = min(rows, 512)
    if rows % block_m != 0:
      raise ValueError(f"rows ({rows}) must be divisible by {block_m}.")
    w_tiled = jax.numpy.broadcast_to(w.reshape(1, dim), (8, dim))
    return pl.pallas_call(
        _fused_rms_scale_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, dim), x.dtype),
        grid=(rows // block_m,),
        in_specs=[
            pl.BlockSpec((block_m, dim), lambda i: (i, 0)),
            pl.BlockSpec((8, dim), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((block_m, dim), lambda i: (i, 0)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",)
        ),
        interpret=interpret,
    )(x, w_tiled)

  return _fn


def export_tpu_mlir_smoke(rows: int = 1024, dim: int = 1024) -> int:
  """Verifies Mosaic TPU MLIR lowering on CPU without a physical TPU."""
  import torch_tpu  # pylint: disable=g-import-not-at-top,unused-import

  abs_dev = mesh_lib.AbstractDevice(
      device_kind="TPU7x", num_cores=1, platform="tpu"
  )
  abs_mesh = mesh_lib.AbstractMesh((), (), abstract_device=abs_dev)

  def _jax_tpu_fn(x: jax.Array, w: jax.Array) -> jax.Array:
    with jax.sharding.use_abstract_mesh(abs_mesh):
      return build_rms_scale_jax(interpret=False)(x, w)

  tpu_op = torch.tpu.pallas.jax_op("demo::rms_scale", _jax_tpu_fn)
  xm = torch.empty((rows, dim), dtype=torch.bfloat16, device="meta")
  wm = torch.empty((1, dim), dtype=torch.bfloat16, device="meta")
  ym = tpu_op(xm, wm)
  assert ym.shape == (rows, dim)
  return rows * dim


def reference_rms_scale_fp32(
    x: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
  """Exact `float32` PyTorch reference implementation."""
  xf = x.float()
  rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
  return (xf * rstd * w.float()).to(x.dtype)


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument(
      "--device", default="auto", choices=("auto", "tpu", "cpu")
  )
  parser.add_argument("--rows", type=int, default=4096)
  parser.add_argument("--dim", type=int, default=4096)
  args = parser.parse_args(argv)

  use_tpu = False
  if args.device != "cpu":
    try:
      import torch_tpu  # pylint: disable=g-import-not-at-top,unused-import

      use_tpu = args.device == "tpu" or torch.accelerator.is_available()
    except ImportError:
      use_tpu = False

  if use_tpu:
    op = torch.tpu.pallas.jax_op(
        "custom::rms_scale", build_rms_scale_jax(interpret=False)
    )
    compiled_op = torch.compile(op, backend="tpu")
    x = torch.randn(args.rows, args.dim, dtype=torch.bfloat16, device="tpu")
    w = torch.randn(1, args.dim, dtype=torch.bfloat16, device="tpu")
    y = compiled_op(x, w)
    torch.accelerator.synchronize()
    ref = reference_rms_scale_fp32(x.cpu(), w.cpu())
    cos = torch.nn.functional.cosine_similarity(
        y.cpu().float().flatten(), ref.float().flatten(), dim=0
    ).item()
    print(f"TPU execution OK: shape={tuple(y.shape)} cos_sim_vs_fp32={cos:.6f}")
  else:
    x_np = torch.randn(256, 256, dtype=torch.float32)
    w_np = torch.randn(1, 256, dtype=torch.float32)
    y_jax = build_rms_scale_jax(interpret=True)(
        jax.numpy.asarray(x_np.numpy()), jax.numpy.asarray(w_np.numpy())
    )
    y_pt = torch.from_numpy(jax.numpy.asarray(y_jax).__array__().copy())
    ref = reference_rms_scale_fp32(x_np, w_np)
    torch.testing.assert_close(y_pt, ref, rtol=1e-5, atol=1e-5)
    print("CPU Pallas interpret parity OK (rtol=1e-5, atol=1e-5).")


if __name__ == "__main__":
  main()
