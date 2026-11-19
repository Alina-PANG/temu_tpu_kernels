#!/usr/bin/env python3
"""Block-size and GQA packing ablation for RPAv3 head_dim=64 on TPU v7x."""

import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernel_bench as kb  # pylint: disable=g-import-not-at-top


def main() -> None:
  out_dir = "/results/rpa_tune"
  os.makedirs(out_dir, exist_ok=True)
  ctx = kb.Ctx(
      out_dir=out_dir,
      iters=15,
      warmup=3,
      trace_iters=3,
      dtype=jnp.dtype("bfloat16"),
      impls=None,
      scopes=None,
  )
  rpa_mod = "tpu_inference.kernels.ragged_paged_attention.v3.kernel_hd64"
  lib = kb._import(rpa_mod)  # pylint: disable=protected-access
  kernel = lib.ragged_paged_attention_hd64
  batch, ctx_len, dim, page_size = 64, 2048, 64, 256
  pages_per_seq = -(-ctx_len // page_size)
  num_pages = batch * pages_per_seq
  tokens = batch * ctx_len
  pairs = batch * (ctx_len * (ctx_len + 1) // 2)

  # Test MHA (hq=8, hkv=8, GQA=1 -> 2x q_packing pad) vs GQA-4 (hq=8, hkv=2).
  configs = [
      (8, 8, None, None),
      (8, 8, 8, 64),
      (8, 8, 8, 128),
      (8, 8, 8, 256),
      (8, 8, 4, 256),
      (8, 8, 8, 512),
      (8, 8, 4, 512),
      (8, 8, 2, 512),
      (8, 8, 4, 1024),
      (8, 2, None, None),
      (8, 2, 8, 128),
      (8, 2, 8, 256),
      (8, 2, 8, 512),
      (8, 2, 4, 512),
  ]
  for q_heads, kv_heads, bkvp, bq in configs:
    tag = "auto" if bq is None else f"bkvp{bkvp}_bq{bq}"
    name = f"rpa3_hd64_prefill_b64_ctx2048_hq{q_heads}_hkv{kv_heads}_{tag}"
    record = dict(
        kernel="paged_tune",
        case=name,
        impl="rpa_v3_hd64",
        mode="prefill",
        batch=batch,
        seq=ctx_len,
        heads=q_heads,
        kv_heads=kv_heads,
        head_dim=dim,
        page_size=page_size,
        bkvp=bkvp,
        bq=bq,
    )
    cache_shape = lib.get_kv_cache_shape(
        num_pages, page_size, kv_heads, dim, ctx.dtype
    )
    arrays = [
        kb.rand((tokens, q_heads, dim), ctx.dtype, 0),
        kb.rand((tokens, kv_heads, dim), ctx.dtype, 1),
        kb.rand((tokens, kv_heads, dim), ctx.dtype, 2),
        kb.rand(cache_shape, ctx.dtype, 3),
        np.full((batch,), ctx_len, np.int32),
        np.arange(num_pages, dtype=np.int32),
        (np.arange(batch + 1) * ctx_len).astype(np.int32),
        np.array([0, 0, batch], np.int32),
    ]
    flops = 4.0 * pairs * q_heads * dim

    def fn(
        q,
        k,
        v,
        cache,
        kv_lens,
        pages,
        cu_q_lens,
        dist,
        bkvp=bkvp,
        bq=bq,
    ):
      return kernel(
          q,
          k,
          v,
          cache,
          kv_lens,
          pages,
          cu_q_lens,
          dist,
          num_kv_pages_per_block=bkvp,
          num_queries_per_block=bq,
      )

    def feedback(call_args, out):
      return call_args[:3] + [out[1]] + call_args[4:]

    kb.time_case(ctx, record, fn, arrays, "core", (3,), feedback, flops=flops)

  kb.print_summary(ctx.results)


if __name__ == "__main__":
  main()
