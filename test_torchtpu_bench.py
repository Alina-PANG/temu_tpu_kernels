#!/usr/bin/env python3
"""Parameterized unit and hygiene tests for the TorchTPU kernel bench suite.

Follows Google unit-testing style with parameterized test generation and a
friendly test-name generator so test failures clearly identify the scenario.
"""

from __future__ import annotations

import importlib
import json
import pathlib
import tempfile
from typing import Any
import unittest

import torch

import attn_ops
import other_ops
import torchtpu_kernel_bench as kb

_PKG_DIR = pathlib.Path(__file__).resolve().parent

# Hex-encoded internal markers so literal strings do not appear in repo files.
_BANNED_TOKENS = tuple(
    bytes.fromhex(h).decode("ascii")
    for h in (
        "786c615f7470755f647666735f705f7374617465",
        "626f6461626f7267",
        "7562656e63682d6c6f6773",
        "636c6f75642d7470752d7368617265642d6361706163697479",
        "736f757263652e636f72702e676f6f676c652e636f6d",
        "616c696e6170687a2d7370656369616c697374",
        "7765696b756f2d6473763431",
        "6d6c2d776f726b6c6f61642d62656e63686d61726b73",
    )
)


def _friendly_case_name(
    testcase_cls: type[Any], idx: int, params: dict[str, Any]
) -> str:
  """Generates readable parameterized test method names."""
  del testcase_cls
  label = params.get("name", f"case_{idx}")
  clean = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(label))
  return f"test_{clean}"


def _parameterize(cases: list[dict[str, Any]]):
  """Lightweight parameterized decorator with friendly test naming."""

  def decorator(func):
    def wrapper_factory(params: dict[str, Any]):
      def test_method(self):
        kwargs = {k: v for k, v in params.items() if k != "name"}
        return func(self, **kwargs)

      return test_method

    func._param_cases = [  # pylint: disable=protected-access
        (_friendly_case_name(object, i, p), wrapper_factory(p))
        for i, p in enumerate(cases)
    ]
    return func

  return decorator


class _ParamMeta(type(unittest.TestCase)):
  """Expands `_param_cases` methods on `unittest.TestCase` subclasses."""

  def __new__(mcs, name, bases, namespace):
    expanded = {}
    for attr_name, val in list(namespace.items()):
      param_cases = getattr(val, "_param_cases", None)
      if param_cases is not None:
        del namespace[attr_name]
        for friendly_name, fn in param_cases:
          expanded[f"{attr_name}__{friendly_name}"] = fn
    namespace.update(expanded)
    return super().__new__(mcs, name, bases, namespace)


class TorchTpuBenchTest(unittest.TestCase, metaclass=_ParamMeta):
  """Unit tests for standalone backward timing, precision, and hygiene."""

  def test_standalone_custom_pallas_example_cpu(self) -> None:
    """Runs the standalone Pallas op example on CPU when JAX is installed."""
    try:
      demo = importlib.import_module("examples.standalone_custom_pallas_op")
    except ImportError:
      self.skipTest("JAX not installed in host Python; tested in Docker build.")
    demo.main(["--device", "cpu"])

  @_parameterize([
      {"name": "sdpa_flash_cpu", "impl": "torchtpu_flash", "dim": 64},
      {"name": "dpa_cpu", "impl": "dpa", "dim": 128},
      {"name": "ref_cpu", "impl": "ref", "dim": 64},
  ])
  def test_standalone_bwd_does_not_rerun_fwd(
      self, impl: str, dim: int
  ) -> None:
    """Verifies `bwd_setup` runs forward once and `bwd_step` matches `fp32`."""
    torch.manual_seed(7)
    q = torch.randn(2, 4, 64, dim, dtype=torch.float32)
    k = torch.randn(2, 4, 64, dim, dtype=torch.float32)
    v = torch.randn(2, 4, 64, dim, dtype=torch.float32)
    do = torch.randn(2, 4, 64, dim, dtype=torch.float32)

    fwd_fn, bwd_setup = attn_ops.build_attention_ops(
        impl, seq=64, heads=4, kv_heads=4, dim=dim, blocks=None,
        device_type="cpu",
    )
    out = fwd_fn(q, k, v)
    bwd_step = bwd_setup(q, k, v, do)
    # Run standalone backward multiple times to verify repeatability without
    # `retain_graph` exhaustion errors.
    dq1, dk1, dv1 = bwd_step()
    dq2, dk2, dv2 = bwd_step()

    ref_out, ref_grads = attn_ops.reference_attention_fp32(q, k, v, do)
    assert ref_grads is not None
    torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dq1, ref_grads[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dk1, ref_grads[1], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dv1, ref_grads[2], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dq1, dq2)
    torch.testing.assert_close(dk1, dk2)
    torch.testing.assert_close(dv1, dv2)

  @_parameterize([
      {"name": "torch_rms_norm", "impl": "torch_rms_norm"},
      {"name": "rmsnorm_f32acc", "impl": "rmsnorm_f32acc"},
      {"name": "rmsnorm_bf16", "impl": "rmsnorm_bf16"},
  ])
  def test_rmsnorm_standalone_vjp_matches_autograd(self, impl: str) -> None:
    """Verifies standalone RMSNorm VJP matches standard PyTorch autograd."""
    torch.manual_seed(11)
    x = torch.randn(16, 64, dtype=torch.float32)
    w = torch.randn(64, dtype=torch.float32)
    dy = torch.randn(16, 64, dtype=torch.float32)

    fwd_fn, bwd_setup = other_ops.build_rmsnorm_ops(impl, 64, "cpu")
    bwd_step = bwd_setup(x, w, dy)
    dx, dw = bwd_step()

    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    out = fwd_fn(xr, wr)
    ref_dx, ref_dw = torch.autograd.grad(out, (xr, wr), dy)
    torch.testing.assert_close(dx, ref_dx, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dw, ref_dw, rtol=1e-4, atol=1e-4)

  @_parameterize([
      {"name": "fp32_exact", "dtype": torch.float32, "tol": 1e-6},
      {"name": "bf16_exact", "dtype": torch.bfloat16, "tol": 0.0},
  ])
  def test_moe_weighted_before_unsort_is_exact(
      self, dtype: torch.dtype, tol: float
  ) -> None:
    """Verifies `e2e_opt` weighted-before-unsort matches `e2e` output."""
    cases, chk = other_ops.build_moe_suite(
        "torchtpu_ragged_dot",
        tokens=64,
        hidden=32,
        inter=16,
        experts=4,
        top_k=2,
        dtype=dtype,
        device=torch.device("cpu"),
    )
    self.assertLessEqual(chk["reduce_opt_max_abs_diff"], tol)
    stage_map = {name: fn for name, fn, _ in cases}
    out_base = stage_map["e2e"]()
    out_opt = stage_map["e2e_opt"]()
    torch.testing.assert_close(out_base, out_opt, rtol=tol, atol=tol)

  @_parameterize([
      {
          "name": "smoke_all_suites",
          "kernel": "temu_table,attention,paged,rmsnorm,moe",
      },
      {"name": "filtered_dpa_only", "kernel": "temu_table"},
  ])
  def test_cli_smoke_emits_valid_jsonl(self, kernel: str) -> None:
    """Runs `main()` on CPU and validates emitted `results.jsonl` records."""
    with tempfile.TemporaryDirectory() as tmp:
      argv = [
          "--device", "cpu",
          "--preset", "smoke",
          "--kernel", kernel,
          "--iters", "2",
          "--warmup", "1",
          "--trace_mode", "none",
          "--out", tmp,
      ]
      if kernel == "temu_table":
        argv.extend(["--impls", "dpa,ref"])
      kb.main(argv)
      jsonl = pathlib.Path(tmp) / "results.jsonl"
      self.assertTrue(jsonl.is_file())
      rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
      self.assertGreater(len(rows), 0)
      for r in rows:
        self.assertNotIn("error", r, msg=f"Unexpected error in {r}")
        self.assertGreater(r["primary_ms_p50"], 0.0)

  @_parameterize([
      {
          "name": str(p.relative_to(_PKG_DIR)),
          "file_path": p,
          "check_80col": p.suffix in (".py", ".sh", ".yaml", ".md")
          or p.name == "Dockerfile",
      }
      for p in sorted(_PKG_DIR.rglob("*"))
      if p.is_file()
      and "__pycache__" not in p.parts
      and ".git" not in p.parts
      and (
          p.suffix in (".py", ".sh", ".yaml", ".md", ".jsonl")
          or p.name in ("Dockerfile", "LICENSE")
      )
      and p.name != "test_torchtpu_bench.py"
  ])
  def test_external_hygiene_and_80col_limit(
      self, file_path: pathlib.Path, check_80col: bool
  ) -> None:
    """Ensures no internal flags/URLs leak and code/docs stay <= 80 chars."""
    text = file_path.read_text(encoding="utf-8")
    low = text.lower()
    for token in _BANNED_TOKENS:
      self.assertNotIn(
          token.lower(),
          low,
          msg=f"Banned internal token '{token}' found in {file_path}",
      )
    if check_80col:
      for lineno, line in enumerate(text.splitlines(), 1):
        # Allow GFM table rows (`| ... |`) and reference link URLs (`[id]: ...`)
        # in Markdown to exceed 80 chars so tables render cleanly on GitHub.
        if file_path.suffix == ".md" and (
            line.startswith("|") or line.startswith("[")
        ):
          continue
        self.assertLessEqual(
            len(line),
            80,
            msg=(
                f"{file_path.name}:{lineno} exceeds 80 chars"
                f" ({len(line)}): {line}"
            ),
        )


if __name__ == "__main__":
  unittest.main()
