"""
Correctness and gradient sanity tests for DoRALinear.

Primary tests (CPU, no external deps):
  - init invariant: W' = W0 at step 0  (B=0, m = ||W0||_c)
  - forward matches manual Eq.(5) computation
  - gradients flow only through m, lora_A, lora_B; base.weight is frozen
  - detach trick: denominator norm treated as constant during backward
  - merge / unmerge round-trip

peft comparison (skipped if peft is unavailable, dropout=0):
  - outputs match peft.LoraConfig(use_dora=True)
  - gradients on m, lora_A, lora_B match peft
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from code.dora_layers import DoRALinear, apply_dora_to_module

# Small dimensions so tests are fast on CPU.
IN_F, OUT_F, RANK = 8, 6, 2
ALPHA = 4.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fresh_linear(bias: bool = False, seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(IN_F, OUT_F, bias=bias)


def _fresh_dora(bias: bool = False, dropout: float = 0.0, seed: int = 0) -> DoRALinear:
    return DoRALinear(_fresh_linear(bias=bias, seed=seed), rank=RANK, alpha=ALPHA, dropout=dropout)


def _x(batch: int = 4, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, IN_F)


def _nonzero_dora(seed: int = 0) -> DoRALinear:
    d = _fresh_dora(seed=seed)
    torch.manual_seed(7)
    d.lora_A.data = torch.randn_like(d.lora_A)
    d.lora_B.data = torch.randn_like(d.lora_B)
    return d


# ---------------------------------------------------------------------------
# Init invariant
# ---------------------------------------------------------------------------

class TestInit:
    def test_forward_equals_base_at_init(self):
        """B=0 at init => W_eff = W0 => W' = W0 => output matches base layer."""
        d = _fresh_dora()
        x = _x()
        with torch.no_grad():
            assert torch.allclose(d(x), d.base(x), atol=1e-5)

    def test_magnitude_equals_base_row_norms(self):
        d = _fresh_dora()
        expected = d.base.weight.norm(p=2, dim=1)
        assert torch.allclose(d.m.detach(), expected, atol=1e-6)

    def test_lora_b_zero_at_init(self):
        d = _fresh_dora()
        assert d.lora_B.abs().max().item() == 0.0

    def test_not_merged_at_init(self):
        assert not _fresh_dora().merged


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------

class TestForward:
    def test_matches_manual_eq5(self):
        """Output equals W' x where W' = m * (W0 + s*BA) / ||W0 + s*BA||_c (row-wise)."""
        d = _nonzero_dora()
        x = _x()
        with torch.no_grad():
            W0 = d.base.weight
            W_eff = W0 + d.scaling * (d.lora_B @ d.lora_A)
            c = W_eff.norm(p=2, dim=1, keepdim=True)
            W_prime = (d.m.unsqueeze(1) / c) * W_eff
            expected = F.linear(x, W_prime, d.base.bias)
            got = d(x)
        assert torch.allclose(got, expected, atol=1e-5), \
            f"max diff = {(got - expected).abs().max():.2e}"

    def test_merged_equals_unmerged(self):
        d = _nonzero_dora()
        x = _x()
        with torch.no_grad():
            unmerged = d(x).clone()
            d.merge()
            merged = d(x)
        assert torch.allclose(merged, unmerged, atol=1e-5)

    def test_bias_passed_through(self):
        d = _fresh_dora(bias=True)
        x = _x()
        with torch.no_grad():
            out = d(x)
        assert out.shape == (x.shape[0], OUT_F)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradients:
    def test_trainable_params_get_grads(self):
        d = _nonzero_dora()
        d(_x()).sum().backward()
        assert d.m.grad is not None,      "m.grad is None"
        assert d.lora_A.grad is not None, "lora_A.grad is None"
        assert d.lora_B.grad is not None, "lora_B.grad is None"

    def test_base_weight_frozen_no_grad(self):
        d = _nonzero_dora()
        d(_x()).sum().backward()
        assert not d.base.weight.requires_grad
        assert d.base.weight.grad is None

    def test_trainable_grads_nonzero(self):
        """Grads must carry real signal (not accidentally all-zero)."""
        d = _nonzero_dora()
        d(_x()).sum().backward()
        assert d.m.grad.abs().sum().item() > 0,      "m.grad is all-zero"
        assert d.lora_A.grad.abs().sum().item() > 0, "lora_A.grad is all-zero"
        assert d.lora_B.grad.abs().sum().item() > 0, "lora_B.grad is all-zero"

    def test_detach_trick_does_not_block_grad(self):
        """Grads on lora_A/B must flow through the numerator even with .detach() on the norm."""
        d = _nonzero_dora()
        d(_x()).sum().backward()
        # Numerator W_eff = W0 + s*BA carries grad to lora_A and lora_B.
        assert d.lora_A.grad is not None
        assert d.lora_B.grad is not None


# ---------------------------------------------------------------------------
# Merge / unmerge
# ---------------------------------------------------------------------------

class TestMergeUnmerge:
    def test_merge_sets_flag(self):
        d = _nonzero_dora(); d.merge()
        assert d.merged

    def test_unmerge_clears_flag(self):
        d = _nonzero_dora(); d.merge(); d.unmerge()
        assert not d.merged

    def test_unmerge_restores_base_weight(self):
        d = _nonzero_dora()
        original = d.base.weight.detach().clone()
        d.merge()
        d.unmerge()
        assert torch.allclose(d.base.weight, original, atol=1e-6)

    def test_output_stable_after_round_trip(self):
        d = _nonzero_dora()
        x = _x()
        with torch.no_grad():
            before = d(x).clone()
            d.merge(); d.unmerge()
            after = d(x)
        assert torch.allclose(after, before, atol=1e-5)

    def test_double_merge_noop(self):
        d = _nonzero_dora(); d.merge()
        x = _x()
        with torch.no_grad():
            out1 = d(x).clone()
        d.merge()
        with torch.no_grad():
            out2 = d(x)
        assert torch.allclose(out1, out2)

    def test_unmerge_before_merge_is_noop(self):
        """unmerge() before merge() should return silently (guard: not self.merged)."""
        d = _fresh_dora()
        original = d.base.weight.detach().clone()
        d.unmerge()  # should not raise
        assert not d.merged
        assert torch.allclose(d.base.weight, original)


# ---------------------------------------------------------------------------
# apply_dora_to_module
# ---------------------------------------------------------------------------

class _TwoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.fc1 = nn.Linear(IN_F, OUT_F)
        self.fc2 = nn.Linear(OUT_F, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestApplyDoraToModule:
    def test_replaces_both_targets(self):
        m = _TwoLayer()
        apply_dora_to_module(m, ["fc1", "fc2"], RANK, ALPHA)
        assert isinstance(m.fc1, DoRALinear)
        assert isinstance(m.fc2, DoRALinear)

    def test_leaves_non_target_intact(self):
        m = _TwoLayer()
        apply_dora_to_module(m, ["fc1"], RANK, ALPHA)
        assert isinstance(m.fc1, DoRALinear)
        assert isinstance(m.fc2, nn.Linear)

    def test_base_weights_frozen(self):
        m = _TwoLayer()
        apply_dora_to_module(m, ["fc1", "fc2"], RANK, ALPHA)
        assert not m.fc1.base.weight.requires_grad
        assert not m.fc2.base.weight.requires_grad

    def test_trainable_param_count(self):
        m = _TwoLayer()
        _, trainable = apply_dora_to_module(m, ["fc1", "fc2"], RANK, ALPHA)
        # fc1: m(OUT_F) + A(RANK*IN_F) + B(OUT_F*RANK)
        # fc2: m(4)     + A(RANK*OUT_F) + B(4*RANK)
        expected = (OUT_F + RANK * IN_F + OUT_F * RANK) + (4 + RANK * OUT_F + 4 * RANK)
        assert sum(p.numel() for p in trainable) == expected

    def test_trainable_list_excludes_base_weight(self):
        m = _TwoLayer()
        _, trainable = apply_dora_to_module(m, ["fc1"], RANK, ALPHA)
        ids = {id(p) for p in trainable}
        assert id(m.fc1.base.weight) not in ids

    def test_forward_still_works_after_injection(self):
        m = _TwoLayer()
        apply_dora_to_module(m, ["fc1", "fc2"], RANK, ALPHA)
        out = m(_x())
        assert out.shape == (4, 4)


# ---------------------------------------------------------------------------
# peft DoRA comparison (skipped if peft unavailable)
# ---------------------------------------------------------------------------

try:
    import peft as _peft_lib
    _HAS_PEFT = True
except ImportError:
    _peft_lib = None
    _HAS_PEFT = False


class _SingleFC(nn.Module):
    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.fc = lin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make_peft_dora(base_linear: nn.Linear):
    """Wrap base_linear in a minimal model and apply peft DoRA (dropout=0)."""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=RANK, lora_alpha=ALPHA, lora_dropout=0.0,
        target_modules=["fc"], bias="none", use_dora=True,
    )
    return get_peft_model(_SingleFC(base_linear), cfg)


def _sync_peft_params(peft_fc, scratch: DoRALinear) -> None:
    """Copy m / lora_A / lora_B from scratch into the peft DoRA fc layer."""
    with torch.no_grad():
        peft_fc.lora_A["default"].weight.copy_(scratch.lora_A)
        peft_fc.lora_B["default"].weight.copy_(scratch.lora_B)
        peft_fc.lora_magnitude_vector["default"].copy_(scratch.m)


@pytest.mark.skipif(not _HAS_PEFT, reason="peft not installed — skipping peft comparison tests")
class TestPeftDoRAComparison:
    """Scratch DoRA outputs and gradients must match peft DoRA at identical weights."""

    @staticmethod
    def _setup():
        """Return (scratch, peft_model, peft_fc) with identical weights."""
        torch.manual_seed(0)
        base1 = nn.Linear(IN_F, OUT_F, bias=False)
        scratch = DoRALinear(base1, rank=RANK, alpha=ALPHA, dropout=0.0)
        torch.manual_seed(7)
        scratch.lora_A.data = torch.randn_like(scratch.lora_A)
        scratch.lora_B.data = torch.randn_like(scratch.lora_B)

        # Identical base weights via same seed.
        torch.manual_seed(0)
        base2 = nn.Linear(IN_F, OUT_F, bias=False)
        peft_model = _make_peft_dora(base2)
        peft_fc = peft_model.base_model.model.fc
        _sync_peft_params(peft_fc, scratch)

        return scratch, peft_model, peft_fc

    def test_outputs_match(self):
        scratch, peft_model, _ = self._setup()
        x = _x()
        with torch.no_grad():
            out_s = scratch(x)
            out_p = peft_model(x)
        assert torch.allclose(out_s, out_p, atol=1e-4), \
            f"max output diff = {(out_s - out_p).abs().max():.2e}"

    def test_gradients_match(self):
        scratch, peft_model, peft_fc = self._setup()
        x1 = _x(); x2 = x1.clone()

        scratch(x1).sum().backward()
        peft_model(x2).sum().backward()

        def _chk(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
            assert a is not None and b is not None, f"{name} grad is None"
            assert torch.allclose(a, b, atol=1e-4), \
                f"{name} grad max diff = {(a - b).abs().max():.2e}"

        _chk("m",      scratch.m.grad,      peft_fc.lora_magnitude_vector["default"].grad)
        _chk("lora_A", scratch.lora_A.grad, peft_fc.lora_A["default"].weight.grad)
        _chk("lora_B", scratch.lora_B.grad, peft_fc.lora_B["default"].weight.grad)
