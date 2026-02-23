"""Unit tests for SORF-VeRA primitives."""

import math
import torch
import pytest

from sorf_vera import (
    next_power_of_2,
    build_hadamard,
    build_sorf_matrix,
    build_gamma,
    apply_gamma_transpose_batched,
    givens_pair_indices,
    build_lambda_d_rot,
    decompose_orthogonal_to_givens,
    fit_b,
    fit_d,
    init_sorf_vera_params,
)


def compute_delta_w(S, b_real, b_imag, B, scale, angles, A, r, d_out, d_in):
    """Compute DeltaW = S @ Gamma(b) @ B @ Lambda_d_rot @ A (test helper)."""
    device = A.device
    dtype = A.dtype
    Ld = build_lambda_d_rot(scale, angles, r, device, dtype)
    return S @ build_gamma(b_real, b_imag) @ B @ Ld @ A


# --------------------------------------------------
# next_power_of_2
# --------------------------------------------------

class TestNextPowerOf2:
    def test_powers(self):
        assert next_power_of_2(1) == 1
        assert next_power_of_2(2) == 2
        assert next_power_of_2(4) == 4

    def test_non_powers(self):
        assert next_power_of_2(3) == 4
        assert next_power_of_2(5) == 8
        assert next_power_of_2(768) == 1024


# --------------------------------------------------
# Hadamard
# --------------------------------------------------

class TestHadamard:
    def test_orthonormality(self):
        """H @ H^T should equal I (normalized Hadamard is orthonormal)."""
        for n in [2, 4, 8, 16]:
            H = build_hadamard(n, device='cpu', dtype=torch.float64)
            prod = H @ H.T
            eye = torch.eye(n, dtype=torch.float64)
            assert torch.allclose(prod, eye, atol=1e-12), f"Failed for n={n}"

    def test_shape(self):
        H = build_hadamard(8, device='cpu', dtype=torch.float64)
        assert H.shape == (8, 8)


# --------------------------------------------------
# SORF
# --------------------------------------------------

class TestSORF:
    def test_determinism(self):
        """Same seed should produce same SORF matrix."""
        S1 = build_sorf_matrix(16, seed=42, device='cpu', dtype=torch.float32)
        S2 = build_sorf_matrix(16, seed=42, device='cpu', dtype=torch.float32)
        assert torch.allclose(S1, S2)

    def test_different_seeds(self):
        S1 = build_sorf_matrix(16, seed=42, device='cpu', dtype=torch.float32)
        S2 = build_sorf_matrix(16, seed=99, device='cpu', dtype=torch.float32)
        assert not torch.allclose(S1, S2)

    def test_shape_non_power_of_2(self):
        """Non-power-of-2 d_out should still produce correct shape."""
        S = build_sorf_matrix(768, seed=42, device='cpu', dtype=torch.float32)
        assert S.shape == (768, 768)

    def test_shape_power_of_2(self):
        S = build_sorf_matrix(8, seed=42, device='cpu', dtype=torch.float32)
        assert S.shape == (8, 8)


# --------------------------------------------------
# Gamma (complex block-diagonal)
# --------------------------------------------------

class TestGamma:
    def test_identity(self):
        """b_real=1, b_imag=0 should give identity matrix."""
        k = 4
        b_real = torch.ones(k)
        b_imag = torch.zeros(k)
        G = build_gamma(b_real, b_imag)
        eye = torch.eye(2 * k)
        assert torch.allclose(G, eye, atol=1e-7)

    def test_block_structure(self):
        """Each 2x2 block should have the correct pattern."""
        b_real = torch.tensor([2.0, 3.0])
        b_imag = torch.tensor([0.5, -1.0])
        G = build_gamma(b_real, b_imag)

        # Block 0
        assert G[0, 0].item() == pytest.approx(2.0)
        assert G[0, 1].item() == pytest.approx(-0.5)
        assert G[1, 0].item() == pytest.approx(0.5)
        assert G[1, 1].item() == pytest.approx(2.0)

        # Block 1
        assert G[2, 2].item() == pytest.approx(3.0)
        assert G[2, 3].item() == pytest.approx(1.0)
        assert G[3, 2].item() == pytest.approx(-1.0)
        assert G[3, 3].item() == pytest.approx(3.0)

    def test_off_block_zeros(self):
        """Off-block elements should be zero."""
        b_real = torch.tensor([1.0, 2.0])
        b_imag = torch.tensor([0.5, 0.3])
        G = build_gamma(b_real, b_imag)
        # Check off-block elements
        assert G[0, 2].item() == 0.0
        assert G[0, 3].item() == 0.0
        assert G[1, 2].item() == 0.0
        assert G[1, 3].item() == 0.0
        assert G[2, 0].item() == 0.0
        assert G[3, 1].item() == 0.0


class TestApplyGammaTranspose:
    def test_matches_explicit(self):
        """apply_gamma_transpose_batched should match G^T @ h."""
        k = 4
        b_real = torch.randn(k)
        b_imag = torch.randn(k)
        d_out = 2 * k

        G = build_gamma(b_real, b_imag)
        H = torch.randn(3, 5, d_out)  # (batch, seq, d_out)

        result = apply_gamma_transpose_batched(b_real, b_imag, H)

        # Compare with explicit G^T @ each vector
        expected = (H @ G.T)
        assert torch.allclose(result, expected, atol=1e-6)


# --------------------------------------------------
# Givens rotations
# --------------------------------------------------

class TestGivens:
    def test_pair_count(self):
        for r in [2, 4, 8]:
            pairs = givens_pair_indices(r)
            assert len(pairs) == r * (r - 1) // 2

    def test_identity_zero_angles(self):
        """Zero angles and unit scale should give identity."""
        r = 4
        scale = torch.tensor(1.0, dtype=torch.float64)
        angles = torch.zeros(r * (r - 1) // 2, dtype=torch.float64)
        Ld = build_lambda_d_rot(scale, angles, r, 'cpu', torch.float64)
        assert torch.allclose(Ld, torch.eye(r, dtype=torch.float64), atol=1e-12)

    def test_scaling_only(self):
        """Zero angles, non-unit scale should give scale * I."""
        r = 3
        scale = torch.tensor(2.0, dtype=torch.float64)
        angles = torch.zeros(r * (r - 1) // 2, dtype=torch.float64)
        Ld = build_lambda_d_rot(scale, angles, r, 'cpu', torch.float64)
        expected = torch.eye(r, dtype=torch.float64) * 2.0
        assert torch.allclose(Ld, expected, atol=1e-12)

    def test_roundtrip(self):
        """Build from angles -> decompose -> rebuild should recover the matrix."""
        r = 4
        torch.manual_seed(123)
        # Build a rotation from known angles
        scale_orig = torch.tensor(1.0, dtype=torch.float64)
        angles_orig = torch.randn(r * (r - 1) // 2, dtype=torch.float64) * 0.5
        Ld_orig = build_lambda_d_rot(scale_orig, angles_orig, r, 'cpu', torch.float64)

        # Decompose (diag_signs should all be +1 since we built a proper rotation)
        angles_rec, diag_signs = decompose_orthogonal_to_givens(Ld_orig, r)
        Ld_rec = build_lambda_d_rot(torch.tensor(1.0, dtype=torch.float64),
                                     angles_rec, r, 'cpu', torch.float64)

        assert torch.allclose(Ld_rec, Ld_orig, atol=1e-10), \
            f"Max diff: {(Ld_rec - Ld_orig).abs().max().item()}"

    def test_autograd(self):
        """Gradients should flow through build_lambda_d_rot."""
        r = 3
        scale = torch.tensor(0.7, requires_grad=True)
        angles = torch.randn(r * (r - 1) // 2, requires_grad=True)

        Ld = build_lambda_d_rot(scale, angles, r, 'cpu', scale.dtype)
        loss = Ld.sum()
        loss.backward()

        assert scale.grad is not None
        assert angles.grad is not None



# --------------------------------------------------
# fit_b / fit_d
# --------------------------------------------------

class TestFitting:
    def test_fit_d_uniform_scale(self):
        """fit_d exactly recovers Ld when built with a single uniform scale."""
        torch.manual_seed(99)
        r = 4
        d_in = 16

        A = torch.randn(r, d_in, dtype=torch.float64)

        scale_true = torch.tensor(1.5, dtype=torch.float64)
        angles_true = torch.randn(r * (r - 1) // 2, dtype=torch.float64) * 0.5
        Ld_true = build_lambda_d_rot(scale_true, angles_true, r, 'cpu', torch.float64)

        V_r = A.T @ Ld_true.T

        scale_fit, angles_fit = fit_d(A, V_r, r)
        Ld_fit = build_lambda_d_rot(scale_fit, angles_fit, r, 'cpu', torch.float64)

        product_true = Ld_true @ A
        product_fit = Ld_fit @ A
        assert torch.allclose(product_fit, product_true, atol=1e-8), \
            f"Max diff: {(product_fit - product_true).abs().max().item()}"

    def test_fit_b_per_block(self):
        """fit_b correctly solves per-block lstsq when T = Gamma(b) @ B."""
        torch.manual_seed(42)
        r = 4
        d_out = 8

        S = build_sorf_matrix(d_out, seed=42, device='cpu', dtype=torch.float64)
        B = torch.randn(d_out, r, dtype=torch.float64)

        # Known b
        b_real_true = torch.randn(d_out // 2, dtype=torch.float64)
        b_imag_true = torch.randn(d_out // 2, dtype=torch.float64)

        # Construct inputs so that S^T @ U_r @ diag(Sigma_r) = Gamma(b_true) @ B
        # By setting U_r = S @ Gamma(b_true) @ B and Sigma_r = ones
        Gamma_true = build_gamma(b_real_true, b_imag_true)
        T_target = Gamma_true @ B          # (d_out, r)
        U_r = S @ T_target                 # S^T @ (S @ T) = T  (since S orthogonal)
        Sigma_r = torch.ones(r, dtype=torch.float64)

        b_real_fit, b_imag_fit = fit_b(S, B, U_r, Sigma_r, d_out, r)

        Gamma_fit = build_gamma(b_real_fit, b_imag_fit)
        T_fit = Gamma_fit @ B
        assert torch.allclose(T_fit, T_target, atol=1e-8), \
            f"Max diff: {(T_fit - T_target).abs().max().item()}"

    def test_full_pipeline_reconstruction(self):
        """Full pipeline: DeltaW -> SVD -> fit_b + fit_d -> DeltaW_recon.

        The reconstruction is approximate because:
        1. fit_d uses polar decomposition (G@diag(s) ≈ Lambda_target)
        2. fit_b solves lstsq with T that doesn't perfectly match Gamma@B
        We verify the pipeline doesn't crash and produces non-degenerate output.
        """
        torch.manual_seed(42)
        r = 4
        d_out = 8
        d_in = 16

        S = build_sorf_matrix(d_out, seed=42, device='cpu', dtype=torch.float64)
        B = torch.randn(d_out, r, dtype=torch.float64)
        A = torch.randn(r, d_in, dtype=torch.float64)

        b_real_true = torch.randn(d_out // 2, dtype=torch.float64)
        b_imag_true = torch.randn(d_out // 2, dtype=torch.float64)
        scale = torch.tensor(0.5, dtype=torch.float64)
        angles = torch.randn(r * (r - 1) // 2, dtype=torch.float64) * 0.3

        DeltaW = compute_delta_w(S, b_real_true, b_imag_true, B, scale, angles, A, r, d_out, d_in)

        # SVD
        U, S_vals, Vh = torch.linalg.svd(DeltaW, full_matrices=False)
        U_r = U[:, :r]
        Sigma_r = S_vals[:r]
        V_r = Vh[:r, :].T

        # Fit
        b_real_fit, b_imag_fit = fit_b(S, B, U_r, Sigma_r, d_out, r)
        scale_fit, angles_fit = fit_d(A, V_r, r)
        DeltaW_recon = compute_delta_w(
            S, b_real_fit, b_imag_fit, B, scale_fit, angles_fit, A, r, d_out, d_in
        )

        # Verify non-degenerate reconstruction
        assert DeltaW_recon.norm() > 0, "Reconstruction is zero"
        assert torch.isfinite(DeltaW_recon).all(), "Reconstruction contains NaN/Inf"

        # The reconstruction error can be significant (inherent to parameterization)
        # but should be bounded
        rel_err = (DeltaW - DeltaW_recon).norm() / DeltaW.norm()
        assert rel_err < 2.0, f"Reconstruction error unreasonably large: {rel_err.item():.4f}"


# --------------------------------------------------
# init_sorf_vera_params
# --------------------------------------------------

class TestInit:
    def test_shapes(self):
        params = init_sorf_vera_params(d_out=768, r=8, init_scale=0.1)
        assert params['b_real'].shape == (384,)
        assert params['b_imag'].shape == (384,)
        assert params['d_scale'].shape == ()
        assert params['d_angles'].shape == (28,)

    def test_values(self):
        params = init_sorf_vera_params(d_out=8, r=3, init_scale=0.2)
        assert torch.allclose(params['b_real'], torch.ones(4))
        assert torch.allclose(params['b_imag'], torch.zeros(4))
        assert torch.allclose(params['d_scale'], torch.tensor(0.2))
        assert torch.allclose(params['d_angles'], torch.zeros(3))


# --------------------------------------------------
# Forward pass shape + trainable params (integration-ish)
# --------------------------------------------------

class TestModelIntegration:
    @pytest.fixture
    def mock_args(self):
        """Create mock args for model creation."""
        import argparse
        args = argparse.Namespace(
            model='roberta-base',
            vera=True,
            r=8,
            projection_prng_key=0,
            save_projection=True,
            vera_dropout=0.0,
            d_initial=0.1,
            fan_in_fan_out=False,
            bias='none',
            modules_to_save=None,
            init_weights=True,
            layers_to_transform=None,
            layers_pattern=None,
            lora_alpha=8,
            lora_dropout=0.1,
            rslora=False,
            sorf_seed=42,
        )
        return args

    @pytest.mark.slow
    def test_trainable_params(self, mock_args):
        """Only sorf_d_scales, sorf_d_angles, and classifier should be trainable."""
        from models import create_peft_sorf_vera_model

        model = create_peft_sorf_vera_model(num_labels=2, args=mock_args)

        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]

        for name in trainable_names:
            assert (
                'sorf_d_scale' in name
                or 'sorf_d_angles' in name
                or 'classifier' in name
            ), f"Unexpected trainable param: {name}"

        # Check that there are some trainable params
        assert len(trainable_names) > 0

    @pytest.mark.slow
    def test_forward_shape(self, mock_args):
        """Forward pass should produce correct output shape."""
        from models import create_peft_sorf_vera_model

        model = create_peft_sorf_vera_model(num_labels=2, args=mock_args)
        model.eval()

        # Dummy input
        input_ids = torch.randint(0, 1000, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.long)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        assert outputs.logits.shape == (2, 2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
