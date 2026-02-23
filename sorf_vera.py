"""
SORF-VeRA core math utilities.

Implements all primitives for the SORF-VeRA federated aggregation approach:
- SORF (Structured ORthogonal Random Features) matrix construction
- Complex block-diagonal Gamma(b) parameterization
- Givens rotation parameterization of orthogonal matrices
- Full DeltaW computation
- Server-side closed-form fitting (fit_b, fit_d)
"""

import math
import torch
import scipy.linalg


# --------------------------------------------------
# SORF construction
# --------------------------------------------------

def next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def build_hadamard(n, device, dtype):
    """Build a normalized Hadamard matrix of size n (must be power of 2).

    Returns (n, n) tensor equal to scipy.linalg.hadamard(n) / sqrt(n).
    """
    assert n%2 == 0 and n > 0, "Hadamard size must be a power of 2"
    H = scipy.linalg.hadamard(n).astype('float64')
    H = torch.tensor(H, device=device, dtype=dtype) / math.sqrt(n)
    return H


def _random_signs(size, generator):
    """Return a vector of +1/-1 with the given PRNG."""
    bits = torch.randint(0, 2, (size,), generator=generator)
    return 2.0 * bits.float() - 1.0


def build_sorf_matrix(d_out, seed, device, dtype):
    """Build SORF matrix S of effective shape (d_out, d_out).

    S = D1 @ H @ D2 @ H @ D3  where Di are random sign diagonals and
    H is a normalized Hadamard matrix.

    If d_out is not a power of 2, we pad to next_power_of_2(d_out),
    build the full SORF, then truncate to (d_out, d_out).
    """
    n = next_power_of_2(d_out)

    gen = torch.Generator()
    gen.manual_seed(seed)

    d1 = _random_signs(n, gen)
    d2 = _random_signs(n, gen)
    d3 = _random_signs(n, gen)

    H = build_hadamard(n, device='cpu', dtype=torch.float64)

    D1 = torch.diag(d1.double())
    D2 = torch.diag(d2.double())
    D3 = torch.diag(d3.double())

    S_full = D1 @ H @ D2 @ H @ D3
    S = S_full[:d_out, :d_out].to(device=device, dtype=dtype).contiguous()
    return S


# --------------------------------------------------
# Complex block-diagonal Gamma(b)
# --------------------------------------------------

def build_gamma(b_real, b_imag):
    """Build block-diagonal Gamma matrix from b_real and b_imag.

    Each pair (a_j, c_j) = (b_real[j], b_imag[j]) produces a 2x2 block:
        [[a_j, -c_j],
         [c_j,  a_j]]

    Args:
        b_real: (d_out//2,) tensor
        b_imag: (d_out//2,) tensor

    Returns:
        (d_out, d_out) block-diagonal matrix
    """
    k = b_real.shape[0]
    d_out = 2 * k
    G = torch.zeros(d_out, d_out, device=b_real.device, dtype=b_real.dtype)

    for j in range(k):
        a = b_real[j]
        c = b_imag[j]
        i = 2 * j
        G[i, i] = a
        G[i, i + 1] = -c
        G[i + 1, i] = c
        G[i + 1, i + 1] = a

    return G


def apply_gamma_transpose_batched(b_real, b_imag, H):
    """Compute H @ Gamma(b)^T in a vectorized way (right-multiply by Gamma^T).

    Gamma has blocks [[a, -c], [c, a]], so Gamma^T has blocks [[a, c], [-c, a]].
    For right-multiply w = H @ Gamma^T, per block j:
        w[..., 2j]   = a_j * H[..., 2j] - c_j * H[..., 2j+1]
        w[..., 2j+1] = c_j * H[..., 2j] + a_j * H[..., 2j+1]

    Args:
        b_real: (d_out//2,) tensor
        b_imag: (d_out//2,) tensor
        H: (..., d_out) tensor

    Returns:
        (..., d_out) tensor
    """
    even = H[..., 0::2]  # (..., d_out//2)
    odd = H[..., 1::2]   # (..., d_out//2)

    a = b_real  # (d_out//2,)
    c = b_imag  # (d_out//2,)

    new_even = a * even - c * odd
    new_odd = c * even + a * odd

    out = torch.empty_like(H)
    out[..., 0::2] = new_even
    out[..., 1::2] = new_odd
    return out


# --------------------------------------------------
# Givens rotations
# --------------------------------------------------

def givens_pair_indices(r):
    """Return all unique (p, q) plane pairs for Givens rotations in R^r.

    Each pair (p, q) with p < q identifies a 2D plane: the corresponding
    Givens rotation rotates only basis vectors e_p and e_q by some angle,
    leaving all others unchanged. Any r×r rotation matrix can be built from
    one rotation per plane, so the r*(r-1)/2 pairs here form the schedule of
    learnable angles used by build_lambda_d_rot.
    """
    pairs = []
    for p in range(r - 1):
        for q in range(p + 1, r):
            pairs.append((p, q))
    return pairs


def build_lambda_d_rot(scale, angles, r, device, dtype):
    """Build Lambda_d_rot = scale * G(angles).

    Applies Givens rotations (from angles) then uniform scaling by a single
    magnitude, matching the proof's parameterization: Λ_d = G(θ) · diag(m^{1/r}).
    Iteration is reversed relative to decompose_orthogonal_to_givens so that
    the roundtrip decompose -> build recovers the original orthogonal matrix.
    All operations use torch ops for autograd compatibility.

    Args:
        scale: scalar tensor — single uniform magnitude
        angles: (r*(r-1)//2,) tensor — Givens angles
        r: rank
        device, dtype: target device and dtype

    Returns:
        (r, r) tensor
    """
    # Start from identity
    M = torch.eye(r, device=device, dtype=dtype)

    pairs = givens_pair_indices(r)
    # Iterate in reverse order for consistency with decompose_orthogonal_to_givens
    for idx in reversed(range(len(pairs))):
        p, q = pairs[idx]
        theta = angles[idx]
        c = torch.cos(theta)
        s = torch.sin(theta)

        # Clone the rows to avoid in-place gradient corruption
        row_p = M[p].clone()
        row_q = M[q].clone()

        M = M.clone()
        M[p] = c * row_p - s * row_q
        M[q] = s * row_p + c * row_q

    # Uniform scaling: scale * G(angles)
    M = M * scale.to(device=device, dtype=dtype)

    return M


def decompose_orthogonal_to_givens(O, r):
    """Decompose an orthogonal matrix O into Givens angles + residual diagonal.

    Uses QR-style elimination: for each (p,q) pair in canonical order,
    compute theta = atan2(O[q,p], O[p,p]) and apply inverse rotation.
    After all pairs, the remaining matrix is diagonal with ±1 entries.

    Args:
        O: (r, r) orthogonal matrix
        r: size

    Returns:
        (angles, diag_signs):
            angles: (r*(r-1)//2,) tensor of Givens angles
            diag_signs: (r,) tensor of ±1 residual diagonal entries
    """
    W = O.clone()
    pairs = givens_pair_indices(r)
    angles = torch.zeros(len(pairs), device=O.device, dtype=O.dtype)

    for idx, (p, q) in enumerate(pairs):
        theta = torch.atan2(W[q, p], W[p, p])
        angles[idx] = theta

        # Apply inverse rotation to W (eliminate W[q,p])
        c = torch.cos(theta)
        s = torch.sin(theta)

        row_p = W[p].clone()
        row_q = W[q].clone()

        W[p] = c * row_p + s * row_q
        W[q] = -s * row_p + c * row_q

    diag_signs = torch.diag(W)
    return angles, diag_signs



# --------------------------------------------------
# Server-side fitting (closed-form)
# --------------------------------------------------

def fit_b(S, B, U_r, Sigma_r, d_out, r):
    """Fit b parameters (b_real, b_imag) given SVD factors.

    Target: S @ Gamma(b) @ B = U_r @ diag(Sigma_r)
    => Gamma(b) @ B = S^{-1} @ U_r @ diag(Sigma_r)

    We solve S @ T = U_r @ diag(Sigma_r) via linalg.solve (exact even
    when S is a truncated SORF that is not perfectly orthogonal), then
    fit per-block lstsq:
        For block j (rows 2j, 2j+1):
            T[2j:2j+2, :] = [[a_j, -c_j], [c_j, a_j]] @ B[2j:2j+2, :]
        Rearranging into lstsq: B_block^T @ [a_j, c_j]^T columns

    Args:
        S: (d_out, d_out) SORF matrix
        B: (d_out, r) shared VeRA B matrix
        U_r: (d_out, r) left singular vectors
        Sigma_r: (r,) singular values
        d_out: output dimension
        r: rank

    Returns:
        (b_real_new, b_imag_new) each (d_out//2,)
    """
    device = S.device
    dtype = S.dtype

    # T = S^{-1} @ (U_r @ diag(Sigma_r)),  shape (d_out, r)
    # Using linalg.solve instead of S.T to handle non-orthogonal
    # truncated SORF matrices (when d_out is not a power of 2).
    T = torch.linalg.solve(S, U_r * Sigma_r.unsqueeze(0))

    k = d_out // 2
    b_real_new = torch.zeros(k, device=device, dtype=dtype)
    b_imag_new = torch.zeros(k, device=device, dtype=dtype)

    for j in range(k):
        i = 2 * j
        # T_block: (2, r), B_block: (2, r)
        T_block = T[i:i + 2, :]    # (2, r)
        B_block = B[i:i + 2, :]    # (2, r)

        # We need: [[a, -c], [c, a]] @ B_block = T_block
        # Rearranging per-column of T_block:
        #   T_block[0, l] = a * B_block[0, l] - c * B_block[1, l]
        #   T_block[1, l] = c * B_block[0, l] + a * B_block[1, l]
        # This is a least-squares system for (a, c) with 2r equations:
        #   [B[0,l], -B[1,l]] [a]   [T[0,l]]
        #   [B[1,l],  B[0,l]] [c] = [T[1,l]]  for each l

        b0 = B_block[0, :]   # (r,)
        b1 = B_block[1, :]   # (r,)

        # Build system matrix (2r, 2) and rhs (2r,)
        # Row pairs: for each l in range(r):
        #   row 2l:   [b0[l], -b1[l]]
        #   row 2l+1: [b1[l],  b0[l]]
        col0 = torch.stack([b0, b1], dim=1).reshape(-1)     # (2r,)
        col1 = torch.stack([-b1, b0], dim=1).reshape(-1)    # (2r,)
        A_mat = torch.stack([col0, col1], dim=1)             # (2r, 2)

        rhs = torch.stack([T_block[0, :], T_block[1, :]], dim=1).reshape(-1)  # (2r,)

        # Solve lstsq
        result = torch.linalg.lstsq(A_mat, rhs)
        ac = result.solution  # (2,)
        b_real_new[j] = ac[0]
        b_imag_new[j] = ac[1]

    return b_real_new, b_imag_new


def fit_d(A, V_r, r):
    """Fit d parameters (scale, angles) given SVD factors.

    Lambda_d_target = V_r^T @ pinv(A),  shape (r, r)
    We approximate Lambda_target ≈ scale * G(angles) using polar
    decomposition: SVD(Lambda_target) = P @ D @ Q^T, then O = P @ Q^T is
    the nearest orthogonal matrix. scale is the mean of the singular values
    (the Frobenius-optimal uniform magnitude).

    Args:
        A: (r, d_in) shared VeRA A matrix
        V_r: (d_in, r) right singular vectors (columns)
        r: rank

    Returns:
        (scale_new, angles_new): scalar tensor and (r*(r-1)//2,) tensor
    """
    device = A.device
    dtype = A.dtype

    # Lambda_d_target = V_r^T @ pinv(A),  (r, d_in) @ (d_in, r) -> (r, r)
    A_pinv = torch.linalg.pinv(A)          # (d_in, r)
    Lambda_target = V_r.T @ A_pinv         # (r, r)

    # SVD of target
    P, D_vals, QT = torch.linalg.svd(Lambda_target)
    # P: (r,r), D_vals: (r,), QT: (r,r)

    # Nearest orthogonal matrix (polar factor)
    O = P @ QT

    # Decompose O into Givens angles (residual diag_signs are discarded
    # because we use a single uniform scale instead of per-axis values)
    angles_new, _diag_signs = decompose_orthogonal_to_givens(O, r)

    # Single magnitude: mean of singular values (Frobenius-optimal uniform scale)
    scale_new = D_vals.mean()

    return scale_new, angles_new


# --------------------------------------------------
# Init helper
# --------------------------------------------------

def init_sorf_vera_params(d_out, r, init_scale=0.1):
    """Initialize SORF-VeRA parameters.

    Returns dict with:
        b_real: ones(d_out//2)  — so Gamma = I initially
        b_imag: zeros(d_out//2)
        d_scale: scalar init_scale — single uniform magnitude
        d_angles: zeros(r*(r-1)//2)
    """
    k = d_out // 2
    num_angles = r * (r - 1) // 2
    return {
        'b_real': torch.ones(k),
        'b_imag': torch.zeros(k),
        'd_scale': torch.tensor(init_scale),
        'd_angles': torch.zeros(num_angles),
    }
