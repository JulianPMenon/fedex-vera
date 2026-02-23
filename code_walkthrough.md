# Code Walkthrough: SORF-VeRA Federated Learning

This document walks through the implementation in the order ideas are presented in the proof paper.

---

## 1. FedEx-LoRA (`fed_agg.py:85`)

The paper shows that naively averaging $A$ and $B$ separately introduces an error because:

$$\mathbb{E}[B_i] \cdot \mathbb{E}[A_i] \neq \mathbb{E}[B_i \cdot A_i]$$

The function `aggregate_models_ours` computes the **residual correction** $\Delta W_\text{res}$:

$$\Delta W_\text{res} = \underbrace{\frac{1}{k}\sum_{i=1}^k B_i A_i}_{M} - \underbrace{\left(\frac{1}{k}\sum_{i=1}^k B_i\right)\left(\frac{1}{k}\sum_{i=1}^k A_i\right)}_{\bar{B}\,\bar{A}}$$

```python
# B_ws: stacked B matrices from all clients, shape (num_clients, d_out, r)
# A_ws: stacked A matrices from all clients, shape (num_clients, r, d_in)
M       = sum(B_ws[i] @ A_ws[i] for i in range(num_clients)) / num_clients
# B_avg, A_avg: element-wise means across the client dimension
residue = M - (B_avg @ A_avg)
# base_key: state-dict key for W_0, e.g. "roberta.encoder.layer.0.attention.self.query.base_layer.weight"
global_state[base_key] += residue.T * scaling   # absorbed into W_0
```

Because the residue has rank $k \cdot r$ and is untrainable, it is folded directly into the frozen base weight $W_0$.

---

## 2. FedSVD / Freeze-A (`fed_agg.py:58`, `models.py:75`)

The paper describes reinitialising $A$ and $B$ via SVD and then only training (and transmitting) $B$:

$$U \Sigma V^T = \text{SVD}(B \cdot A), \qquad A \leftarrow V^T, \quad B \leftarrow U\Sigma$$

The code implements the "Freeze-A" variant as `aggregate_models_ffa` — only the $B$-side is averaged:

```python
# k: a parameter name string from the model's state dict (e.g. "...lora_B.default.weight")
# cs: one client's full state dict  (client_states is the list of all of them)
if "lora_B" in k or "vera_lambda_b" in k or "classifier" in k:
    global_state[k] = torch.stack([cs[k].float() for cs in client_states]).mean(0)
```

The model counterpart `create_peft_FFA_model` (`models.py:75`) freezes $A$ at creation time:

```python
for name, param in model.named_parameters():
    if "lora_A" in name:
        param.requires_grad = False
```

---

## 3. VeRA (`models.py:37`)

Standard VeRA defines the weight update as:

$$\Delta W = \Lambda_b \cdot B \cdot \Lambda_d \cdot A, \qquad \Lambda_b = \text{diag}(b),\quad \Lambda_d = \text{diag}(d)$$

where $A \in \mathbb{R}^{r \times d_\text{in}}$ and $B \in \mathbb{R}^{d_\text{out} \times r}$ are frozen random matrices, and $b \in \mathbb{R}^{d_\text{out}}$, $d \in \mathbb{R}^r$ are trainable vectors. This is created via PEFT's `VeraConfig` in `create_peft_model`.

---

## 4. Naive Aggregation Fails for VeRA (`fed_agg.py:145`)

The proof by contradiction shows:

$$\frac{1}{k}\sum_{i=1}^k \Lambda_{b,i}\, B\, \Lambda_{d,i}\, A \;\neq\; \left(\frac{1}{k}\sum_{i=1}^k \Lambda_{b,i}\right) B \left(\frac{1}{k}\sum_{i=1}^k \Lambda_{d,i}\right) A$$

`aggregate_models_ours_vera_fedex` applies the FedEx correction to VeRA's diagonal matrices:

```python
# lambda_bs: list of Λ_b matrices (client vectors b expanded to full diag matrices), shape (d_out, d_out)
# lambda_ds: list of Λ_d matrices (client vectors d expanded to full diag matrices), shape (r, r)
# B, A: frozen shared VeRA random matrices for this layer
M       = sum(lambda_bs[i] @ B @ lambda_ds[i] @ A for i in range(num_clients)) / num_clients
# lambda_b_avg_mat, lambda_d_avg_mat: averaged vectors re-wrapped as diagonal matrices
residue = M - (lambda_b_avg_mat @ B @ lambda_d_avg_mat @ A)
# args.fedex_lr: step size controlling how much of the residue is folded into W_0
global_state[base_key] += args.fedex_lr * residue
```

Same structure as LoRA FedEx, with $\Lambda_b$, $\Lambda_d$ in place of $B$, $A$.

---

## 5. Naive FedVeRA Hits a Wall

The paper shows that solving

$$\Lambda_d \cdot A = V^T, \qquad \Lambda_b \cdot B = U\Sigma$$

for **diagonal** $\Lambda_d$, $\Lambda_b$ is impossible: $V^T$ is a rotation matrix and $A$ is random — no diagonal matrix can produce the required rotation. **There is no code for this** since it is a dead end established by the proof.

---

## 6. Naive Rotational Basis / Givens Rotations (`sorf_vera.py:142`)

The paper proposes replacing diagonal entries with Givens rotation angles. The number of angles required is:

$$\text{num\_rotations} = \frac{r(r-1)}{2}$$

This is tractable for small $r$ (the rank). The code implements this for the $d$ vector:

- **`givens_pair_indices(r)`** (`sorf_vera.py:142`) — generates all $\frac{r(r-1)}{2}$ index pairs $(p, q)$
- **`build_lambda_d_rot(scale, angles, r, ...)`** (`sorf_vera.py:158`) — constructs $\Lambda_{d,\text{rot}} = m \cdot G(\theta)$ by composing Givens rotations then applying a single uniform magnitude $m$, matching the proof's parameterization
- **`decompose_orthogonal_to_givens(O, r)`** (`sorf_vera.py:196`) — the inverse: decomposes an orthogonal matrix into Givens angles and a residual diagonal via QR-style elimination

---

## 7. SORF Mixing Matrix (`sorf_vera.py:44`)

The paper defines the SORF mixer as:

$$S_r := D_1 H_r D_2 H_r D_3 \in \mathbb{R}^{r \times r}$$

where $H_r$ is the normalized Walsh–Hadamard matrix ($H_r H_r^T = I_r$) and $D_1, D_2, D_3$ are diagonal sign matrices with entries in $\{-1, +1\}$ sampled from a shared seed.

The code builds this in three steps:

```python
# n: next power of 2 >= d_out (Hadamard requires power-of-2 size)
# d_out: actual output dimension of the layer (e.g. 768 for RoBERTa)
n  = next_power_of_2(d_out)
H  = build_hadamard(n, ...)              # H / sqrt(n), orthonormal, shape (n, n)
# d1, d2, d3: random {-1, +1} vectors drawn from shared seed; D1=diag(d1), etc.
d1 = _random_signs(n, gen)
S  = D1 @ H @ D2 @ H @ D3               # full SORF, shape (n, n)
S  = S_full[:d_out, :d_out]             # truncate back to (d_out, d_out)
```

The seed ensures every client reconstructs the identical $S$ without transmitting it.

---

## 8. Complex Vectors as Rotation + Scaling (`sorf_vera.py:77`)

The paper extends the rank space by identifying $\mathbb{R}^{2r}$ with $\mathbb{C}^r$. Each complex number $b_j = a_j + ic_j$ is embedded into a $2\times 2$ rotation–scaling block:

$$\Phi(b_j) = \begin{pmatrix} a_j & -c_j \\ c_j & a_j \end{pmatrix} = \rho_j\, R(\theta_j), \qquad \rho_j = |b_j|, \quad \theta_j = \arg(b_j)$$

These blocks are assembled into the block-diagonal embedding:

$$\Gamma(b) = \text{blkdiag}\!\left(\Phi(b_1), \dots, \Phi(b_r)\right) \in \mathbb{R}^{2r \times 2r}$$

**`build_gamma(b_real, b_imag)`** (`sorf_vera.py:77`) constructs this matrix explicitly. For efficiency in the forward pass, **`apply_gamma_transpose_batched`** (`sorf_vera.py:107`) applies $\Gamma^T$ without materialising the full matrix, using even/odd index slicing:

$$w_{2j}   = a_j\, h_{2j} - c_j\, h_{2j+1}, \qquad w_{2j+1} = c_j\, h_{2j} + a_j\, h_{2j+1}$$

> **Design choice:** SORF + complex blocks is used for $b$ (size $d_\text{out}$, large). Full Givens rotations are used for $d$ (size $r$, small). This matches the paper's closing remark: *"the SORF approach can be only used for vector $b$ because of its much bigger size while $d$ enjoys the full precision of the naive rotation basis."*

---

## 9. Full $\Delta W$ and the Forward Pass (`sorf_vera.py:238`, `models.py:118`)

The paper defines:

$$\Delta W = S \cdot \Gamma(b) \cdot B \cdot S \cdot \Gamma(d) \cdot A$$

The implementation in `compute_delta_w` uses the hybrid parameterisation (Givens for $d$, SORF+complex for $b$):

$$\Delta W = S \cdot \Gamma(b) \cdot B \cdot \Lambda_{d,\text{rot}} \cdot A$$

computed right-to-left:

| Step | Operation | Shape |
|------|-----------|-------|
| 1 | $\Lambda_{d,\text{rot}} \cdot A$ | $(r, d_\text{in})$ |
| 2 | $B \cdot (\text{step 1})$ | $(d_\text{out}, d_\text{in})$ |
| 3 | $\Gamma(b) \cdot (\text{step 2})$ | $(d_\text{out}, d_\text{in})$ |
| 4 | $S \cdot (\text{step 3})$ | $(d_\text{out}, d_\text{in})$ |

The **forward pass** `_sorf_vera_forward` (`models.py:118`) processes batched inputs as right-multiplies (transposed order) for efficiency:

$$h = x \xrightarrow{A^T} \xrightarrow{\Lambda_{d,\text{rot}}^T} \xrightarrow{B^T} \xrightarrow{\Gamma(b)^T} \xrightarrow{S^T} \Delta W \cdot x$$

---

## 10. Server-Side Reinitialization (`sorf_vera.py:277`, `sorf_vera.py:349`)

The paper sets up the SVD equations the server must approximately solve:

$$U\Sigma V^T = \text{SVD}\!\left(S \cdot \Gamma(b_i) \cdot B \cdot \Lambda_{d,\text{rot},i} \cdot A\right)$$

$$S \cdot \Gamma(b_{i+1}) \cdot B = U_r \cdot \text{diag}(\Sigma_r), \qquad \Lambda_{d,\text{rot},i+1} \cdot A = V_r^T$$

### `fit_b` (`sorf_vera.py:277`)

Solves $S \cdot \Gamma(b) \cdot B = U_r\,\text{diag}(\Sigma_r)$ for $b$:

1. Compute $T = S^{-1}\bigl(U_r\,\text{diag}(\Sigma_r)\bigr)$ via `linalg.solve` (exact even for truncated, non-orthogonal $S$)
2. For each $2\times2$ block $j$ (covering rows $2j$ and $2j{+}1$ of the $d_\text{out}$-dimensional space), solve the per-block least squares. Here $B_{0,l}$ and $B_{1,l}$ are the two rows of the shared VeRA matrix $B$ within this block, indexed over the $r$ rank columns $l$:

$$\begin{pmatrix} B_{0,l} & -B_{1,l} \\ B_{1,l} & \phantom{-}B_{0,l} \end{pmatrix} \begin{pmatrix} a_j \\ c_j \end{pmatrix} = \begin{pmatrix} T_{0,l} \\ T_{1,l} \end{pmatrix} \quad \forall\, l \in \{1,\dots,r\}$$

where $(a_j, c_j)$ are the real and imaginary parts of the complex $b$ entry for block $j$, and $T_{0,l}, T_{1,l}$ are the corresponding rows of $T = S^{-1}(U_r\,\text{diag}(\Sigma_r))$.

### `fit_d` (`sorf_vera.py:353`)

Solves $\Lambda_{d,\text{rot}} \cdot A = V_r^T$ for scale and angles:

1. Compute target: $\Lambda_\text{target} = V_r^T \cdot A^+$ where $A^+ = \text{pinv}(A)$ is the pseudoinverse of the frozen VeRA $A$ matrix — this gives the $(r \times r)$ matrix that $\Lambda_{d,\text{rot}}$ should ideally equal
2. Polar decomposition via SVD: $P\,D\,Q^T = \text{SVD}(\Lambda_\text{target})$, then $O = P Q^T$ is the nearest orthogonal matrix to $\Lambda_\text{target}$ in Frobenius norm
3. Decompose $O$ into Givens angles (the new `angles`); the single uniform magnitude is $\text{scale} = \text{mean}(D)$ — the Frobenius-optimal scalar approximation to the singular values

---

## 11. The Full Federated Loop (`fed_train_glue.py`, `fed_agg.py:252`)

The orchestration follows the cycle described in the paper:

```
Server initialises b, d, seed
│
└─► each round:
      │
      ├─ Broadcast b (frozen), d to clients
      │
      ├─ Clients train d only (b frozen)       ← requires_grad on d_scale, d_angles only
      │
      ├─ Clients upload updated d
      │
      └─ Server aggregation (aggregate_models_sorf_vera):
           │
           ├─ Average Λ_d matrices across clients        ← average the built (r×r) matrices,
           │   avg_Ld = mean(                              not the raw params — because
           │     build_lambda_d_rot(                       composing Givens rotations is nonlinear
           │       cs[scale_key],   # per-client d_scale   in the angles, so mean(angles) ≠
           │       cs[angles_key],  # per-client d_angles    angles of mean(Λ_d)
           │       r, device, dtype
           │     ) for each client cs
           │   )
           │
           ├─ Compute ΔW = S @ Γ(b) @ B @ avg_Ld @ A
           │
           ├─ Truncated SVD: ΔW ≈ U_r Σ_r V_r^T
           │
           ├─ fit_b → new b_real, b_imag
           │
           └─ fit_d → new scale, angles
```

**Model creation** (`models.py:164`): `create_peft_sorf_vera_model` monkey-patches every VeRA layer with the buffer $S$, parameters `sorf_b_real/sorf_b_imag` (frozen), `sorf_d_scale/sorf_d_angles` (trainable), and replaces the forward method.

**Trainability** is enforced in a final pass:

```python
for param_name, param in model.named_parameters():
    if 'sorf_d_scale' in param_name or 'sorf_d_angles' in param_name:
        param.requires_grad = True
    elif 'classifier' in param_name:
        param.requires_grad = True
    else:
        param.requires_grad = False
```

---

## Summary

| Paper concept | Code location | Key function/class |
|---|---|---|
| FedEx-LoRA residue | `fed_agg.py:85` | `aggregate_models_ours` |
| FedSVD / Freeze-A | `fed_agg.py:58`, `models.py:75` | `aggregate_models_ffa`, `create_peft_FFA_model` |
| VeRA model | `models.py:37` | `create_peft_model` (with `VeraConfig`) |
| VeRA FedEx correction | `fed_agg.py:145` | `aggregate_models_ours_vera_fedex` |
| Givens rotations + uniform scale for $d$ | `sorf_vera.py:142` | `build_lambda_d_rot`, `decompose_orthogonal_to_givens` |
| SORF mixing matrix | `sorf_vera.py:44` | `build_sorf_matrix` |
| Complex block-diagonal $\Gamma(b)$ | `sorf_vera.py:77` | `build_gamma`, `apply_gamma_transpose_batched` |
| Full $\Delta W$ computation | `sorf_vera.py:238` | `compute_delta_w` |
| SORF-VeRA forward pass | `models.py:118` | `_sorf_vera_forward` |
| Server refit of $b$ | `sorf_vera.py:277` | `fit_b` |
| Server refit of $d$ | `sorf_vera.py:349` | `fit_d` |
| Full federated loop | `fed_train_glue.py:189`, `fed_agg.py:252` | `federated_learning`, `aggregate_models_sorf_vera` |
