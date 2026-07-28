# Chapter 16 — The paper, equation by equation

> **What this chapter is.** A cross-reference from Côté et al. (2026), *Generalized
> Aberrations for Processing-Aware Optical Design* (ACM TOG), to the exact file, line, and
> variable name that implements each equation. Use it in two directions: paper → code when
> you read the paper and want to see the mechanism, and code → paper when you are reading
> source and want to know which claim it supports.
>
> Every mapping below was checked against the source. Where the paper and the shipped code
> **disagree**, that is stated explicitly rather than smoothed over — §16.12 collects those,
> and they are the most useful part of the chapter for a defense.

---

## 16.1 The master table

| Eq. | Paper form | Implementation | Chapter |
|---|---|---|---|
| (1) | `x* = arg min L(x)` | `imaging_system.py` `training_step` (555) | 12 |
| (2) | `L(x) = ½‖ℓ(x)‖²` | `optimizers.py` `least_squares_loss`; `imaging_system.py:1032+` | 7, 9 |
| (3) | `ℓ_TRA = (ε(x) − ε̄(x))/√(fwp)` | `residuals.py:61` `TransverseRayAberrationResiduals` | 7 |
| (4) | `ℓ_ESR = [‖ℓ_TRA‖] ∈ ℝ¹` | **not implemented** — the ablation baseline | 7, 16.12 |
| (5) | GTRA linearization, `w` and `ε′` | `imaging_system.py:474–513` | 8, 15 |
| (6) | `ℓ_GTRA = √w (ε(x) − ε′)` | `imaging_system.py:513` `residual_vector` | 8 |
| (7) | `ℓ_MSE = [‖I′(x) − I‖] ∈ ℝ¹` | `e2e_vector_mode: False` branch (`imaging_system.py:218`) | 8 |
| (8) | Hartmann dispersion `n(λ) = A + C/(λ − B)` | `optics.py:735` `hartmann_dispersion` | 2 |
| (9) | Even-asphere sag | `ray_tracing.py:285` `evaluate_aspherical_profile` | 2, 5 |
| (10) | DOE phase `φ(ρ) = Σ pᵢ ρ^{2i}` | `misc_surfaces.py:130` `compute_phase_gradient` | 13 |
| (11) | LM update `Δx = −(JᵀJ + λD²)⁻¹Jᵀℓ₀` | `optimizers.py:60–150` | 9 |
| (12) | Airy pattern `U ∝ 2J₁(·)/(·)` | `simulation.py:667` | 10 |
| (13) | Spatially-varying convolution + noise | `simulation.py:467` `SVOLAConvolution` | 10 |

---

## 16.2 Eq. (1)–(2): the objective

The paper writes the design problem as `x* = arg min L(x)` with

```
L(x) = ½‖ℓ(x)‖²,     ℓ ∈ ℝᵏ
```

**Notation → code:**

| paper | code | measured (toy) |
|---|---|---|
| `x` | flat variable vector | `[31]`, 20 trainable |
| `n` (number of variables) | `n_variables` | **20** |
| `k` (number of residuals) | `n_residuals` | **5704** |
| `ℓ` | `residual_vector` | `[5704]` |
| `L` | `least_squares_loss` | `1.418987e−04` |

The paper says `x ∈ ℝⁿ` "typically consists of several dozens of heterogeneous variables …
including (aspheric) surface profiles, material properties, and spacings." Chapter 14 §14.3
measured exactly that decomposition: 5 spacings, 3 curvatures, 12 aspheric coefficients, 6
glass values.

The condition the paper stresses — `k ≥ n` "for effective optimization" — is `5704 ≥ 20`
here, satisfied by a factor of 285. **That factor is the entire subject of the paper.**

## 16.3 Eq. (3): transverse ray aberrations

```
ℓ_TRA(x) = (1/√(fwp)) (ε(x) − ε̄(x)) ∈ ℝ^{2fwp}
```

Implemented in `residuals.py:61`, `TransverseRayAberrationResiduals.compute_loss`:

```python
delta_xy = xy - xy_centroid          # residuals.py:82
```

**Notation → code, exactly:**

| paper | code | toy value |
|---|---|---|
| `ε(x)` | `xy` | `[2, 11, 256, 1, 1]` |
| `ε̄(x)` | `xy_centroid` | `[2, 11, 1, 1, 1]` |
| `f` (fields) | `n_fields` | **11** |
| `w` (wavelengths) | `n_wavelengths` | **1** |
| `p` (pupil samples) | `n_rays` | **256** |
| `2fwp` | residual length | **5632** = 2×11×1×256 |

The `1/√(fwp)` normalization is the "rays should have a weight of 1 on average" comment at
`residuals.py:84-86`. The paper's remark that "for a given field, `‖ℓ_TRA‖` can be
recognized as the effective spot radius" is what Chapter 14's `rms_spot_size` array measures.

**A measured confirmation of the paper's setup.** The paper's Table 1 reports ESR = **16.6
µm** for LM + `ℓ_TRA`. Chapter 14 measured the 11 per-field RMS spot sizes of the shipped
`optimized_spot.yml` design; their root-mean-square across fields is

```
√( mean( rms_spot_size² ) ) = 16.6029 µm
```

**16.60 vs the paper's 16.6.** The shipped design *is* the design in Table 1, and "effective
spot radius" is the RMS over fields of the per-field RMS. That single number authenticates
the config against the publication.

## 16.4 Eq. (4) and (7): the naive baselines

```
ℓ_ESR(x) = [‖ℓ_TRA(x)‖] ∈ ℝ¹          (4)
ℓ_MSE(x) = [‖I′(x) − I‖] ∈ ℝ¹          (7)
```

These are the **straw men** — the paper's demonstration that collapsing a vector residual to
a scalar destroys LM. Eq. (7) is reachable in the shipped code: set `e2e_vector_mode: False`
and `imaging_system.py:218` takes the scalar branch, producing a `1 × n` Jacobian.

**Eq. (4) has no implementation.** There is no `EffectiveSpotRadiusResiduals` class. If you
wanted to reproduce the `LM + ℓ_ESR` row of Table 1 (33.0 µm), you would have to write it.
Worth knowing before you go looking.

The paper's point, in its own terms: using `ℓ_ESR` (`k = 1`) instead of `ℓ_TRA` (`k = 2fwp`)
"causes rank deficiency in J … by discarding per-ray sensitivities." Chapter 15 §15.5
measured `lstsq_rank = 14` with the *full* residual set at the optimum — so rank deficiency
is a matter of degree, and the paper's `k = 1` case is its extreme.

## 16.5 Eq. (5): the central result

This is the paper's contribution, and the mapping is worth giving line by line. The paper:

```
L_TD(x) ≈ L₀ + ∇L₀ᵀ(ε(x) − ε₀)
        ≈ ½ · (‖∇L₀‖²/(2L₀)) · ‖ε(x) − (ε₀ − 2L₀∇L₀/‖∇L₀‖²)‖²
             └────── w ──────┘        └────────── ε′ ──────────┘
```

The code, `imaging_system.py:474–513`:

| paper symbol | line | code |
|---|---|---|
| `L₀`, `∇L₀` | 477–479 | `torch.func.grad_and_value(self.scalar_e2e_loss, ...)` |
| `‖∇L₀‖²` | 484 | `grad_norm_squared = grad_valid @ grad_valid` |
| `ε′` | 487 | `xy_control = xy.detach() - grad * 2 * scalar_loss / grad_norm_squared` |
| `w` | 510 | `weight = (compensation_factor * grad_norm_squared / (2*scalar_loss)).sqrt()` |
| `ℓ_GTRA` | 513 | `residual_vector = weight * (xy - xy_control)` |

Line 487 is `ε′ = ε₀ − 2L₀∇L₀/‖∇L₀‖²` character for character. Line 510 is `√w` — note the
`.sqrt()`, because the paper's `w` multiplies the *squared* norm while the code stores the
factor applied to the residual.

**A notation trap worth being ready for.** The paper's `w = ‖∇L₀‖²/(2L₀)`; the code's
`weight_w = 0.1573027223`. These differ, and the difference is a square root:

```
w_paper = 2.225276e−04 / (2 × 0.0044965702) = 0.02474414833
√w_paper                                    = 0.15730272829
code weight_w                               = 0.1573027223
relative error                                3.81e−08
```

**The code variable named `weight_w` holds `√w`, not `w`.** If a supervisor asks you to
confirm the weight and you quote `0.157`, you are quoting `√w`. Say which.

And `ε′` reproduces too. With `2L₀/‖∇L₀‖² = 40.41359544`:

```
grad max  0.0010466933 → shift −0.04230064   vs measured xy_control_shift min −0.0423007011
grad min −0.0011020704 → shift +0.04453863   vs measured xy_control_shift max +0.0445387363
```

Chapter 15 verified the identity itself: `½‖ℓ‖² = 0.004496567` against `L₀ = 0.0044965702`,
relative error **7.249e−07**.

### What the code adds that the paper's Eq. (5) does not show

`imaging_system.py:489–507` clips `ε′` to a box around the centroid, then computes

```python
compensation_factor = ‖xy − xy_control‖² / ‖xy − clipped_xy_control‖²
```

so that `½‖ℓ‖²` still equals `L₀` exactly after clipping. The paper does not mention this;
it is a practical guard keeping the virtual target inside the PSF grid. **Chapter 8 covers
the derivation.** If asked "what is in the code that is not in the paper?", this is a good
answer.

### The two approximations, and their honest scope

The paper is explicit that Eq. (5) involves two approximations: a "locally
gradient-preserving linearization" of the scalar loss, and a second step that "recognizes
that `‖ε(x) − ε₀‖` is small near convergence."

So the surrogate is exact in **value and gradient** at the current iterate, and degrades as
`ε` moves away. Chapter 15 measured how far it moves: `xy_control_shift` reaches 0.0445 mm
against a corner RMS spot of 0.0300 mm — displacement of roughly 1.5 spot radii. That is the
quantitative answer to "how good is the approximation?", and it is why the residual is
**rebuilt every iteration**.

## 16.6 Eq. (6): the residual, and why it plugs in

```
ℓ_GTRA(x) = √w (ε(x) − ε′) ∈ ℝ^{2fwp}
```

Measured: `e2e_residual_vector [5632]` — **the same 2fwp as `ℓ_TRA`**. This is the whole
architectural payoff: a scalar image loss becomes a residual block of the same shape as a
classical aberration term, so it concatenates into the same vector and feeds the same solver.

The paper's remark that "GTRA generalize TRA … recovering TRA as the special case with `w =
1/wpf` and `ε′ = ε̄`" is worth checking against the code: set the image loss to the spot
variance and `xy_control` becomes the centroid. The two residual classes are then the same
computation.

`ε` and `ε′` are both treated as constants when the Jacobian is formed — the paper says so,
and `xy.detach()` at lines 487 and 503 is where.

## 16.7 Eq. (8): Hartmann dispersion

```
n(λ) = A + C/(λ − B)
```

`optics.py:735` `hartmann_dispersion`. The docstring is unusually good and worth reading in
full; the substance is that `A`, `B`, `C` are **derived** from the three design variables
`(nd, vd, dpgf)` rather than being variables themselves:

- `pgf = m·vd + o + dpgf`, `optics.py:771`
- `B` from the deviation from normal partial dispersion (Schott TIE-29), `:772`
- `C` from the Abbe-number definition, `:773`
- `A` from `n(λ_d) = nd`, `:774`
- `n = A + C/(λ − B)` — Eq. (8) itself, `:775`

Constants: `wc = 656.3`, `wd = 587.6`, `wf = 486.1`, `wg = 435.8` nm; `m = −0.001682`, `o =
0.6438`.

**Why this matters for the method.** Optimizing `(nd, vd)` continuously — the paper's glass
mesh, Fig. 5 — requires a dispersion model that is *smooth and differentiable* in those two
numbers. A catalog lookup is not. Hartmann's three-parameter form makes glass a continuous
design variable, and Chapter 13's `BindMaterialsCallback` is how it returns to reality.

Toy config: `vd = 1e6`, so `C → 0` and `n(λ) ≡ 1.5`. **Zero dispersion by construction.**

## 16.8 Eq. (9): the even-asphere sag

```
z̃(ρ) = c ρ² / (1 + √(1 − (1+k) c²ρ²)) + Σᵢ₌₂ a_{2i} ρ^{2i}
```

`ray_tracing.py:285` `evaluate_aspherical_profile`. Note the packing convention, documented
at `optics.py:199`: **the conic constant `k` is `a[..., 0]`** — the first "aspheric
coefficient" is not a coefficient at all. Chapter 2 covers it; it is an easy misreading when
you first look at the `[4, 1, 4]` shape.

Toy: 4 aspheric surfaces × 4 slots = 16 stored, and the paper's setup says "12 aspheric
coefficients (degrees 4, 6, and 8)" — so 4 of the 16 are conic constants and **12 are
optimized**, which is exactly `total_trainable`'s aspheric contribution in Chapter 14 §14.3.
The paper's sentence and the measured mask agree.

## 16.9 Eq. (10): diffractive phase

```
φ(ρ) = Σᵢ₌₁ pᵢ ρ^{2i}
```

`misc_surfaces.py:130` `compute_phase_gradient`. The code never evaluates `φ` — the ray
tracer needs `dφ/dρ`, so it differentiates the polynomial analytically via the substitution
`ξ = (ρ/ρ_max)²`, documented in the docstring at `:133-142`. `n_diffractive = 0` in the toy
config, so this path is untraced here. Chapter 13 covers it.

## 16.10 Eq. (11): the LM update

```
Δx = −(JᵀJ + λD²)⁻¹ Jᵀℓ₀
```

`optimizers.py:60–150`. **Three divergences between this equation and the code, all
deliberate, all worth knowing.**

**1. `JᵀJ` is never formed.** The paper writes the normal equations; the code solves the
equivalent stacked least-squares problem

```
[   J   ] Δx = −[ ℓ₀ ]
[ √λ·D  ]       [ 0  ]
```

`augmented_matrix [5724, 20]`, measured. The reason is conditioning: Chapter 15 measured `J`
spanning ±486 while `JᵀJ` spans ±2.25e6. Forming the normal equations squares the condition
number. `pseudo_hessian_JTJ` exists in the trace for diagnosis only.

**2. `D²` is a running maximum, not the instantaneous diagonal.** The paper: "we set the
entries of `D²` to the diagonal entries of `JᵀJ`," citing Nocedal & Wright. The code,
`optimizers.py:87-90`:

```python
beta * torch.max(damping_terms, group["damping_terms"]) + (1 - beta) * damping_terms
```

With `beta = 0.99` this is a **ratchet**: `torch.max` means the damping scale can rise fast
and decay only at 1% per iteration. Not an EMA, despite looking like one. Measured spread
`0.3719` to `1501.38`.

**3. The acceptance test is not what the paper says.** The paper: "a step is accepted if and
only if `L(x + Δx) < L(x)`," and adds that this "explains the monotonic decrease of LM curves
in Fig. 4." The code, `optimizers.py:143`:

```python
if not math.isfinite(loss_ratio) or loss_ratio > group["tolerance"]:
```

with `tolerance` **defaulting to 2.0**. A step that *doubles* the loss would be accepted at
the default. The toy config sets `tolerance: 1.` — recovering exactly the paper's rule — and
so does `telephoto`. But `microscope` and `c_mount` use `1.25`, and `wide_angle` uses `1.5`.

**So the paper's claim of monotonic decrease holds for the toy and telephoto configs and
does not hold for three of the shipped production configs.** Chapter 15 §15.8 measured the
toy behaviour: iteration 11 has ratio 1.1102 and is rejected — as `tolerance: 1.` requires.
Under `wide_angle`'s 1.5 it would have been accepted.

**4. The damping factors.** The paper: reduce λ "by a factor of 3" on success, increase "by
a factor of 2" on failure, citing Transtrum & Sethna. The code defaults are
`lam_decrease_factor: 2.0` and `lam_increase_factor: 2.0` — symmetric. The production
configs (`microscope`, `c_mount`, `telephoto`, `wide_angle`) all set `lam_decrease_factor:
3.`, matching the paper. **The toy config does not**, which is why Chapter 15's convergence
table shows λ halving (1 → 1/2 → 1/4 …) rather than dividing by three.

None of these four is an error. They are the difference between a paper's description and a
configurable implementation — but if you cite the paper's Eq. (11) while running the toy
config, three of the four details you would state are wrong for that run.

## 16.11 Eq. (12)–(13): imaging simulation

```
U_Airy(r) ∝ 2J₁(π r NA/λ) / (π r NA/λ)          (12)
I′(x′,y′) = ∬ I(x′−x, y′−y) h(x,y) dx dy + n    (13)
```

Eq. (12) is `simulation.py:667`:

```python
diffraction_kernels = 2 * torch.special.bessel_j1(airy_input_r) / airy_input_r
```

with `na = 1/(2·f_number)` at `:622` and the Airy radius `0.61/scale_pix` at `:654`, kernel
sized to at least 3 Airy radii per side (`:656`).

**A detail the paper compresses.** There are two modes, `"airy"` and `"airy_field"`
(`:668-671`), and the default flag at `simulation.py:375` is **`airy_field`**. In that mode
the geometric PSF is square-rooted to an amplitude, convolved with the Airy *amplitude*, and
squared back (`:446`, `:456-458`). This is a coherent-amplitude convolution rather than an
intensity convolution — physically the better approximation, and not stated in the paper's
Eq. (12). The paper calls it a "heuristic," which it is: the code is choosing amplitude
convolution as that heuristic.

Toy config sets no `diffraction_f_number`, and the paper's toy setup says "we do not consider
diffraction effects in this particular problem." **The two agree** — this path is inactive in
everything Chapters 14–15 measured.

Eq. (13) is `SVOLAConvolution` (`simulation.py:467`), Chapter 10: patch-wise FFT convolution
with overlap-add, `psf_grid_shape` = 9×9 giving the 81 grid entries measured in Chapter 15,
and the paper's "spatially varying convolution over 9 × 9 sensor regions" matching exactly.

## 16.12 Paper vs. code: the divergence list

Collected, because these are the questions with the most defensible answers.

| # | paper says | code does | severity |
|---|---|---|---|
| 1 | `Δx = −(JᵀJ + λD²)⁻¹Jᵀℓ₀` | stacked `[k+n, n]` least squares, `JᵀJ` never formed | equivalent, better conditioned |
| 2 | `D² = diag(JᵀJ)` | `beta`-ratcheted running **max** of column norms | deliberate, undocumented |
| 3 | accept iff loss decreases | accept iff `ratio ≤ tolerance`, **default 2.0** | config-dependent; 3 of 5 production configs are non-monotonic |
| 4 | λ ÷ 3 on success, × 2 on failure | defaults ÷ 2 and × 2; production configs set ÷ 3 | toy config diverges |
| 5 | `w = ‖∇L₀‖²/(2L₀)` | `weight_w` stores **√w** | notation only |
| 6 | Eq. (5) as written | plus clipping of `ε′` and a compensation factor | code is richer |
| 7 | Eq. (12) Airy intensity | default mode convolves **amplitude**, then squares | code is more physical |
| 8 | `ℓ_ESR` (Eq. 4) as a baseline | **not implemented** | reproduce Table 1 row and you must write it |
| 9 | "f = w = 11 … p = 512 pupil locations, 61 952 rays" | toy uses `p = 256`, `w = 1` → **2816 rays** | toy ≠ the paper's main experiments |
| 10 | MAE for IRM steps, MSE for lens steps | matches (`imaging_system.py`) | agrees |

Item 9 deserves emphasis. The paper's §3.4 describes the *general* configuration used "in
most experiments": 11 fields × 11 wavelengths × 512 pupil points = 61,952 rays. **The toy
problem is not that.** It is monochromatic (`w = 1`) with 256 pupil points, giving 2816
rays. Both are correct — different experiments — but quoting 61,952 while showing a toy trace
would be an error.

## 16.13 Notation quick-reference

| paper | code | Chapter |
|---|---|---|
| `x` | flat variable vector, `LensParameterization` | 3 |
| `n` | `n_variables` = 20 | 9 |
| `k` | `n_residuals` = 5704 | 9 |
| `ℓ` | `residual_vector` | 7 |
| `L` | `least_squares_loss` | 7 |
| `L_TD` | `scalar_e2e_loss` | 8 |
| `ε` | `xy`, `xy_spot_diagrams` | 4, 10 |
| `ε̄` | `xy_centroid` | 7 |
| `ε₀` | `xy.detach()` | 8 |
| `ε′` | `xy_control` | 8 |
| `w` | `weight_w²` (code holds `√w`) | 8 |
| `∇L₀` | `grad`, `grad0` | 8 |
| `‖∇L₀‖²` | `grad_norm_squared` | 8 |
| `J` | `jacobian` `[5704, 20]` | 9 |
| `D²` | `damping_terms` (squared) | 9 |
| `λ` | `lm_parameter` | 9 |
| `f, w, p` | `n_fields`, `n_wavelengths`, `n_rays` | 4 |
| `I` | `batch` / sharp image | 10 |
| `I′` | `blurred_image` | 10 |
| `I″` | restored image | 11 |
| `h` | `psf_grid`, `rgb_psfs` | 10 |
| `n(λ)` | `hartmann_dispersion` | 2 |
| `z̃(ρ)` | `evaluate_aspherical_profile` | 5 |
| `φ(ρ)` | `compute_phase_gradient` | 13 |

## 16.14 What to take away

1. **Eq. (5) is `imaging_system.py:474–513`**, and line 487 is `ε′` character for character.
2. The code's `weight_w` is **`√w`**, not the paper's `w`. Verified: `√0.02474414833 =
   0.15730273` vs measured `0.1573027223`.
3. **The paper's Table 1 ESR of 16.6 µm reproduces**: RMS over fields of the shipped design's
   per-field RMS spot is **16.6029 µm**. The config is the publication's design.
4. Table 1's `Proposed` MSE of **0.0033** matches the measured `blur_mse_vs_sharp =
   0.0033178618`.
5. Eq. (11) is implemented as a **stacked least-squares solve**, not the normal equations —
   equivalent mathematics, `J` spanning ±486 instead of `JᵀJ` spanning ±2.25e6.
6. **The paper's monotonic-decrease claim is config-dependent.** `tolerance` defaults to
   2.0; toy and telephoto set 1.0 (matching the paper), while microscope, c_mount and
   wide_angle permit loss increases of 25–50%.
7. **The toy config's λ schedule diverges from the paper** (÷2, not ÷3); the production
   configs match.
8. `D²` is a `beta = 0.99` **ratchet on the running max** of column norms, not
   `diag(JᵀJ)` as written.
9. `ℓ_ESR` (Eq. 4) **has no implementation**.
10. The default diffraction mode convolves **amplitude** and squares, which is more physical
    than Eq. (12) as printed — and inactive in the toy problem, consistent with the paper.
11. The paper's 61,952 rays describe its main experiments; the toy problem traced in
    Chapters 14–15 uses **2816**.
12. The code contains a clipping-plus-compensation guard on `ε′` that the paper does not
    mention — a good answer to "what is in the code but not the paper?"

---

*Previous: [Chapter 15 — Execution trace: one training iteration](15_trace_training.md)*
*Next: [Chapter 17 — Mapping to your own implementation](17_your_implementation.md)*
