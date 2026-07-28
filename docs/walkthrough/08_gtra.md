# Chapter 8 — Generalized transverse ray aberrations: the paper's core idea

> **What this chapter answers.** This is the contribution. If you understand nothing
> else, understand this chapter. How do you feed an *image* loss to a *least-squares*
> optimizer? What is `xy'`? Why is the weight `‖∇L‖/√(2L₀)`? And is the claimed identity
> actually true in the shipped code — measured, not asserted?

---

## 8.1 The problem, stated precisely

You now have both halves of the tension.

**Half one (Chapters 6–7, 9).** Lens design has an excellent optimizer:
Levenberg–Marquardt. It converges in tens of iterations because it builds a curvature
estimate `JᵀJ` from first derivatives. But it requires the objective in the form

```
L(x) = ½ ‖l(x)‖²        l : ℝⁿ → ℝᵐ
```

**A vector.** Give LM a scalar and you have `m = 1`, `J` is a single row, `JᵀJ` has rank
1, and LM degenerates into a badly-scaled gradient step.

**Half two (Chapters 10–11).** End-to-end optimization means the objective is
"reconstructed image quality" — trace rays, build PSFs, convolve, denoise, compare to
ground truth. That is unavoidably a **scalar**. There is no natural way to write "MSE
between two images after restoration" as a vector of optical residuals.

So: the good optimizer needs a vector; the objective you care about is a scalar.
Existing work resolves this by giving up the good optimizer and using Adam. This paper
resolves it by **manufacturing a vector** whose least-squares problem is locally
equivalent to the scalar one.

## 8.2 The construction

`evaluate_generalized_transverse_ray_aberrations` (`imaging_system.py:445-516`). The
docstring states the whole thing in six lines (`imaging_system.py:454-460`):

> *The loss is evaluated at the spot diagram "xy0": L(xy0) === L0.*
> *The gradient of the loss is computed w.r.t. the spot diagram "xy0": grad(L)(xy0) === grad0.*
> *Then, the scalar loss is locally approximated as a function L(xy) = L(xy0) + (xy − xy0) @ grad(L)(xy0).*
> *An approximation gives L(xy) = 1/2 ‖l(xy)‖²*
> *where l(xy) = w * (xy − xy')*
> *with xy' = xy0 − grad0 * 2L0 / ‖grad0‖²*
> *and w = ‖grad0‖ / sqrt(2L0)*

Read it as three moves.

**Move 1 — pick the right intermediate variable.** Not the lens parameters `x`, but the
**spot diagram** `xy` — the ray coordinates at the image plane. Everything downstream of
`xy` (PSF, convolution, restoration, MSE) is treated as a black box that supplies a
value and a gradient. Everything upstream (lens → rays → `xy`) keeps its full,
exactly-differentiable structure.

This is the pivot on which the whole method turns. `xy` is the natural interface: it is
the last purely *geometric* quantity, and it is where the classical transverse ray
aberration lives. Hence the name — a *generalized* transverse ray aberration.

**Move 2 — linearize the black box.**

```
L(xy) ≈ L₀ + ⟨∇L₀, xy − xy₀⟩
```

A first-order Taylor expansion in `xy`. Cheap: one forward and one backward pass through
the image pipeline gives both `L₀` and `∇L₀`.

**Move 3 — find the quadratic that matches it.** We want `l(xy) = w·(xy − xy′)` such
that `½‖l‖²` reproduces the linear model. Two unknowns, `xy′` and `w`; impose two
conditions.

*Value at `xy₀`:*

```
½ w² ‖xy₀ − xy′‖² = L₀
```

*Gradient at `xy₀`:*

```
∇_xy [½ w² ‖xy − xy′‖²]|_{xy₀} = w² (xy₀ − xy′) = ∇L₀
```

From the gradient condition, `xy₀ − xy′ = ∇L₀ / w²`. Substituting into the value
condition:

```
½ w² · ‖∇L₀‖²/w⁴ = L₀   →   w² = ‖∇L₀‖² / (2L₀)
```

giving the two formulas in the docstring:

```
w   = ‖∇L₀‖ / √(2L₀)
xy′ = xy₀ − ∇L₀ · 2L₀ / ‖∇L₀‖²
```

**A geometric reading.** `xy′` is a *virtual target spot diagram*. It is the point you
reach by stepping from the current spots along the negative loss gradient, by exactly
the distance at which the linear model predicts zero loss. The residual is then "how far
are my rays from that target", scaled so the sum of squares reproduces the true loss.

The construction has turned "minimize a scalar image loss" into **"move each ray toward
its own target position"** — precisely the form classical lens design has always used,
except the targets come from an image-quality gradient instead of a paraxial ideal.

## 8.3 The code, line by line

```python
# Compute L0 and grad0 (note that they are returned as constants)
with torch.inference_mode(False):
    with torch.no_grad():
        grad, (scalar_loss, _) = torch.func.grad_and_value(
            self.scalar_e2e_loss, 0, has_aux=True
        )(xy.detach(), batch, self.lens_e2e_loss_fn)
```

`imaging_system.py:474-479`. Several things at once:

- `torch.func.grad_and_value(f, 0, has_aux=True)` — functional autograd. Differentiates
  argument **0** (`xy`) and returns value and gradient in one pass. Using the functional
  API rather than `backward()` keeps this gradient out of the main graph entirely.
- `xy.detach()` — the gradient is with respect to the spot positions **as data**. The
  lens's own graph is deliberately cut here.
- `with torch.no_grad()` — the comment says *"ignore potential optimizable parameters in
  IRM"*. The image restoration model may have its own trainable weights; when computing
  the lens's residual you do not want to accumulate gradients into them. The two
  optimizations are separated (Chapter 12).
- `with torch.inference_mode(False)` — inference mode is a stronger no-grad that
  *forbids* recording. Lightning may put validation under it; this re-enables the
  autograd machinery locally so `grad_and_value` can work at all.

**`L₀` and `∇L₀` come out as constants.** They carry no graph. This is the crucial
property: the expensive image pipeline is evaluated *once*, and the resulting residual
vector is a cheap analytic function of `xy`.

```python
ray_valid = xy.isfinite().all(dim=0)
grad_valid = grad.where(ray_valid, 0.0).view(-1)
grad_norm_squared = grad_valid @ grad_valid
```

`imaging_system.py:481-484`. `‖∇L₀‖²` computed with **failed rays excluded** — recall
from Chapter 5 that `compute_spot_diagrams` marks invalid rays as `inf`
(`imaging_system.py:547-549`). A single stray value here would corrupt both `xy′` and
`w` for every ray in the system, since the norm is global.

```python
xy_control = xy.detach() - grad * 2 * scalar_loss / grad_norm_squared
```

`imaging_system.py:487`. The virtual target `xy′`, exactly as derived. The variable is
named `xy_control` — think of it as a control point the rays are pulled toward.

```python
weight = (compensation_factor * grad_norm_squared / (2 * scalar_loss)).sqrt()
residual_vector = weight * (xy - xy_control)
residual_vector = residual_vector[ray_valid.broadcast_to(xy.shape)]
```

`imaging_system.py:510-514`. Note carefully: `xy_control` is a **constant** (built from
detached quantities), but `xy` on line 513 is the *live, differentiable* spot diagram.
So the residual vector is differentiable with respect to the lens parameters through the
ray tracer — full geometric structure retained — while the image pipeline appears only
through the two constants.

**That is the whole trick.** One expensive evaluation of the image loss, converted into
a per-ray least-squares problem that LM can attack with its full machinery.

## 8.4 Clipping, and why it needs a compensation factor

`imaging_system.py:489-507`:

```python
if boundaries is not None and xy_centroid is not None:
    xy_min = xy_centroid - boundaries
    xy_max = xy_centroid + boundaries
    clipped_xy_control = xy_control.clip(min=xy_min, max=xy_max)
    compensation_factor = torch.dot(*((xy.detach() - xy_control).view(-1),) * 2) \
                        / torch.dot(*((xy.detach() - clipped_xy_control).view(-1),) * 2)
    xy_control = clipped_xy_control
```

**Why clip.** `xy′` is derived from a *linear* extrapolation. If `‖∇L₀‖` is small
relative to `L₀`, the step `2L₀/‖∇L₀‖²` is enormous and the target lands far outside the
PSF grid the simulator supports (Chapter 10). The optimizer would then chase rays into a
region where the loss model is meaningless.

**Why compensate.** Clipping shortens `xy₀ − xy′`, which would reduce `½‖l‖²` below
`L₀` and silently under-report the loss. The factor

```
compensation = ‖xy₀ − xy′‖² / ‖xy₀ − xy′_clipped‖²
```

is folded into `weight` **inside the square root** (`imaging_system.py:510`), so
`w² → compensation · w²` and the identity `½‖l‖² = L₀` is preserved exactly. The comment
at `imaging_system.py:500-501` states this intent.

The direction of each residual changes under clipping; the total magnitude does not.
That is the right trade: keep the loss value honest, accept a slightly rotated descent
direction.

## 8.5 Verifying the identity — measured, not asserted

The construction is only valid if `½‖l(xy₀)‖² = L₀` really holds in the shipped code. I
instrumented it (`instrument_trace.py`, section `6_generalized_aberrations`) on the
end-to-end toy config with a real 1024×1024 image batch.

| quantity | measured |
|---|---|
| `batch_images` | `[1, 3, 1024, 1024]`, range 0.0–1.0 |
| `xy_spot_diagrams` | `[2, 11, 256, 1, 1]`, grad ✓ |
| `L0_scalar_e2e_loss` | **0.0044965702** |
| `grad0` | `[2, 11, 256, 1, 1]`, range ±1.1e−3, **no grad** |
| `grad_norm_squared` | **2.225276e−4** |
| `xy_control` | `[2, 11, 256, 1, 1]`, **no grad** |
| `xy_control_shift` (= `xy₀ − xy′`) | max magnitude **0.0445 mm** |
| `weight_w` | **0.1573027223** |
| `e2e_residual_vector` | **[5632]**, grad ✓ |
| `½‖l‖²` | **0.004496567** |

Check the two derived quantities by hand against the formulas:

```
w   = √(‖∇L₀‖² / 2L₀) = √(2.225276e−4 / (2 × 0.0044965702))
    = √(0.024745…)     = 0.157305        ✓  matches 0.1573027
step = 2L₀/‖∇L₀‖²      = 0.0089931/2.2253e−4 = 40.41
```

and the identity itself:

```
½‖l(xy₀)‖² = 0.004496567
L₀         = 0.0044965702
absolute error = 3.3e−09
relative error = 7.249e−07
```

**The identity holds to seven significant figures** — float32 rounding, nothing more.
The construction is implemented exactly as the paper describes it.

I also reproduced the entire computation by hand from `L₀` and `∇L₀` — recomputing
`xy′`, `w`, and `l` independently — and compared against what the shipped method
returns:

```
reference_matches_manual : True
```

Both vectors are `[5632]` with identical min/max (−0.0070060645, +0.0066540153). **You
can state in a defense that you verified the paper's central identity numerically
against the released implementation, and it reproduces to 7e−7 relative.**

## 8.6 Reading the numbers

**Why 5632 and not 5704?** The GTRA vector is `2 × 11 × 256 = 5632` — two coordinates
per ray, all rays valid. It is exactly the same length as the classical
`transverse_ray_aberration` block from Chapter 7, because it is the same object with
different targets. The extra 72 entries in the full vector are the manufacturability
hinges.

**`xy_control_shift` max is 0.0445 mm.** The virtual targets sit up to 45 µm from where
the rays currently land — comparable to the 30 µm spot spread measured in Chapter 5.
The linearization is being asked to extrapolate about one spot diameter. That is a
reasonable regime for a first-order model, and it is why the clipping guard exists for
when it is not.

**`grad0` has `requires_grad = False`.** Confirmed in the trace. The image pipeline
contributes no graph; only its value and slope.

## 8.7 What the approximation costs

Be able to state the limitations, because this is where a sharp supervisor will push.

**It is first-order in `xy` only.** The true `L(xy)` has curvature; the model does not
capture it. The residual is exact in value and gradient at `xy₀`, and degrades away from
it.

**It must be rebuilt every iteration.** `xy′` and `w` are only valid at the current spot
diagram. Chapter 9 shows the optimizer recomputing them each outer iteration — the
image pipeline runs once per iteration, not once per trial step. Trial steps within an
iteration reuse the same `xy′`, which is precisely what makes them cheap.

**It requires `L₀ > 0` and `‖∇L₀‖ > 0`.** Both appear in denominators
(`imaging_system.py:487, 510`). At a perfect reconstruction both vanish and the
construction is undefined. Not a practical concern — the loss never reaches zero — but
it is the honest answer to "what if the loss is zero".

**It assumes the loss depends on the lens only through `xy`.** True for the geometric
PSF model of Chapter 10. It would need re-examination for a diffractive model where
wavefront phase matters independently of ray positions.

## 8.8 Why this beats the alternatives

| approach | optimizer | image-aware | convergence |
|---|---|---|---|
| classical spot-size residuals | LM | no | fast |
| Adam on scalar e2e loss | first-order | yes | slow, hyperparameter-sensitive |
| **GTRA** | **LM** | **yes** | **fast** |

The first row is Chapter 7: a proper vector, but the objective is geometric spot size,
which is a *proxy* for image quality — and a poor one when a restoration network sits
downstream, because a network can undo some blur but not others.

The second row is what most published end-to-end work does. It works, but it inherits
every difficulty of first-order optimization on a stiff, badly-scaled problem: learning
rates per parameter group, thousands of iterations, sensitivity to initialization.

The third row keeps LM's convergence and the end-to-end objective simultaneously, by
inserting the linearization at exactly the point where the pipeline stops being
geometric.

## 8.9 The scalar fallback

`e2e_vector_mode` has three states (`imaging_system.py:104-122`):

| value | name registered | behaviour |
|---|---|---|
| `None` | — | end-to-end disabled; classical residuals only |
| `False` | `e2e_scalar_loss` | scalar loss, one residual element |
| `True` | `generalized_transverse_ray_aberration` | the GTRA vector |

The `False` mode is the ablation baseline: the same pipeline, the same optimizer, but
the loss enters as a single element. It exists so the paper can measure what the vector
lift buys. The toy config `configs/toy/defaults.yml` sets `e2e_vector_mode: None`
(measured: `e2e_enabled = False`); `configs/toy/opt_scalar_mse.yml` selects the scalar
mode.

Note the assertion at `imaging_system.py:106-109`: enabling either end-to-end mode
requires an `optics_simulator`, since the loss cannot be evaluated without a PSF model.

## 8.10 What to take away

1. LM needs a **vector**; end-to-end image quality is a **scalar**. That incompatibility
   is the problem the paper solves.
2. The pivot is the **spot diagram `xy`** — the last geometric quantity in the pipeline.
3. Linearize the image loss in `xy`, then find the quadratic `½w²‖xy − xy′‖²` matching
   its value **and** gradient. Two conditions, two unknowns, closed form.
4. `xy′` is a **virtual target spot diagram**: step along `−∇L₀` to where the linear
   model predicts zero loss. `w = ‖∇L₀‖/√(2L₀)`.
5. `xy′` and `w` are **constants**; `xy` in `l = w(xy − xy′)` is **live**. Full ray-trace
   gradients survive; the image pipeline costs one forward and one backward pass.
6. Clipping bounds the extrapolation; the compensation factor keeps `½‖l‖² = L₀` exact.
7. **Measured:** `L₀ = 0.0044965702`, `½‖l‖² = 0.004496567`, relative error
   **7.25e−07**. Independently reproduced from `L₀` and `∇L₀`, matching the shipped
   method. The paper's central identity is verified.
8. Costs: first-order in `xy`, must be rebuilt each iteration, needs `L₀ > 0` and
   `‖∇L₀‖ > 0`.

---

*Previous: [Chapter 7 — The residual library](07_residuals.md)*
*Next: [Chapter 9 — The Levenberg–Marquardt optimizer](09_lm_optimizer_part1.md)*
