# Chapter 17 — Mapping `eisoptx` to your own `e2e-gtra-optics`

> **What this chapter is.** A side-by-side of the authors' released code and your
> reimplementation: what corresponds to what, where you diverged deliberately, where you
> diverged accidentally, and what is genuinely missing. This is the chapter to read before a
> meeting where you have to say *"my implementation does X, theirs does Y, and here is why."*
>
> Every claim below was checked against both trees.

---

## 17.1 The two codebases at a glance

|  | `eisoptx` (Côté et al.) | `e2e-gtra-optics` (yours) |
|---|---|---|
| total lines | 9,761 | 6,138 |
| core (non-viz, non-test) | ~4,000 | ~2,300 |
| visualization | 1,758 | 941 |
| tests | none shipped | **1,260** (`test_core.py`, `test_solves.py`) |
| config system | Lightning CLI + jsonargparse YAML | `PipelineConfig` dataclass + registries |
| training frame | PyTorch Lightning | plain loop (`JointOptimizer`) |
| precision | `float64` throughout | `float32` default |
| lens representation | sequence string parsed to events | `list[Surface]` dataclasses |
| ray shape | `[2, f, w, p, 1]` 5-D | `(F, W, P, 2)` 4-D |
| optimizer | `LMOptimizer(torch.optim.Optimizer)` | `LevenbergMarquardt` (standalone class) |

**You have tests and they do not.** That is worth saying out loud in a meeting: 1,260 lines
of test against a reference implementation shipping none. It is also the reason your
divergences are mostly *deliberate* — a test pins behaviour, so a change is a decision.

## 17.2 The correspondence table

| concept | `eisoptx` | yours | verdict |
|---|---|---|---|
| spot diagram container | raw tensor `[2,f,w,p,1]` | `SpotDiagram` dataclass, `base.py:28` | **yours is better structured** |
| lens object | `Lens`, `optics.py` | `RotationallySymmetricOptics`, `raytrace.py:206` | equivalent |
| surface | dict from sequence string | `Surface` dataclass, `raytrace.py:144` | yours is more explicit |
| dispersion (Eq. 8) | `hartmann_dispersion`, `optics.py:735` | `hartmann_index`, `raytrace.py:43` | **same model** |
| asphere sag (Eq. 9) | `evaluate_aspherical_profile`, `ray_tracing.py:285` | `asphere_sag`, `raytrace.py:62` | equivalent |
| sag derivative | inside marching iteration | `asphere_dsag_du`, `raytrace.py:78` | yours is separated out |
| Snell | `apply_snell_aspherical`, `ray_tracing.py:383` | `refract`, `raytrace.py:95` | equivalent |
| diffractive surface (Eq. 10) | `misc_surfaces.py` (442 lines) | `diffract` **stub**, `raytrace.py:126` | **gap** |
| pupil sampling | `ray_initialization.py` (663 lines) | `concentric_pupil`, `raytrace.py:183` | yours is simpler; see §17.6 |
| paraxial / ABCD | `paraxial_ray_tracing.py` (97) | `paraxial.py` (257) | **yours is more thorough** |
| solves | `optics.py` `curvature_solve` | `apply_solves`, `raytrace.py:409` | equivalent, see §17.5 |
| parameterization | `LensParameterization` + scaling | `_pack`/`_unpack`, `raytrace.py:373` | **theirs has variable scaling** |
| TRA residual (Eq. 3) | `residuals.py:61` | `tra_residuals`, `gtra.py:43` | **equivalent, verified** |
| GTRA lift (Eq. 5–6) | `imaging_system.py:474–513` | `gtra.py:60–169` | **equivalent, verified** |
| geometric constraints | `residuals.py` classes, wired in | `constraints.py` (321 lines), **not wired** | **§17.8 — real gap** |
| LM | `optimizers.py` | `lm.py` | see §17.4 |
| KDE PSF (Eq. 12) | `ray_analysis.py` `compute_psfs` | `kde_psf`, `kde_psf.py:71` | equivalent kernel |
| diffraction Airy | `simulation.py:604–680` | `airy_field`, `kde_psf.py:118` | yours is present but unused |
| spatially-varying conv (Eq. 13) | `SVOLAConvolution`, 9×9 patches | `ConvolutionImaging` — **shift-invariant** | **§17.7 — the biggest gap** |
| restoration | NAFNet (1.49 M) + Wiener + chainer | `TinyUNet`, `unet.py:30` | yours is a placeholder |
| joint loop | `imaging_system.py` `training_step` | `JointOptimizer.step`, `joint.py:137` | **§17.3** |

## 17.3 The joint loop: where you match and where you differ

Your `_optics_step` (`joint.py:97–120`) is the paper's algorithm, correctly:

```python
spot = optics.spot_from_theta(theta)
eps0 = spot.detach()
eps_leaf = eps0.clone().requires_grad_(True)         # a fresh leaf at eps
spot_obj = optics.spot_object_from_flat(eps_leaf)
L = self._task_loss_from_spot(spot_obj, scene, target)
grad_L, = torch.autograd.grad(L, eps_leaf)            # ONE backward pass
```

**The leaf trick is the right idea and it is cleaner than theirs.** `eisoptx` uses
`torch.func.grad_and_value` on a closure; you detach the spot, make it a leaf, rebuild a
`SpotDiagram` around it, and take a plain `autograd.grad`. Both give `∂L/∂ε` in one backward
pass through the image pipeline. Yours is easier to read and easier to test — and you did
test it.

Then:

```python
def residual(th):
    eps = optics.spot_from_theta(th)
    return gtra_residuals(eps, L_val, grad_L, eps0=eps0, ...)
lm = LevenbergMarquardt(residual, theta.detach())
theta_new = lm.run(self.cfg.lm_iters_per_step)
```

`L_val` is a float and `grad_L` a detached tensor, so the closure holds them constant — which
is exactly what Chapter 8 says the lift requires. **Correct.**

### Divergence 1 — inner LM iterations per outer step

`lm_iters_per_step: int = 3` (`joint.py:44`), and the demo uses 2 (`demo_toy_e2e.py:137`).

`eisoptx` does **one** LM step per training step. That is not a detail: the GTRA linearization
is exact in value and gradient only at the iterate where it was built (Chapter 16 §16.5).
Running 2–3 LM iterations against a *frozen* `(w, ε′)` means iterations 2 and 3 are optimizing
a surrogate built at a point you have already left.

Chapter 15 measured how fast that surrogate goes stale: `xy_control_shift` reaches 0.0445 mm
against a 0.0300 mm corner spot — the target sits ~1.5 spot radii away. After one accepted
step you have consumed a real fraction of that. Your inner loop is *not wrong* — it will
still descend, because the residual is a valid least-squares objective — but it is descending
on a stale model, and the extra iterations buy less than they cost.

**This is a defensible answer either way, but you need the answer.** If asked: *"I amortize
the backward pass through the image pipeline over 2–3 LM iterations, trading surrogate
fidelity for cost; the reference rebuilds every iteration."* Then say what you measured. If
you have not measured it, that is the experiment to run — it is one config change.

### Divergence 2 — the network step sees a detached spot

Yours, `joint.py:127-131`:

```python
with torch.no_grad():
    spot = self.optics.forward()              # optics frozen
capture = self.bridge.simulate(spot, scene, add_noise=True)
```

`eisoptx` wraps the PSF build in `torch.inference_mode(True)` for the same reason (Chapter
12 §12.7). **You and they agree**, by different mechanisms. The consequence is the same and
worth knowing: no lens gradient flows during the network half in either implementation.

### Divergence 3 — one scene per step, and the same scene for both halves

`scene_sampler()` returns one `(C,H,W)` pair; `eisoptx` batches through a Lightning
`DataLoader` with field-limit-tagged crops (Chapter 13 §13.5). Both use the **same sample for
both halves**, which matches. But you have no batch dimension and no field-of-view tagging,
because §17.7's shift-invariant convolution does not need it.

## 17.4 LM: four differences, three of which favour theirs

Your `lm.py` is 140 lines against their ~200, and it is a faithful implementation. The
differences:

**1. You form `JᵀJ` explicitly.** `lm.py:88`:

```python
JTJ = JT @ J                                   # (N,N)
...
A = JTJ + it_lam * D2
delta = torch.linalg.solve(A, -g)
```

`eisoptx` solves the stacked system `[J; √λD] Δ = −[ℓ; 0]` and never forms `JᵀJ` (Chapter 16
§16.10). Chapter 15 measured the difference: `J` spans ±486, `JᵀJ` spans ±2.25e6. **Forming
the normal equations squares the condition number.** In `float32` — your default — that is a
real precision loss, roughly halving your significant digits in the solve.

This is the one change with the clearest payoff. The fix is small:

```python
A = torch.cat([J, torch.diag(self.damping) * it_lam**0.5], dim=0)   # (M+N, N)
b = torch.cat([-r0, torch.zeros(self.N, dtype=r0.dtype)])
delta = torch.linalg.lstsq(A, b, driver='gelsd').solution
```

`gelsd` also gives you the minimum-norm solution on rank-deficient systems, which matters:
Chapter 15 measured `lstsq_rank = 14` of 20 at the converged optimum. Your
`torch.linalg.solve` on a singular `A` raises, and your fallback `lstsq` uses the default
driver.

**2. Your damping stores `√diag(JᵀJ)` and squares it; theirs stores column norms directly.**
`lm.py:78-79` vs `optimizers.py:87-90`. These are the same quantity — a column norm *is*
`√diag(JᵀJ)` — and both apply the same `beta`-ratcheted running maximum. **You match the
paper's Supp. S64 and you documented the floor (S65), which `eisoptx` does not have.** Your
`floor` is a genuine improvement: it stops a momentarily-insensitive parameter from getting
zero damping.

**3. Your acceptance test is `loss_new < TF * loss0` with `TF = 1`.** That is the paper's
rule exactly (Chapter 16 §16.10, divergence 3). `eisoptx` defaults `tolerance` to **2.0** and
only the toy and telephoto configs set it to 1.0. **You match the paper; the reference's
default does not.** Say that if the acceptance rule comes up.

**4. Your inner loop retries with increasing λ up to 30 times within one `step()`.**
`lm.py:96`. `eisoptx` proposes one step per `.step()` call and raises λ for the *next* call.
Yours converges in fewer outer iterations; theirs gives the outer loop a chance to log and
schedule between proposals. Neither is wrong. Yours costs up to 30 residual evaluations in a
bad iteration — cheap here because the residual is ray-trace-only, which is exactly the
property the lift buys.

**And one thing yours has that theirs does not.** `lm.py:105-114`:

```python
# Reject non-finite proposals BEFORE the loss comparison. A NaN theta traces to
# NaN ray coordinates, which the tracer sanitizes with nan_to_num -> the residual
# comes back all zeros and the loss is 0.0, i.e. the best step LM has ever seen.
```

That is a real bug you found and guarded, with the mechanism written down. `eisoptx` checks
`math.isfinite(loss_ratio)` (`optimizers.py:143`) — which catches a NaN *loss*, but not a NaN
`theta` that has been laundered into a finite zero loss by `nan_to_num`. **Your guard is
strictly stronger.** If a supervisor asks whether you just transcribed the reference, this
comment is the answer.

## 17.5 The lift itself: verified equivalent

Your `gtra_weight_and_target` (`gtra.py:60-75`):

```python
g2 = grad_L.dot(grad_L).clamp_min(eps)
w = g2 / (2.0 * L.clamp_min(eps))
eps_prime = eps0 - 2.0 * L * grad_L / g2
```

`eisoptx` (`imaging_system.py:484,487`):

```python
grad_norm_squared = grad_valid @ grad_valid
xy_control = xy.detach() - grad * 2 * scalar_loss / grad_norm_squared
```

**Identical**, term for term. Note the naming: your `w` is the paper's `w`, and you take
`torch.sqrt(w)` at `gtra.py:166`. `eisoptx`'s variable *called* `weight` already holds `√w`
(Chapter 16 §16.5). **Your naming matches the paper; theirs does not.** With `L₀ =
0.0044965702` and `‖∇L₀‖² = 2.225276e−04`, your `w` would be `0.02474414833` and theirs
`0.1573027223` — the same physics, and you should be ready to say which is which.

Your `clamp_min(eps)` guards on both `g2` and `L` are additions; `eisoptx` has no guard and
would divide by zero at a perfect optimum. Small, correct.

### The one substantive divergence in the lift

`clip_control_values` (`gtra.py:78-96`) does a per-coordinate clamp, and your own docstring is
honest about it:

> "v1 uses a simple per-coordinate clamp, which is the leading-order version of the paper's
> rescaling; the exact energy-preserving rescale is a documented refinement."

`eisoptx` does more (Chapter 8): it clips, then computes

```python
compensation_factor = ‖xy − xy_control‖² / ‖xy − clipped_xy_control‖²
```

and folds it into the weight so that `½‖ℓ‖² = L₀` **exactly** after clipping. Without it, the
surrogate's value no longer matches the true loss — the gradient direction survives, but the
scale is wrong, and the LM step length is computed against a loss that is too small.

**This is the highest-value thing to port**, and it is ~4 lines. You already know it is
missing and said so in a docstring; closing it converts a known approximation into a verified
identity. Chapter 15 measured the identity holding to a relative error of **7.249e−07** in
the reference.

### `tra_control_values` — an idea theirs does not have

`gtra.py:99-123` returns the `(w, ε′)` that make GTRA *equal* TRA, so you can drive TRA
through the GTRA code path rather than as a separate branch. `eisoptx` has two separate
classes. **Yours is the better factoring** and it directly demonstrates the paper's
"GTRA generalize TRA" claim as executable code rather than prose. Show this one.

## 17.6 Ray initialization: 663 lines vs 18

`concentric_pupil` (`raytrace.py:183-200`) samples the full disk on rings of `6i` points with
angular jitter. That is a reasonable sampler and the jitter is a good touch — it decorrelates
the sampling pattern from the surface geometry.

What `eisoptx`'s 663 lines add:

- **iterative ray aiming** for vignetting and pupil aberration (the paper's §3.4 cites Côté
  et al. 2023b for this),
- telecentric and finite-conjugate configurations,
- alternative pupil patterns,
- interior aperture stops.

Your `aim_rays` (`raytrace.py:712`) is a stub. **When does this matter?** When the stop is
not the first surface, or when the pupil is aberrated enough that a ray aimed at the
paraxial pupil misses the real one. Chapter 4 noted that the toy config's stop *is* the first
surface, so aiming is trivially satisfied — which is why your simplification works for the
problem you are running and would break on the paper's telephoto study.

**Say it that way.** Not "I did not implement ray aiming," but "the stop is the first surface
in my test configuration, so exact aiming is the identity; a shifted stop would need the
iterative scheme from Côté et al. 2023b."

## 17.7 The imaging model: the biggest gap

`ConvolutionImaging.psf` (`imaging.py:57-84`) returns **one** PSF — `(1, G, G)` — selected at
`self.field_index`, and `convolve` applies it to the whole scene.

`eisoptx` builds a 9×9 grid of field-dependent PSFs, interpolates and rotates them for
rotational symmetry, and applies `SVOLAConvolution` patch-wise with overlap-add (Chapter 10).

**This is not a refinement, it is a different physical model.** Spatially varying blur is the
entire reason task-driven design differs from classical design: the paper's Fig. 4a result is
that GTRA finds a design with *radially narrow, angularly wide* PSFs — a trade-off that only
exists because blur varies across the field. **A shift-invariant simulator cannot represent
the paper's headline result.**

Two consequences to be honest about:

1. Your pipeline demonstrates the *machinery* of GTRA (lift, LM, joint loop) but not the
   *phenomenon* it was built for.
2. Your `grid_half_extent` clipping bound is a single number, whereas field-dependent PSF
   grids need per-field bounds.

Your `offgrid_fraction` + warning (`imaging.py:69-82`) is excellent practice and shows you
understood the failure mode — a renormalized clipped PSF looks plausible and quietly
falsifies every PSNR downstream. `eisoptx` has `distribute_unaccounted_energy` for the same
problem but warns about nothing.

**The upgrade path**, if you want it: `kde_psf` already returns `(F, W, G, G)` — all fields.
You are discarding `F−1` of them at `imaging.py:83`. A patch-wise convolution over field
zones would use them.

## 17.8 The constraint library that is never called

`constraints.py` is 321 lines: `ray_path_residuals`, `ray_angle_residuals`,
`surface_normal_residuals`, `distortion_residuals`, a `GeometricConstraints` bundler with a
`from_optics` classmethod and a `report`. It mirrors `eisoptx`'s residual classes closely and
it is tested.

**And nothing in `e2e_optics/` calls it.** `joint.py`'s `residual(th)` returns
`gtra_residuals(...)` alone. Grepping the package for `geometric_residuals` or
`GeometricConstraints` outside `constraints.py` finds only the `__init__.py` re-export.

Chapter 14 measured what this costs. The reference's residual vector is

```
5704 = 5632 (GTRA/TRA) + 14 (ray_path) + 0 (ray_angle) + 58 (surface_normal)
```

72 of 5704 entries — 1.3% — are geometric constraints. They are 1.3% of the *length* and
100% of the reason the design stays manufacturable: without them nothing stops surfaces
interpenetrating or edge thicknesses going negative. An unconstrained LM will happily walk
into a physically impossible lens with an excellent merit function.

**The fix is one line in `joint.py`:**

```python
def residual(th):
    eps, probes = optics.spot_from_theta(th, probes=True)
    return torch.cat([
        gtra_residuals(eps, L_val, grad_L, eps0=eps0, ...),
        self.constraints(probes),
    ])
```

`_trace_packed` already accepts `probes=True` (`raytrace.py:434`) and its docstring says the
probes are "reused to derive element apertures and geometric constraints." **The wiring was
designed and never connected.** This is the single highest-priority gap in your
implementation, and it is the one a supervisor is most likely to find by asking "what stops
the optimizer producing a nonsense lens?"

## 17.9 Restoration

| | `eisoptx` | yours |
|---|---|---|
| network | NAFNet, 1,493,939 params | `TinyUNet`, width 16 |
| Wiener stage | `ParametrizedWienerDeconvolution`, learnable log-SNR | `WienerDeconv` — `forward` is a `pragma: no cover` stub |
| chaining | `ImageRestorationChainer` (NAFNet→Wiener→NAFNet) | none |
| PSF passed to net | yes (and ignored by NAFNet) | `psf=None` explicitly |
| loss asymmetry | MAE for net, MSE for lens | single `loss_fn` |

`TinyUNet` is a placeholder and labelled as one. Two things worth knowing:

**The MAE/MSE asymmetry is a real design choice you are missing.** The paper (§4): MAE for
IRM steps "in line with standard practice," MSE for lens steps "for additional stability."
Your `JointOptimizer` takes one `loss_fn` for both. The reason it matters: MSE's gradient is
proportional to the error, so the GTRA lift's `∇L₀` is smoother and the linearization more
reliable; MAE's gradient is a sign, which makes `‖∇L₀‖²` nearly constant and the weight `w`
poorly scaled. **Passing MAE into a GTRA lift is the kind of thing that quietly degrades
convergence.** Two `loss_fn` arguments would fix it.

**Your `psf=None` is more honest than theirs.** Chapter 11 found that `NAFNet.forward(inp,
*args)` accepts a PSF and ignores it — so in `eisoptx` only the Wiener stage actually uses
the PSF, and in a NAFNet-only config the restoration network never sees the optics at all
except through the blurred image. You pass `None` and mean it.

## 17.10 What is not in your implementation at all

| missing | `eisoptx` location | matters? |
|---|---|---|
| spatially-varying convolution | `simulation.py:467` | **yes — §17.7** |
| geometric constraints wiring | `residuals.py`, `imaging_system.py` | **yes — §17.8** |
| clipping compensation factor | `imaging_system.py:489-507` | **yes — §17.5** |
| glass optimization + catalog mesh | `callbacks.py:62-313` | only if you optimize glass |
| diffractive/metasurface surfaces | `misc_surfaces.py` (442 lines) | **yes for your thesis topic** |
| iterative ray aiming | `ray_initialization.py` | only with a non-first stop |
| Airy diffraction in the pipeline | `simulation.py:604-680` | you have `airy_field`, unused |
| variable scaling | `parameterization.py` | see below |
| CODE-V export | `callbacks.py` | for commercial validation |
| `float64` | `precision: 64` | see below |

**Variable scaling and precision are the two quiet ones.** `eisoptx` scales variables so a
curvature and a spacing have comparable magnitude before they enter the parameterization, and
runs everything in `float64`. You run `float32` with unscaled variables *and* form `JᵀJ`.
Those three compound: unscaled variables give a badly conditioned `J`, `JᵀJ` squares that
condition number, and `float32` has ~7 digits to absorb it. Chapter 15 measured the reference's
damping terms spanning `0.3719` to `1501.38` — four orders of magnitude across 20 variables,
*with* Marquardt scaling active. **Of the three, switching to `float64` is a one-line change
and the cheapest insurance.**

## 17.11 A prioritized list

If you are going to spend effort, in this order:

1. **Wire the geometric constraints into the LM residual** (§17.8). One line; the probes
   already exist; without it the optimizer is unconstrained.
2. **Add the clipping compensation factor** (§17.5). Four lines; converts a documented
   approximation into an exact identity you can verify to 1e−07 as Chapter 15 does.
3. **Switch to `float64`** and stop forming `JᵀJ` (§17.4). Both are small and both address
   conditioning, which is the thing most likely to make your LM behave differently from theirs
   for reasons you cannot explain.
4. **Separate the lens loss from the network loss** (MSE / MAE, §17.9).
5. **Measure whether `lm_iters_per_step > 1` helps** (§17.3). This is an experiment, not a
   fix, and the result is publishable either way.
6. **Field-dependent PSFs and patch-wise convolution** (§17.7). The largest piece of work and
   the one that unlocks the paper's actual phenomenon.
7. Diffractive surfaces (`misc_surfaces.py` as the reference), if metasurfaces stay in scope
   for your thesis.

## 17.12 What to say when asked "is this just a reimplementation?"

It is not, and there are specific answers:

1. **The NaN-launder guard** (`lm.py:105-114`) is a bug the reference has, that you found,
   fixed, and documented the mechanism for.
2. **The damping floor** (`lm.py:78`) implements Supp. S65, which `eisoptx` does not.
3. **`TF = 1` acceptance** matches the paper; `eisoptx`'s default `tolerance = 2.0` does not.
4. **`tra_control_values`** (`gtra.py:99`) makes the paper's "GTRA generalizes TRA" claim
   executable instead of prose.
5. **`offgrid_fraction` + the clipping warning** (`imaging.py:69`) catches a silent
   falsification mode that `eisoptx` does not warn about.
6. **`w` is named `w`** and holds `w`, where the reference's `weight` holds `√w`.
7. **1,260 lines of tests** against a reference that ships none.
8. **A cleaner lift interface**: the detach-and-relift-as-leaf pattern (`joint.py:104-109`)
   against their `torch.func` closure.

And the honest counterweight, which you should offer before it is asked: the imaging model is
shift-invariant, the constraints are unwired, and the clipping is leading-order. All three are
known, all three are documented in your own docstrings, and §17.11 orders them.

## 17.13 What to take away

1. Yours is **6,138 lines to their 9,761**, with 1,260 lines of tests against their zero.
2. **The lift is verified equivalent**, term for term — `gtra.py:60-75` against
   `imaging_system.py:484-487`.
3. Your `w` is the paper's `w`; their `weight` is `√w`. **Yours matches the paper's notation.**
4. Your LM matches the paper on **acceptance (`TF=1`)** and **damping floor (S65)** where the
   reference's defaults do not, and carries a **NaN guard the reference lacks**.
5. Your LM forms `JᵀJ` explicitly, squaring a condition number the reference measured at
   ±486 → ±2.25e6, **in `float32`**. Highest-value correctness fix.
6. **The constraint library is written, tested, and never called.** 72 of 5704 residuals in
   the reference; 1.3% of the length and all of the manufacturability.
7. **The clipping compensation factor is missing** — the surrogate's value no longer equals
   `L₀`, so the LM step length is computed against a loss that is too small.
8. **The imaging model is shift-invariant**, so the paper's headline result — radially narrow,
   angularly wide PSFs — is not representable in your pipeline as it stands.
9. `lm_iters_per_step = 2–3` optimizes a surrogate built at a point already left; the
   reference rebuilds every iteration. Defensible, but you need the measurement.
10. Ray aiming is a stub, and that is **fine for a first-surface stop** — say it that way.
11. Missing MAE/MSE asymmetry: **feeding an MAE gradient into the GTRA lift gives a poorly
    scaled `w`**, since MAE's gradient is a sign.
12. `float64` is a one-line change and the cheapest of all the fixes.

---

*Previous: [Chapter 16 — The paper, equation by equation](16_paper_map.md)*
*Next: [Appendix A — Supervisor defense bank](A_defense_bank_part1.md)*
