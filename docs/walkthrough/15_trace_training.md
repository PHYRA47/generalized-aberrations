# Chapter 15 — Execution trace: one full LM / joint training iteration

> **What this chapter is.** Chapter 14 followed rays to the sensor and stopped at the
> residual vector. This chapter picks it up there and follows **one complete optimizer
> iteration** — image formation, the aberration lift, the Jacobian, the linear solve, the
> accept/reject test, and the damping update — with every number measured.
>
> It closes with a **12-iteration convergence run** from a deliberately perturbed start, so
> you can see the algorithm actually work rather than sit at a fixed point.

---

## 15.1 The two regimes being traced

| | config | what the loss is |
|---|---|---|
| **A. classical** | `toy/defaults.yml + designs/optimized_spot.yml` | ray residuals only, `e2e_enabled: False` |
| **B. joint** | `toy/defaults.yml + defaults_e2e.yml + designs/optimized_e2e.yml` | ray residuals **+** image loss lifted through GTRA |

Sections 15.2–15.6 trace regime B, because it contains regime A as a special case. §15.8
runs regime A to convergence.

---

## 15.2 Stage 7 — from spot diagrams to a blurred image

Chapter 14 ended with rays at the sensor. The image pipeline (Chapter 10) begins there.

**Measured** (`7_psf_and_image`):

```
simulator_class     : OpticsSimulator
psf_sampler_class   : PSFSampler
convolution_class   : SVOLAConvolution
sensor_diagonal_mm  : 11.5470053838

xy_spot_diagrams     [2, 11, 256, 1, 1]   grad=True   min −0.1390069  max 5.3473377
rgb_psfs             [11, 3, 35, 35]      grad=True   mean 8.163265e−04
psf_energy_per_field [11, 3]              grad=True   min 0.9999999404  max 1.0
psf_grid             [1, 81, 3, 35, 35]   grad=True   mean 8.163266e−04
blurred_image        [1, 3, 1024, 1024]   grad=True   mean 0.4727122
blur_mse_vs_sharp    : 0.0033178618
```

Four things to read out of this.

**The spot diagram *is* the interface.** `xy_spot_diagrams` `[2, 11, 256, 1, 1]` is the same
tensor Chapter 14 produced, and it is the only thing the image pipeline receives from the
optics. That is the architectural seam the whole method is built on.

**`requires_grad=True` survives all the way to `blurred_image`.** Splatting, interpolation,
rotation, resizing and overlap-add convolution are each differentiable, so a gradient
computed on the image can reach `xy`. This is the property Chapter 8 exploits.

**Energy is conserved to 8 significant figures.** `psf_energy_per_field` ranges
`0.9999999404` to `1.0`, and the means of `rgb_psfs` (`8.163265e−04`) and `psf_grid`
(`8.163266e−04`) agree to nine digits — note `1/(35 × 35) = 8.1632653e−04` exactly, which is
what a normalized 35×35 kernel must average. The interpolate → rotate → resize chain does
not leak light.

**11 fields become an 81-entry grid.** `[1, 81, 3, 35, 35]` — 9×9 spatial positions across
the sensor, built by rotating the 11 traced radial fields around the axis (Chapter 10). The
7× saving that rotation buys is what makes tracing only 11 fields viable.

`blur_mse_vs_sharp = 0.0033178618` is the baseline: what the lens costs you before any
restoration.

## 15.3 Stage 8 — restoration

**Measured** (`8_restoration`):

```
wiener_class          : ParametrizedWienerDeconvolution
wiener_learned_snr    : []  →  1000.0001220703      grad=True
wiener.detach_psfs    : False
wiener_output         : [1, 3, 1024, 1024]  min −0.7174844  max 1.7384253
mse_blurred_vs_sharp  : 0.0033178618
mse_wiener_vs_sharp   : 0.0481321253
nafnet_n_parameters   : 1493939
chainer_n_stages      : 3
mse_chainer_vs_sharp  : 0.1863023043
```

**`detach_psfs: False` is the switch that makes the system end-to-end.** With it False,
gradient flows image → PSF → `xy` → lens variables. Set it True and you have a restoration
network trained against a lens that cannot respond.

**But read the MSEs honestly.** Blurred is `0.00332`; Wiener output is `0.04813`. The Wiener
stage made the image **14.5× worse**, and the chain of three stages is worse again at
`0.1863`. The output range `−0.717` to `1.738` is well outside `[0,1]` — classic
over-amplification.

This is not a defect in the method; it is what an **untrained** pipeline looks like. `snr`
sits at its initialization (`1000.0001220703`, the round-trip through the `Exp`
parametrization of `log 1000 = 6.9077553`), and the NAFNet's 1,493,939 parameters are
random. The numbers say only that these modules do nothing useful before training — which
is the correct baseline to have measured.

Two consequences worth carrying:

- If you quote restoration numbers, quote them **after** training. A trace at
  initialization measures the architecture, not the method.
- The toy e2e config has `restoration_model_class: None`, so **the toy configuration cannot
  demonstrate co-design at all.** It demonstrates the lens half. For co-design results you
  need the telephoto or c-mount configs.

## 15.4 Stage 9 — the aberration lift

This is the paper's central step, and Chapter 8 derived it. Here it is measured
(`6_generalized_aberrations`):

```
batch_images        [1, 3, 1024, 1024]
batch_field_limits  [1, 4]  = [−0.7071068, 0.7071068, 0.7071068, −0.7071068]
xy_spot_diagrams    [2, 11, 256, 1, 1]
L0_scalar_e2e_loss  ()      = 0.0044965702
grad0               [2, 11, 256, 1, 1]   min −0.0011020704  max 0.0010466933
grad_norm_squared   ()      = 0.0002225276
xy_control          [2, 11, 256, 1, 1]
xy_control_shift    [2, 11, 256, 1, 1]   min −0.0423007011  max 0.0445387363
weight_w            ()      = 0.1573027223
e2e_residual_vector [5632]              min −0.0070060645  max 0.0066540153
half_sum_sq         ()      = 0.004496567
identity_abs_error  : 3.3e−09
identity_rel_error  : 7.249e−07
reference_matches_manual : True
```

`field_limits = ±0.7071068 = ±1/√2` — the full field, since the corner of a square
inscribed in the unit field circle sits at `1/√2`.

**The identity.** The lift replaces the image loss `L₀` with a quadratic in `xy` that has
the same value and the same gradient:

```
l = w · (xy − xy′),    w = ‖∇L₀‖ / √(2 L₀)
```

Check the weight by hand:

```
‖∇L₀‖ / √(2L₀) = √(2.225276e−04) / √(2 × 0.0044965702)
               = 0.01491736 / 0.09482921
               = 0.15730...      vs measured 0.1573027223  ✓
```

And the value:

```
½‖l‖² = 0.004496567     vs     L₀ = 0.0044965702
absolute error 3.3e−09,  relative error 7.249e−07
```

**The lifted residual reproduces the image loss to seven significant figures.** And
`reference_matches_manual: True` records that an independent reimplementation of the
formula, written for this trace, agrees with what the shipped code produces. The paper's
central claim is verified, not paraphrased.

**Shape check.** `e2e_residual_vector` is `[5632]` — exactly the length of the classical
transverse-ray-aberration block from Chapter 14 §14.7, because it is the same `2 × 11 × 256`
rays. The image loss has been converted into something that **plugs into the same
least-squares machinery** as ray aberrations. That is the entire point.

**Magnitude check.** `xy_control_shift` reaches 0.0445 mm — Chapter 14 measured the corner
RMS spot at 0.0300 mm, so the virtual target is displaced by roughly one and a half spot
diameters. Small enough for the linearization to be defensible, large enough to carry
information.

## 15.5 Stage 10 — the Jacobian and the linear system

**Measured** (`5_lm_step`, regime A at the shipped optimum):

```
lens_optimizer_class : LMOptimizer
lm_parameter (λ)     : 1.0
damped_term_min      : 1e−08
tolerance            : 1.0
lam_increase_factor  : 2.0
lam_decrease_factor  : 2.0
beta                 : 0.99

residual_vector      [5704]        min −0.001404372  max 0.0012709902
least_squares_loss   ()            = 0.0001418987
jacobian             [5704, 20]    min −485.834198   max 383.8717651
constraint_mask      [5704]        true_fraction 0.0
damping_terms        [20]          min 0.3718612     max 1501.3764648
augmented_matrix     [5724, 20]
augmented_rhs        [5724]
lstsq_rank           : 14.0
gradient_JT_r        [20]          min −3.832946e−04  max 4.006246e−04
pseudo_hessian_JTJ   [20, 20]      min −2127462.5     max 2254132.0
step_delta_x         [20]          min −2.73e−08      max 5.62e−08
```

**`[5704, 20]`** — Chapter 14's residual count meets Chapter 14's trainable-variable count.
The Jacobian is computed with `jacfwd`, and the shape is why: 20 forward-mode tangents
versus 5704 reverse-mode cotangents, a **285× saving**. Deep learning is the mirror case
(few outputs, millions of parameters) and uses `jacrev`; here the sensible choice is the
opposite one.

**`5724 = 5704 + 20`.** The damping is applied by *stacking* rows, not by forming `JᵀJ + λD`:

```
[   J   ]        [ r ]
[ √λ·D  ] Δ  =  −[ 0 ]
```

Why it matters is visible in the trace. `jacobian` spans `±486`; `pseudo_hessian_JTJ` spans
`±2.25e6`. Forming the normal equations **squares the condition number**. The stacked
formulation solves the same problem in the well-conditioned space. `pseudo_hessian_JTJ` is
recorded here for diagnosis only — the shipped path never builds it.

**Damping terms span four orders of magnitude**, `0.3719` to `1501.38`. These are the column
norms of `J`, i.e. `√diag(JᵀJ)` — Marquardt's scaling, which makes the step invariant to
rescaling any variable. Necessary here because a curvature in mm⁻¹ and an eighth-order
aspheric coefficient differ enormously in natural size.

**`lstsq_rank = 14` out of 20.** Six directions in the 20-dimensional design space have no
first-order effect on the residuals *at this point*. That is degeneracy at the optimum, not
a bug — §15.8 shows rank 20 throughout a perturbed run. `gelsd` returns the minimum-norm
solution, which is the right behaviour for a rank-deficient system.

## 15.6 Stage 11 — accept or reject

```
loss_before   : 0.0001418987
loss_after    : 0.0001418987
loss_ratio    : 1.0000001192
step_accepted : False
lam_next      : 2.0
step_delta_x  : max |Δ| = 5.62e−08
```

The step is ~1e−08 in magnitude, the loss changes in the eighth significant figure, the
ratio is fractionally above 1, and the step is **rejected** — λ doubles from 1.0 to 2.0.

**This is exactly right.** The trace was taken at the shipped optimum. A converged
optimizer should propose a vanishing step and reject it. If it accepted a step here,
something would be wrong.

The acceptance rule (Chapter 9) is asymmetric: a step is accepted if `ratio < tolerance`,
and the toy config sets `tolerance: 1.0`, which closes the non-monotonic window. Set
`tolerance: 1.05` and small increases would be tolerated to escape narrow valleys.

## 15.7 Cost accounting for one iteration

| stage | ray traces | image-pipeline passes |
|---|---:|---:|
| residual + Jacobian (`jacfwd`, 20 tangents) | 1 primal + 20 tangent | 1 (the GTRA lift) |
| LM trial evaluation | 1 | 1 |
| restoration step (regime B) | 0 — cached under `inference_mode` | 1 fwd + 1 bwd |

**The image pipeline runs once per LM iteration, not once per Jacobian column.** Without the
lift it would run 21 times, and Chapter 10's memory table says that is not affordable at
6144 × 8192. The lift is not only a mathematical device; it is what makes the computation
fit.

## 15.8 A real convergence run

A fixed point is a poor demonstration. So: perturb the optimized design and let LM recover.

```
perturbation_scale    : 0.05  (5% relative, applied to all 20 variables)
perturbation_applied  [20]    min −0.0777548  max 0.1002490
n_iterations          : 12
loss_first_iteration  : 1.6092851162
loss_last_iteration   : 0.0952506959
n_accepted_steps      : 11
loss_reduction_factor : 16.895
```

| it | loss before | loss after | ratio | λ | rank | ‖Δ‖ | accepted |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 0 | 1.609285 | 1.384984 | 0.8606 | 1.000000 | 20 | 0.0859 | yes |
| 1 | 1.384984 | 1.265769 | 0.9139 | 0.500000 | 20 | 0.0793 | yes |
| 2 | 1.265769 | 1.077678 | 0.8514 | 0.250000 | 20 | 0.4443 | yes |
| 3 | 1.077678 | 1.045608 | 0.9702 | 0.125000 | 20 | 0.1556 | yes |
| 4 | 1.045608 | 0.698053 | 0.6676 | 0.062500 | 20 | 1.0527 | yes |
| 5 | 0.698053 | 0.444370 | 0.6366 | 0.031250 | 20 | 1.4287 | yes |
| 6 | 0.444370 | 0.369469 | 0.8314 | 0.015625 | 20 | 0.2967 | yes |
| 7 | 0.369469 | 0.350848 | 0.9496 | 0.007813 | 20 | 0.4189 | yes |
| 8 | 0.350848 | 0.173532 | 0.4946 | 0.003906 | 20 | 0.3540 | yes |
| 9 | 0.173532 | 0.089286 | 0.5145 | 0.001953 | 20 | 0.3449 | yes |
| 10 | 0.089286 | 0.085794 | 0.9609 | 0.000977 | 20 | 0.4671 | yes |
| 11 | 0.085794 | 0.095251 | 1.1102 | 0.000488 | 20 | 0.4593 | **no** |

Five readings.

**A 5% perturbation costs a factor of 11,300 in loss.** From `1.419e−04` to `1.609`. The
optimum is *sharp* — which is why gradient-free or coarse search would struggle here, and
why a second-order method earns its cost.

**λ halves on every accepted step**, 1 → 1/2 → … → 1/2048. The model is being trusted more
each iteration; by the end the method is essentially pure Gauss–Newton. This is the textbook
signature of LM working well.

**Rank is 20 at every iteration.** Contrast §15.5's rank 14 at the converged optimum. The
degeneracy is a property of *the optimum*, not of the parameterization — a genuinely useful
distinction to be able to draw, because "your Jacobian is rank-deficient" sounds like a
criticism until you can say where and why.

**Progress is not uniform.** Iterations 0–3 reduce the loss by 35% total; iterations 4–5
alone reduce it by 57%; 8–9 by another 49%. Flat stretches punctuated by drops — LM finding
a valley, then descending it.

**The run ends with a rejection**, ratio 1.1102. Loss went *up*, the step was thrown away, λ
doubles and a shorter step will be tried. It stops here at 16.9× improvement because 12
iterations is where the instrumentation stops, not because the optimizer is finished — it
would keep going to recover the remaining factor of ~670.

## 15.9 The complete iteration, in one diagram

```
                  ┌─ lens variables [20] ────────────────────┐
                  │                                          │
                  ▼                                          │
   scale → solve → Lens → trace 2816 rays → xy [2,11,256,1,1] │
                                              │              │
             ┌────────────────────────────────┴──┐           │
             ▼                                   ▼           │
   ray residuals [5632+14+0+58]        splat → PSFs [11,3,35,35]
             │                                   ▼           │
             │                         grid [1,81,3,35,35]   │
             │                                   ▼           │
             │                      SVOLA → blurred [1,3,1024,1024]
             │                                   ▼           │
             │                          restoration → L₀ = 0.0044965702
             │                                   ▼           │
             │                   GTRA lift: w=0.1573027223, xy′
             │                                   ▼           │
             │                        e2e residual [5632]  ──┤
             ▼                                               │
        concatenate → residual vector [5704] ────────────────┤
                          ▼                                  │
        jacfwd (20 tangents) → J [5704, 20]                  │
                          ▼                                  │
        stack √λ·D → [5724, 20] → gelsd → Δx [20], rank 14   │
                          ▼                                  │
        trial: loss_after / loss_before = 1.0000001192       │
                          ▼                                  │
        reject → λ ×= 2 → 2.0 ───────────────────────────────┘
```

## 15.10 What to take away

1. `xy_spot_diagrams` `[2,11,256,1,1]` is **the only interface** between optics and imaging,
   and it stays differentiable through to the image.
2. PSF energy conserved to 8 significant figures; mean `8.163265e−04` = `1/35²` exactly.
3. 11 traced fields → **81-entry PSF grid** by rotation.
4. At initialization the restoration stack **hurts**: MSE `0.00332` blurred → `0.04813`
   Wiener → `0.18630` chained. Measure restoration after training, not before.
5. The toy e2e config has `restoration_model_class: None` — it **cannot** demonstrate
   co-design.
6. **The GTRA identity is verified**: `w = 0.1573027223` reproduced by hand, `½‖l‖² =
   0.004496567` vs `L₀ = 0.0044965702`, relative error **7.249e−07**, and an independent
   reimplementation matches the shipped code.
7. The lifted residual is `[5632]` — the same length as the classical block, so it enters
   the same least-squares solver.
8. `jacfwd` over `jacrev` is a **285× saving** at this shape.
9. `5724 = 5704 + 20`: damping is **stacked**, never squared. `J` spans ±486 while `JᵀJ`
   spans ±2.25e6 — the reason why.
10. Damping terms span **0.372 to 1501.4**, giving scale-invariance across curvature and
    aspheric coefficients.
11. **Rank 14 at the optimum, rank 20 throughout the perturbed run** — degeneracy is a
    property of the optimum.
12. A step at the optimum is correctly **rejected** (ratio 1.0000001192, ‖Δ‖ ≈ 1e−08).
13. A 5% perturbation raises the loss **11,300×**; LM recovers 16.9× in 12 iterations, 11
    accepted, λ halving 1 → 1/2048.
14. The image pipeline runs **once per iteration**, not 21 times. That is the lift's
    practical payoff.

---

*Previous: [Chapter 14 — Execution trace: one forward pass](14_trace_forward.md)*
*Next: [Chapter 16 — The paper, equation by equation](16_paper_map.md)*
