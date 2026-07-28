# Chapter 0 — Orientation: what this codebase is, and how to read it

> **Who this is for.** You are about to defend this code in front of a supervisor.
> That means you need two different things at once: a *mental model* strong enough to
> answer "why is it done that way?", and *specific line references* strong enough to
> answer "show me where". This walkthrough gives you both. Every claim about behaviour
> in these chapters is either a file/line citation or a number measured by
> `instrument_trace.py` on the shipped toy configuration.

---

## 0.1 The one-paragraph version

`eisoptx` is a **differentiable optical design engine**. You hand it a lens described as
a string of surface types plus numeric parameters; it traces real rays through that
lens with PyTorch tensors, so every ray coordinate at the sensor carries a gradient
back to every curvature, thickness and glass variable. On top of that it puts a
**Levenberg–Marquardt optimizer** that reshapes the design to minimize a vector of
residuals — spot size, focal length, manufacturability constraints — and, crucially,
an **image-quality term** that has been converted from a scalar loss into a residual
vector so that LM can consume it. That conversion is the paper's contribution.

## 0.2 The problem the paper actually solves

This is the single most important idea in the repository, so it is worth being precise
about the difficulty before looking at any code.

Classical lens design is a **least-squares** problem. You have a long vector of
residuals `l(x)` — how far each ray lands from where it should, how far the focal
length is from target — and you minimize `½‖l(x)‖²`. Least-squares structure is not a
cosmetic detail: it is what lets you use **Levenberg–Marquardt**, which builds a local
quadratic model from the Jacobian `J = ∂l/∂x` and takes enormous, well-scaled steps.
Lens design has used LM (and its Damped Least Squares cousin) since the 1960s because
first-order gradient descent is hopeless on these problems — the parameters have wildly
different scales and the curvature is severe.

End-to-end optics-algorithm co-design wants something different. It wants to minimize
the quality of the **final processed image**: run the lens, form a PSF, blur an image,
push it through a restoration network, compare to ground truth. That is a **scalar**
loss `L`. Scalars have no residual vector and no Jacobian — only a gradient. So the
entire end-to-end literature falls back to Adam/SGD, and inherits their slow, fragile
convergence on optical parameters.

So you are stuck between:

| | optimizer | converges on optics | can use image loss |
|---|---|---|---|
| classical lens design | Levenberg–Marquardt | yes, excellently | **no** |
| end-to-end co-design | Adam / SGD | poorly | yes |

**The paper's move:** manufacture a residual vector whose sum of squares *equals* the
scalar image loss, and whose Jacobian is available. Then LM can optimize image quality
directly. These are the *generalized aberrations* of the title — they play the role
that transverse ray aberrations play in classical design, but they are derived from an
arbitrary downstream loss rather than from geometry.

The construction, given spot diagram `xy₀`, loss `L₀ = L(xy₀)` and gradient
`g₀ = ∇L(xy₀)`:

```
xy' = xy₀ − g₀ · 2L₀ / ‖g₀‖²          the "virtual target" the rays are driven toward
w   = ‖g₀‖ / sqrt(2L₀)                 a scalar weight
l(xy) = w · (xy − xy')                 the generalized aberrations
```

Evaluate that at `xy = xy₀` and you get `½‖l(xy₀)‖² = L₀` exactly. **I verified this
identity numerically**: on the toy configuration, `L₀ = 0.0044965702` and
`½‖l‖² = 0.004496567`, a relative error of `7.2 × 10⁻⁷` (float32 rounding). See
Chapter 8 for the derivation line by line, and `trace_dump.json` section
`6_generalized_aberrations` for the values.

The trick is that `l` depends on `xy` (which carries gradients to the lens parameters)
while `xy'` and `w` are **constants** — computed once and detached. So LM sees a
genuine least-squares problem, gets a real Jacobian, and takes real LM steps, but the
objective it descends is the image loss.

## 0.3 What is in the box

9,761 lines of Python, in one package:

| file | lines | what it is |
|---|---:|---|
| `eisoptx/utils/visualization.py` | 1758 | plotting; layout diagrams, spot diagrams, PSF grids |
| `eisoptx/imaging_system.py` | 1159 | **the core.** LightningModule tying everything together |
| `eisoptx/optimization/residuals.py` | 892 | the residual library — every optimization target |
| `eisoptx/modeling/simulation.py` | 848 | spot diagram → PSF → blurred image |
| `eisoptx/modeling/optics.py` | 776 | `Lens`, sequence parsing, paraxial properties, the trace loop |
| `eisoptx/optimization/parameterization.py` | 754 | variables ↔ lens; solves; the glass model |
| `eisoptx/utils/callbacks.py` | 727 | the training schedule (toggles, glass annealing) |
| `eisoptx/modeling/ray_initialization.py` | 663 | where rays start; pupil sampling; ray aiming |
| `eisoptx/modeling/ray_tracing.py` | 498 | surface intersection, Snell, failure handling |
| `eisoptx/modeling/misc_surfaces.py` | 442 | metasurfaces and diffractive elements |
| `eisoptx/data/datasets.py` | 240 | image loading, field-limit collation |
| `eisoptx/image_restoration/nafnet.py` | 236 | the restoration network |
| `eisoptx/modeling/ray_analysis.py` | 177 | centroids, spot size, PSF kernel density estimation |
| `eisoptx/image_restoration/wiener.py` | 169 | differentiable Wiener deconvolution |
| `eisoptx/optimization/optimizers.py` | 158 | **the LM optimizer** |
| `eisoptx/modeling/paraxial_ray_tracing.py` | 97 | ABCD matrices |
| `eisoptx/main.py` | 76 | the CLI |

Note the shape of that table. The two files that carry the paper's actual contribution
— `imaging_system.py` and `optimizers.py` — are 1,317 lines together. Everything else
is the optical engine they need in order to work.

## 0.4 The data flow, once, in full

Read this twice. Every later chapter is a zoom into one arrow.

```
  configs/*.yml
       │  jsonargparse instantiates classes named in the YAML
       ▼
  ImagingSystemModule ──────────────────────────────────────┐
       │                                                     │
       │ .parameterization                                   │
       ▼                                                     │
  LensParameterization                                       │
       │  flat variable vector [31], 20 trainable            │
       │  scale → unpack → apply solves                      │
       ▼                                                     │
  Lens object   (c, s, nd, vd, a, ...)                       │
       │                                                     │
       │ RayInitialization: sample the entrance pupil        │
       ▼                                                     │
  r, d  [3, 11 fields, 256 rays, 1 wavelength, 1 lens]       │
       │                                                     │
       │ Lens.trace_rays — generator, 11 events              │
       ▼                                                     │
  xy at sensor  +  ray_status  +  per-event intermediates    │
       │                                                     │
       ├──────────────► residual terms (spot, path, normals) │
       │                                                     │
       └─► PSFSampler ─► psf grid ─► SVOLA blur ─► restore   │
                                          │                  │
                                          ▼                  │
                                    scalar loss L₀           │
                                          │                  │
                          generalized aberration lift        │
                                          ▼                  │
                              extra residual entries ────────┤
                                                             │
                          all residuals concatenated  ◄──────┘
                                     │  [5704]
                                     ▼
                       CustomClosure.get_least_squares_quantities
                                     │  jacfwd → J [5704, 20]
                                     ▼
                              LMOptimizer.step
                                     │  solve [J; √λD] δ = −[l; 0]
                                     ▼
                        accept/reject, adapt λ, write back
```

## 0.5 The measured toy system

Everything above is concrete. On `configs/toy/defaults.yml` +
`configs/toy/designs/optimized_spot.yml`:

| quantity | value | where measured |
|---|---|---|
| lens sequence | `s-aRa-aRa-` | `1_config_and_lens.sequence_string` |
| events in the trace | 11 | `3_ray_trace.per_event_steps.n_events` |
| refractive interfaces | 4 | `1_config_and_lens.n_interfaces` |
| aspheric surfaces | 4 | `1_config_and_lens.n_aspherical` |
| total variables | 31 | `1_config_and_lens.variable_layout` |
| **trainable** variables | **20** | same |
| fields × rays | 11 × 256 = 2816 | `3_ray_trace` |
| rays surviving to sensor | 2816 (100%) | `3_ray_trace.ray_valid_final` |
| effective focal length | 10.0 exactly | `1_config_and_lens.lens.efl` |
| **residual vector length** | **5704** | `4_residuals.residual_vector_table` |
| **Jacobian** | **[5704, 20]** | `5_lm_step.jacobian` |
| constraints | 0 | `5_lm_step.n_constraints` |

The EFL is exactly 10.0 because a **curvature solve** forces it there on every
construction — the last curvature is not a free variable but is computed from the
others (Chapter 6). This is why 20 variables are trainable out of 31.

## 0.6 Three things that will confuse you, stated up front

**1. The Lagrange/KKT branch of the optimizer is dead code.**
`optimizers.py:93-113` implements constrained optimization with Lagrange multipliers.
It never runs. Every residual class in the shipped library declares
`constraint = False` (verified by grep across the whole package), so
`constraints.numel()` is always 0 and control always takes the unconstrained branch at
`optimizers.py:114`. Measured: `n_constraints = 0`. Do not let a supervisor's question
about "how are constraints handled?" catch you — the answer is *as weighted soft
penalties, not as hard constraints*; the hard-constraint machinery exists but is
unused.

**2. `configs/toy/designs/starting_point.yml` is 0 bytes.** The README's suggested
starting design is an empty file. Real starting values live inline in
`configs/toy/defaults.yml`.

**3. `configs/toy/defaults_e2e.yml:14-15` points at a directory that does not exist**
(`configs/ablation/sample_image`). `test` silently passes anyway, because
`test_dataloader` returns a dummy batch (`datasets.py:142`) and never opens the
folder. `fit` dies with a zero-length dataset. The working path is
`configs/toy/sample_image`. All e2e numbers in this walkthrough use that override.

## 0.7 How to read this walkthrough

**If you have one hour before a meeting:** Chapter 0 (this one), Chapter 8
(generalized aberrations), Chapter 9 (LM), Chapter 17 (defense bank).

**If you want to understand the physics:** 2 → 3 → 4 → 5 → 6.

**If you want to understand the machine learning:** 8 → 9 → 12.

**If you are defending your own implementation:** 16, then whichever chapter it
points you to.

Chapters are numbered by dependency, not by importance. Long chapters are split into
`_partN.md` files; the index (`README.md`) lists every part in order.

## 0.8 Conventions in these chapters

- `file.py:123` means exactly that line in the current checkout.
- Numbers in **bold** were measured, not inferred. Their source key in
  `trace_dump.json` is given.
- Tensor shapes are written as the code writes them:
  `[3, n_fields, n_rays, n_wavelengths, n_lens]`.
- Where the authors' code and your `e2e-gtra-optics` differ, you will see a
  **↔ your repo** note pointing to Chapter 16.
- Where I think something is a defect rather than a design choice, it is marked
  **⚠ defect** and backed by an experiment you can re-run.

---

*Next: [Chapter 1 — The configuration is the program](01_configuration.md)*
