# Chapter 13 — The periphery: everything the core depends on but isn't

> **What this chapter answers.** What is in the other 4,300 lines? What is
> `misc_surfaces.py` and why does the paper need metasurfaces? How does the data loader
> handle field-of-view tiling, and why does it exist at all? What do the callbacks
> actually schedule? Which of these do you need to read, and which can you skip?

---

## 13.1 Scope, honestly stated

Chapters 2–12 covered the core in line-level detail. This chapter is a **survey**: enough
to know what each remaining module does, why it exists, and where to look if you ever need
it. Depth is spent in proportion to how likely you are to be asked about it.

```
                                            lines   treatment
eisoptx/utils/visualization.py               1758   survey — largest file, no physics
eisoptx/utils/callbacks.py                    727   moderate — schedules real algorithms
eisoptx/modeling/ray_initialization.py        663   partly covered (Ch. 4); remainder here
eisoptx/modeling/misc_surfaces.py             442   moderate — the paper's DOE/metasurface work
eisoptx/data/datasets.py                      240   moderate — field tiling matters
eisoptx/modeling/paraxial_ray_tracing.py       97   short and complete
eisoptx/utils/utils.py                         60   trivial
```

**Roughly 44% of the codebase is visualization and callbacks** — infrastructure, not
method. That ratio is itself worth knowing: the scientific content of this repository is
concentrated in about 4,000 lines.

## 13.2 `paraxial_ray_tracing.py` — 97 lines of first-order optics

The smallest module and the one you should actually read end to end. It is pure ABCD matrix
optics, the foundation Chapter 6's solves rest on.

Every element is a 2×2 matrix acting on `[height, angle]`:

```python
def interface_abcd(mu, c):        # refraction at a surface
def propagation_abcd(d):          # travel through a space
def interface_propagation_abcd(c, t, mu):
def diffraction_abcd(p, wavelength_ratio):
def reduce_abcd(abcd):            # multiply the chain
def reduce_abcd_cumulative(abcd)  # partial products
```

`reduce_abcd_cumulative` is the one that matters for Chapter 6: it returns every *partial*
product, so the solve can split the system at an arbitrary surface into "everything before"
and "everything after" without recomputing. That is what makes `curvature_solve` a closed
form rather than an iteration.

`diffraction_abcd` shows the module is not limited to refractive systems — a diffractive
surface has a paraxial power that scales with wavelength, which is exactly the strong
negative dispersion §13.5 depends on.

**This module is fully differentiable and exact.** There is no approximation in first-order
optics; it is a different, linearized model of the same physics. If a supervisor asks how
you know the EFL is right, the answer is that it comes from the ABCD product, not from a
fit to traced rays.

## 13.3 `misc_surfaces.py` — the generalized-refraction surfaces

442 lines implementing surfaces that are **not** refractive interfaces. This is where the
repository's title — *generalized aberrations* — earns its "generalized."

```
MiscSurfaceModel                      (abstract base)
├── DispersionEngineeredMetasurface   (265 lines — an idealized exploratory model)
└── GeneralizedSnellModel             (base for phase-gradient surfaces)
    └── DiffractiveOpticalElement     (ideal DOE)
```

The physics is the **generalized law of refraction**, stated in the class docstring:

```
n' sin(i') − n sin(i) = (λ/2π) · ∂φ/∂ρ
```

An ordinary surface has `n' sin i' = n sin i` — Snell's law. Add a surface with a spatially
varying phase `φ(ρ)` and the phase gradient contributes an extra term. That covers
diffractive optical elements, metasurfaces, and holographic elements in one formalism.

For a flat interface in air it reduces to a direction-cosine update
(`misc_surfaces.py:295-297`):

```
l' = l + (λ/2π)(∂φ/∂ρ) · x/ρ
m' = m + (λ/2π)(∂φ/∂ρ) · y/ρ
```

Two implementation points worth carrying.

**The phase gradient is autograd, not algebra** (`misc_surfaces.py:271-276`):

```python
self.compute_scaled_phase_gradient = torch.func.grad(
    lambda *inputs: self.compute_phase(*inputs).sum(), 0)
```

A subclass writes down `compute_phase` and gets `∂φ/∂ρ` for free. Add a new surface type by
specifying only its phase function — no hand-differentiation, no chance of an inconsistent
derivative. This is a genuinely nice piece of design and a good example of what
differentiable programming buys beyond optimization.

**The DOE phase is an even polynomial in ρ** (`misc_surfaces.py:429-441`):

```
φ(ρ) = (2π/λ₀) · (p₀ρ² + p₁ρ⁴ + p₂ρ⁶ + …)
```

Structurally identical to the aspheric sag polynomial of Chapter 3 — even powers only, for
rotational symmetry and smoothness at the axis. The `p` coefficients are trainable exactly
like aspheric coefficients, so a DOE slots into the parameterization with no special
handling. The `1/λ₀` scaling is what gives diffractive surfaces their characteristic strong
negative dispersion, which is why they are attractive for chromatic correction in compact
systems.

**Relevance to you:** none of this is exercised by the toy config, but it is likely
directly relevant to your own thesis topic. If you are asked "could this method handle a
metasurface?", the answer is that it already does, through `GeneralizedSnellModel`, and
the machinery is generic in the phase function.

## 13.4 `datasets.py` — and the tiling trick

Three of the four classes are unremarkable: `SingleFolderDataset` walks a directory,
`ImageFolderDataModule` wraps it in a Lightning data module, `NoneDataModule` is a
placeholder that yields `(None,)` so the lens can be optimized with no images at all
(the toy config's choice — *"For siloed lens optimization"*).

`FieldLimitCollateFn` (`datasets.py:179-240`) is the one with real content, and it solves a
problem specific to this kind of work.

**The problem.** The paper's telephoto sensor is 6144 × 8192. You cannot fit that through
the pipeline of Chapter 10 at training time. So you train on 1024² crops. But a crop is
*from somewhere* — a corner crop should be blurred with corner PSFs, a centre crop with
centre PSFs. A naive crop loses that information and you would train the network on the
wrong blur.

**The solution.** The collate function assigns each crop a **field-of-view tile**, and
passes `field_lims` alongside the image. That is the `field_lims` argument threaded through
`compute_psf_grid` in Chapter 10 — it tells the simulator which part of the field this crop
came from, so the correct sub-grid of PSFs is applied.

Three sampling modes:

| mode | behaviour |
|---|---|
| `random` | uniform random tile per sample |
| `pseudorandom` | stratified — `(arange(n) + rand(n))/n`, so one batch spans the whole field |
| `deterministic` | tile determined by dataset index; reproducible |

`pseudorandom` is the interesting one. Pure random sampling can give a batch that is
entirely centre tiles, and the gradient for that step then contains no information about
corner performance. Stratifying guarantees every batch covers the field. This is variance
reduction, and it matters more here than in ordinary image restoration because the *thing
being optimized* varies across the field.

One more detail (`datasets.py:203-206`):

```python
tile_radial_distance = np.linalg.norm(tile_idx + 0.5 - tile_layout / 2, axis=-1)
tile_order = np.argsort(-tile_radial_distance.flatten())
```

Tiles are ordered **outermost first**. Combined with the stratified sampler, early indices
map to corner tiles — where aberrations are worst. The hardest fields are seen first and
most systematically.

`compute_field_lims` (`utils.py:39-60`) converts a tile index into normalized field limits
`(x0, x1, y0, y1)`, dividing by the diagonal so the coordinates match the relative field
convention of Chapter 4. Note the sign flip on `y` — image coordinates run downward,
field coordinates upward.

## 13.5 `callbacks.py` — where the training *schedule* lives

727 lines, and unlike visualization this is not decoration: several of these callbacks
implement scheduling that changes what problem is being solved.

**`ToggleIRMOptimizerCallback`** (`callbacks.py:14-60`). Disables or re-enables the
restoration optimizer at a given step. Accepts either an integer step or a **float fraction
of `max_steps`** — so a schedule written as `0.3` transfers between runs of different
length. Chapter 12 showed `get_optimizers` reading the `irm_optimization_disabled` flag;
this is what sets it.

Why you would want it: joint optimization from step zero means the lens is being optimized
against a *random* network. Freezing the lens while the network warms up — or the reverse —
is a curriculum, and the paper explores such schedules.

**`ToggleGlassOptimizationCallback`** and **`BindMaterialsCallback`** (`callbacks.py:62-247`)
together solve the discrete-glass problem raised in Chapter 3. Recall the trick: glass is
optimized as *continuous* `(n, V)` coordinates, because real catalog glasses form a
discrete set that gradient descent cannot search.

But you must eventually build the lens from real glass. `BindMaterialsCallback` snaps the
continuous coordinates to the nearest catalog entries and freezes them — and its docstring
notes *"The lens optimizer state is reset to the default values,"* because the LM damping
state is meaningless after a discontinuous jump in the parameters.

The `end_step` argument is the sophisticated part: materials can be bound **one at a time**
over a range of steps, so the remaining free glasses can re-optimize to compensate for each
snap. Binding all glasses simultaneously would take a large uncompensated hit; sequential
binding lets the design recover between steps. This is a real algorithmic technique, and
it is a plausible supervisor question — *"how do you get from continuous glass back to a
manufacturable design?"*

**`IncreaseGlassVariableResidualsWeightCallback`** (`callbacks.py:248-313`) ramps the weight
on `GlassVariableResiduals` (Chapter 7) during training — a continuation method. Early on,
the glass coordinates roam freely; later, they are pushed increasingly hard toward the
feasible catalog region, so that when binding happens the jump is small. Pair it with
`BindMaterialsCallback` and the two form a complete strategy.

**`CodeVSeqFileCallback`** (`callbacks.py:545-577`) exports the design as a CODE-V `.seq`
file. Chapter 12 flagged this as the bridge to commercial validation. `make_codev_file`
has a `use_private_catalog` flag, indicating the authors work with proprietary glass
catalogs.

**`ExtendedLoggingCallback`** and **`ConfigFileCallback`** are logging; the latter dumps the
fully-resolved config periodically so every run is reproducible from its own log directory.

## 13.6 `visualization.py` — 1758 lines you can mostly skip

The largest file in the repository and the one with the least method in it. Fourteen
`Visualization` subclasses, dispatched by `VisualizationCallback` every N steps, each
rendering a matplotlib figure into TensorBoard via `plot_to_image`.

What you get: `LensLayout` (the cross-section with rays), `SpotDiagrams`, `RayFanPlot`,
`GlassPlot` (the n–V diagram, showing where the continuous glass variables sit relative to
the catalog), `PupilSampling`, `PSFs` / `RGBPSFs` / `CombinedRGBPSFs` / `PSFGrid`,
`OpticsSimulation` (before/after images), `MetricPlot`, and
`DispersionEngineeredPhase`.

Two reasons to open this file at all:

1. **`GlassPlot`** is how you *see* the glass optimization of §13.5 — continuous variables
   drifting across the n–V plane toward catalog points. If you present the glass method,
   present this plot.
2. **`compute_rms_size`** (`visualization.py:1359`) and `plot_layout` contain geometry you
   might want to reuse. Note that a diagnostic RMS computation living in the visualization
   module is a small architectural smell — if you cite an RMS number, cite the one from
   `ray_analysis.py` (Chapter 5), not this one, and check they agree before you rely on
   either.

Otherwise: this is a plotting library. Read it when a figure looks wrong.

## 13.7 `ray_initialization.py` — the 663-line remainder

Chapter 4 covered the parts the toy config exercises: field generation, the pupil sampler,
the shell structure, ray aiming. The rest of the module handles cases the toy config does
not reach — alternative pupil sampling patterns, telecentric and finite-conjugate
configurations, and the aperture-stop handling that Chapter 4 noted is trivially satisfied
when the stop is the first surface.

If you change `pupil_position_z` or move the stop to an interior surface, the ray-aiming
iteration Chapter 4 described as unnecessary becomes load-bearing, and that is when to
read this module properly.

## 13.8 Reading priority

If you have limited time, in order:

1. **`paraxial_ray_tracing.py`** — 97 lines, underpins Chapter 6, read it whole.
2. **`callbacks.py:62-313`** — the glass binding and weight-ramping strategy; a likely
   question with a non-obvious answer.
3. **`datasets.py:179-240`** — field tiling; explains a parameter (`field_lims`) that
   appears throughout Chapter 10.
4. **`misc_surfaces.py`** — only if DOEs or metasurfaces are relevant to your own work
   (they likely are).
5. **`visualization.py`** — only when a figure misbehaves.

## 13.9 What to take away

1. About 44% of the repository is visualization and callbacks; the scientific core is
   ~4,000 lines.
2. `paraxial_ray_tracing.py` is exact first-order optics via ABCD matrices;
   `reduce_abcd_cumulative` is what makes Chapter 6's solves closed-form.
3. `misc_surfaces.py` implements the **generalized law of refraction**, covering DOEs and
   metasurfaces; phase gradients come from `torch.func.grad` on a user-supplied phase
   function, and DOE phase is an even polynomial in ρ, exactly like aspheric sag.
4. `FieldLimitCollateFn` tags each training crop with **which part of the field of view it
   came from**, so a crop is blurred with the right PSFs; `pseudorandom` stratifies tiles
   across a batch, and tiles are ordered outermost-first.
5. The callbacks implement the **discrete-glass strategy**: optimize continuously, ramp a
   penalty toward the catalog, then bind to real glasses one at a time with an optimizer
   state reset.
6. `ToggleIRMOptimizerCallback` accepts fractional steps, making curricula portable across
   run lengths.
7. A CODE-V sequence file is exported every run — the path to commercial validation.
8. `visualization.py` is a plotting library with one diagnostic (`compute_rms_size`) that
   arguably belongs elsewhere; prefer `ray_analysis.py` for numbers you cite.

---

*Previous: [Chapter 12 — The training loop](12_training_loop.md)*
*Next: [Chapter 14 — Execution trace: one forward pass](14_trace_forward.md)*
