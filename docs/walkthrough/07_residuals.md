# Chapter 7 — The residual library: what "good lens" means numerically

> **What this chapter answers.** What is a residual, and why a *vector* rather than a
> scalar? How do 16 residual classes become one 5704-element vector? Why is
> `sqrt(weight)` applied and not `weight`? Why do some residuals return an **empty**
> tensor, and why is that the *correct* behaviour rather than a bug?

---

## 7.1 Residual vs. loss — the distinction the whole paper rests on

A conventional deep-learning loss is a **scalar**. `residuals.py` produces **vectors**.

The optimizer minimizes

```
L(x) = ½ ‖l(x)‖²        where l(x) ∈ ℝᵐ  is the residual vector
```

If you only ever computed `L`, you would need gradient descent. Because you keep the
**vector** `l`, you can compute the Jacobian `J = ∂l/∂x` and use Gauss–Newton:

```
L(x + Δ) ≈ ½‖l + JΔ‖²   →   (JᵀJ) Δ = −Jᵀl
```

`JᵀJ` is a **curvature estimate obtained from first derivatives only** — the
Gauss–Newton approximation to the Hessian. That is the entire reason least-squares
optimizers converge in tens of iterations where Adam needs thousands. It is available
*only* if you keep the vector.

Everything in this chapter is in service of that: each residual class must return a
vector whose **sum of squares** is a meaningful cost.

## 7.2 The base class

`Residuals` (`residuals.py:9-37`) is 29 lines:

```python
class Residuals(torch.nn.Module):
    def __init__(self, weight: float | None, reduce_to_scalar: bool = False, **kwargs):
        self.weight = weight
        self.reduce_to_scalar = reduce_to_scalar
        super().__init__()

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if self.reduce_to_scalar:
            return (loss**2).sum().sqrt()
        else:
            return loss
```

Three class attributes are the plug-in contract, declared by each subclass:

| attribute | purpose |
|---|---|
| `name` | dictionary key; also the log label |
| `constraint` | `True` → hard constraint (KKT branch of LM) |
| `data_keys` | which intermediate ray-trace quantities this residual needs |

`reduce_to_scalar` deserves attention: it collapses the vector to `‖l‖₂` — a **single**
element whose square is the same sum of squares. The cost is unchanged, but the
Jacobian loses all but one row for this term. Setting it `True` deliberately throws away
Gauss–Newton structure for that residual. The toy config sets it `False`
(`configs/toy/defaults.yml`, `TransverseRayAberrationResiduals.reduce_to_scalar: False`)
— and that is the right choice.

## 7.3 `data_keys` — a pull-based collection protocol

Residuals need intermediate quantities from *inside* the trace: incidence cosines,
propagation distances, surface normals. Collecting all of them always would be wasteful.
Instead, each residual **declares** what it needs, and the imaging system takes the
union (`imaging_system.py:93-96`):

```python
self.data_collection_keys = set(
    k for residual in self.residuals for k in residual.data_keys
)
```

Then during the trace (`imaging_system.py:180-195`):

```python
rt_info = {k: [] for k in keys}
for r, d, ray_status, event_info in lens.trace_rays(r0, d0, wavelengths, yield_on="all"):
    for k, v in event_info.items():
        if k in self.data_collection_keys:
            rt_info[k].append(v)
rt_info = {k: torch.stack(v, dim=-1) if len(v) > 0 else torch.tensor([]) ...}
```

This is where the **generator form of `trace_rays`** from Chapter 2 pays off. The trace
yields at every event; the imaging system filters by the declared key set and stacks
along a new trailing axis. Add a new residual class with a new `data_key`, and the
collection machinery picks it up with no changes anywhere else.

Measured for the toy config (`4_residuals`):

```
registered_residual_names : ['transverse_ray_aberration', 'ray_path',
                             'ray_angle', 'surface_normal']
data_collection_keys      : ['cos2_prime', 'cos2_theta', 'cos_n', 'delta_z']
```

Four residuals requesting four intermediate quantities. `transverse_ray_aberration`
declares `data_keys = ()` — it only needs the final ray positions, which are returned
by the trace anyway.

One detail at `imaging_system.py:83-86`:

```python
# Residuals (duplicated entries are overwritten by the last occurrence)
self.residuals = list({residual.name: residual for residual in residuals}.values())
```

Deduplication **by name**. Combined with the `residuals+` append operator from
Chapter 1, this is how an experiment config overrides one residual from the base config:
append a new instance with the same `name` and the later one wins.

## 7.4 The image-quality term

`TransverseRayAberrationResiduals` (`residuals.py:61-98`) is the one that actually
makes the lens sharp. Its core is one line:

```python
delta_xy = xy - xy_centroid
```

**Distance from each ray to its field's centroid.** Not to the paraxial ideal image
point — to the centroid of where the rays actually landed. Minimizing the sum of
squares of these minimizes the RMS spot size.

This choice has a large consequence, which Chapter 5 measured: the residual is blind to
**where the centroid is**. A design can be perfectly sharp and badly distorted, and this
term cannot tell. The toy lens has **−14.22% distortion at the corner field** and its
loss is completely indifferent to that.

The normalization (`residuals.py:95-97`) is worth reading closely:

```python
weighted_valid_rays = (ray_valid_weighted**2).sum() / 2
residuals = delta_xy_weighted[ray_valid] / weighted_valid_rays.sqrt()
```

Dividing by `sqrt(N)` means the **sum of squares** is `‖Δxy‖²/N` — a mean-square, not a
sum. Without it, the image-quality term's magnitude would scale with the ray count and
every weight in the config would need retuning when you change `n_r`. The `/2` accounts
for x and y both being counted in `ray_valid`.

Two more details:
- `weights = weights / weights.mean()` then `weights.sqrt()` (`residuals.py:86-89`) —
  weights are applied in the **amplitude** domain because they get squared in the sum of
  squares. The comment says exactly this. Same reasoning appears at
  `imaging_system.py:1013-1015`.
- `delta_xy_weighted[ray_valid]` — boolean indexing, so **failed rays are dropped
  entirely** rather than contributing zeros. This is why the residual vector length can
  change between iterations if rays start failing.

## 7.5 The hinge pattern — manufacturability constraints

Most of the remaining classes share one shape. `RayAngleResiduals`
(`residuals.py:296-305`):

```python
residuals = (self.threshold - cos2).clip(min=0)
residuals = residuals[residuals > 0] / np.sqrt(n_rays)
```

That is a **one-sided hinge**: zero when satisfied, growing when violated. It expresses
an inequality (`angle ≤ 60°`) in a least-squares framework, which natively handles only
equalities.

The threshold conversion (`residuals.py:294`) avoids trigonometry in the hot loop:

```python
self.threshold = np.cos(np.deg2rad(max_angle)) ** 2
```

`max_angle: 60` becomes `cos²(60°) = 0.25`, compared directly against the `cos2_theta`
the tracer already computed. No `arccos` anywhere.

**Now the important line** (`residuals.py:304`):

```python
residuals = residuals[residuals > 0] / np.sqrt(n_rays)
```

Only strictly positive entries are kept. `RayPathResiduals` says why explicitly
(`residuals.py:221`):

> *"Return only positive values; zero values do not have a gradient due to the ramp
> function"*

A satisfied hinge contributes exactly zero to the cost **and** exactly zero to the
Jacobian — it is a flat region of a ramp. Keeping those entries would add thousands of
identically-zero rows to `J`, inflating the QR/SVD cost for no information. Dropping
them is a pure efficiency win with no change to the mathematics.

**The consequence is that residual vectors have data-dependent length.** Measured:

| residual | length | why |
|---|---|---|
| `transverse_ray_aberration` | **5632** | 2 × 11 fields × 256 rays, all valid |
| `ray_path` | **14** | only violated path bounds |
| `ray_angle` | **0** | *no ray exceeds 60°* |
| `surface_normal` | **58** | only violated normal bounds |

**`ray_angle` returns an empty tensor, and that is the design working correctly.** The
shipped lens satisfies the 60° incidence bound everywhere, so there is nothing to
penalize. An empty tensor concatenates harmlessly and contributes zero. If you see an
empty residual, the constraint is inactive — not broken.

`SurfaceNormalResiduals` adds an explicit guard for the degenerate case
(`residuals.py:332-333`):

```python
if cos_n.numel() == 0:
    return torch.tensor([], device=cos_n.device)
```

— needed because a lens with no aspheres never produces a `cos_n`, and `np.prod(())` of
an empty shape would misbehave.

## 7.6 `RayPathResiduals` — the most configurable class

`residuals.py:159-256`. It bounds the **horizontal path length** `delta_z` of each ray
through each gap, with six different cutoffs (`residuals.py:166-190`):

| cutoff | meaning |
|---|---|
| `min_cutoff` / `max_cutoff` | general, for airspaces |
| `min_cutoff_refractive` / `max_cutoff_refractive` | absolute, inside glass |
| `min_cutoff_refractive_relative` / `max_cutoff_refractive_relative` | as a *fraction of central thickness* |
| `other_min_cutoffs` / `other_max_cutoffs` | per-surface overrides |

Toy config: `min_cutoff: .5`, `max_cutoff: .inf`, `min_cutoff_refractive: .5`,
`max_cutoff_refractive: 6.`, `min_cutoff_refractive_relative: 0.3333`,
`max_cutoff_refractive_relative: 3.`.

Read that as real manufacturing rules:
- **≥ 0.5 mm everywhere** — nothing thinner than half a millimetre can be made or
  mounted.
- **Edge thickness between ⅓× and 3× the centre thickness** — this is the classic
  constraint preventing knife-edge lenses (too thin at the rim to grind) and extreme
  meniscus shapes (too thick at the rim to mould).

The relative cutoffs are computed against the actual spacings
(`residuals.py:240-250`), so the bound *moves as the design changes* — it is a genuine
ratio constraint, not a fixed number.

The two-sided hinge is `residuals.py:218-220`:

```python
residuals = torch.maximum((min_cutoff - delta_z).clip(min=0),
                          (delta_z - max_cutoff).clip(min=0))
```

Only one branch can be active at a time, so `maximum` picks whichever bound is violated.

Note `get_cutoffs` (`residuals.py:232-236`) filters `event["s"] is not None` — it skips
the placeholder propagation event from Chapter 2. A concrete example of why that
"empty" event needed to exist as a distinct case.

## 7.7 The rest of the library

The remaining classes are not enabled in the toy config but are part of what you should
be able to speak to:

| class | line | enforces |
|---|---|---|
| `SpotSizeResiduals` | 40 | per-field RMS spot size (a *reduced* alternative to TRA) |
| `MarginalRayPathResiduals` | 259 | path bounds for marginal rays only |
| `RayBoundaryResiduals` | 101 | rays must stay inside a box at the image plane |
| `GlassVariableResiduals` | 340 | keep `(nd, vd)` in the physically realizable region |
| `GlassMeshDistanceResiduals` | 373 | distance to the convex hull of a real glass catalog |
| `FocalLengthResiduals` | 463 | EFL target *as a penalty* (the alternative to Chapter 6's solve) |
| `TotalTrackLengthResiduals` | 481 | TTL as a penalty |
| `ImageHeightResiduals` | 514 | image height target |
| `DistortionResiduals` | 541 | **the term that would fix the toy lens's −14% distortion** |
| `RimmerRelativeIlluminationResiduals` | 579 | vignetting/illumination falloff (analytic) |
| `RelativeIlluminationResiduals` | 618 | illumination falloff (ray-counted) |
| `GroupDelayResiduals` | 661 | for diffractive/metasurface designs |

Note the pattern: `FocalLengthResiduals` and `TotalTrackLengthResiduals` **duplicate**
constraints available as solves. That is deliberate. Use the solve when you want the
constraint exact and can spare the variable; use the residual when the constraint is a
soft target, when several must trade off, or when a solve would be singular.

`GlassMeshDistanceResiduals` is the most involved (`residuals.py:373-460`, plus the
simplex-distance helpers at `residuals.py:699-870`). It builds a Delaunay
triangulation of the glass catalog in `(nd, vd, dpgf)` space and penalizes distance to
that mesh — the soft counterpart to Chapter 3's hard straight-through quantization.

## 7.8 Assembly into one vector

`ImagingSystemModule.compute_residual_vector` (`imaging_system.py:1005-1029`):

```python
residuals_dict, weight_dict, constraint_dict, logs = ...
weighted_residuals_dict = {k: np.sqrt(weight_dict[k]) * residuals_dict[k] for k in keys}
self.residual_vector = torch.cat([weighted_residuals_dict[k].view(-1) for k in keys])
self.constraint_mask = torch.tensor(
    [constraint_dict[k] for k in keys for _ in range(residuals_dict[k].numel())])
```

**`sqrt(weight)`, not `weight`.** Because the cost is `½‖l‖²`, multiplying a residual
block by `√w` multiplies its contribution to the cost by exactly `w`. Applying `w`
directly would give `w²`. The comment on `imaging_system.py:1013` says so. This is the
single most common way to get weighting wrong in a least-squares codebase, and it is a
good detail to have ready.

The `constraint_mask` is built by **repeating each residual's flag once per element**,
so it aligns element-for-element with the concatenated vector. That is what lets the
LM optimizer in Chapter 9 slice the vector into objective and constraint parts.

Measured assembly for the toy config:

| block | numel | weight | √weight | ‖l‖ (weighted) | sum of squares |
|---|---|---|---|---|---|
| `transverse_ray_aberration` | 5632 | 1.0 | 1.000 | 0.0166029 | 2.757e−4 |
| `ray_path` | 14 | 20.0 | 4.472 | 0.0004213 | 1.775e−7 |
| `ray_angle` | 0 | 10.0 | 3.162 | 0.0 | 0.0 |
| `surface_normal` | 58 | 10.0 | 3.162 | 0.0006776 | 4.591e−7 |
| **total** | **5704** | | | | **2.764e−4** |

Two readings.

**Image quality dominates by three orders of magnitude** (2.757e−4 vs 6.4e−7 combined
for the constraints). At the shipped optimum the manufacturability terms are *barely
active* — they are guardrails, not objectives. They matter during optimization, where a
step that would violate them gets pushed back; at convergence they should be near zero,
and they are.

**5632 + 14 + 0 + 58 = 5704** — that is the measured `residual_vector` length, and
therefore the number of rows in the Jacobian `[5704, 20]` of Chapter 9.

## 7.9 The constraint flag is dead code in every shipped config

`constraint = False` on **all sixteen** classes. I grepped the whole package: nothing
sets it `True`.

The consequence is significant for reading Chapter 9. `LMOptimizer` has a full
Karush–Kuhn–Tucker branch for hard constraints — Lagrange multipliers, an augmented
system, the works. **It never executes in any bundled configuration.** Measured:
`constraint_mask` is all-`False`, and the KKT path is not entered.

Treat it as designed-for-but-unused infrastructure. If your supervisor asks "what does
the constraint machinery do", the honest answer is: it implements hard constraints via
KKT, no shipped residual requests it, and the mechanism is untested by any config in
the repository.

## 7.10 What to take away

1. Residuals are **vectors** because `JᵀJ` — the Gauss–Newton Hessian — is only
   available if you keep them. `reduce_to_scalar: True` throws that away.
2. `data_keys` is a pull-based protocol: each residual declares what it needs, the
   trace yields everything, the imaging system collects the union.
3. The image-quality term measures spread **about the centroid** — sharp but distorted
   designs are invisible to it, which is exactly what the toy lens is (−14.2%).
4. The hinge pattern turns inequalities into least-squares terms; satisfied entries are
   **dropped** because they have zero value and zero gradient.
5. An **empty** residual vector means the constraint is inactive. `ray_angle` measured
   0 elements — the lens respects the 60° bound everywhere.
6. Weights are applied as **`sqrt(weight)`** because the cost squares them.
7. Measured composition: 5632 + 14 + 0 + 58 = **5704**, with image quality dominating
   the constraints by ~10³.
8. No shipped residual sets `constraint = True`; the LM optimizer's KKT branch is
   unreachable in every bundled config.

---

*Previous: [Chapter 6 — Paraxial solves](06_solves.md)*
*Next: [Chapter 8 — Generalized transverse ray aberrations](08_gtra.md)*
