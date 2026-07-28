# Chapter 5 — Ray tracing: the differentiable core

> **What this chapter answers.** How does a ray find an aspheric surface if there is no
> closed-form solution? How can a 16-iteration Newton loop be differentiable without
> exploding the graph? What happens when a ray fails — and why does the code never
> throw? What is `ray_status`?

---

## 5.1 The interpreter loop

`Lens.trace_rays` (`optics.py:379-497`) walks the event list compiled in Chapter 2 and
dispatches on `event["type"]`:

| event | what it updates | line |
|---|---|---|
| `r` (refraction) | **direction** `d` — Snell's law | `optics.py:407` |
| `p` (propagation) | **position** `r` — march to next surface | `optics.py:454` |
| `d` (diffraction) | direction — phase gradient | `optics.py:446` |
| `m` (metasurface) | direction — learned model | `optics.py:436` |
| `s` (stop) | nothing | — |

The rhythm alternates: propagation moves the ray to a surface, refraction bends it. The
stop event does nothing at trace time — it mattered at *initialization* (Chapter 4),
where it defined the pupil.

First, a reshape (`optics.py:393-394`):

```python
r = r.reshape(3, -1, len(wavelengths), len(self))
d = d.reshape(3, -1, len(wavelengths), len(self))
```

The field and pupil axes are **flattened together**. Inside the trace, a ray is a ray;
which field it belongs to is irrelevant to the physics. The original shape is restored
at yield time (`optics.py:493`). Measured: input `[3, 11, 256, 1, 1]` → internally
`[3, 2816, 1, 1]` → yielded back as `[3, 11, 256, 1, 1]`.

## 5.2 `cum_z_distance` — the coordinate trick

Watch this variable (`optics.py:396, 456, 420, 457`):

```python
cum_z_distance = 0
...
elif event["type"] == "p":
    s = r.new_zeros(1) if event["s"] is None else self.s[event["s"]]
    cum_z_distance = cum_z_distance + s
    r0 = rt.shift_rays(r, cum_z_distance)
```

Rays are stored in a **global** coordinate system, but every surface computation wants
coordinates **local** to that surface's vertex. Rather than transforming the ray array
at each step, the code accumulates the running z-offset and applies it on the fly with
`shift_rays` (`ray_tracing.py:445`), which just subtracts from the z component.

Why this matters for gradients: `cum_z_distance` is a **sum of spacing tensors**, and
it is differentiable. Every surface's position depends on every preceding spacing.
Change `s[0]` and all four surfaces move. That dependency is captured automatically by
this accumulation — no manual chain rule anywhere in the file.

## 5.3 Spherical intersection is closed-form; aspheric is not

For a sphere, "where does this ray hit?" is a quadratic — solved directly in
`find_marching_distance_spherical` (`ray_tracing.py:42-70`), which conveniently also
returns `cos_theta` (needed by Snell) as a by-product.

An asphere has profile

```
z(ρ) = cρ²/(1 + sqrt(1 − (1+k)c²ρ²)) + a₁ρ⁴ + a₂ρ⁶ + a₃ρ⁸
```

with `ρ² = x² + y²`. Substituting a parametric ray gives a **transcendental** equation.
There is no closed form; you must iterate. This is the single most computationally
significant fact about the ray tracer, and it is where all the interesting engineering
in `ray_tracing.py` lives.

## 5.4 The iterate-then-differentiate pattern

`find_marching_distance_aspherical` (`ray_tracing.py:105-156`) is four stages. Read the
`.detach()` calls — they are the point:

```python
# 1. bracket the valid region
t_min, t_max = find_marching_distance_boundaries(
    *(tensor.detach() for tensor in (r, d, c, a)), eps=eps)

# 2. initial guess
dist = approximate_marching_distance(
    *(tensor.detach() for tensor in (t_min, t_max, r, d, c, a)), eps=eps)

# 3. refine — up to 16 Newton iterations
dist = refine_marching_distance(
    *(tensor.detach() for tensor in (dist, t_min, t_max, r, d, c, a)),
    max_iter=max_iter, tol=tol)

# 4. ONE final iteration, WITH autograd
dist, is_inside, delta_z, _ = update_marching_distance(dist, r, d, c, a)
dist = dist.clip(min=t_min, max=t_max)
```

Stages 1–3 are fully detached (`approximate_marching_distance` and
`refine_marching_distance` also carry `@torch.no_grad()` decorators at
`ray_tracing.py:159, 220`). Stage 4 repeats **one** Newton step with gradients enabled.

**Why this is correct, not a shortcut.** At convergence, the Newton update is a fixed
point: one more step from the converged value returns (numerically) the same value.
But autograd does not care about the value — it records the *operation*. That single
differentiable step is the implicit-function derivative of the intersection condition,
evaluated at the solution. So you get the exact gradient of the true root.

**What it saves.** Differentiating through all 16 iterations would build a graph 16×
deeper, cost 16× the backward memory, and give the *same* answer. Worse, it would
differentiate the transient convergence path — numerically noisy — instead of the
converged root.

This is the standard "implicit differentiation of a fixed point" trick, and it recurs
whenever a differentiable pipeline contains an iterative solver. **This is a very
likely supervisor question** ("you have a loop inside your forward pass — how do you
backprop through it?"), and `ray_tracing.py:139-147` is your citation.

The initial guess (`ray_tracing.py:160-217`) is itself careful: it samples 5 candidate
distances spread proportionally to ray height (`ray_tracing.py:188-194`), refines each
once, then picks by a merit that prefers the **nearest** intersection
(`ray_tracing.py:204`, `merit = 1/|dist|`). That last detail handles the case where a
steeply curved asphere is crossed more than once — you want the first surface the ray
meets, not an aphysical downstream root.

## 5.5 Snell's law, vector form

`apply_snell_spherical` (`ray_tracing.py:346-380`):

```python
cos2_prime = 1 - mu**2 * (1 - cos_theta**2)
tir = cos2_prime - eps < 0
cos_prime = (cos2_prime.where(~tir, 1.0)).sqrt()
g = cos_prime - mu * cos_theta
d = mu * d - g * (c * r - get_z_mask(d))
```

with `mu = n1/n2`. The first line is Snell in cosine form. If `cos2_prime < 0` the
refraction is impossible — **total internal reflection**.

Look at how TIR is handled:

```python
cos_prime = (cos2_prime.where(~tir, 1.0)).sqrt()
```

The offending values are **replaced by 1.0 before the square root**, not after. This is
essential and easy to get wrong: `sqrt(negative)` gives NaN, and NaN in the forward
pass poisons the entire backward pass — including for the rays that were fine, once
they mix in a sum. By substituting a safe dummy value first, the arithmetic stays
finite; the ray is separately *marked* as failed and its contribution removed later.

**Never produce a NaN you intend to mask afterwards.** Mask the input, not the output.

For aspheres, `apply_snell_aspherical` (`ray_tracing.py:383-420`) must first construct
the surface normal explicitly (`ray_tracing.py:401-405`):

```python
derivative, *_ = evaluate_aspherical_profile((r[:2]**2).sum(dim=0), c, a,
                                             compute_derivative=True)
n = -torch.cat((2 * r[:2] * derivative, -torch.ones_like(r[0:1])), dim=0)
n = n / n.norm(dim=0)
```

This is `∇(z − z(ρ))` with `∂ρ²/∂x = 2x`, normalized. It returns `cos_n` — the
z-component of the normal — which is exactly what `SurfaceNormalResiduals` consumes to
keep surfaces manufacturable (Chapter 7).

## 5.6 `ray_status`: five states, monotonic severity

`ray_status` (`optics.py:399`) is an **integer** tensor, one entry per ray:

| code | meaning | set at |
|---|---|---|
| 0 | fine | initialized `optics.py:399` |
| 1 | backtracked (negative propagation distance) | `optics.py:482` |
| 2 | total internal reflection | `optics.py:434` |
| 3 | direction went backward (`d_z < 0`) | `optics.py:433` |
| 4 | missed the surface / aspheric solve failed | `optics.py:471` |

Measured on the toy system: `ray_status_final` has `min = 0, max = 0`, and
`ray_valid_final` is `true_fraction = 1.0` over all **2816** rays. Every ray survives
this design. That is a property of a converged, well-behaved toy lens — do not assume
it holds mid-optimization, which is precisely why the machinery exists.

The **ordering is load-bearing**. Look at line 435:

```python
r, d = rt.reset_bad_rays(r, d, ray_status < 2, normalize=True)
```

The validity test is `< 2`. Codes 0 and 1 are *survivable*: a backtracking ray is
suspicious and gets penalized by a residual, but it still traces. Codes 2, 3, 4 are
*fatal* and the ray is reset. The numbering is a severity scale, and the threshold is
encoded in the comparison.

There is one subtle line worth reading twice (`optics.py:482`):

```python
ray_status = ray_status.where(~backtrack | (ray_status > 0), 1)
```

Write 1 only where the ray backtracked **and** its status is still 0. The `| (status >
0)` clause protects an already-recorded worse failure from being downgraded to 1.
Status is **monotonic** — once a ray fails badly, later events cannot make it look
better. Without that guard, a ray that hit TIR at surface 2 and then backtracked at
surface 3 would end up reported as merely "backtracked".

## 5.7 `reset_bad_rays` — keeping the graph finite

`ray_tracing.py:423-442`:

```python
r = r.where(ray_valid, 0.0)
d = d.where(ray_valid, get_z_mask(d))
if normalize:
    with torch.no_grad():
        norm = d.norm(dim=0)
    d = d / norm
```

Failed rays are parked at the origin travelling along `+z`. The docstring is explicit:
*"The goal is to avoid NaNs in the forward/backward pass."* A failed ray keeps flowing
through the tensor pipeline as a well-defined dummy; downstream code excludes it by
mask, not by exception.

**The design principle:** never raise, never NaN. An optimizer that throws when a
candidate design loses a ray cannot explore. LM proposes bad steps *on purpose* and
needs a finite loss back in order to reject them. Ray failure must be a **number**, not
a control-flow event.

Note the renormalization detail (`ray_tracing.py:439-441`): the norm is computed under
`no_grad` and then divided out. So the division rescales the value but contributes no
gradient of its own — the normalization is treated as a projection, not as part of the
physics. Cheap and stable.

## 5.8 Dispersion

`hartmann_dispersion` (called at `optics.py:413`) turns `(nd, vd, dpgf)` plus wavelength
into a refractive index. The Hartmann formula is an empirical fit that reproduces glass
catalog behaviour from these three numbers.

Toy config: `wavelengths = [550.]`, `w0 = 550.`, so
`wavelength_ratios = [1.0]` — measured exactly. And with `vd = 1e6`, dispersion is
suppressed anyway. Both refractive indices are exactly 1.5 at all sampled wavelengths.
The dispersion machinery runs but is a no-op here by construction.

## 5.9 What the trace produces

Measured at the sensor (`3_ray_trace`):

| quantity | shape | value |
|---|---|---|
| `r_at_sensor` | `[3, 11, 256, 1, 1]` | grad ✓ |
| `x_at_sensor` | `[11, 256, 1, 1]` | min −0.0745, max +0.0144 |
| `y_at_sensor` | `[11, 256, 1, 1]` | min −0.0144, max +4.9606 |
| `y_centroid` | `[11, 1, 1, 1]` | 0.0, 0.526, 1.051, …, 4.952 |
| `rms_spot_size` | `[11]` | 0.00995 … 0.02996 |

Two readings of that table.

**The `y` range is field position, not blur.** `y_at_sensor` reaching 4.96 mm is the
corner field landing near the edge of the 11.55 mm-diagonal sensor. The blur is the
*spread about the centroid*: `transverse_ray_aberrations_y` spans only
`[−0.0299, +0.0246]` mm — about 30 µm.

**Spot size grows 3× from axis to corner:**

| field | angle | RMS spot (mm) |
|---|---|---|
| 0 | 0° | 0.00995 |
| 5 | 15° | 0.01161 |
| 10 | 30° | 0.02996 |

A textbook off-axis aberration curve — nearly flat to ~12°, then rising steeply. Two
elements cannot correct 30° of field.

**Distortion.** Compare the measured centroid to the paraxial prediction
`y = f·tan(θ)` with `f = 10`:

| field | angle | `y_centroid` | `10·tan θ` | distortion |
|---|---|---|---|---|
| 0 | 0° | 0.0000 | 0.0000 | — |
| 2 | 6° | 1.0510 | 1.0510 | −0.00% |
| 5 | 15° | 2.5962 | 2.6795 | **−3.11%** |
| 8 | 24° | 4.0571 | 4.4523 | **−8.88%** |
| 10 | 30° | 4.9523 | 5.7735 | **−14.22%** |

**−14% barrel distortion at the corner.** This is a real, quantitative property of the
shipped toy design that you can state in a defense. Nothing in the toy residual set
penalizes distortion — `TransverseRayAberrationResiduals` measures spread *about the
centroid* and is blind to where the centroid sits. The design is sharp but geometrically
distorted, and that is a direct consequence of what the loss does and does not measure.
(`DistortionResiduals` exists in the library — Chapter 7 — it is simply not enabled
here.)

## 5.10 What to take away

1. The trace is an interpreter over the event list; `p` updates position, `r` updates
   direction.
2. `cum_z_distance` keeps rays global and surfaces local, and makes every surface
   position differentiably depend on all preceding spacings.
3. Aspheric intersection has no closed form: iterate **detached**, then take **one**
   differentiable Newton step. Same gradient, 1/16 the graph.
4. TIR is handled by substituting a safe value *before* `sqrt`, never by masking a NaN
   afterwards.
5. `ray_status` is a monotonic severity scale 0–4; `< 2` is the survivable threshold,
   and the `| (ray_status > 0)` guard stops downgrades.
6. Failed rays are parked, never raised. The optimizer must get a finite number back
   from a bad design.
7. The shipped toy design loses **zero** of 2816 rays, has 10–30 µm spots, and
   **−14.2% distortion at the corner** — which nothing in its loss penalizes.

---

*Previous: [Chapter 4 — Ray initialization](04_ray_initialization.md)*
*Next: [Chapter 6 — Paraxial solves](06_solves.md)*
