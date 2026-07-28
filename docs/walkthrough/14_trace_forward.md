# Chapter 14 — Execution trace: YAML → CLI → lens → rays → sensor

> **What this chapter is.** One complete forward pass, followed from the command line to
> the sensor, with **every number measured** by running the code. No shape is asserted from
> a docstring. If a value appears here it came out of `instrument_trace.py`, and
> `trace_dump.json` in this directory holds the raw record.
>
> Read this chapter with Chapters 1–5 open. Those explain *why*; this one shows *what
> actually happens*, in order.

---

## 14.0 The run being traced

```bash
python -m eisoptx.main fit \
  -c configs/toy/defaults.yml \
  -c configs/toy/designs/optimized_spot.yml
```

Environment: `torch 2.12.0`, seed `0`, `precision: 64`. Everything below is from
`trace_dump.json`, sections `0_meta` through `4_residuals`.

---

## 14.1 Stage 0 — YAML to constructor arguments

Two config files, merged left to right: `defaults.yml` gives the architecture, the design
file overrides the variable values. Lightning's CLI instantiates each `class_path` with its
`init_args`.

Before instantiation, the three argument links of Chapter 12 §12.8 fire. The one that
matters:

```
sensor_diagonal = 2 · target_efl · tan(hfov)
                = 2 · 10.0 · tan(30°)
                = 11.5470053838 mm
```

**This number is in no configuration file.** It is derived. That is the first thing worth
knowing about the trace: some of what the module receives was computed by the CLI, not
written by a human.

The result is an `ImagingSystemModule` holding a `LensParameterization`, a
`RayInitialization`, an `OpticsSimulator`, and an optimizer spec.

## 14.2 Stage 1 — the sequence string becomes a structure

**Measured** (`1_config_and_lens`):

```
sequence_string : s-aRa-aRa-
n_interfaces    : 4
n_propagations  : 5
n_refractive    : 2
n_aspherical    : 4
n_diffractive   : 0
stop_idx        : 1
sequence_events : ['p','s','p','r','p','r','p','r','p','r','p']
```

Read `s-aRa-aRa-` character by character (Chapter 2):

| char | meaning |
|---|---|
| `s` | aperture stop |
| `-` | propagation (air gap) |
| `a` | aspheric refractive surface |
| `R` | refractive **medium** — glass between two surfaces |
| `a` | aspheric surface |

So: stop, gap, [asphere / glass / asphere], gap, [asphere / glass / asphere], gap to sensor.
**Two elements, four surfaces, all aspheric.**

The `sequence_events` list is the *interpreter program* of Chapter 5 — 11 events,
alternating propagate and interact:

```
p  s  p  r  p  r  p  r  p  r  p
0  1  2  3  4  5  6  7  8  9  10
```

Five `p` (propagations), one `s` (stop), four `r` (refractions). `n_propagations = 5`
matches; `n_interfaces = 4` counts the refractive surfaces. **The ray tracer executes
exactly this list.** Everything downstream is a consequence of parsing this one string.

## 14.3 Stage 2 — variables to a lens

The flat parameter vector, measured:

```
total_numel     : 31
total_trainable : 20
```

Layout (`variable_layout`):

| key | shape | numel | slice | trainable |
|---|---|---|---|---|
| `s` spacings | `[5,1]` | 5 | `[0:5]` | 5 |
| `g` glass | `[2,1,3]` | 6 | `[5:11]` | **0** |
| `c` curvatures | `[4,1]` | 4 | `[11:15]` | **3** |
| `a` aspheric | `[4,1,4]` | 16 | `[15:31]` | 12 |
| `d`, `m` | `[0,1,0]` | 0 | — | 0 |

**31 stored, 20 trainable.** Three groups are held back:

- **glass (6)** — the toy config uses a fictitious material, `nd = 1.5`, `vd = 1e6`. An
  Abbe number of a million means **zero dispersion**. Combined with a single wavelength
  (§14.4), the toy system has no chromatic behaviour whatsoever, by construction.
- **one curvature of four** — the last one, because `solve_idx = -1`, `solve_type =
  focal_length`. Chapter 6: it is *computed* from the EFL target, so it must not also be
  optimized.
- **`d` and `m`** are empty: no diffractive or metasurface elements.

`20 = 5 + 3 + 12` — and 20 is exactly the variable count in the LM Jacobian of §14.9. The
chain closes.

### Scaling

Chapter 3's `scale_factor = 2.5`, applied as `c × 2.5` and `s ÷ 2.5`:

| | variable | scaled (physical) |
|---|---|---|
| `s[0]` | 0.5291371942 | 1.3228429556 mm |
| `s[4]` | 4.0861597061 | 10.2153987885 mm |
| `c[0]` | −0.5445552468 | −0.2178221047 mm⁻¹ |
| `c[3]` | −0.2304074019 | −0.0921629593 mm⁻¹ |

Check: `0.5291371942 × 2.5 = 1.32284298…` ✓ and `−0.5445552468 / 2.5 = −0.2178220987…` ✓
(to float32 print precision). The scaling is exactly what Chapter 3 said.

### After solves

```
lens.c_after_solves : [-0.2178221047, -0.1592064947, 0.1029609889, -0.0921629593]
lens.efl            : [10.0]
lens.bfl            : [10.2538299561]
```

`efl = 10.0` **exactly**, because `c[3]` was solved for it. Chapter 6 measured the residual
sensitivity `|∂EFL/∂vars| = 2.98e−07` — the pin holds under differentiation, not just at
this point.

The full ABCD product:

```
lens.get_abcd(reduce=True) : [[0.0038431287, 9.9876174927],
                              [-0.1000000015, 0.3222140372]]
```

**`C = −0.1000000015`, and `EFL = −1/C = 9.99999985 ≈ 10.0`.** That is the first-order
optics of Chapter 13 §13.2 checking out against the solve of Chapter 6, independently.

`get_abcd(reduce=False)` is `[11,1,2,2]` — **one matrix per sequence event**, the same 11
events as §14.2. The paraxial model and the real trace walk the identical program.

## 14.4 Stage 3 — ray initialization

**Measured** (`2_ray_initialization`):

```
hfov_deg              : 30.0
n_fields              : 11
wavelengths_nm        : [550.0]
pupil_sampling_mode   : skew_uniform_jittered
ray_aiming_steps      : 0
epd_computed          : 5.0
pupil_position_z      : [0.0]
r_initial             : [3, 11, 256, 1, 1]
d_initial             : [3, 11, 256, 1, 1]
```

`[3, 11, 256, 1, 1]` = `[xyz, n_fields, n_rays, n_wavelengths, n_lens]`. **2816 rays.**

- **11 fields**, 0° to 30° in 3° steps.
- **256 rays**, from Chapter 4's 16 shells: `1+3+5+…+31 = 16² = 256`. Verified.
- **1 wavelength.** With `vd = 1e6` glass, the system is monochromatic twice over.
- `epd_computed = 5.0` with `efl = 10.0` → **f/2**.
- `pupil_position_z = 0.0` and `ray_aiming_steps = 0` — the stop *is* the first surface, so
  aiming is unnecessary rather than approximate (Chapter 4).

`r_initial` ranges ±2.4788086414. The pupil semi-radius is `5.0/2 = 2.5`, so the outermost
sampled ray sits at `2.4788/2.5 = 99.15%` of the pupil edge — the jitter pulls it just
inside, as designed.

## 14.5 Stage 4 — the trace, event by event

This is the measured progression through all 11 events (`per_event_steps`). `r.min`/`r.max`
are over all three components; because `z` accumulates, `r.max` is effectively the global
z-position of the ray front.

| # | event | r.min | r.max | r.mean | `d.grad` | status | ok | data emitted |
|---|---|---|---|---|---|---|---|---|
| 0 | `p` | −2.478809 | 2.478809 | 0.352992 | False | 0 | 2816 | — |
| 1 | `s` | −2.478809 | 2.478809 | 0.352992 | False | 0 | 2816 | — |
| 2 | `p` | −2.478809 | 2.758776 | 0.768771 | False | 0 | 2816 | `delta_z` |
| 3 | `r` | −2.478809 | 2.758776 | 0.768771 | **True** | 0 | 2816 | `cos_n`, `cos2_theta`, `cos2_prime` |
| 4 | `p` | −3.022732 | 4.505201 | 2.073889 | True | 0 | 2816 | `delta_z` |
| 5 | `r` | −3.022732 | 4.505201 | 2.073889 | True | 0 | 2816 | `cos_n`, `cos2_theta`, `cos2_prime` |
| 6 | `p` | −3.076614 | 6.885920 | 2.624723 | True | 0 | 2816 | `delta_z` |
| 7 | `r` | −3.076614 | 6.885920 | 2.624723 | True | 0 | 2816 | `cos_n`, `cos2_theta`, `cos2_prime` |
| 8 | `p` | −2.688461 | 11.007553 | 4.510008 | True | 0 | 2816 | `delta_z` |
| 9 | `r` | −2.688461 | 11.007553 | 4.510008 | True | 0 | 2816 | `cos_n`, `cos2_theta`, `cos2_prime` |
| 10 | `p` | −0.074524 | 21.223282 | 7.922042 | True | 0 | 2816 | `delta_z` |

Five readings from this table.

**1. Propagation moves positions; refraction moves directions.** Events 3, 5, 7, 9 leave
`r` completely unchanged — identical min, max and mean to the preceding `p`. Refraction
happens *at* a surface, so the ray is already there. Only `d` changes. Conversely `p`
events change `r` and leave `d` alone. Chapter 5's two primitives, visible in the data.

**2. `d.requires_grad` flips at event 3.** Before the first refraction, ray directions are
pure field angles — geometry, no lens parameters. At the first refractive surface, Snell's
law brings `n` and the surface normal (hence `c` and the aspheric coefficients) into the
computation, and directions become differentiable functions of the design. `r` is already
differentiable at event 0 because pupil positions depend on the entrance pupil, which
depends on the lens.

**Event 3 is where the lens enters the gradient graph.** Everything before it is a constant
with respect to the design.

**3. `r.max` traces the physical layout.** 2.48 → 2.76 → 4.51 → 6.89 → 11.01 → 21.22. The
jumps between refractions are the air gaps and glass thicknesses; the last, largest jump
(11.01 → 21.22) is the back focal distance. `bfl = 10.2538299561`, and `21.223 − 11.008 =
10.215` — matching `s[4] = 10.2153987885`, the last spacing. The trace agrees with the
prescription.

**4. `r.min` goes to −0.0745 at the end.** Off-axis rays converge from wide positions to a
tight spot; the final min is the largest negative `x` at the sensor. The beam has collapsed
from ±2.5 mm at the pupil to tens of microns at the image.

**5. Zero ray failures at every event.** `status_counts` is `{ok_0: 2816, backtrack_1: 0,
tir_2: 0, backward_3: 0, miss_4: 0}` throughout. The optimized design is well-behaved; no
rays are clipped, none total-internal-reflect. Chapter 5 explained the machinery for when
they do — this run never needs it.

The `delta_z`, `cos_n`, `cos2_theta`, `cos2_prime` columns are the **data collection**
channel: the tracer emits these as it goes so the residual library can consume them without
a second pass (Chapter 7).

## 14.6 Stage 5 — at the sensor

```
r_at_sensor  : [3, 11, 256, 1, 1]   min −0.0745244026   max 21.2232818604
x_at_sensor  : [11, 256, 1, 1]      min −0.0745244026   max  0.0143616199
y_at_sensor  : [11, 256, 1, 1]      min −0.0143558979   max  4.9605784416
ray_valid    : true_fraction 1.0
```

`y` reaching 4.96 is the corner field's image height; `x` stays within ±0.075 because
fields lie in the y–z plane (Chapter 4) and only aberration spreads rays in `x`.

**Centroids and spot sizes, per field:**

| field | angle | `y_centroid` | paraxial `10·tan θ` | distortion | RMS spot (mm) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0° | 0.0000000 | 0.00000 | — | 0.0099527 |
| 1 | 3° | 0.5264055 | 0.52408 | +0.44% | 0.0099686 |
| 2 | 6° | 1.0510405 | 1.05104 | +0.00% | 0.0100253 |
| 3 | 9° | 1.5721134 | 1.58384 | −0.74% | 0.0101813 |
| 4 | 12° | 2.0877895 | 2.12557 | −1.78% | 0.0106083 |
| 5 | 15° | 2.5961735 | 2.67949 | −3.11% | 0.0116144 |
| 6 | 18° | 3.0952878 | 3.24920 | −4.74% | 0.0135224 |
| 7 | 21° | 3.5830297 | 3.83864 | −6.66% | 0.0164709 |
| 8 | 24° | 4.0570951 | 4.45229 | −8.88% | 0.0203395 |
| 9 | 27° | 4.5147543 | 5.09525 | −11.39% | 0.0248302 |
| 10 | 30° | 4.9522867 | 5.77350 | **−14.22%** | 0.0299637 |

Two findings, both worth having ready.

**Spot size grows 3.0× from axis to corner** — 9.95 µm to 29.96 µm RMS. Monotonic, and the
growth is slow until about 15° then accelerates. That is the normal signature of field
aberrations.

**The design has −14.2% barrel distortion at the corner, and nothing penalizes it.** This
is not a bug; it is a consequence of the residual set. Chapter 7 established that
`TransverseRayAberrationResiduals` measures spread **about the centroid**, so a field whose
centroid is in the wrong place but whose rays are tightly bunched scores perfectly.
`DistortionResiduals` exists in the library and is **not enabled** in this config.

If asked "is this a good lens?", the honest answer is: it is a good *spot-size* lens and a
poor *distortion* lens, and that is exactly what it was told to be.

Derived quantities:

```
transverse_ray_aberrations_x : [11, 256, 1, 1]   ±0.0745 / +0.0144
transverse_ray_aberrations_y : [11, 256, 1, 1]   −0.0299 / +0.0246
```

These are `xy − centroid` — the raw material of the residual vector.

## 14.7 Stage 6 — residuals

**Measured** (`4_residuals`):

```
registered_residual_names : ['transverse_ray_aberration', 'ray_path', 'ray_angle', 'surface_normal']
data_collection_keys      : ['cos2_prime', 'cos2_theta', 'cos_n', 'delta_z']
e2e_enabled               : False
```

| residual | numel | weight | ‖·‖₂ (raw) | Σ(·²) raw |
|---|---:|---:|---:|---:|
| `transverse_ray_aberration` | 5632 | 1.0 | 0.0166029222 | 2.756570e−04 |
| `ray_path` | 14 | 20.0 | 0.0004212540 | 1.774549e−07 |
| `ray_angle` | 0 | 10.0 | 0.0 | 0.0 |
| `surface_normal` | 58 | 10.0 | 0.0006775917 | 4.591305e−07 |
| **total** | **5704** | | | **2.762936e−04** |

`5632 = 2 × 11 × 256` — x and y aberration for every ray. It is **98.7%** of the residual
vector; the constraint-like residuals are 72 numbers against 5632.

`ray_angle` is registered with weight 10 but contributes **zero elements**. It only fires on
surfaces that violate its angle limit; none do. A residual that is configured, active, and
empty — worth knowing before you go looking for its 0 entries in the Jacobian.

### The weighting, verified numerically

Here the trace is more informative than it first looks, and the discrepancy is the lesson.

Half the raw sum above is `0.5 × 2.762936e−04 = 1.381468e−04`. But the optimizer reports

```
least_squares_loss : 0.0001418987
```

**These do not match.** The gap is not error — it is the weighting. The `4_residuals`
section records residuals **before** weighting; the vector the optimizer receives has
Chapter 7's `√weight` already applied. Squaring `√w · r` gives `w · r²`, so:

```
Σ = 1.0 × 2.756570e−04  +  20 × 1.774549e−07  +  10 × 4.591305e−07
  = 2.837974037694e−04
½Σ = 1.4189870188e−04      vs reported 1.418987e−04
relative error 1.33e−08
```

**Exact to eight decimal places.** And an independent confirmation from the extreme values:
the largest raw `ray_path` residual is `0.000284202`, and

```
0.000284202 × √20 = 0.0012709900     vs  weighted vector max 0.0012709902
```

If the code applied `weight` rather than `√weight`, that entry would be `0.005684` and the
loss would be `1.416e−03` — an order of magnitude out. **The measurement settles it: it is
`√weight`.** A `weight: 20.` in a YAML file contributes a factor of 20 to the *loss*, and
4.472 to the residual.

This is worth rehearsing, because "what does `weight: 20` mean?" is a natural supervisor
question and the intuitive answer is wrong.

## 14.8 The complete forward-pass shape ladder

Every shape in one place, measured:

```
config YAML
   ↓  CLI links: sensor_diagonal = 11.5470053838
flat variables                        [31]  (20 trainable)
   ↓  scale_lens_parameters (c×2.5, s÷2.5)
lens parameters              s[5,1] c[4,1] a[4,1,4] nd[2,1] vd[2,1]
   ↓  curvature_solve  →  efl = 10.0 exactly
Lens object                           abcd [11,1,2,2] → [1,2,2]
   ↓  ray_initialization
r_initial, d_initial          [3, 11, 256, 1, 1]      2816 rays
   ↓  11-event trace  (p s p r p r p r p r p)
r_at_sensor                   [3, 11, 256, 1, 1]      0 failures
   ↓  centroid subtraction
transverse aberrations x,y    [11, 256, 1, 1] each
   ↓  flatten + √weight + concat
residual vector                       [5704]
   ↓  0.5·Σ(·²)
least-squares loss                    scalar   1.418987e−04
```

## 14.9 Where the optimizer picks it up

The next chapter follows the LM iteration. The handoff, measured (`5_lm_step`):

```
residual_vector : [5704]
jacobian        : [5704, 20]      ← 5704 residuals from §14.7, 20 variables from §14.3
```

Both numbers were derived independently, several stages apart, and they meet exactly. That
is the strongest confirmation available that the chain above is understood correctly.

## 14.10 What to take away

1. `sensor_diagonal = 11.5470053838` is **derived by the CLI**, not configured.
2. `s-aRa-aRa-` compiles to an **11-event program**; the tracer and the paraxial ABCD chain
   both execute it.
3. **31 stored variables, 20 trainable** — glass frozen (fictitious `vd = 1e6`), one
   curvature reserved for the EFL solve.
4. `EFL = 10.0` confirmed twice: by the solve, and independently by `−1/C = 9.99999985`
   from the ABCD product.
5. **2816 rays** = 11 fields × 256 pupil points; `r_initial` reaches 99.15% of the pupil
   semi-radius.
6. **The lens enters the gradient graph at event 3**, the first refraction. Everything
   before is design-independent.
7. `p` events change positions only, `r` events change directions only — visible in the
   per-event table as identical `r` statistics across each refraction.
8. **Zero ray failures** at every event.
9. Spot size **9.95 µm → 29.96 µm** (3.0×) from axis to corner.
10. **−14.22% distortion at the corner, unpenalized** — a consequence of centroid-relative
    residuals plus a disabled `DistortionResiduals`.
11. Residual vector **5704 = 5632 + 14 + 0 + 58**; the ray-aberration block is 98.7% of it.
12. **`√weight`, verified:** the raw sum of squares (`2.762936e−04`) does *not* halve to the
    reported loss; applying the weights does, to a relative error of **1.33e−08**. Confirmed
    independently by `0.000284202 × √20 = 0.0012709900` matching the weighted vector's
    maximum `0.0012709902`.
13. Loss `1.418987e−04`, and the Jacobian is `[5704, 20]` — both endpoints of the chain
    agree.

---

*Previous: [Chapter 13 — The periphery](13_periphery.md)*
*Next: [Chapter 15 — Execution trace: one LM/joint iteration](15_trace_training.md)*
