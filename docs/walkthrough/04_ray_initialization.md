# Chapter 4 — Ray initialization: where 2816 rays come from

> **What this chapter answers.** Where does a ray start? Why 256 rays per field and not
> 250? What is the `[3, 11, 256, 1, 1]` shape? What is ray aiming, and why is it off in
> the toy config? Why does only *half* the pupil get sampled?

---

## 4.1 The job

Before you can trace, you must decide **which** rays to trace. A lens has an infinite
ray manifold; you sample it. Two independent choices:

1. **Which fields** — the angles the light arrives from. On-axis, 10° off-axis, …
2. **Which pupil points** — where each ray crosses the entrance pupil.

The Cartesian product of these gives the ray set. Toy config: 11 fields × 256 pupil
points = **2816 rays**, measured in `2_ray_initialization` as
`r_initial: [3, 11, 256, 1, 1]`.

## 4.2 The universal tensor shape

Every ray quantity in this codebase has the same five-axis layout:

```
[3, n_fields, n_rays, n_wavelengths, n_lens]
 │      │        │          │            └── batched optics (Chapter 2), = 1
 │      │        │          └── chromatic sampling, = 1 (monochromatic toy)
 │      │        └── pupil samples, = 256
 │      └── field angles, = 11
 └── x, y, z — the spatial vector components
```

Measured: `r_initial` and `d_initial` are both `[3, 11, 256, 1, 1]`.

Put the vector components **first** rather than last. That is not the PyTorch
convention (channels usually trail), and it is a deliberate choice: it makes
`r[0], r[1], r[2]` read as `x, y, z`, and makes cross products and dot products over
`dim=0` natural in the tracing code. When you read `ray_tracing.py` and see `dim=0`
reductions everywhere, this is why.

## 4.3 Field directions

`initialize_ray_directions` (`ray_initialization.py:421-436`):

```python
rel_fields = np.linspace(1, 0, n_fields)[::-1].tolist()
u = torch.tensor(rel_fields).to(hfov) * hfov.deg2rad()
cy = u.sin().view(-1, 1, 1, 1)
cx = torch.zeros_like(cy)
cz = (1 - cx**2 - cy**2).sqrt()
d = torch.stack(torch.broadcast_tensors(cx, cy, cz))
```

Three things to notice.

**Fields lie in the y–z plane only** (`cx = 0`). The system is assumed **rotationally
symmetric**, so a field at azimuth 0 tells you everything about a field at any other
azimuth. This assumption threads through the whole codebase — most visibly in the PSF
simulator (Chapter 10), which samples PSFs radially and then *rotates* them onto the
2-D sensor grid.

**Directions are unit vectors by construction**: `cz = sqrt(1 − cx² − cy²)`. Measured
`r_norm_check` has `max = 1.0`; the direction vectors are normalized to machine
precision.

**The reversal `[::-1]` is deliberate**, and the comment says why
(`ray_initialization.py:429-430`): *"Upside down so the field is maximal if only one"*.
With `n_fields = 1` you get the **corner** field, not the axis. If you are debugging
with a single field, you are looking at the hardest case, not the easiest. That is a
good default for optical design, and a trap if you assume single-field means on-axis.

For the toy config, `n_fields = 11` and `hfov = 30°` gives field angles
0°, 3°, 6°, …, 30°.

## 4.4 Pupil sampling: why exactly 256

`configs/toy/defaults.yml:45-48`:

```yaml
pupil_sampling_mode: skew_uniform_jittered
pupil_sampling_kwargs:
  n_r: 16
  n_i: 1
```

`n_r` is **not** a ray count — it is the number of concentric shells. The ray count
comes from `ray_initialization.py:598`:

```python
rays_per_shell = np.arange(1, 2 * n_r, 2) * n_i
```

That is the odd numbers: `[1, 3, 5, 7, …, 31]`. And the sum of the first `n` odd
numbers is `n²`:

```
1 + 3 + 5 + … + 31 = 16² = 256      ✓ measured n_rays = 256
```

This is not numerology. Shell `k` has area proportional to `(k+1)² − k² = 2k + 1` — an
odd number. Giving each shell a ray count proportional to its **area** is exactly what
uniform sampling of a disk requires. The odd-number progression *is* equal-area
sampling.

The shell radii follow the same logic (`ray_initialization.py:605-609`):

```python
shell_limits = np.linspace(0, 1, n_r + 1)
shell_mean_r = np.sqrt((outer_r**2 + inner_r**2) / 2)
```

`sqrt((r_out² + r_in²)/2)` is the **RMS radius** of the annulus — the radius that
splits it into two equal areas. Again: equal area, not equal radius. Placing rays at
the arithmetic mean radius would over-weight the pupil centre and systematically
under-estimate spot size, because outer rays carry more aberration.

> **Defense note.** "Why 256 rays?" is a question with a real answer here: `n_r = 16`
> equal-area shells with odd-numbered ray counts, giving `n_r² = 256` rays whose
> density is uniform over the pupil disk. It is not an arbitrary power of two.

## 4.5 Half the pupil, and the jitter

The docstring at `ray_initialization.py:588` says the pattern spans **"the right half
of the pupil"**, and the angles confirm it (`ray_initialization.py:612-614`):

```python
theta = np.array([(i / n - 0.5) * np.pi for n in rays_per_shell
                  for i in (np.arange(n) + 0.5)])
```

`(i/n − 0.5)·π` for `i/n ∈ (0,1)` spans `(−π/2, +π/2)` — a half-disk. Combined with the
`cx = 0` field choice of §4.3, this halves the ray count for free: the system is
symmetric about the y–z plane, so the left half of the pupil is the mirror image of the
right. Measured `r_initial` confirms the symmetric span in x: `min = −2.479`,
`max = +2.479`, with the entrance pupil radius being 2.5 mm.

Then there is the jitter (`ray_initialization.py:616-648`), 33 lines of heuristics. The
comments are honest about their nature — *"heuristic to sample outer edge of pupil and
to move rays apart from each other"*. What it does:

- **Radial jitter**, odd shells only, applied **in pairs**: one ray pushed outward, its
  partner pushed inward by the matching amount (`ray_initialization.py:618, 621-623`).
  Pairing is what keeps the mean radius of the shell unchanged — the bias cancels.
- **Angular jitter**, odd shells stretched and even shells compressed toward the axes
  by up to a quarter of the shell's angular spacing (`ray_initialization.py:635-638`).

Why bother? A perfectly regular polar grid has two failure modes: rays align into
spokes (so the spot diagram shows structure that is an artifact of the sampling, not of
the lens), and no ray ever lands exactly on the pupil rim, where aberration is
worst. The jitter breaks the spokes and reaches the edge. The docstring concedes the
pattern is *"slightly biased"* — an accepted trade for edge coverage.

This function is **deterministic**. Despite the name, there is no randomness — it is a
fixed perturbation pattern computed from indices. Same rays every call, every run. For
an LM optimizer this is essential: a Jacobian computed against a *different* ray set
each iteration would make the accept/reject test meaningless.

> **⚠ Worth knowing.** `assert n_r % 2 == 0` (`ray_initialization.py:597`) — the shell
> count must be even. The pairwise radial jitter needs it.

## 4.6 Entrance pupil position and diameter

Measured (`2_ray_initialization`):

```
epd_computed      = 5.0        # entrance pupil diameter, mm
pupil_position_z  = [0.0]      # relative to the first surface
```

The config sets `aperture_type: epd, aperture: 5.` directly, so the EPD is given rather
than derived. The alternative is `f_number`, in which case
`EPD = EFL / f_number` — the note recorded in the trace. With `EFL = 10` and
`EPD = 5`, this toy lens is **f/2**.

`pupil_position_z = 0` because the stop is the first element of the sequence
(`s-aRa-aRa-`) with nothing in front of it. With no optics before the stop, the
entrance pupil **is** the stop, in its physical place. In a lens with elements ahead of
the stop, the entrance pupil is the stop's paraxial image through them, and this value
would be nonzero — computed via the ABCD matrices of Chapter 2.

Rays are then scaled from relative to absolute units by
`scale_to_epd` (`ray_initialization.py:656-663`):

```python
return [x * epd / 2 for x in args]
```

Relative pupil coordinates live in `[−1, 1]`; multiply by the pupil **radius** to get
millimetres. Measured `r_initial` extreme is 2.4788, just inside the 2.5 mm radius —
the jittered outer shell reaches 99.15% of the rim.

## 4.7 Ray aiming — off here, essential at scale

`ray_aiming_steps: 0` in the toy config, so `ray_aiming`
(`ray_initialization.py:260-354`) does not run. You still need to understand it.

**The problem it solves.** "Entrance pupil" is a *paraxial* concept. A ray launched to
pass through the paraxial entrance pupil at relative height 1.0 will, after real
refraction through the elements ahead of the stop, generally **miss** the edge of the
physical stop — it may be clipped, or may leave part of the stop unfilled. This is
pupil aberration. At wide apertures and large fields it is significant, and it corrupts
the spot diagram: your sampling no longer matches the light the lens actually passes.

**The fix.** Iterate. Trace a ray, see where it actually crosses the stop plane, adjust
the launch position, repeat. `ray_aiming_step` (`ray_initialization.py:372-420`) is one
Newton-style correction; `ray_aiming_steps` is how many to run.

**Why the toy config sets 0.** The stop is the *first* surface — there is nothing ahead
of it to aberrate the pupil. The launch position at the stop is exact by construction.
Zero iterations is not a shortcut; it is the correct answer for this topology. The
paper-scale configs, where elements sit in front of the stop, enable it.

Cost matters too: each aiming step is a full differentiable trace, and it sits inside
the Jacobian computation. Turning it on multiplies the cost of every LM iteration.

## 4.8 Field stop correction

`field_stop_correction` (`ray_initialization.py:223-259`) handles vignetting by a field
stop. Measured `field_stop_position = None` — not used in the toy config. Mentioned for
completeness; it belongs to the same family as ray aiming, adjusting *which* rays are
considered valid before tracing begins.

## 4.9 The measured output

From `2_ray_initialization`:

| quantity | value |
|---|---|
| `hfov_deg` | 30.0 |
| `n_fields` | 11 |
| `wavelengths_nm` | [550.0] |
| `pupil_sampling_mode` | `skew_uniform_jittered` |
| `epd_computed` | 5.0 |
| `pupil_position_z` | [0.0] |
| `ray_directions_only` | `[3, 11, 1, 1, 1]` |
| `r_initial` | `[3, 11, 256, 1, 1]`, min −2.4788, max +2.4788 |
| `d_initial` | `[3, 11, 256, 1, 1]`, min 0.0, max 1.0 |
| `r_norm_check` | max 1.0 — directions are unit vectors |

Note `ray_directions_only` has `n_rays = 1` before broadcasting: direction depends only
on the field, position only on the pupil point. They are broadcast against each other
to form the full 2816-ray set. This is memory-efficient and expresses the physics —
a field is a direction, a pupil point is a position.

## 4.10 What to take away

1. Ray tensors are `[3, n_fields, n_rays, n_wavelengths, n_lens]`, components first.
2. Fields lie in the y–z plane; rotational symmetry is assumed throughout the codebase.
3. `linspace(1,0)[::-1]` means a single-field run gives you the **corner** field.
4. `n_r = 16` shells → `1+3+…+31 = 256` rays; odd counts and RMS shell radii are
   equal-area sampling, not arbitrary numbers.
5. Only the right half-pupil is sampled, exploiting the same symmetry.
6. The "jitter" is deterministic — a fixed pattern, so the ray set is identical across
   LM iterations.
7. Ray aiming is off in the toy config because the stop is the first surface, which
   makes it unnecessary rather than approximate.

---

*Previous: [Chapter 3 — Parameterization and the glass model](03_parameterization.md)*
*Next: [Chapter 5 — Ray tracing](05_ray_tracing.md)*
