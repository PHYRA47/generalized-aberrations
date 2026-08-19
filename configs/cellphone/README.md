# Cellphone camera lens — Ansys KA-01995 Part 1, replicated in `eisoptx`

## What the article actually contains

The Ansys/Zemax article **"Designing Cell phone Camera Lenses Part 1: Optics"**
does not publish its own prescription. It takes the design from
**US 2019/0129149 A1, "Wide FOV 5 element lens system," lens system 710**
(Tables 1, 2A, 2B, 3), enters it in OpticStudio, notes that the MTF target is
missed as-published, and then reoptimizes with real catalog plastics.

So the numeric source of truth is the patent, not the article. Both were used:
the patent for the prescription, the article for design targets and the
material substitution.

> The article's own downloadable archive (`710_MobilePhoneLens.zip` →
> `710_original.zar`, `710_reoptimized_MTF_materials_QType.zar`) was retrieved,
> but `.zar` stores LZW-compressed entries, so the OpticStudio files were not
> decoded. Every number below comes from the patent tables.

## Prescription (patent Table 1) — `*` marks aspheric surfaces

| Surf | R [mm] | Thickness [mm] | nd | vd | Element |
|---|---|---|---|---|---|
| 1 | Inf | 0.800 | 1.525 | 54.5 | cover glass |
| 2 | -194.000 | 0.150 |  |  | cover glass back |
| 3/4 | Inf | 0.000 |  |  | STOP (at L1 front vertex) |
| 5* | 2.049 | 0.527 | 1.545 | 56.0 | L1 |
| 6* | -182.726 | 0.231 |  |  |  |
| 7* | 33.602 | 0.251 | 1.678 | 19.5 | L2 |
| 8* | 6.457 | 0.098 |  |  |  |
| 9* | -3.661 | 0.876 | 1.545 | 56.0 | L3 |
| 10* | -0.989 | 0.050 |  |  |  |
| 11* | 4.898 | 0.350 | 1.678 | 19.5 | L4 |
| 12* | 2.044 | 0.097 |  |  |  |
| 13* | 0.777 | 0.301 | 1.545 | 56.0 | L5 |
| 14* | 0.592 | 0.386 |  |  |  |
| 15 | Inf | 0.210 | 1.517 | 64.2 | IR cut filter |
| 16 | Inf | 0.475 |  |  | to sensor |
| 17 | Inf | -- |  |  | sensor |

Design data (patent Table 3): **f = 2.399 mm, F/# = 2.0, full FOV = 95°,
TTL/(2·ImaH) = 0.764**. TTL surface 1 → sensor = **4.802 mm**.

## The two conventions that decide whether this transfers correctly

**1. Aspheric coefficients are on the PHYSICAL radial coordinate `r` [mm].**
This is the one real trap. The article presents OpticStudio's *Extended Asphere*
formula, whose polynomial is defined on a **normalized** ρ = r/r_max. The patent
tabulates on r in mm, and `eisoptx`'s `evaluate_aspherical_profile` also uses r
in mm:

    z = c·r² / (1 + √(1 − (1+k)c²r²)) + Σ pᵢ · r^(2(i+2))

so the patent's `A4…A20` transfer **directly, with no rescaling**. Reading them
as normalized coefficients is geometrically impossible — it would put a −1.19 mm
polynomial departure on surface 6, whose base sag is only −0.001 mm.

**2. The stop's negative spacing must be collapsed.** The patent places the stop
on a dummy surface +0.055 mm past the cover glass, then steps **−0.055 mm** back
to the L1 front vertex. Carried literally, `eisoptx` flags `delta_z < 0` as
"backtrack" on every ray (45% backtrack, 20% valid). A stop coincident with the
L1 front vertex is optically identical and traces cleanly.

## Mapping onto the config schema

| Config field | Value |
|---|---|
| `lens_sequence` | `R-s-aRa-aRa-aRa-aRa-aRa-R-` |
| `a` | 10 surfaces × [K, A4…A20] |
| `target_efl` / `solve_type` / `solve_idx` | `2.399` / `focal_length` / `-3` |
| `total_track_length_solve` | `4.802` |
| `scale_factor` | `0.59975` (= EPD/2) |
| `aperture_type` / `aperture` | `epd` / `1.1995` (= f/2.0) |
| `hfov` | `47.5` |

The patent gives **10 coefficients per surface (K + A4…A20)**, more than the
7 the existing `telephoto`/`wide_angle` configs carry; `LensParameterization`
infers the count from the array, so no code change is needed.

Sensor block, from the article's stated specs (2.5 µm pixel, ~200 cyc/mm
Nyquist, 2–2.7 mm semi-diagonal): `shape: [1080, 1920]` gives a 2.377 µm pixel
and 210 cyc/mm Nyquist against a `sensor_diagonal` of 5.236 mm (auto-linked from
f and hfov). `psf_abs_size: 16.638e-03` = 7 px. `diffraction_f_number: 2.0`
gives a 1.34 µm Airy radius (the article quotes 1.4 µm).

## Verification

Independent of any solve, the patent's Table 3 **dimensionless invariants**
audit every transcribed radius — all agree to better than 0.5%:

| Quantity | Patent | Computed |
|---|---|---|
| \|R1+R2\|/\|R1−R2\| | 0.978 | 0.9778 |
| \|R5+R6\|/\|R5−R6\| | 1.740 | 1.7403 |
| (R7+R8)/(R7−R8) | 2.433 | 2.4324 |
| (R9+R10)/(R9−R10) | 7.383 | 7.4000 |
| f_sys/f_L1 … f_L5 | 0.646, 0.203, 1.080, −0.440, 0.225 | 0.6446, 0.2027, 1.0765, −0.4407, 0.2240 |

The dispersion model also reproduces the patent's `nd` exactly at 587.56 nm.

Through the repo's own CLI (`test` on both configs): **EFL 2.399 mm, TTL 4.802 mm,
100% valid rays**.

## float64 precision

`trainer.precision: 64` casts *module parameters*, but tensors built inside pupil
sampling and ray initialization used to follow torch's **global default dtype**
(float32). At this scale — sub-mm airspaces, radii down to 0.592 mm — float32
caused the aspheric marching solve to fail and silently discard every field
beyond ~30° (40× error in `loss/distortion`, 33% `ray_miss`). This is now fixed
at the source: tensor construction in `LensParameterization`/`GlassModel` is
explicitly `dtype=torch.float64`, and the remaining dtype mismatches in
`OpticsSimulator`/`PSFSampler` are resolved with `.to(...)` casts (see commits
`4594571` and `55181b1`). No global dtype override or wrapper script is needed —
run the CLI directly.

## Run

    export PYTHONPATH=$PWD KMP_AFFINITY=disabled OMP_NUM_THREADS=4
    python -m eisoptx.main test -c configs/cellphone/defaults.yml \
                            -c configs/cellphone/designs/patent_710.yml

`patent_710.yml` uses the patent's idealized indices;
`patent_710_real_materials.yml` uses the article's substitution (N-BK7 cover and
IR filter, APL5014C for L1/L3/L5, EP10000 for L2/L4).

## Caveats

- The article reports the patent prescription **misses the MTF target as-published**;
  these configs reproduce the *starting point*, so they are a design to optimize
  from, not a converged solution. `loss/distortion` ≈ 0.0106 vs ~1e-5 for the
  repo's converged designs is expected.
- Aspheric **clear semi-diameters are not in the patent tables**, so no aperture
  limits are set; my sag comparison used estimated semi-apertures.
- The reoptimized Q-type variant from the article was not transferred (its `.zar`
  was not decoded, and `eisoptx` uses the even-asphere form).
