# Chapter 1 — The configuration is the program

> **What this chapter answers.** Where does execution start? How does a YAML file
> become live Python objects? Why is there almost no code in `main.py`? Which knobs
> actually matter, and which ones lie?

---

## 1.1 The entry point is 76 lines, and that is the point

`eisoptx/main.py` is the whole command-line interface. Here is the operative part:

```python
# eisoptx/main.py:73-76
if __name__ == "__main__":
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    torch.set_float32_matmul_precision("high")
    lightning_cli = CustomCLI(ImagingSystemModule, auto_configure_optimizers=False)
```

That is it. There is no argument parsing, no experiment dispatch, no `if mode ==
"train"`. The reason is **jsonargparse**, the configuration layer under PyTorch
Lightning's CLI. It reads the *type annotations* on `ImagingSystemModule.__init__`
and builds a command-line interface from them automatically. A YAML key like:

```yaml
model:
  lens_parameterization:
    class_path: eisoptx.optimization.parameterization.LensParameterization
    init_args:
      lens_sequence: s-aRa-aRa-
```

means: *import that class, call it with those keyword arguments, and pass the result
as the `lens_parameterization` argument of `ImagingSystemModule`.* The YAML is not
data being read by a program — **the YAML is the program**, and Python is its
runtime. This is worth internalizing because it explains a property of the repository
that is otherwise baffling: there are no experiment scripts. Every experiment in the
paper is a YAML file.

`auto_configure_optimizers=False` matters: it tells Lightning not to build optimizers
from the config, because `ImagingSystemModule.configure_optimizers`
(`imaging_system.py:908`) does something Lightning's default cannot — it builds two
independent optimizers (lens and restoration model) and can return `None` for either,
freezing those parameters (`imaging_system.py:913-916`).

## 1.2 The three argument links — derived quantities

`CustomCLI.add_arguments_to_parser` (`main.py:24-43`) is the only real logic in the
file. It declares three **links**: values that are computed from other config values
rather than specified.

**Link 1 — experiment name from the lens sequence** (`main.py:26-29`):

```python
parser.link_arguments(
    "model.lens_parameterization.init_args.lens_sequence",
    "trainer.logger.init_args.name",
)
```

The TensorBoard run is named after the optical prescription. Your logs directory is
literally organized by lens topology.

**Link 2 — sensor size from focal length and field of view** (`main.py:31-38`):

```python
parser.link_arguments(
    ("model.lens_parameterization.init_args.target_efl",
     "model.ray_initialization.init_args.hfov"),
    "model.optics_simulator.init_args.sensor_diagonal",
    lambda efl, hfov: float(2 * efl * np.tan(np.deg2rad(hfov))),
)
```

This is the optical relation `diagonal = 2·f·tan(HFOV)`. It is **not** a free
parameter. For the toy config `efl = 10.0`, `hfov = 30°`:

```
2 × 10.0 × tan(30°) = 2 × 10.0 × 0.57735 = 11.547
```

Measured: `sensor_diagonal_mm = **11.5470053838**` (`7_psf_and_image`). Matches.

Why this matters for your defense: a supervisor may ask "how do you ensure the sensor
matches the lens?" The answer is that you cannot get it wrong — it is derived, and any
attempt to set it in YAML is overridden by the link.

**Link 3 — wavelengths propagate to the simulator** (`main.py:40-43`). The ray
initialization owns the wavelength list; the PSF simulator receives a copy. One source
of truth.

> **↔ your repo.** Your `e2e-gtra-optics` computes the sensor size inside the sensor
> class rather than at config-link time. Same physics, different binding moment — see
> Chapter 16.

## 1.3 Config composition: later files win

```bash
python -m eisoptx.main fit -c configs/toy/defaults.yml -c configs/toy/defaults_e2e.yml
```

Files are merged left to right. This is how the ablations in the paper are expressed:
one base config, then a small delta. `configs/toy/opt_scalar_mse.yml` is *two lines*:

```yaml
model:
  e2e_vector_mode: False
```

That single flag is the paper's central ablation — generalized aberrations
(`True`) versus the scalar-loss baseline (`False`). The entire experimental
comparison is one boolean.

There is one special merge operator. In `configs/toy/defaults_e2e.yml:5`:

```yaml
  residuals+:
    - class_path: eisoptx.optimization.residuals.TransverseRayAberrationResiduals
```

The `+` suffix means **append to the list** rather than replace it. Without it, the
e2e config would wipe out the four residual terms defined in the base config.

## 1.4 Reading the toy config as an optical prescription

`configs/toy/defaults.yml` in full, grouped by what it means physically.

### The lens itself (`lines 2-37`)

```yaml
lens_sequence: s-aRa-aRa-
c:  [0., 0., 0., 0.]              # curvatures — flat plates to start
s:  [.5, 1., .5, 2., 9.]          # spacings, mm
nd: [1.5, 1.5]                    # refractive indices — two elements
vd: [1e6, 1e6]                    # Abbe numbers — 1e6 means "no dispersion"
a:  [[-1., 0., 0., 0.], ...]      # aspheric coefficients, 4 surfaces
```

Read `s-aRa-aRa-` left to right (parsing in Chapter 2): `s` = aperture stop, `-` =
propagation, `a` = aspheric, `R` = refractive interface. So: stop, gap, **element 1**
(aspheric front, glass, aspheric back), gap, **element 2**, gap to sensor. Two
elements, four aspheric surfaces, stop in front.

`vd: [1e6, 1e6]` is a tell. Abbe number is `(nd−1)/(nF−nC)`; setting it to a million
makes the denominator vanish — **zero dispersion**. Combined with
`wavelengths: [550.]` (line 50), this is a deliberately monochromatic toy: no chromatic
aberration to worry about, so the numbers stay interpretable.

### What is allowed to change (`lines 19-30`)

```yaml
freeze:
  s: False                              # all spacings free
  c:
    default: False
    toggle_row_col_list: [-1]           # ...except the last curvature
  g: True                               # glass frozen entirely
  a:
    default: False
    toggle_row_col_list: [[null, 0]]    # ...except column 0 of every asphere
  d: True
  m: True
```

`freeze` is per-array, with per-element exceptions. `toggle_row_col_list` **flips** the
default for the listed entries. So `c` is free *except* index `-1`, and `a` is free
*except* column 0 of all rows (`null` = every row).

Why those two exceptions?

- **Last curvature frozen** because `solve_type: focal_length` with `solve_idx: -1`
  (lines 32-33) *computes* it to force EFL = 10. It is determined, not free.
- **Aspheric column 0 frozen** because that column is the conic constant, initialized
  to `-1.` — the code holds it fixed and optimizes only the higher-order terms.

Measured consequence: **31 total variables, 20 trainable**
(`1_config_and_lens.variable_layout`). The arithmetic: 5 spacings + 4 curvatures + 16
aspheric + 6 glass = 31. Frozen: 6 glass + 1 curvature + 4 aspheric conics = 11.
`31 − 11 = 20`. ✓

### How rays are launched (`lines 38-50`)

```yaml
aperture_type: epd
aperture: 5.                     # 5 mm entrance pupil → f/2 at EFL 10
hfov: 30.
n_fields: 11
pupil_sampling_mode: skew_uniform_jittered
pupil_sampling_kwargs: {n_r: 16, n_i: 1}
ray_aiming_steps: 0
wavelengths: [550.]
```

`n_r: 16` gives **256 rays per field** (Chapter 4 explains the 16 → 256 expansion),
and 11 fields gives **2816 rays** total. Measured and confirmed:
`3_ray_trace.ray_valid_final` shows 2816 entries, `true_fraction = 1.0` — every ray
survives.

`ray_aiming_steps: 0` disables the Newton iteration that would correct for pupil
aberration. Fine for a toy; the paper-scale configs enable it.

### What is being optimized (`lines 51-74`)

```yaml
residuals:
  - TransverseRayAberrationResiduals:  weight: 1.
  - RayPathResiduals:                  weight: 20.   (min .5, max 6. refractive)
  - RayAngleResiduals:                 weight: 10.   (max_angle 60°)
  - SurfaceNormalResiduals:            weight: 10.   (max_angle 30°)
```

One term measures **image quality**; three enforce **manufacturability** — glass not
too thin or thick, rays not hitting surfaces at absurd angles, surfaces not too steeply
curved. Weights say the constraints matter 10–20× more per unit than spot size, which
is the standard lens-design posture: constraints are near-hard, image quality is the
thing you actually improve.

Measured residual vector composition (`4_residuals.residual_vector_table`):

| term | entries | note |
|---|---:|---|
| `transverse_ray_aberration` | 5632 | 2 axes × 2816 rays |
| `ray_path` | 14 | violations only |
| `surface_normal` | 58 | violations only |
| `ray_angle` | 0 | **no violations at the optimum** |
| **total** | **5704** | |

The constraint terms contribute **only where violated**. `ray_angle` contributes zero
entries because at the shipped optimum, no ray exceeds 60°. This is an important
behaviour to be able to explain: the residual vector **changes length** between
iterations as constraints activate and deactivate.

### The optimizer (`lines 90-96`)

```yaml
lens_optimizer:
  class_path: eisoptx.optimization.optimizers.LMOptimizer
  init_args:
    lm_parameter: 1.        # initial damping λ
    damped_term_min: 1e-8   # floor on the diagonal scaling
    tolerance: 1.           # accept a step if loss_ratio ≤ 1
    lam_eps: 1e-12
```

`tolerance: 1.` means *accept any step that does not make things worse*. Chapter 9
covers the accept/reject logic; measured behaviour is 11 accepted out of 12 from a
perturbed start.

### Numerical precision (`line 107`)

```yaml
precision: 64
```

**Double precision.** This is unusual in deep learning and essential here. Ray tracing
involves differences of large nearly-equal numbers (surface sag vs. ray height), and
the LM solve is on a matrix that becomes rank-deficient near the optimum. In float32
the Jacobian rank estimate is unreliable. Note this if asked "why is it slow?" — it is
64-bit on CPU by design.

## 1.5 ⚠ Defect: the README documents key names that do not exist

The README's Image-Driven Optimization section names:

- `model.end_to_end_vector_mode`
- `model.end_to_end_loss_weight`

The actual constructor arguments (`imaging_system.py:35-36`) are:

- `e2e_vector_mode`
- `e2e_loss_weight`

This is not a cosmetic mismatch — the CLI **rejects** the README's names. Verified:

```
$ python -m eisoptx.main test -c configs/toy/defaults.yml \
      --model.end_to_end_vector_mode=True
error: unrecognized arguments: --model.end_to_end_vector_mode=True
```

If you follow the README you get an error. The shipped configs use the correct short
names, so the configs work and only the prose is wrong.

## 1.6 ⚠ Defect: the toy e2e config points at a missing folder

`configs/toy/defaults_e2e.yml:14-15`:

```yaml
    train_folder: configs/ablation/sample_image
    val_folder: configs/ablation/sample_image
```

`configs/ablation/` does not exist anywhere in the repository. The behaviour splits by
command:

| command | outcome | why |
|---|---|---|
| `test` | **passes silently** | `test_dataloader` returns a dummy batch (`datasets.py:142`) and never touches the folder |
| `fit` | **crashes** — `num_samples=0` | the train dataloader really does try to list the directory |

The folder that does exist is `configs/toy/sample_image`. With
`--data.train_folder=configs/toy/sample_image --data.val_folder=configs/toy/sample_image`
the toy end-to-end run trains cleanly. Every e2e number in this walkthrough uses that
override, recorded in `trace_dump.json` under `0_meta.e2e_path_override`.

The silent pass under `test` is the dangerous part: a reviewer running the documented
test command sees success and concludes the data path is fine.

## 1.7 ⚠ Defect: `starting_point.yml` is empty

```
$ ls -la configs/toy/designs/
-rw-r--r--  1090  optimized_e2e.yml
-rw-r--r--  1091  optimized_spot.yml
-rw-r--r--     0  starting_point.yml     ← 0 bytes
```

The design the README implies you start from is an empty file. Starting values are
inline in `configs/toy/defaults.yml` (flat curvatures, zero aspheres). Practical
consequence for Chapter 9: to demonstrate the optimizer taking *accepted* steps, I
perturb the shipped optimum rather than load a starting design, because there is no
starting design to load.

## 1.8 Side effects of running the CLI

Two things get written into the repository on any `fit`:

- `logs/logs_ablation/` — TensorBoard and CSV logs (config lines 111, 114)
- `lightning_logs/` — Lightning's own default directory

`.gitignore` covers `logs/` but **not** `lightning_logs/`. Running the documented
commands leaves untracked files in a clean checkout.

## 1.9 What to take away

1. The YAML is the program; `main.py` only wires up three derived values.
2. `sensor_diagonal = 2·efl·tan(hfov)` is derived and cannot be set by hand.
3. `residuals+` appends; plain `residuals` replaces.
4. The paper's headline ablation is one boolean, `e2e_vector_mode`.
5. `freeze` + solves is why 31 variables become 20 trainable.
6. Three documentation/config defects: wrong README key names, missing sample-image
   folder, empty starting-point file.

---

*Previous: [Chapter 0 — Orientation](00_orientation.md)*
*Next: [Chapter 2 — The Lens object and sequence parsing](02_lens_object.md)*
