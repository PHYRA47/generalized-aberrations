# Cellphone camera lens — layout and compose order

US 2019/0129149 A1, system 710. Five plastic aspheres + cover glass + IR filter.

## Layout

| path | role |
|---|---|
| `defaults.yml` | System defaults: sensor, field points, wavelengths, LM optimizer, trainer, and all residuals — including the two weightless monitors. Every run starts here. |
| `defaults_e2e.yml` | Shared by every rung that forms images: DIV2K loader, restoration chain, e2e loss, 25 000-step budget. |
| `vis.yml` | Figure overlay — layout, spot diagrams, PSF grid, ray fans. Optional; append last. |
| `designs/param_*.yml` | The lens **as given**: a prescription, never optimized. `param_zemax710.yml` is the agreed baseline. |
| `designs/opt_*.yml` | Optimization **experiments**: what to vary (drop an element, minimize TTL, materials). |
| `ladder/rung1_raw.yml` | The one genuinely rung-specific delta: nulls the restoration net so rung 1 measures the lens alone. |

The split is by scope, not by topic: **top level is shared by every run, `ladder/` holds
per-rung deltas.** `defaults_e2e.yml` sits at top level because all four rungs use it, and
that also keeps the name it has in `telephoto/`, `c_mount/` and `microscope/`. `designs/`
keeps the flat shape those folders use; the `param_` / `opt_` prefixes already separate
prescriptions from experiments.

The aperture and illumination monitors used to be a separate `residuals_monitor.yml`
overlay, kept out of `defaults.yml` so the baseline config stayed bit-identical to what
version_8 ran. That reason expired once version_9 demonstrated they are inert — both carry
`weight: null`, so they are logged and never enter the loss, and every other `loss/*` value
came back bit-identical with them added. They now live in `defaults.yml`, which means you
cannot forget to append them and every run in the ladder is monitored by default.

## Compose order

Configs compose left to right and **later files override earlier ones**. This is not
cosmetic — see the traps below. Every command starts with:

    export PYTHONPATH=$PWD
    BASE="-c configs/cellphone/defaults.yml -c configs/cellphone/designs/param_zemax710.yml"

| rung | question it answers | subcommand | append after `$BASE` |
|---|---|---|---|
| baseline | does our forward model reproduce Zemax? | `test` | *nothing* |
| 1 | what does the lens alone give? | `validate` | `-c configs/cellphone/defaults_e2e.yml -c configs/cellphone/ladder/rung1_raw.yml` |
| 2 | ...plus a trained restoration net? | `fit` | `-c configs/cellphone/defaults_e2e.yml --model.lens_optimizer=null` |
| 3 | ...training lens and net together? | `fit` | `-c configs/cellphone/defaults_e2e.yml` |
| 4 | fewer elements / shorter track? | `fit` | rung 3 plus a `designs/opt_*.yml` — deferred |

Rung 2 needs no config file of its own: `--model.lens_optimizer=null` is the documented
way to disable the lens half (`imaging_system.py:563`), and it is cleaner than freezing all
six variable keys because the LM solver has no guard against having zero free parameters.
The composed config is written into each run's log by `ConfigFileCallback`, so a
command-line flag is recorded just as durably as a file would be.

Rung 1 must use `validate`, not `fit`: with both optimizers absent,
`imaging_system.py:572` raises "At least one optimizer must be provided when fitting",
and rung 1 trains nothing by definition. `validate` still yields the optical metrics,
because `validation_step` calls `test_step(None, 0)` on the first batch.

## Four traps this layout exists to prevent

**1. The `trainer` block lives in `defaults_e2e.yml`.** It sets `max_steps: 25000`;
`defaults.yml` sets 1000. Omit the overlay, or compose it before `defaults.yml`, and you
silently get a 1000-step run that looks successful.

**2. `T_max` must equal `max_steps`.** `irm_lr_scheduler.T_max` is the cosine schedule's
horizon. They agree at 25000 in the committed files. When overriding the step count,
override **both** or the learning rate barely decays:

    --trainer.max_steps=5000 --model.irm_lr_scheduler.init_args.T_max=5000

**3. `residuals+` overrides by name, and `--print_config` hides it.** `defaults_e2e.yml`
appends a second `TransverseRayAberrationResiduals` with `weight: null`, so the composed
config lists TRA twice — once at weight 1.0, once weightless. That looks like a bug and
isn't: `imaging_system.py:85` does `{residual.name: residual for residual in
residuals}.values()`, deduplicating by name and keeping the **last**. So the weightless one
wins and TRA is monitor-only in every e2e rung — which is the intended semantics, since
rung 3 is supposed to optimize the image, not the classical aberration term. `--print_config`
prints the raw pre-dedup list, so trust `residual.name` over the printed length. The
practical rule: **appending a residual whose name already exists replaces it.**

**4. `s[2] = 0.0` is the stop on the L1 vertex, not a too-thin airspace.** Every design in
`designs/` puts the aperture stop on L1's front vertex via a collapsed dummy pair, straight
out of the Zemax export. `RayPathResiduals.min_cutoff` is 0.05 mm, so before the exemption
this was a permanent weight-20 violation — `loss/ray_path` sat at 0.0222 in rungs 1 and 2
and never moved, because those rungs froze the optics. Rung 3 frees the spacings, and the
optimizer's two ways to relieve it are opening the gap or bending L1's front surface: a
design change driven by a modeling artifact, not by image quality. **Do not fix this by
freezing `s[2]`** — the cutoff hinges on per-ray `delta_z` through the airspace, which
depends on both `s[2]` *and* L1's front sag (`residuals.py:191-270`), so freezing the
spacing removes the cheap remedy and leaves only the surface-bending one. The fix is
`other_min_cutoffs: [[2, -.inf]]` in `defaults.yml`, following `demo_tele4p.yml`'s
`[[0, -.inf]]` (stop at a negative spacing) and `telephoto/defaults.yml`'s `[[-2, -.inf]]`
(fixed IR-filter thickness). Index 2 is into the propagation-event list, which runs in the
same order as the `s` array. Consequence: re-running rung 1 or 2 now logs
`loss/ray_path` near 0 instead of 0.0222 — a monitor change only, since those rungs hold
the optics fixed and the term never entered a gradient.

Verify any composition without creating a log directory by appending `--print_config`.

## Measured cost

RTX 3090, batch 32, crop 120, `precision: 64`, DIV2K.

| configuration | s/step | 25 000 steps |
|---|---|---|
| optics only (`NoneDataModule`, no restoration) | 1.00 | 6.9 h |
| restoration only (`--model.lens_optimizer=null`) | 2.67 | 18.5 h |
| full end-to-end | 5.72 | 39.7 h |

The end-to-end step is 17.5 % LM solve, 46.7 % image path, 35.8 % the cost of
`e2e_vector_mode: True` forcing the LM Jacobian to backprop through the restoration
chain. Validation costs 7.5 s per pass over all 100 DIV2K validation images —
negligible at `val_check_interval: 100`.

**Agreed budget:** rungs 1–3 at 25 000 steps, about 58 h total. Rung 4 is deferred until
rung 3's result is known, since a sweep over lens variants is only worth 40 h per variant
if training the lens beats leaving it fixed.
