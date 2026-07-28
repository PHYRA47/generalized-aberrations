# Chapter 12 — The training loop: how the two halves are interleaved

> **What this chapter answers.** What happens on one training step, in order? How do two
> optimizers with completely different characters share a loop? Why is
> `automatic_optimization = False`? What is the closure really for, and what is the
> `xy_updated` / `psfs_updated` cache protecting? Where does the CLI wire the config into
> the module?

---

## 12.1 The frame: PyTorch Lightning, manual mode

`ImagingSystemModule` (`imaging_system.py:20`) is a `pl.LightningModule`, so the outer loop
— epochs, batching, device placement, logging, checkpointing — is Lightning's. What is
*not* Lightning's is the optimization itself:

```python
self.automatic_optimization = False        # imaging_system.py:65
```

This single line hands back control of `zero_grad` / `backward` / `step`. It is necessary,
not stylistic. Lightning's automatic mode assumes one loss, one backward pass, one
optimizer step. This system has:

- two optimizers with **different algorithms** (LM and Adam),
- one of which needs a **closure it can call repeatedly** (Chapter 9's trial-and-revert),
- and a **Jacobian**, not a gradient, obtained by forward-mode differentiation.

None of that fits the automatic path. So the module does it by hand, and
`configure_optimizers` returns a list that `get_optimizers` unpacks itself.

## 12.2 One training step, in order

`training_step` (`imaging_system.py:555-593`) is short enough to read whole:

```python
lens_optimizer, irm_optimizer = self.get_optimizers()
if lens_optimizer is None and irm_optimizer is None:
    raise RuntimeError("At least one optimizer must be provided when fitting.")

logs = {}
if lens_optimizer is not None:
    logs = {**logs, **self.lens_optimization_step(batch, lens_optimizer)}
if irm_optimizer is not None:
    logs = {**logs, **self.irm_optimization_step(batch, irm_optimizer)}

schedulers = self.lr_schedulers()
if schedulers is not None:
    self.lr_scheduler_step(schedulers, None)

for k, v in logs.items():
    self.log(k, v)
```

**Alternating optimization**, and the docstring says so: *"The training step is divided into
two parts: lens optimization and image restoration model optimization... The same input
batch is used for both steps."*

Three things to notice.

**Both halves are optional, and the same code covers all three regimes.** Set
`lens_optimizer: null` and you are training a restoration network against a fixed lens —
the paper's baseline. Set `irm_optimizer: null` and you are doing classical lens design;
that is the toy config. Set both and you are doing co-design. **The ablation study is
configuration, not code.**

**The same batch feeds both.** The lens step and the network step see identical images. The
alternative — a fresh batch each — would add gradient noise between the two updates for no
benefit.

**Order matters, slightly.** The lens moves first, then the network is updated against the
*already-moved* lens. So the network always chases a lens it has just seen change. The
reverse order would have the lens optimize against a stale network. Neither is obviously
right; this is the choice the code makes, and it is worth being able to state.

This is coordinate descent on a joint objective: fix the network, improve the lens; fix the
lens, improve the network; repeat. There is no convergence guarantee for the joint problem
— which is worth saying out loud, because a supervisor may well ask.

## 12.3 The lens half

`lens_optimization_step` (`imaging_system.py:595-632`):

```python
lens = self.lens                                    # snapshot BEFORE the update
with lens_optimizer.toggle_model():
    closure = CustomClosure(self, lens_optimizer, batch)
    lens_optimizer.zero_grad()
    scalar_loss = lens_optimizer.step(closure=closure)
```

`toggle_model()` is a Lightning context manager that sets `requires_grad = False` on every
parameter *not* belonging to this optimizer. During the lens step, the 1.5 million NAFNet
parameters are frozen. **This is what makes `CustomClosure`'s assertion hold** — Chapter 9
noted that `LMOptimizer` requires exactly one trainable parameter tensor, and this is where
that condition is manufactured. Without `toggle_model`, the assertion at
`imaging_system.py:1091` would fire the moment a restoration model was configured.

Note `lens = self.lens` on line 607, with its comment *"For logging purposes, initialize
lens before it is updated."* The logged first-order data (EFL, image height, ray status)
describes the lens **as it entered the step**, not as it leaves. That matters when you read
TensorBoard curves: the metric at step *n* is the state the loss at step *n* was computed
from, which is the consistent choice, but it means the final logged lens is one step stale.

And line 630:

```python
self.xy_updated = self.psfs_updated = False
```

Cache invalidation. The lens has moved; every cached ray trace and PSF is now wrong. §12.5
covers what that cache is for.

## 12.4 The closure: what LM actually calls

`CustomClosure` (`imaging_system.py:1032-1104`) has three entry points, and knowing which
one gets called when clarifies the whole design.

| method | caller | returns |
|---|---|---|
| `__call__()` | a generic PyTorch optimizer | scalar loss, and populates `.grad` |
| `evaluate_least_squares_loss()` | LM's trial evaluation (Ch. 9 §9.8) | scalar loss only |
| `get_least_squares_quantities()` | LM's step computation | residual vector, loss, **Jacobian**, constraint mask |

`__call__` exists so the closure remains a valid PyTorch closure — swap in LBFGS or SGD and
it still works. LM never uses it. LM uses the other two: the expensive one once per
iteration to build the step, the cheap one once per trial to judge it.

Underneath sits `LossWrapper` (`imaging_system.py:982-1029`), and its existence is a pure
consequence of `torch.func`. `jacfwd` needs a **pure function** `parameters → residual
vector`, but a `LightningModule` is a stateful object whose forward pass returns a scalar.
`LossWrapper` adapts it: a `torch.nn.Module` whose `forward(batch)` returns the residual
vector, so `functional_call` can swap in candidate parameter values.

That is also why it stashes results on `self`:

```python
self.residual_vector = ...
self.constraint_mask = ...
self.logs = logs
```

`jacfwd` returns only the Jacobian, discarding the function's actual output. Rather than
recompute the residual vector, the wrapper leaves it on the instance for
`get_least_squares_quantities` to pick up. Slightly impure, entirely pragmatic — and
correct only because `jacfwd`'s last evaluation is at the true parameter values.

Line 1013-1016 is the detail Chapter 7 flagged and it is worth repeating here:

```python
# Apply sqrt on the weights since the loss is the sum of squares
weighted_residuals_dict = {k: np.sqrt(weight_dict[k]) * residuals_dict[k] for k in keys}
```

**`√weight` on the residual, because the loss squares it.** A `weight: 10.` in a config
contributes a factor of 10 to the loss, not 100. Misreading this is an easy way to
misreport what a configuration does.

## 12.5 The restoration half

`irm_optimization_step` (`imaging_system.py:634-657`):

```python
with irm_optimizer.toggle_model():
    with torch.inference_mode(True):
        rgb_psfs = self.try_build_simulation_model()
    e2e_loss, _ = self.scalar_e2e_loss(None, batch, self.irm_e2e_loss_fn, rgb_psfs=rgb_psfs)
    irm_optimizer.zero_grad()
    self.manual_backward(e2e_loss)
    irm_optimizer.step()
```

Ordinary deep learning — with one line that is not.

**`torch.inference_mode(True)` around the PSF construction.** During the network's step the
lens is frozen, so the PSFs are constants. Building them inside `inference_mode` means no
graph is recorded for the entire ray-trace-and-splat pipeline. Given Chapter 10's memory
table, that is the difference between holding the whole optics graph in memory and holding
none of it.

But it also has a semantic consequence worth stating precisely: **during the restoration
step, no gradient flows to the lens.** Chapter 11 established that `detach_psfs: False` is
what makes the system end-to-end — and it is, but only during the *lens* step. In the
network step the coupling is deliberately cut. The two directions of influence happen in
different halves of the loop:

- lens step: image loss → PSFs → lens variables (via GTRA, Chapter 8)
- network step: image loss → network weights only

Note also `self.irm_e2e_loss_fn` here versus `self.e2e_loss_fn` in the lens path — Chapter
11 §11.10's MAE-versus-MSE split is enforced right here, by which attribute each half
reads.

## 12.6 The cache, and why it exists

Three flags and three slots (`imaging_system.py:156-157`):

```python
self.last_xy = self.last_psfs = self.last_rgb_psfs = None
self.xy_updated = self.psfs_updated = False
```

`try_build_simulation_model` (`imaging_system.py:872-906`) reads them as a four-branch
ladder:

```python
if xy is not None:                      # 1. differentiable path — always rebuild
    psfs, rgb_psfs = self.optics_simulator.build_optics_model_from_xy(xy)
elif self.psfs_updated:                 # 2. PSFs still valid — reuse
    psfs, rgb_psfs = self.last_psfs, self.last_rgb_psfs
elif self.xy_updated:                   # 3. rays still valid — re-splat only
    psfs, rgb_psfs = self.optics_simulator.build_optics_model_from_xy(self.last_xy)
elif hasattr(self.optics_simulator, "psfs"):   # 4. FixedPSFsOpticsSimulator
    psfs, rgb_psfs = self.optics_simulator.build_optics_model()
else:                                   # 5. cold — full ray trace
    psfs, rgb_psfs = self.optics_simulator.build_optics_model(lens, self.ray_initialization)
```

The cost ordering is the point: branch 2 is free, branch 3 skips the ray trace, branch 5
pays for everything. Within one training step the lens is traced **once**, and the
restoration step, the validation metrics, and the visualization callbacks all reuse that
result.

The `xy is not None` branch is deliberately *not* cached, and that is the important part.
Passing `xy` explicitly is the **differentiable** path — it means "I need this connected to
the graph." A cached tensor cannot serve that purpose, because:

```python
self.last_psfs = psfs.detach()
self.last_rgb_psfs = rgb_psfs.detach()
```

**Everything stored is detached.** The cache is for logging, validation, and visualization
— never for gradients. If it were not detached, holding `last_psfs` across a step would
keep the entire previous graph alive and leak memory unboundedly.

## 12.7 Validation

`validation_step` (`imaging_system.py:659-734`) does what training does not: it measures
**image quality**, not loss.

```python
metrics = {
    "psnr": torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0, ...),
    "ssim": torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0, ...),
}
```

PSNR and SSIM, logged for both the aberrated and the restored image — so you can read off
what the restoration network contributed, separately from what the lens contributed. For a
co-design paper that separation is the headline table.

`if batch_idx == 0: self.test_step(None, 0)` runs the optics-only diagnostics (spot
diagrams, first-order data, ray status) once per validation pass rather than per batch,
since they do not depend on the images at all.

Toy config: `val_check_interval: 100`, `check_val_every_n_epoch: null` — validate every 100
*steps*, ignoring epochs entirely. Sensible when the "dataset" is
`NoneDataModule(n_samples=100)`, a placeholder that yields nothing so that the lens can be
optimized with no images at all.

## 12.8 From YAML to module: the CLI

`eisoptx/main.py` is 76 lines and does one interesting thing — `add_arguments_to_parser`
(`main.py:24-43`) declares three **argument links**, which are values *derived* from other
config values rather than typed in:

```python
parser.link_arguments(
    ("model.lens_parameterization.init_args.target_efl",
     "model.ray_initialization.init_args.hfov"),
    "model.optics_simulator.init_args.sensor_diagonal",
    lambda efl, hfov: float(2 * efl * np.tan(np.deg2rad(hfov))),
)
```

This is why `sensor_diagonal` appears nowhere in the toy config and yet the trace reports
**11.5470053838** — Chapter 10 needed it and derived it by hand as `2 · 10 · tan(30°)`.
The CLI computes it at parse time. The other two links name the TensorBoard run after the
lens sequence, and forward the wavelengths from ray initialization to the simulator so they
cannot disagree.

**This is a real defense point.** If someone asks "where is the sensor size configured?",
the answer is: it is not, it is *linked* — sensor size is a consequence of focal length and
field of view, and the code enforces that rather than trusting you to keep three numbers
consistent.

Two more lines worth knowing. `torch.set_float32_matmul_precision("high")` at `main.py:75`
— though the toy config sets `precision: 64`, so the whole system runs in **float64**.
That is unusual for deep learning and entirely correct for optics: ray tracing accumulates
over surfaces, and Chapter 6's solves need the precision. It is also why Chapter 11's Adam
config carries the `foreach: False` workaround. And `auto_configure_optimizers=False` at
`main.py:76` tells Lightning not to build optimizers itself, since `configure_optimizers`
returns a hand-built list.

## 12.9 What one full iteration costs

Assembling the cost model from Chapters 9 and 10, for a joint step with the telephoto
config:

| stage | ray traces | image pipeline passes |
|---|---|---|
| build residual vector + Jacobian (`jacfwd`, 20 tangents) | 1 primal + 20 tangent | 1 (GTRA linearization) |
| LM trial evaluation | 1 | 1 |
| restoration step | 0 (cached, `inference_mode`) | 1 forward + 1 backward |

The lens half dominates, and the single most important architectural decision in the paper
is visible right here: **the image pipeline runs once per LM iteration, not once per
Jacobian column.** Without Chapter 8's aberration lift it would run 21 times, and Chapter
10 §12.10's memory table says that is not affordable at `6144 × 8192`.

## 12.10 Callbacks: what runs beside the loop

The toy config attaches five (`configs/toy/defaults.yml:112-125`):

| callback | purpose |
|---|---|
| `CodeVSeqFileCallback` | writes a CODE-V `.seq` file — the design exports to commercial software |
| `ConfigFileCallback` | dumps the resolved config every 10 steps |
| `CustomProgressBar` | disables tqdm when not on a terminal |
| `ModelCheckpoint` | checkpoints every 100 steps |
| `VisualizationCallback` | lens layout, spot diagrams, PSFs → TensorBoard every 10 steps |

`CodeVSeqFileCallback` is the one to notice. It means every optimization run produces a
design importable into CODE-V, which is how these results get validated against a
commercial ray tracer. If you ever need to prove the code traces rays correctly, that
export is the mechanism.

## 12.11 What to take away

1. `automatic_optimization = False` because there are two optimizers, one needs a
   re-callable closure, and the lens step needs a **Jacobian**, not a gradient.
2. One step = lens update, then restoration update, **same batch** — alternating
   (coordinate-descent) optimization, with no joint convergence guarantee.
3. Either half can be switched off in YAML. Classical design, network-only training, and
   co-design are **the same code path with different configs**.
4. `toggle_model()` freezes the other half's parameters, which is what makes LM's
   one-parameter-tensor assertion hold.
5. `LossWrapper` exists to give `torch.func.jacfwd` a pure function; it stashes the
   residual vector on `self` because `jacfwd` discards the primal output.
6. Weights enter as **`√weight`** because the loss squares the residual.
7. The restoration step builds PSFs under `inference_mode` — **no lens gradient during the
   network half**. The coupling runs in the lens half only.
8. The `xy_updated` / `psfs_updated` cache traces the lens **once per step**; everything
   stored is `.detach()`ed and is for logging only.
9. Validation logs PSNR/SSIM for aberrated *and* restored images — the separation a
   co-design result needs.
10. The CLI **links** `sensor_diagonal = 2·EFL·tan(HFOV)` rather than accepting it as
    input: **11.5470053838** for the toy config, never written in any YAML file.
11. `precision: 64` — the whole system runs in double, for ray-tracing accuracy.
12. A CODE-V sequence file is written every run, which is the bridge to commercial
    validation.

---

*Previous: [Chapter 11 — Image restoration](11_image_restoration.md)*
*Next: [Chapter 13 — The periphery](13_periphery.md)*
