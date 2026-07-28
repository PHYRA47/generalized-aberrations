# Chapter 11 — Image restoration: the algorithm half of "end-to-end"

> **What this chapter answers.** What exactly is being co-designed with the lens? Why a
> Wiener filter *sandwiched between* two neural networks and not just one network? Why is
> the SNR stored as its logarithm? Why does `detach_psfs` exist, and what breaks if you
> set it wrong? And why does the measured restoration make the image *worse*?

---

## 11.1 The half of the system the lens is being optimized against

Everything up to Chapter 10 was optics. This chapter is the other half of "end-to-end
optimization of optics *and algorithm*" — and it is the reason the whole GTRA machinery
of Chapter 8 exists.

The claim the paper makes is that a lens optimized *jointly with* its restoration
algorithm can be worse as a lens and better as a system. That claim is empty unless the
restoration model is a real, trainable, differentiable thing. Three files provide it:

| file | lines | contents |
|---|---|---|
| `eisoptx/image_restoration/wiener.py` | 169 | Wiener deconvolution + filter helpers |
| `eisoptx/image_restoration/nafnet.py` | 236 | NAFNet, a U-Net-shaped restoration CNN |
| `eisoptx/image_restoration/models.py` | 31 | `ImageRestorationChainer` — composes them |

436 lines total. Small, because the interesting content is the *coupling* to the optics,
not the network architecture.

## 11.2 The interface contract

Every restoration model in this codebase has the same signature:

```python
def forward(self, im, psf_grid): ...
```

**It receives the PSF grid, not just the image.** This is the structural difference
between "deblurring" and "co-designed restoration." A generic deblurring network sees only
a blurry image and must infer the blur. Here the model is *told* the blur — and,
crucially, the PSF grid is **still attached to the autograd graph**, so gradients flow
from the image loss back through the PSFs to the spot diagram and on to the lens
variables.

That connection is the physical channel through which "the algorithm's preferences"
reach the lens. Break it and the two halves optimize independently.

## 11.3 The Wiener stage

`WienerDeconvolution` (`wiener.py:8-62`) is the classical, non-learned core of the
restoration chain — with two learned scalars' worth of adaptation.

**Reducing the PSF grid** (`wiener.py:44-48`):

```python
if psf_grid.dim() == 5:
    effective_psf = psf_grid.mean(dim=1)
```

Chapter 10 produced 81 spatially-varying PSFs. Wiener deconvolution is a **frequency-domain**
operation and requires shift invariance, so the 81 are averaged into one effective PSF.
This is a real approximation and worth naming: the deconvolution is correct at no point in
the field and approximately right everywhere. The NAFNets on either side are what clean up
the resulting spatially-varying residual error — which is precisely why the chain has the
shape it does.

**The filter** (`wiener.py:106`):

```python
wiener_f = kernel_f.conj() / (kernel_f.abs() ** 2 + 1.0 / snr)
```

The textbook Wiener filter. Compare naive inverse filtering, `1/K`, which explodes wherever
`K ≈ 0` — and an aberrated PSF has zeros throughout its spectrum. The `1/snr` term in the
denominator bounds the gain:

```
|K| ≫ 1/√snr  →  wiener_f ≈ 1/K      (invert the blur)
|K| ≪ 1/√snr  →  wiener_f ≈ K̄·snr   (suppress; don't amplify noise)
```

The SNR is the **regularization knob** that decides where the crossover sits. Too high and
you amplify noise into visible artifacts; too low and you barely deblur.

**The residual formulation** (`wiener.py:109-115`):

```python
im_out_f = im_f * (wiener_f - 1)
im_out = torch.fft.irfft2(im_out_f)
im_out = im_out[..., pad_h:pad_h + im_h, pad_w:pad_w + im_w]
return im_out + im
```

Algebraically identical to `irfft2(im_f * wiener_f)`, but computed as a **correction added
to the original**. Two benefits:

1. The image was padded with `mode="replicate"`, and the FFT treats the result as
   periodic. That mismatch produces wrap-around artifacts. In this formulation the
   artifacts live entirely in the *correction term*, which is cropped before being added
   to the clean, unpadded original — so they are attenuated rather than baked in.
2. It makes the module a **residual block**. If the filter does nothing (`wiener_f ≈ 1`),
   the output is exactly the input. Well-behaved at initialization and stable to train
   around.

**Verified:** feeding a delta-function PSF with `snr = 1e9` reproduces the input image
with maximum absolute error **exactly 0.0**. The identity case is exact.

## 11.4 Why the SNR is stored as a logarithm

`wiener.py:25-27`:

```python
if learn_snr:
    self.snr = torch.nn.Parameter(torch.tensor(math.log(snr_initial_value)))
    torch.nn.utils.parametrize.register_parametrization(self, "snr", Exp())
```

The stored parameter is `log(snr)`; `Exp()` maps it back on every access. Three reasons,
all worth knowing:

1. **Positivity is structural.** A negative SNR would put a pole in the filter. Exponential
   parameterization makes negative values unreachable — no clamping, no penalty term.
2. **Multiplicative updates.** Adam's additive step in log-space is a *multiplicative* step
   in SNR. Moving from 100 to 1000 costs the same as 1000 to 10000. Correct for a quantity
   that spans orders of magnitude.
3. **Conditioning.** With `snr = 1000`, `∂output/∂snr` is tiny; `∂output/∂log(snr)` is
   O(1). The gradient is on the same scale as the network's other parameters, so one
   learning rate serves both.

`register_parametrization` is the modern PyTorch idiom for this — it renames the raw tensor
to `parametrizations.snr.original` and installs `Exp` as a property. A side effect worth
recognizing: **the class is dynamically renamed**, which is why the trace reports
`ParametrizedWienerDeconvolution` rather than `WienerDeconvolution`. If you `grep` for that
class name you will not find it; it does not exist in the source.

**Verified:** `snr_initial_value: 1000.` stores raw **6.907755374908447**, and
`log(1000) = 6.907755278982137`. Exponentiating recovers **1000.0001220703125** — the
float32 round-trip error, and exactly the value the trace reports as
`wiener_learned_snr`.

## 11.5 `detach_psfs` — the switch that defines the experiment

`wiener.py:41-42`:

```python
if self.detach_psfs:
    psf_grid = psf_grid.detach()
```

Two words of code, and they decide what experiment you are running.

| setting | gradient path | meaning |
|---|---|---|
| `False` (paper's e2e config) | image loss → PSF grid → spot diagram → lens | The Wiener stage's *preference about the blur* reaches the lens |
| `True` | image loss → network weights only | The lens is optimized only through the NAFNets' path |

With `detach_psfs: False`, the optimizer can improve the image loss by making the PSF
**easier to deconvolve** — more spectral energy away from zeros — rather than smaller.
That is the co-design effect the paper is about, and it is why an end-to-end optimized lens
can have a *worse* RMS spot size and a *better* restored image.

`configs/telephoto/defaults_e2e.yml:24` sets `detach_psfs: False` explicitly. **Measured:
`wiener.detach_psfs: False`.** If a supervisor asks "where in the code does the algorithm
influence the lens?", this line and §11.2's interface are the two places to point.

## 11.6 NAFNet

`nafnet.py` is a faithful implementation of NAFNet (Chen et al., ECCV 2022). It is
**not** a contribution of this paper — it is an off-the-shelf restoration backbone. Know
what it is, don't over-invest in it.

The name means *Nonlinear Activation Free* network, and the claim is that you can drop
ReLU/GELU entirely if you replace them with a multiplicative gate:

```python
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
```

Split the channels in half, multiply elementwise. Nonlinear (it is a product of two learned
features), halves the channel count, and has no saturating region — so no vanishing
gradients and no dead units. Every place a conventional network would put an activation,
NAFNet puts this.

`NAFBlock` (`nafnet.py:101-196`) is the repeated unit:

```
x → LayerNorm2d → 1×1 conv (expand ×2) → 3×3 depthwise conv → SimpleGate
  → × SimplifiedChannelAttention → 1×1 conv → × β → + input
  → LayerNorm2d → 1×1 conv → SimpleGate → 1×1 conv → × γ → + input
```

Points worth noting:

**`groups=dw_channel`** on `conv2` (`nafnet.py:129`) — depthwise convolution. Spatial
mixing and channel mixing are separated, which is where most of the parameter savings
come from.

**Simplified channel attention** (`nafnet.py:137-144`) is `AdaptiveAvgPool2d(1)` followed by
a 1×1 conv — a global-average-pooled per-channel scale. Squeeze-and-excitation with the
nonlinearity and the bottleneck removed.

**`self.beta` and `self.gamma` initialized to zeros** (`nafnet.py:167-168`). Both residual
branches start at **exactly zero**, so a freshly constructed NAFBlock is the **identity
function**. Stacking 20 of them still gives the identity. This is LayerScale, and it is why
deep NAFNets train stably from scratch — the network begins as a no-op and learns how much
of each block to admit.

**`x = x + inp`** at `nafnet.py:88` — a global skip from input to output, so the whole
network learns a *correction* to the blurred image rather than the image itself. Same
principle as the Wiener stage's residual formulation.

**`check_image_size`** pads to a multiple of `padder_size = 2^(number of encoder levels)`,
then crops back. With 4 encoder levels, `padder_size = 16` — **verified**. Necessary for
U-Net downsampling; the crop makes it invisible to the caller.

**Verified parameter count** for the shipped configuration (`width: 16`,
`middle_blk_num: 1`, `enc_blk_nums: [1,1,1,4]`, `dec_blk_nums: [1,1,1,1]`):
**1,493,939 parameters**, matching the trace exactly. Small by modern standards, and
deliberately so — this must run inside a training loop that also traces rays.

## 11.7 The chain, and why it has three stages

`ImageRestorationChainer` (`models.py:4-31`) is 31 lines: hold a `ModuleList`, average the
PSF grid once, apply each model in sequence.

The shipped configuration (`configs/telephoto/defaults_e2e.yml:13-31`) is:

```
NAFNet  →  WienerDeconvolution  →  NAFNet
```

**This ordering is a design decision worth being able to defend.** The reasoning:

**Stage 1 (NAFNet).** The Wiener filter assumes additive Gaussian noise and a known,
shift-invariant PSF. Reality violates all three. A network first can denoise and
pre-condition the image into something closer to what the filter expects.

**Stage 2 (Wiener).** Does the heavy lifting of inverting the blur, using **actual
knowledge of the PSF**. This is the physics injection — a network would need enormous
capacity and data to learn what one FFT and a division do exactly. Getting this for free
is why the hybrid beats a pure network at this size.

**Stage 3 (NAFNet).** Cleans up what the Wiener filter necessarily gets wrong: ringing at
edges, amplified noise, and the spatially-varying error introduced by averaging 81 PSFs
into one.

Physics where physics is exact; learning where the model is wrong. The alternative — one
large network — would need to rediscover deconvolution from data.

Note `ImageRestorationChainer` passes the same `effective_psf` to every stage, and NAFNet's
`forward(self, inp, *args)` simply **ignores** it (`nafnet.py:62`). The NAFNets do not see
the PSF at all; only the Wiener stage uses it. So the entire lens-to-algorithm gradient
path runs through **one** stage of three.

## 11.8 What the measurements actually show — and why

From `8_restoration`, all with **randomly initialized networks**:

| stage | MSE vs. sharp |
|---|---|
| blurred (no restoration) | **0.0033178618** |
| after Wiener alone | **0.0481321253** |
| after full 3-stage chain | **0.1863023043** |

**Restoration makes the image 14× worse, and the full chain 56× worse.** Both numbers must
be read correctly, and the trace's own note flags the second: *"NAFNets are randomly
initialized here, so this is not a trained result."*

For the **chain**, the explanation is trivial: two untrained NAFNets are two random
nonlinear maps. Nothing else could happen. (Note that despite the zero-initialized `β`/`γ`
making each *block* an identity, `intro`, `ending`, `ups`, and `downs` are randomly
initialized ordinary convolutions, so the network as a whole is not.)

For the **Wiener stage alone**, the explanation is more interesting and is a real result:
`snr_initial_value = 1000` is the *untrained default*, and it is far too aggressive for
this system. With `1/snr = 0.001`, the filter inverts spectral components down to
`|K| ≈ 0.03`, amplifying frequencies where the toy PSF has almost no energy. The output
range shows it directly: `wiener_output` spans **−0.717 to +1.738**, well outside the
valid `[0, 1]` image range — classic Wiener ringing and overshoot.

This is exactly the quantity the training loop is supposed to learn. `learn_snr = True`
exists because the correct SNR depends on the lens, the noise level, and the image
statistics, and none of those are known in advance.

⚠ **The note attached to `mse_wiener_vs_sharp` in `trace_dump.json` reads "lower than the
blurred MSE means the Wiener stage helped." It did not help** — 0.0481 > 0.0033. The note
describes how to interpret the number, not what the number says. The measurement stands;
read the comparison yourself.

**If you show these numbers to your supervisor, present them as an untrained-baseline
sanity check that the pipeline runs and is differentiable — not as a result.** A trained
result requires the training loop of Chapter 12 to actually converge.

## 11.9 The toy config has no restoration at all

**Measured: `restoration_model_class: None`.**

`configs/toy/defaults_e2e.yml:9` sets `image_restoration_model:` with no value. The toy
end-to-end config runs the optics pipeline and computes an image loss on the **blurred
image directly**, with no restoration stage.

That is a legitimate and useful configuration — it isolates the GTRA machinery, letting you
verify the lift on an image loss without a network in the way. Chapter 8's verified
identity was measured in exactly this setting. But it means **the toy config cannot
demonstrate co-design.** With no restoration model, "optimize the lens for the algorithm"
degenerates to "optimize the lens for MSE against a sharp image," which mostly means "make
the PSF small" — the same objective as classical design.

To see the paper's actual claim you need `configs/telephoto/defaults_e2e.yml` or
`configs/c_mount/defaults_e2e_restored.yml`. Know which config produced any figure you
present.

## 11.10 Two optimizers, and why

From `configs/telephoto/defaults_e2e.yml:33-43`:

```yaml
irm_optimizer:
  class_path: torch.optim.Adam
  init_args: {lr: 0.001, betas: [0.9, 0.9], foreach: False}
irm_lr_scheduler:
  class_path: torch.optim.lr_scheduler.CosineAnnealingLR
  init_args: {T_max: 25000, eta_min: 0.}
```

The lens gets LM; the restoration model gets Adam with cosine annealing. Chapter 9 §9.12
gave the structural reason — `LMOptimizer` asserts exactly one trainable parameter tensor,
and a NAFNet has hundreds. But the deeper reason is that they are different problems: 20
stiff, coupled, expensively-evaluated variables against 1.5 million cheap, well-conditioned
ones. The tool matches the problem in each case. Chapter 12 covers how the two are
interleaved.

Two further details in that config: `betas: [0.9, 0.9]` lowers the second-moment decay from
the usual 0.999, appropriate when the loss landscape shifts under you because the lens is
changing too. And `foreach: False` carries the comment *"To avoid bug with Pytorch 2.1,
float64 precision and CUDA usage"* — the fused multi-tensor Adam path was buggy for
float64 on CUDA, and this codebase runs in double precision. A version-specific workaround
worth knowing about if you upgrade PyTorch.

Also note `irm_e2e_loss_fn: mae` against `e2e_loss_fn: mse`: **the lens and the network are
trained on different losses.** MSE for the lens because Chapter 8's GTRA lift needs a
smooth, differentiable-everywhere objective; MAE for the network because L1 is standard in
image restoration and produces less over-smoothing. And `noise_sigma: 0.005` adds sensor
noise in the e2e config (the toy config sets `0`) — without noise the Wiener filter's SNR
term has nothing to regularize against and the optimal SNR would be unbounded.

## 11.11 What to take away

1. Every restoration model takes `(im, psf_grid)` with the PSF grid **still attached to the
   graph** — that is the channel through which the algorithm influences the lens.
2. Wiener deconvolution is applied with **one averaged PSF** (81 → 1), an approximation the
   surrounding networks are there to repair.
3. The filter `K̄/(|K|² + 1/snr)` is bounded inverse filtering; SNR sets where inversion
   stops and suppression starts.
4. It is written as a **residual correction**, which keeps FFT wrap-around artifacts out of
   the base image and makes the module an identity when the filter is trivial.
   **Verified exact** on a delta PSF.
5. SNR is stored as `log(snr)` with an `Exp` parametrization: positivity by construction,
   multiplicative updates, O(1) gradients. Verified round-trip 6.907755 → 1000.00012.
   The class is dynamically renamed to `ParametrizedWienerDeconvolution`.
6. **`detach_psfs: False` is what makes it end-to-end.** `True` reduces the experiment to
   separate optimization.
7. NAFNet is off-the-shelf: `SimpleGate` (channel-split product) replaces activations;
   zero-init `β`/`γ` make each block an identity at start. **1,493,939 parameters**,
   verified.
8. The chain is `NAFNet → Wiener → NAFNet`: precondition, invert with real PSF knowledge,
   repair. Only the middle stage sees the PSF.
9. **Measured with untrained weights: MSE 0.0033 (blurred) → 0.0481 (Wiener) → 0.1863
   (chain).** Restoration currently hurts, exactly as expected before training; the
   `snr = 1000` default over-amplifies, visible as output range −0.72 … 1.74.
10. The toy e2e config has **no restoration model** — good for isolating GTRA, useless for
    demonstrating co-design.
11. Lens and network use different optimizers *and different losses* (MSE vs. MAE), for
    structural reasons in both cases.

---

*Previous: [Chapter 10 — PSF simulation](10_psf_simulation.md)*
*Next: [Chapter 12 — The training loop](12_training_loop.md)*
