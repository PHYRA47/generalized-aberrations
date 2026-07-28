# Appendix A — Supervisor defense bank, part 2 (Q26–Q50)

> Optimization, imaging simulation, training, reproducibility, and your own implementation.
> Part 1 (Q1–Q25) covers the method and the optics.

---

## The LM optimizer

**Q26. Write the LM update and say how the code actually solves it.**

The paper's Eq. (11) is `Δx = −(JᵀJ + λD²)⁻¹Jᵀℓ₀`. The code **never forms `JᵀJ`**. It solves
the equivalent stacked least-squares problem

```
[   J   ] Δx = −[ ℓ₀ ]        augmented_matrix [5724, 20], measured
[ √λ·D  ]       [ 0  ]
```

`optimizers.py:60-150`. The reason is conditioning: `J` spans ±486 while `JᵀJ` spans ±2.25e6
(measured, Chapter 15). Forming the normal equations squares the condition number.

**Q27. What is `D²` and why is it not the identity?**

Marquardt's scale-invariant damping — `D²` holds the diagonal of `JᵀJ`, i.e. the squared column
norms of `J`. Using the identity instead would make the step depend on the units you chose for
each variable. With curvatures in mm⁻¹ and spacings in mm, that matters: measured damping terms
span `0.3719` to `1501.38`.

**Q28. The paper says `D² = diag(JᵀJ)`. Does the code do that?**

Not exactly, and this is a good divergence to know. `optimizers.py:87-90`:

```python
beta * torch.max(damping_terms, group["damping_terms"]) + (1 - beta) * damping_terms
```

With `beta = 0.99` this is a **ratchet on the running maximum** — the damping scale can rise in
one iteration and decays at only 1% per iteration. It looks like an EMA and is not one. The
purpose is to stop a parameter that momentarily has small sensitivity from getting almost no
damping and taking an enormous step.

**Q29. When is a step accepted?**

`optimizers.py:143`: rejected if the loss ratio is non-finite or `> tolerance`. **`tolerance`
defaults to 2.0**, meaning a step that doubles the loss would be accepted at the default. The
toy and telephoto configs set `1.0`, which is the paper's stated rule; `microscope` and
`c_mount` use `1.25` and `wide_angle` uses `1.5`.

**Q30. The paper says LM curves decrease monotonically. Is that true?**

**For two of the five shipped configs.** It follows from `tolerance: 1.0`, which toy and
telephoto set. Under `wide_angle`'s `1.5`, a 50% loss increase is accepted and the curve is not
monotonic. Chapter 15 measured the toy behaviour: iteration 11 has ratio `1.1102` and is
rejected — as `tolerance: 1.0` requires, and as `1.5` would not.

**Q31. How is λ adapted?**

Divided by `lam_decrease_factor` on success (`:147`), multiplied by `lam_increase_factor` on
failure (`:149`). The paper says ÷3 and ×2 citing Transtrum & Sethna. The code **defaults to
÷2 and ×2**; the production configs set ÷3 and match the paper, and **the toy config does
not** — which is why Chapter 15's convergence table shows λ halving.

**Q32. What does a rejected step cost?**

One extra residual evaluation, which for GTRA is a ray trace only — the image pipeline is not
re-run, because `w` and `ε′` are frozen. **That cheapness is a direct payoff of the lift** and
is what makes an aggressive rejection policy affordable.

**Q33. `lstsq_rank` was 14 of 20 at convergence. Is that a problem?**

It means 6 directions in parameter space have no first-order effect on the residual at that
point — the design is at a flat spot in those directions. LM handles it because the damping
term `√λD` in the augmented matrix makes the stacked system full rank regardless. Without
damping, `torch.linalg.solve` on `JᵀJ` would fail.

## Imaging simulation

**Q34. How does a spot diagram become a PSF?**

Kernel density estimation with a 2D linear (triangular) kernel of twice the pixel size, binned
onto a virtual detector at the physical pixel pitch. Paper Eq. (12) region, §4.1;
`ray_analysis.py`. The reason for KDE rather than histogramming is differentiability — summing
rays into bins has zero gradient almost everywhere.

**Q35. What are the measured PSF dimensions in your run?**

`35 × 35` at a pixel pitch of `7.9736 µm`. Chapter 10 — and note the YAML comment says
"37×37", which is **stale**. Energy per PSF measured `1.0` to 9 significant figures through
interpolate/rotate/resize.

**Q36. How is diffraction handled?**

A heuristic: each ray is treated as generating its own Airy pattern, `2J₁(·)/(·)`, and the
geometric PSF is convolved with it. `simulation.py:667` is the Bessel evaluation, `:622` is
`NA = 1/(2F#)`, `:654` sizes the kernel to the Airy radius.

**Q37. Intensity or amplitude convolution?**

**Amplitude**, in the default mode. `simulation.py:375` sets `diffraction_mode = "airy_field"`;
`:446` square-roots the geometric PSF to an amplitude, the convolution runs, and `:456-458`
squares back. This is more physical than the paper's Eq. (12) as printed, which reads as an
intensity convolution. It is also **inactive in the toy problem** — no `diffraction_f_number`
is set, and the paper says of that experiment that diffraction is not considered. They agree.

**Q38. How does spatially varying blur work?**

`SVOLAConvolution` (`simulation.py:467`): the image is split into patches, each convolved by
FFT with the PSF interpolated for its field position, and recombined by overlap-add. Measured
81 grid entries (9×9), patches of `113.78 px` with `28.44 px` overlap. Chapter 10. The paper's
setup says "9 × 9 sensor regions" — exact match.

**Q39. Why only 11 fields for a 2D sensor?**

Rotational symmetry. PSFs are traced along one radius and **rotated** to fill the 2D grid — a
7× saving. Chapter 10 verified energy is conserved to 9 significant figures through the
rotation. This is also why the field-limit collate function exists (`datasets.py`): a corner
crop of a 6144×8192 sensor must receive corner PSFs, so crops carry their field provenance.

**Q40. Does the restoration network see the PSF?**

In principle yes; in practice mostly no. `NAFNet.forward(inp, *args)` accepts and **ignores**
it (Chapter 11). Only the Wiener stage uses the PSF. So in a NAFNet-only configuration the lens
gradient reaches the network only through the blurred image.

## Training

**Q41. Walk me through one training step.**

`imaging_system.py:555` `training_step`. In order: build the lens from the current parameters →
trace rays → build the PSF grid → simulate the capture → restore → scalar loss → **one**
backward pass for `∇L₀` → build `(w, ε′)` and freeze them → concatenate GTRA and geometric
residuals → one LM step with a forward-mode ray-tracer Jacobian → accept or reject. Chapter 15
traces it with real numbers.

**Q42. Why `automatic_optimization = False`?**

Because there are two optimizers with incompatible characters — LM needs a closure it can call
repeatedly to evaluate trial steps, Adam does not — and Lightning's automatic path assumes one
`loss.backward()` per step. Chapter 12.

**Q43. What does `toggle_model()` do and why is it required?**

It freezes one half's parameters while the other steps. It is what makes LM's
single-parameter-tensor assertion hold: without it, LM would see the network's parameters in
its group. Chapter 12.

**Q44. Does the lens gradient flow during the network step?**

**No.** The network half wraps the PSF build in `torch.inference_mode`, so the optical
coupling is one-directional per step: the lens step sees the network, the network step sees only
a fixed blurred image. Chapter 12 §12.7.

**Q45. What is `detach_psfs` for?**

It is the switch that defines the experiment. With PSFs detached, the lens receives no gradient
from the image loss and you have the frozen-optics ablation; attached, you have co-design.
Chapter 11.

**Q46. Classical design, network-only training, and co-design — how different is the code?**

Same code path, different config. That is the architectural claim of the package and it holds:
the residual list and the two `*_optimization_disabled` flags select the experiment. Chapter 12.

**Q47. Table 1 reports 16.6 µm and MSE 0.0033. Can you reproduce them?**

Both, from the shipped config. Effective spot radius = RMS over fields of the per-field RMS
spot = **16.6029 µm** against the paper's 16.6. Blurred-vs-sharp MSE = **0.0033178618** against
the paper's 0.0033. Chapter 16 §16.3, §16.14. **These two numbers authenticate the shipped
config as the publication's design** — worth having ready, because it answers "are you sure
you're running the paper's experiment?"

**Q48. What in the paper is not in the code?**

`ℓ_ESR` (Eq. 4) has no implementation — there is no `EffectiveSpotRadiusResiduals`. If you want
to reproduce the `LM + ℓ_ESR` row of Table 1 (33.0 µm), you have to write it. Chapter 16 §16.4.

Also: the paper's §3.4 ray budget — 11 fields × 11 wavelengths × 512 pupil = 61,952 rays —
describes its main experiments. The toy problem is monochromatic with 256 pupil points:
**2816 rays**. Do not quote 61,952 next to a toy trace.

## Your own implementation

**Q49. What are the real gaps in `e2e-gtra-optics`, honestly?**

Three, in priority order (Chapter 17 §17.11):

1. **`constraints.py` is 321 lines, tested, and never called.** Nothing in the package
   concatenates it into the LM residual. In the reference, 72 of 5704 residuals are geometric
   constraints — 1.3% of the length and all of the manufacturability. `_trace_packed` already
   accepts `probes=True` (`raytrace.py:434`) and its docstring says the probes exist for exactly
   this. The wiring was designed and not connected.
2. **`clip_control_values` (`gtra.py:78`) is a plain clamp**, missing the compensation factor,
   so after clipping the surrogate's value no longer equals `L₀` and the LM step length is
   computed against a loss that is too small. Your own docstring says so. ~4 lines to close.
3. **`ConvolutionImaging` is shift-invariant** — `kde_psf` returns all `F` fields and
   `imaging.py:83` discards `F−1` of them. So the paper's headline result (radially narrow,
   angularly wide PSFs) is not representable in your pipeline as it stands.

Plus: you form `JᵀJ` explicitly (`lm.py:88`) in `float32`, and you use one loss for both halves
where the paper uses MSE for the lens and MAE for the network — feeding a sign-valued MAE
gradient into the lift gives a poorly scaled `w`.

**Q50. Is your implementation just a transcription of theirs?**

No, and there are specific answers (Chapter 17 §17.12):

- **A NaN-launder guard the reference lacks** (`lm.py:105-114`): a NaN `theta` traces to NaN
  rays, `nan_to_num` sanitizes them to zeros, the residual comes back all-zero and the loss is
  `0.0` — the best step LM has ever seen. It accepts, and the run reports a perfect merit and
  zero spot size. You guard the *parameters*, not the loss. `eisoptx` checks
  `math.isfinite(loss_ratio)`, which does not catch this.
- **A damping floor** (`lm.py:78`) implementing Supp. S65, which `eisoptx` has no equivalent of.
- **`TF = 1` acceptance**, matching the paper where the reference's default `tolerance = 2.0`
  does not.
- **`tra_control_values`** (`gtra.py:99`) makes "GTRA generalizes TRA" executable rather than
  prose.
- **`offgrid_fraction` + a clipping warning** (`imaging.py:69`) catches a silent falsification
  mode: a renormalized clipped PSF looks plausible and quietly invalidates every PSNR
  downstream. The reference warns about nothing.
- **`w` is named `w` and holds `w`** — the reference's `weight` holds `√w`.
- **1,260 lines of tests** against a reference that ships none.
- **A cleaner lift interface**: detach the spot, make it a leaf, rebuild a `SpotDiagram` around
  it, take a plain `autograd.grad` (`joint.py:104-109`) — easier to read and to test than the
  `torch.func` closure.

Offer the gaps in Q49 before being asked. Knowing your own limitations precisely is the
stronger position.

---

## The three answers to have ready cold

1. **The lift**: `w = ‖∇L₀‖²/(2L₀)`, `ε′ = ε₀ − 2L₀∇L₀/‖∇L₀‖²`, `ℓ = √w(ε−ε′)`,
   `imaging_system.py:484-513`, identity verified to `7.249e−07`.
2. **The computational asymmetry**: backward through the image pipeline **once** per iteration
   for `∇L₀`; forward through the ray tracer **`n = 20`** times for `J`. That asymmetry is why
   the method is affordable.
3. **The two numbers that authenticate the run**: ESR `16.6029 µm` vs the paper's `16.6`, and
   MSE `0.0033178618` vs the paper's `0.0033`.

---

*Previous: [Appendix A part 1 — Q1–Q25](A_defense_bank_part1.md)*
*Back to: [Index](README.md)*
