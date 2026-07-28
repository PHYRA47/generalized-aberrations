# Chapter 10 — PSF simulation: from rays to a blurred image

> **What this chapter answers.** How do 2816 discrete rays become a smooth, *differentiable*
> point-spread function? Why is the PSF 35×35 and not 37×37 as the config comment
> claims? What is SVOLA and why not just one convolution? Where does the energy go, and
> why does the code keep adding constants to the PSF?

---

## 10.1 The gap this chapter closes

Chapter 5 ends with rays at the sensor. Chapter 8 needs `∂L_image/∂xy`. Between them
sits an entire imaging pipeline:

```
spot diagram xy  →  PSFs  →  RGB PSFs  →  PSF grid  →  convolution  →  blurred image
   [2,11,256,1,1]   [11,1,1,35,35]  [11,3,35,35]  [1,81,3,35,35]      [1,3,1024,1024]
```

Every arrow must be differentiable, because Chapter 8's linearization needs the gradient
of the image loss with respect to `xy`. The shapes above are all measured
(`7_psf_and_image`).

Two files do the work: `eisoptx/modeling/simulation.py` (the module structure) and
`eisoptx/modeling/ray_analysis.py` (the splatting kernel).

## 10.2 Rays are discrete; a PSF is continuous

A spot diagram is 256 points. A PSF is an intensity distribution on a pixel grid. The
naive conversion — histogram the points into bins — is **not differentiable**: move a ray
slightly and either nothing changes or a bin count jumps by one. The gradient is zero
almost everywhere and undefined at the bin edges. Useless for optimization.

The fix is **kernel density estimation**. Replace each ray's delta function with a smooth
kernel of finite width, and let neighbouring bins share the energy continuously. Now
moving a ray by a hundredth of a pixel changes the PSF by a hundredth of a pixel's worth
of energy, smoothly, with a well-defined derivative.

`compute_psfs` (`ray_analysis.py:90-158`) does exactly this. It is the bridge between
geometric optics and image formation, and it is short enough to read whole.

## 10.3 The splatting kernel, line by line

```python
dist_x = x[..., None] - x_grid[..., None, None, :]
dist_y = y[..., None, :] - y_grid[..., None]
```

Distance from every ray to every grid coordinate — **separately in x and y**. This is the
key structural choice: the kernel is **separable**, so instead of an `n_rays × 35 × 35`
tensor you build two `n_rays × 35` tensors. For 256 rays that is 17,920 entries instead
of 313,600.

```python
bin_width = span_x / (x_grid.shape[-1] - 1)
sigma_x = sigma_rel * bin_width
```

The kernel width is **one bin**, scaled by `sigma_rel` (toy config: `1.`). Tying the
kernel to the bin size is what makes the smoothing scale-free: change the PSF resolution
and the kernel follows.

Three kernel choices (`ray_analysis.py:130-142`):

| kernel | formula | energy conserving? |
|---|---|---|
| `linear` (toy default) | `(1 − \|d\|/σ)⁺ / σ_rel` | **yes** |
| `cosine` | `cos²(π d / 2σ)⁺ / σ_rel` | **yes** |
| `gaussian` | `exp(−d²/2σ²)` | **no** |

The docstring is explicit about the distinction: *"With either linear or cosine kernel,
each ray contributes equally to the energy... with the Gaussian kernel... the energy of
each ray is not conserved."*

Why: the linear (triangular) and cosine² kernels form a **partition of unity** on a
regular grid — for any ray position, the weights it distributes to neighbouring bins sum
to exactly 1. A Gaussian does not; the sum depends on where the ray falls relative to the
bin centres. That produces a spurious intensity ripple correlated with sub-pixel ray
position, which the optimizer would happily exploit. **The default is `linear` for a
correctness reason, not a speed reason.**

```python
kernels = kernel_y @ kernel_x
```

The separable product, assembled as one matrix multiply — `[.., n_y, n_rays] @ [..,
n_rays, n_x]` contracts over rays and produces the `[n_y, n_x]` PSF in a single BLAS
call. Elegant and fast.

```python
kernels = kernels + kernels.flip(dims=(-1,))
```

**Mirror symmetry.** Chapter 4 established that only the right half of the pupil is
sampled, because the system is rotationally symmetric and fields lie in the y–z plane.
Here the debt is repaid: flipping in x and adding recovers the missing half. Cheap, exact,
and the reason the ray count could be halved in the first place.

```python
kernels = kernels / (x.shape[-1] * 2)
```

Normalize by `2 × n_rays` — 256 rays doubled by the mirror = 512. Each ray carries
exactly `1/512` of the energy. **This is normalization by ray count, not by PSF sum.**
The distinction matters: if rays miss the grid entirely, the PSF sums to *less* than 1,
and that deficit is physically meaningful — it is light that landed outside the PSF
window. Normalizing to unit sum would hide it.

## 10.4 The energy accounting

Which brings us to the pattern that appears three times in this file
(`simulation.py:428-429, 460-462`, and `simulation.py:224-226`):

```python
psf_area = psfs.sum(dim=(-1, -2), keepdim=True)
psfs = psfs + (1 - psf_area) / math.prod(psfs.shape[-2:])
```

Read the comment above it — the authors are unusually candid:

> *"We assume that a PSF area < 1 means that some of the energy was lost because of rays
> not hitting the grid. We redistribute it uniformly over the PSF area. **This has no
> physical meaning**; it's only to penalize energy loss as if it were scattered."*

This is a **modelling decision, not physics**, and you should present it as such.

The situation: a badly aberrated design scatters rays outside the 35×35 PSF window. The
energy is genuinely lost from the model. Two options:

1. Renormalize the PSF to unit sum. The blur then looks *fine* — the optimizer sees no
   penalty for throwing light away. Catastrophic: the optimizer learns to scatter.
2. Add the missing energy back as a **uniform pedestal**. The image gets a flat veil,
   contrast drops, the loss rises. The optimizer is penalized for losing light.

Option 2 is what the code does. It behaves like a physical veiling glare even though the
mechanism is invented. It is the right engineering choice, and the comment says so
honestly.

Note the operation is `+`, not `*` — an additive floor, so it cannot amplify the
existing PSF, only add a background. And it is applied **twice** when diffraction is on:
once after splatting, once after the diffraction convolution, since the convolution's
padding also drops energy at the edges.

**Measured:** `psf_energy_per_field: [11, 3]`, min **0.9999999404**, max **1.0**.
Every field, every channel, unit energy to float32 precision. The shipped toy design loses
no light — consistent with Chapter 5's zero failed rays, and Chapter 5's measured spot
sizes (10–30 µm) being far smaller than the 281 µm PSF window.

## 10.5 Centering — and what it discards

`simulation.py:412-415`:

```python
y_centroid = ra.evaluate_mean_ray_height(y, ray_valid, (1, 2), self.wavelength_weights)
y = y - y_centroid.expand_as(y).where(ray_valid, 0.0)
```

The PSF is centered on the **wavelength-weighted centroid** before splatting. Two
consequences:

**Chromatic aberration survives.** The centroid is computed once across all wavelengths
using the weights. Individual wavelengths keep their own offsets relative to it, so
lateral colour still shows up as channel misregistration in the PSF. Had the code
centered each wavelength separately, chromatic aberration would have been silently
removed.

**Distortion does not survive.** Subtracting the centroid discards the absolute image
height. This is the same blindness Chapter 5 measured (−14.2% barrel) and Chapter 7
explained for the residuals — and it recurs here, in the image pipeline, for the same
structural reason. **The end-to-end loss cannot see distortion either.** In the paper's
full pipeline distortion would have to be handled by a separate residual
(`DistortionResiduals`) or by geometric correction in the restoration stage.

The `.where(ray_valid, 0.0)` guard keeps failed rays at `inf` rather than `inf - finite`;
`build_optics_model` marked them (`simulation.py:127`):

```python
xy = xy.where(ray_status < 2, float("inf"))
```

The same `< 2` survivability threshold as Chapter 5, and the same never-NaN discipline:
`inf` propagates predictably through `isfinite` masking, while `NaN` would poison sums.

## 10.6 Why the PSF is 35×35 — and a stale comment

Toy config:

```yaml
shape: [1024, 1024]
shape_type: image
psf_abs_size: 281.3e-3  # 37x37 pixels
```

The constructor (`simulation.py:55-62`) works out the pixel size from the sensor:

```python
image_diag = torch.tensor(shape).float().norm().item()
pixel_abs_size = self.sensor_diagonal / image_diag
psf_shape = ((np.array(psf_abs_size) / pixel_abs_size) // 2 * 2 + 1).astype(int)
```

Arithmetic:

```
sensor_diagonal = 2 · 10 · tan(30°)          = 11.5470054 mm
image_diag      = ‖(1024, 1024)‖             = 1448.1547 px
pixel_abs_size  = 11.5470054 / 1448.1547     = 7.9736 µm
psf_abs_size / pixel = 281.3 / 7.9736        = 35.279
35.279 // 2 * 2 + 1                          = 35
```

**Measured: `rgb_psfs: [11, 3, 35, 35]`.** The code produces 35×35.

⚠ **The config comment `# 37x37 pixels` is wrong.** Reaching 37 would need
`psf_abs_size ≥ 0.295 mm`. Harmless — the comment is documentation, not code — but if
you cite the config in a defense, cite 35 and be ready to show this arithmetic. The
number that matters is 281.3 µm of sensor, about 2.4× the largest measured spot.

The `// 2 * 2 + 1` forces **odd**, enforced by `assert tuple(self.psf_shape % 2) == (1,1)`
at `simulation.py:367`. An odd kernel has an unambiguous centre pixel; an even one would
introduce a half-pixel shift in the convolution.

## 10.7 Wavelengths to RGB

`build_optics_model_from_xy` (`simulation.py:131-145`):

```python
psfs = self.psf_sampler(xy)
rgb_psfs = torch.einsum("fbwij,wc->fbcij", psfs, self.wavelength_weights)
```

A `[n_wavelengths, 3]` matrix contracts the wavelength axis into three colour channels —
the sensor's spectral response, applied as a linear map.

Toy config is degenerate: one wavelength, weights `[[1.],[1.],[1.]]`, so all three RGB
channels are **identical copies** of the single 550 nm PSF. There is no colour information
in the toy system; the RGB axis exists only so the shapes match the image pipeline.

Contrast `configs/telephoto/defaults.yml:14-32`, which is what the paper actually ran:
**11 wavelengths** from 425 to 675 nm with a realistic photopic weighting, and a full
`[11, 3]` response matrix whose rows are recognizable RGB filter curves (the R row peaks
at 575 nm, B at 525 and below). That configuration produces genuinely different PSFs per
channel and can optimize against lateral colour. **When you present results, be clear
which regime you are in** — the toy config cannot demonstrate chromatic behaviour at all.

## 10.8 One PSF is not enough: the PSF grid

A real lens is **spatially varying** — the corner PSF is nothing like the centre PSF
(Chapter 5: 3× the RMS spot size). A single convolution assumes shift-invariance and
would be wrong everywhere except the field it was sampled at.

`compute_psf_grid` (`simulation.py:147-185`) turns 11 radially-sampled PSFs into a 9×9
spatial grid over the image, in three steps.

**Interpolate** (`simulation.py:187-197`):

```python
psf_weights = get_psf_weights(*self.psf_grid_shape, field_lims, psfs.shape[0])
interpolated_psfs = torch.einsum("fcij,bpf->bpcij", psfs, psf_weights)
```

Each of the 81 patches gets a weighted blend of the 11 sampled fields, by its radial
distance from the image centre. One `einsum`; the weights are prenormalized.

**Rotate** (`simulation.py:199-228`):

```python
angles = torch.arctan2(-x_center[:, None, :], y_center[:, :, None])
rotated_psfs = rotate2d(psfs, angles, "bilinear", "constant", 0.0)
```

This step is the **entire justification for tracing only 11 fields**. The system is
rotationally symmetric, so the PSF at azimuth φ is the PSF at the same radius rotated by
φ. Trace a radial line, rotate to fill the plane. Without it you would need to trace 81
distinct field positions — a 7× cost increase in the most expensive part of the pipeline.

The comment at line 218 notes the rotation preserves area; the pedestal correction is
applied again afterwards because bilinear resampling still leaks energy at the corners.

**Resize** (`simulation.py:230-268`). The PSF was built in *sensor* millimetres; the
convolution needs it in *image* pixels. If the image is not at the sensor's native
resolution, the PSF is bilinearly resampled with `antialias=True` and renormalized to
unit sum. For the toy config the sizes already match, so this is a no-op.

**Measured:** `psf_grid: [1, 81, 3, 35, 35]` — batch 1, 81 patches, RGB, 35×35 each.
Note `mean = 0.0008163266` versus `rgb_psfs` mean `0.0008163265`: identical to 9
significant figures, confirming that interpolation, rotation, and the pedestal correction
together conserve total energy.

## 10.9 SVOLA: spatially-varying convolution

`SVOLAConvolution` (`simulation.py:467-515`) — *spatially varying overlap-add*.

The problem: 81 different kernels, one image. You cannot call `conv2d` once.

The method:

1. Pad the image by the overlap (`mode="replicate"`, avoiding a dark border).
2. Cut it into 81 overlapping patches — 9×9 grid over 1024², so **113.8 px per patch**
   with **28.4 px overlap** at `patch_overlap: 0.25`.
3. Convolve each patch with **its own** PSF (`patch_convolve`).
4. Stitch back with a smooth weight window (`reconstruct_patches`).
5. Remove the padding.

Step 4 is the important one. Butt-jointing the patches would give visible seams wherever
adjacent PSFs differ. The overlap plus a tapered weight window **cross-fades** between
neighbouring PSFs, so the blur varies smoothly across the field. That is what
"overlap-add" buys, and it is why `patch_overlap` is a tunable: too little gives seams,
too much wastes compute.

`try_recompute_operands` (`simulation.py:517+`) caches the patch geometry and only rebuilds
when the image size changes — the geometry depends on the image shape, not on the lens,
so it is invariant across the whole optimization. A guard, not a computation.

**Measured:** `blurred_image: [1, 3, 1024, 1024]`, values in `[0.0338, 0.9669]`,
`requires_grad = True`. **MSE against the sharp image: 0.0033178618.** That number is
the toy system's baseline image degradation, and it is the anchor for Chapter 11.

## 10.10 What differentiability costs here

Everything above is differentiable with respect to `xy`, hence to the lens variables.
Worth knowing where the cost sits:

| stage | tensor | approximate size |
|---|---|---|
| splatting | `kernel_x`, `kernel_y` | 2 × 256 × 35 per field |
| PSFs | `[11, 1, 1, 35, 35]` | 13k |
| PSF grid | `[1, 81, 3, 35, 35]` | 298k |
| patches | 81 × `[3, 142, 142]` | 4.9M |
| image | `[1, 3, 1024, 1024]` | 3.1M |

The patch tensor is the peak, and it is ~370× the PSF grid. At paper scale
(`shape: [6144, 8192]` in the telephoto config) this is the memory bottleneck of the
entire system — which is a second, practical reason the GTRA construction of Chapter 8
runs this pipeline **once per LM iteration** rather than once per Jacobian column.
Twenty forward passes through this would be twenty times that peak.

## 10.11 The alternative: `FixedPSFsOpticsSimulator`

`simulation.py:280-333` provides a subclass whose `build_optics_model` ignores the lens
entirely and returns stored PSFs. It exists for training a restoration network against a
**frozen** optical design — the ablation baseline the paper compares against. If the
optics are not being optimized, there is no reason to re-trace rays every step.

Seeing it in the class hierarchy is a useful reminder of what the main class is for: the
`OpticsSimulator` is expensive *because* it is differentiable, and that expense only buys
something if the lens is a variable.

## 10.12 What to take away

1. Rays → PSF via **kernel density estimation**, not histogramming, because histograms
   have no useful gradient.
2. The kernel is **separable** (two 1-D kernels, one matmul) and **linear by default**
   because triangular kernels conserve each ray's energy exactly; Gaussian does not.
3. `kernels + kernels.flip(-1)` repays the half-pupil sampling of Chapter 4 exactly.
4. Normalization is by **`2 × n_rays`**, so lost light shows up as a PSF sum below 1 —
   and is then added back as a **uniform pedestal**, an admitted non-physical device that
   makes energy loss cost something. Measured energy: **1.0**.
5. Centering on the weighted centroid preserves chromatic aberration but **discards
   distortion** — the same blind spot as the residuals, now in the image pipeline too.
6. PSF size is **35×35**, derived from sensor diagonal and image resolution; the config's
   `# 37x37` comment is stale.
7. Rotational symmetry lets 11 traced fields fill an 81-patch grid — a 7× saving in the
   most expensive stage.
8. **SVOLA** handles spatial variation: 81 overlapping patches, each with its own PSF,
   cross-faded on stitch. 113.8 px patches, 28.4 px overlap.
9. Measured baseline: `blurred_image [1,3,1024,1024]`, **MSE 0.00332** against ground
   truth.

---

*Previous: [Chapter 9 part 2 — the LM optimizer](09_lm_optimizer_part2.md)*
*Next: [Chapter 11 — Image restoration](11_image_restoration.md)*
