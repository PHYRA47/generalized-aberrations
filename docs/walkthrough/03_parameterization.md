# Chapter 3 — Parameterization: variables, scaling, and the glass model

> **What this chapter answers.** What is the optimizer *actually* changing? Why are the
> optimized numbers different from the numbers in the config? Why is a curvature
> multiplied by 2.5 and a spacing divided by it? How can a discrete glass catalog be
> differentiable?

---

## 3.1 The problem: optical parameters have terrible relative scales

Look at the toy lens's physical parameters:

```
spacings   s : 1.32, 3.18, 0.50, 6.00, 10.22      mm         — order 1–10
curvatures c : −0.218, −0.159, 0.103, −0.092      mm⁻¹       — order 0.1
asphere a[3] : 5.89e−6                            mm⁻⁷       — order 10⁻⁶
Abbe      vd : 1e6                                —          — order 10⁶
```

Twelve orders of magnitude between an aspheric coefficient and an Abbe number. Any
optimizer that treats these as one vector will be catastrophically ill-conditioned: a
step size that moves a spacing usefully will annihilate a curvature, and one that
respects the aspheres will never move anything else.

Levenberg–Marquardt is *partially* protected — its damping term uses per-column
Jacobian norms (Chapter 9), which is a form of automatic scaling. But that only helps
with the damping, not with the underlying conditioning of `J` itself. So `eisoptx`
normalizes the variables **before** they ever reach the optimizer.

`LensParameterization` is therefore best understood as a **change of coordinates**: the
optimizer works in a well-conditioned, dimensionless space, and the parameterization
maps that space to physical millimetres on every forward pass.

## 3.2 The scaling rule: dimensional analysis

`get_normalized_variables` (`parameterization.py:716-754`) is the whole story:

```python
a_exp = torch.tensor([0] + [2 * (i + 2) for i in range(lens.a.shape[-1] - 1)])
a = lens.a * scale_factor**a_exp
...
lens_parameters = {
    "s": s / scale_factor,                                   # length¹
    "g": glass_model.g_from_nd_vd_dpgf(lens.nd, lens.vd, lens.dpgf),
    "c": lens.c * scale_factor,                              # length⁻¹
    "a": a,                                                  # mixed
    "d": lens.d * scale_factor**d_exp,
    "m": lens.m,
}
```

The rule is **dimensional**: multiply by `scale_factor` raised to the power of the
quantity's length dimension, so every variable comes out dimensionless and O(1).

- A spacing has dimension length⁺¹ → **divide** by `scale_factor`.
- A curvature has dimension length⁻¹ → **multiply** by `scale_factor`.
- Aspheric coefficient `i` multiplies `r^(2i+2)`, so it has dimension length^−(2i+1) →
  multiply by `scale_factor^(2i+2)`. Hence `a_exp = [0, 4, 6, 8]` for four
  coefficients — index 0 is the conic constant, which is **dimensionless**, so its
  exponent is 0.

`scale_factor: 2.5` in the toy config, and the docstring (`parameterization.py:67`)
says why: *"EPD / 2 is usually a good starting point"*. The entrance pupil is 5 mm, so
`scale_factor = 2.5` = the pupil **radius**. The natural length scale of the problem is
the size of the beam.

**Verified arithmetic.** Physical curvatures, measured
(`1_config_and_lens.lens.c_after_solves`):

```
c_physical = [−0.2178221047, −0.1592064947,  0.1029609889, −0.0921629593]
× 2.5      = [−0.5445552618, −0.3980162368,  0.2574024723, −0.2304073983]
```

Measured normalized variables (`lens_variables[c]`):

```
             [−0.5445552468, −0.3980162442,  0.2574024796, −0.2304074019]
```

Agreement to float32. ✓ And for spacings, `s_physical / 2.5`:

```
[1.3228429556, 3.1832540035, 0.4969364107, 6.0048489571, 10.2153987885] / 2.5
 = [0.52913718, 1.27330160, 0.19877456, 2.40193958, 4.08615952]
measured lens_variables[s]:
   [0.5291371942, 1.2733016014, 0.1987745613, 2.4019396305, 4.0861597061]  ✓
```

Now compare the *ranges*. Physically the values span 0.09 to 10.2 — a factor of 110.
Normalized they span 0.199 to 4.09 — a factor of 21, and everything sits within an
order of magnitude of 1. That is the entire point.

## 3.3 The flat variable vector and the freeze mask

The constructor (`parameterization.py:120-147`) flattens all six variable groups into
one vector, in the fixed order given by `self.variable_keys = ("s", "g", "c", "a", "d", "m")`:

```python
initial_variable_list = torch.cat(
    [lens_variable_dict[k].view(-1) for k in self.variable_keys]
)
...
optimization_mask = torch.cat(
    [~self.freeze_dict[k].view(-1) for k in self.variable_keys]
)
self._lens_variables = torch.nn.Parameter(initial_variable_list[optimization_mask])
self.register_buffer("initial_variables", initial_variable_list)
self.register_buffer("optimization_mask", optimization_mask)
```

Read that carefully — it is the crux of the design:

- **`initial_variables`** is a *buffer* holding **all 31** values.
- **`_lens_variables`** is the only `Parameter`, holding **only the 20 unfrozen** ones.
- **`optimization_mask`** records where the 20 go back among the 31.

The optimizer never sees a frozen variable. This is stronger than
`requires_grad = False`: the frozen values are not in the parameter tensor at all, so
the Jacobian has exactly **20 columns** — measured `jacobian` shape `[5704, 20]`
(`5_lm_step`). No wasted columns, no zero rows in the solve, no rank confusion from
structurally-dead parameters.

Measured layout (`1_config_and_lens.variable_layout`):

| key | shape | numel | slice in the flat vector | trainable |
|---|---|---:|---|---:|
| `s` | [5, 1] | 5 | 0:5 | 5 |
| `g` | [2, 1, 3] | 6 | 5:11 | **0** |
| `c` | [4, 1] | 4 | 11:15 | 3 |
| `a` | [4, 1, 4] | 16 | 15:31 | 12 |
| `d` | [0, 1, 0] | 0 | 31:31 | 0 |
| `m` | [0, 1, 0] | 0 | 31:31 | 0 |
| | | **31** | | **20** |

Reconcile with the config's `freeze` block (Chapter 1):

- `g: True` → all 6 glass variables frozen → 0 trainable ✓
- `c` free except index −1 → 4 − 1 = 3 ✓ (the last curvature is solved, Chapter 6)
- `a` free except column 0 of every row → 16 − 4 = 12 ✓ (column 0 is the conic constant)
- `s` fully free → 5 ✓

Total 20. ✓

## 3.4 The forward pass: variables → `Lens`

Every training iteration, `LensParameterization.forward` rebuilds the lens. The
sequence is:

1. **Scatter** the 20 trainable values back into a length-31 vector using
   `optimization_mask`, taking frozen entries from the `initial_variables` buffer.
2. **Split** the flat vector into `s, g, c, a, d, m` using `variable_shape_dict`.
3. **Unscale** — invert the dimensional scaling of §3.2 to recover millimetres.
4. **Decode glass** — turn `g` into `(nd, vd, dpgf)` (§3.5).
5. **Construct** an `optics.Lens`.
6. **Apply solves** — overwrite `c[solve_idx]` and/or the last spacing (Chapter 6).

Step 6 is why `lens_variables[c]` and `lens.c_after_solves` do not correspond for the
last element. Compare index 3:

```
lens_variables[c][3] = −0.2304074019  → /2.5 = −0.09216296
lens.c_after_solves[3] = −0.0921629593
```

These agree here because the shipped design is *already* converged — the stored value
is the solved value. Mid-optimization they diverge: the stored variable is ignored and
the solve's output is used. This is exactly why `c[-1]` is frozen; leaving it trainable
would let the optimizer push a number that gets thrown away, wasting a Jacobian column
and confusing the rank estimate.

**Everything in this chain is differentiable.** The scatter is indexing, the unscale is
multiplication, the glass decode is arithmetic, the solve is a closed-form expression.
Autograd walks back from a ray coordinate at the sensor all the way to the 20 numbers
in `_lens_variables`.

## 3.5 The glass model — making a discrete catalog differentiable

This is the most interesting piece of engineering in the file, and a likely supervisor
question, because it looks impossible at first: real glasses are a **discrete catalog**
(N-BK7, SF11, …), but gradient descent needs a **continuous** space.

`GlassModel` handles this with a two-level scheme.

**Level 1 — continuous coordinates.** A glass is represented by
`g = (nd, vd, dpgf)`: refractive index, Abbe number, and deviation from normal partial
dispersion. `g_from_nd_vd_dpgf` (used at `parameterization.py:748`) and its inverse map
between this triple and the tensors the ray tracer needs. In this space you can take
arbitrary gradient steps — you just end up at a glass that does not exist.

Measured for the toy config (`lens_variables[g]`):

```
[1.5, 1000000.0, 0.0,   1.5, 1000000.0, 0.0]
 nd    vd        dpgf    nd    vd        dpgf
```

Two identical fictional materials: index 1.5, **Abbe number 10⁶**, zero partial
dispersion deviation. As noted in Chapter 1, `vd = 1e6` means dispersion-free. This
"glass" exists nowhere in nature; it is a deliberate simplification so the toy problem
is monochromatic. Note also that `g` contributes **0 trainable variables** here — the
toy config freezes glass entirely. So in the toy system the glass model is
*infrastructure that is present but inert*. Be careful not to over-claim it in a
defense: point at the paper-scale configs if asked for a live demonstration.

**Level 2 — quantized-continuous variables (`qc_vars`).** When you *do* want
manufacturable glass, setting `qc_vars: True` (with a `glass_file` catalog) switches
on quantization. The constructor guards it (`parameterization.py:81-83`):

```python
if self.glass_model.catalog_g is None:
    assert qc_vars is False, "No glass catalog available."
```

The mechanism is a **straight-through estimator**, the same trick used in quantized
neural networks and VQ-VAE: on the forward pass, snap the continuous `g` to the nearest
catalog entry (`scipy.spatial` handles the nearest-neighbour search — note the import
at `parameterization.py:6`); on the backward pass, pretend the snap was the identity so
gradients flow to the continuous variable. The optimizer moves through a continuous
space while the ray tracer always sees a real, orderable material.

The toy config sets `qc_vars: False` and `glass_file:` (null), so neither is active
here.

> **Defense note.** If asked "how do you handle glass selection, which is discrete?" —
> the answer has three layers: (1) the continuous `(nd, vd, dpgf)` parameterization,
> (2) optional straight-through quantization to a catalog, (3) and in the shipped toy
> configuration, glass is simply frozen and fictional. Do not claim the toy experiment
> demonstrates glass optimization; it does not.

## 3.6 Why `dpgf` exists

`dpgf` is the deviation from "normal" partial dispersion. Most optical glasses lie
close to a line in the `(vd, Pgf)` plane; the ones that deviate — the abnormal
dispersion glasses — are what make apochromatic correction possible. Modelling only
`(nd, vd)` would restrict the optimizer to the normal line and make secondary-spectrum
correction impossible.

The constructor handles its absence gracefully (`parameterization.py:80, 88-89`):

```python
self.glass_model = GlassModel(glass_file, assume_normal_dispersion=dpgf is None)
...
if dpgf is None:
    dpgf = [0.0] * len(nd)
```

Omit `dpgf` from the config and you get normal dispersion with zeros. The toy config
sets `dpgf: [0.0, 0.0]` explicitly.

## 3.7 Assertions worth knowing

`parameterization.py:75-77`:

```python
assert not (self.paraxial_image_solve and self.total_track_length_solve is not None), \
    "Cannot use both paraxial image solve and total track length solve."
```

Both solves would write the **last spacing**. One would silently overwrite the other,
so the constructor refuses. This is the kind of guard worth pointing to if asked about
robustness — the code fails loudly on contradictory configuration rather than
producing a quietly wrong lens.

## 3.8 What to take away

1. The optimizer works in normalized, dimensionless coordinates; the parameterization
   converts to millimetres each forward pass.
2. The scaling exponent is the quantity's **length dimension**: `s/k`, `c·k`,
   `a_i·k^(2i+2)`, conic constant untouched. `k = 2.5` = pupil radius.
3. Frozen variables are *excluded from the parameter tensor*, not merely detached —
   giving a Jacobian with exactly 20 columns.
4. `initial_variables` (31, buffer) + `optimization_mask` + `_lens_variables`
   (20, Parameter) is the scatter/gather scheme; verified against the measured layout.
5. Glass is continuous `(nd, vd, dpgf)`, optionally snapped to a catalog by a
   straight-through estimator; in the toy config it is frozen and fictional
   (`vd = 1e6` = dispersion-free).
6. The whole variables→lens chain is differentiable, including the solves.

---

*Previous: [Chapter 2 — The Lens object and sequence parsing](02_lens_object.md)*
*Next: [Chapter 4 — Ray initialization](04_ray_initialization.md)*
