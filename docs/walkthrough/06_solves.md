# Chapter 6 — Paraxial solves: constraints that cost nothing

> **What this chapter answers.** Why is the EFL exactly 10.0 and not 9.9997? What is a
> "solve" and why is it better than a penalty term? Where does the closed-form formula
> at `optics.py:547` come from? What breaks if you leave the solved variable trainable?

---

## 6.1 The idea: eliminate a variable instead of penalizing it

You want the lens to have EFL = 10 mm. Two ways to get there.

**The obvious way — a penalty.** Add a residual `(EFL − 10)` with a big weight. The
optimizer trades focal-length error against image quality and lands somewhere near 10.
Costs: one more residual, one more weight to tune, a stiffer problem (large weights
worsen conditioning), and you still only get *approximately* 10.

**The way this codebase does it — a solve.** Treat one curvature as *determined* by the
others through the requirement EFL = 10, compute it in closed form on every forward
pass, and remove it from the optimizer's variable list. Costs: nothing. The constraint
is satisfied **exactly**, by construction, at every point in the search — including at
rejected LM trial steps.

This is variable elimination, and it is standard practice in classical lens design
(commercial packages call them "solves" too). It works here because the constraint has
a **closed-form paraxial solution**, which is exactly what the ABCD machinery of
Chapter 2 provides.

## 6.2 Proof that it works

Measured on the toy system, `configs/toy/defaults.yml:31-33`:

```yaml
target_efl: 10.
solve_type: focal_length
solve_idx: -1
```

The shipped design gives `lens.efl = **10.0**` exactly (`1_config_and_lens.lens.efl`).

But "exactly 10.0 at the optimum" could be a coincidence of a converged design. Here is
the experiment that shows the constraint is *structural*. I perturbed one **free**
curvature variable and rebuilt the lens:

```
before:  c = [−0.2178221047, −0.1592064947,  0.1029609889, −0.0921629593]   EFL = 10.0
after:   c = [−0.2178221047, −0.1512064934,  0.1029609889, −0.0994226038]   EFL = 9.99999905
                              └── I moved this               └── the solve moved this
```

I changed `c[1]` by +0.008. I did **not** touch `c[3]`. The solve moved `c[3]` from
−0.09216 to −0.09942 on its own, and the EFL stayed at 10 to within float32 rounding
(`9.999999046` — one part in 10⁷).

Second confirmation, stronger. The gradient of EFL with respect to **all 20 trainable
variables**:

```
|∂EFL/∂vars|_max = 2.98e-07
```

That is float32 zero. **EFL is not a function of the optimization variables at all.**
The optimizer moves in a 20-dimensional space in which the focal length is identically
constant. It cannot violate the constraint even if the loss would reward doing so.

Contrast that with a penalty formulation, where `∂EFL/∂vars` would be large and the
optimizer would spend a real part of every step fighting it.

## 6.3 Where the formula comes from

`Lens.curvature_solve` (`optics.py:499-559`). Three steps.

**Step 1 — find the event index of the target surface** (`optics.py:521-535`). The
config gives `solve_idx = -1`, meaning "the last refractive interface". That is
normalized to a positive index in `update_curvature_inplace`
(`parameterization.py:234-236`):

```python
if solve_idx < 0:
    solve_idx = lens.sequence.n_interfaces + solve_idx     # −1 → 4 − 1 = 3
```

Then the loop walks the event list counting `type == 'r'` events until it reaches
surface 3 — event index 9 in the toy list from Chapter 2.

**Step 2 — split the system into three** (`optics.py:537-544`):

```python
abcd = self.get_abcd()
solve_abcd = abcd[i]                        # the surface being solved
abcd_left  = prt.reduce_abcd(abcd[:i])      # everything before it
abcd_right = prt.reduce_abcd(abcd[i + 1:])  # everything after it
a1, b1, c1, d1 = abcd_left...unbind(-1)
a2, b2, c2, d2 = abcd_right...unbind(-1)
mu = solve_abcd[..., 1, 1]
```

`get_abcd()` returns the **per-event** matrices (not the reduced product). They are
collapsed into a left block and a right block, leaving the unknown surface exposed in
the middle. The system matrix is `M = M_right · M_surface(c) · M_left`, and only
`M_surface` contains the unknown.

**Step 3 — solve for `c`.** The surface's refraction matrix is
`[[1, 0], [−(n₂−n₁)c/n₂, n₁/n₂]]`, i.e. linear in `c` with `mu = n₁/n₂` in the lower
right. So the total system's `C` element is an **affine function of `c`**, and the
requirement `EFL = −1/C` inverts in closed form (`optics.py:547`):

```python
c = (1 / d2 / target_efl / a1 + c2 / d2 + mu * c1 / a1) / (1 - mu)
```

No iteration. One expression, fully differentiable, evaluated fresh every forward pass.

**Note the `(1 − mu)` denominator.** If `mu = 1` — meaning `n₁ = n₂`, no index change
across the surface — the solve is singular. That is physically correct: a surface
between identical media has no optical power no matter how you curve it, so no
curvature can deliver the required focal length. The code does not guard this; a config
that solves on such a surface produces infinities. Worth knowing as a failure mode.

## 6.4 The second solve type: `image_height`

`optics.py:548-556` implements an alternative that fixes the **paraxial chief ray
height** at the image plane instead of the focal length:

```python
c = -(a1*a2*z - a2*b1 + b2*c1*mu*z - b2*d1*mu + target_efl) / (b2*(mu - 1)*(a1*z - b1))
```

where `z` is the pupil position. The docstring (`optics.py:508-514`) makes the
relationship precise and is worth quoting in a defense:

> *"Note that when A = 0 (imaging system), we have B = EFL since AD − BC = 1. In this
> case, the two solve types are perfectly equivalent."*

`A = 0` is the condition for the object plane to be conjugate to the image plane —
i.e. the system is *in focus*. So for a focused system the two solves agree; they
differ only for a defocused one. `image_height` is the more robust choice when the
image distance is itself being optimized, because it pins the actual field coverage
rather than a focal length that may not correspond to the sensor plane.

It carries an extra guard (`optics.py:553-556`): the solved surface must lie **past the
aperture stop**, because the chief-ray height calculation is only meaningful downstream
of the pupil.

## 6.5 The other two solves: the last spacing

`update_last_spacing_inplace` (`parameterization.py:242-261`) implements two more
constraints, both writing the final spacing.

**Paraxial image solve** (`paraxial_image_solve: True`):

```python
new_spacing = lens.s[-1:] + lens.bfl
```

The stored variable becomes a *defocus offset from the paraxial focus*, not an absolute
distance. Look at the matching line in `get_normalized_variables`
(`parameterization.py:742-745`) — the inverse transform:

```python
if paraxial_image_solve:
    new_last_s = lens.s[-1:] - lens.bfl
```

Subtract BFL on the way in, add it back on the way out. The optimizer's variable is
then a small number near zero (the defocus) instead of a large number near 10 (the
absolute back distance). That is much better conditioned, and it means the sensor
tracks the focal plane automatically as other parameters change.

The toy config sets `paraxial_image_solve: False`, so the last spacing is a plain free
variable here. Measured: `lens.s[-1] = 10.2154` while `lens.bfl = 10.2538` — the sensor
sits 38 µm *inside* paraxial focus. That small deliberate defocus is the optimizer
balancing on-axis against off-axis blur, and it is only possible *because* the solve is
off. With `paraxial_image_solve: True` the sensor would be pinned to paraxial focus
exactly.

**Total track length solve** (`total_track_length_solve: <mm>`):

```python
ttl_minus_last_spacing = lens.s[:-1].flip(0).cumsum(dim=0).max(dim=0, keepdims=True)[0]
new_spacing = target_ttl - ttl_minus_last_spacing
```

The last spacing absorbs whatever is left of a fixed total length. This is the
constraint that matters for the compact-device work in the paper: a phone camera has a
hard z-height budget, and this makes it structurally impossible to exceed.

The two are mutually exclusive — both write `s[-1]` — and the constructor asserts so
(`parameterization.py:75-77`), as noted in Chapter 3.

## 6.6 Order of application

`generate_lens_from_lens_parameters` (`parameterization.py:176-203`) applies solves in
a fixed order every forward pass:

```python
lens = optics.Lens(...)                                       # 1. build
if self.total_track_length_solve is not None:
    self.update_last_spacing_inplace(lens, self.total_track_length_solve)   # 2. TTL
if self.solve_idx is not None:
    self.update_curvature_inplace(lens)                       # 3. curvature
if self.paraxial_image_solve:
    self.update_last_spacing_inplace(lens)                    # 4. image
return lens
```

The order is not arbitrary. TTL comes **before** the curvature solve because changing
the last spacing changes the ABCD product the curvature solve reads. The paraxial image
solve comes **last** because it depends on BFL, which depends on the curvature just
solved. Getting this order wrong would give a lens that satisfies neither constraint
exactly.

Note also `update_curvature_inplace` (`parameterization.py:238-240`) does *not* mutate
in place despite its name:

```python
lens.c = torch.cat((lens.c[:solve_idx], new_c[None, ...], lens.c[solve_idx + 1:]), dim=0)
```

It rebuilds the tensor by concatenation. A true in-place write into a tensor that
requires grad would break autograd; concatenation keeps the graph intact and lets the
gradient flow through `new_c` back to the *other* curvatures that determined it. That
is why the perturbation experiment in §6.2 works — `c[3]` is a differentiable function
of `c[0..2]` and the spacings.

## 6.7 Why the solved variable must be frozen

Recall `configs/toy/defaults.yml:21-23`:

```yaml
c:
  default: False
  toggle_row_col_list: [-1]     # freeze the last curvature
```

`solve_idx: -1` and the frozen index are **the same surface**, and that is mandatory.
If you left `c[-1]` trainable:

1. The optimizer would get a Jacobian column for it. But the solve overwrites its value
   on every forward pass, so perturbing it changes nothing downstream — the column
   would be **identically zero**.
2. A zero column makes the least-squares system rank-deficient by one, on top of any
   genuine degeneracy. `gelsd` would still return an answer (it is a
   minimum-norm SVD solve), but the reported rank drops and the diagnostics mislead.
3. The step direction for that variable is meaningless, and the LM damping term
   computed from a zero-norm column hits the `damped_term_min` floor.

Nothing crashes — it just wastes a column and corrupts the rank diagnostic that
Chapter 9 uses to reason about degeneracy. **The freeze list and the solve index must
be kept consistent by hand; nothing in the code checks it.** That is a genuine trap when
writing a new config, and a good thing to be able to say you noticed.

## 6.8 Solves versus the residual library

Both enforce requirements. The division of labour:

| | solve | residual penalty |
|---|---|---|
| satisfied | exactly, always | approximately, at convergence |
| cost | one eliminated variable | one weight to tune |
| needs | closed-form paraxial inverse | nothing |
| suits | equality constraints on paraxial quantities | inequality / ray-based constraints |

You cannot write a solve for "no ray exceeds 60° incidence" — there is no closed-form
curvature that guarantees it. That is inherently a penalty, and it is what Chapter 7 is
about.

## 6.9 What to take away

1. A solve **eliminates** a variable rather than penalizing a deviation; the constraint
   holds exactly at every point of the search, including rejected trial steps.
2. Verified: perturbing a free curvature makes the solve move the solved curvature
   automatically, and `|∂EFL/∂vars|_max = 3e−7` — EFL is constant on the search space.
3. The closed form exists because a refraction matrix is affine in `c`, so the system's
   `C` element is affine in `c` and `EFL = −1/C` inverts directly.
4. `mu = 1` (no index change) makes the solve singular — physically correct, unguarded.
5. `image_height` and `focal_length` coincide when `A = 0` (focused system).
6. Solve order is TTL → curvature → paraxial image, because each depends on the
   previous.
7. The solved curvature **must** appear in the freeze list; nothing enforces this, and
   violating it silently produces a zero Jacobian column.

---

*Previous: [Chapter 5 — Ray tracing](05_ray_tracing.md)*
*Next: [Chapter 7 — The residual library](07_residuals.md)*
