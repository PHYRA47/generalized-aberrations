# Chapter 9 (part 1) — Levenberg–Marquardt: the Jacobian and the step

> **What this chapter answers.** Why LM and not Adam? Why is the Jacobian computed in
> **forward** mode? Why is `JᵀJ` never formed in the branch that actually runs? What is
> the damping term, and why the elaborate scaling? Part 2 covers accept/reject, the
> damping schedule, and the measured convergence run.

---

## 9.1 Why a second-order method at all

Chapter 7 established the form:

```
minimize   L(x) = ½ ‖l(x)‖²,     x ∈ ℝ²⁰,  l ∈ ℝ⁵⁷⁰⁴
```

Twenty variables, five thousand seven hundred residuals. **Vastly overdetermined**, and
that is typical of lens design — you cannot satisfy 5704 conditions with 20 knobs, you
can only balance them.

For this shape of problem, first-order methods are the wrong tool. The variables have
wildly different scales and couplings: curvature affects everything, an eighth-order
aspheric coefficient affects the rim only. The Hessian's condition number is enormous.
Adam's per-parameter adaptive scaling is a crude diagonal approximation to what
Gauss–Newton computes properly.

The Gauss–Newton step comes from linearizing the *residual*, not the loss:

```
l(x + Δ) ≈ l(x) + JΔ
L(x + Δ) ≈ ½‖l + JΔ‖²
```

Setting the derivative to zero: `(JᵀJ)Δ = −Jᵀl`. **`JᵀJ` is a genuine curvature
estimate built from first derivatives only** — that is what makes least-squares special.
For a general scalar objective you would need the true Hessian.

Gauss–Newton alone is unstable when `JᵀJ` is singular or the linearization is poor.
Levenberg–Marquardt adds damping:

```
(JᵀJ + λD)Δ = −Jᵀl
```

with `λ → 0` recovering Gauss–Newton (fast, trusting the model) and `λ → ∞` giving a
short step along `−Jᵀl` (safe gradient descent). **λ interpolates between the two, and
it is adapted every iteration** based on whether the last step actually helped. That
adaptation is the whole algorithm; §9.7 in part 2 shows it happening.

## 9.2 Getting the Jacobian: forward mode

`CustomClosure.get_least_squares_quantities` (`imaging_system.py:1075-1102`):

```python
def compute_residual_vector(p):
    return torch.func.functional_call(
        self.loss_wrapper, dict(zip(parameter_dict.keys(), p)), self.batch)

parameter_dict = {k: v.detach() for k, v in self.loss_wrapper.named_parameters()
                  if v.requires_grad}
assert len(parameter_dict) == 1, "The module should have only one trainable parameter tensor."
parameters = tuple(parameter_dict.values())
# Forward-mode differentiation
jacobian = torch.func.jacfwd(compute_residual_vector, randomness="same")(parameters)[0]
```

**`jacfwd`, not `jacrev`.** This is a deliberate and correct choice, and a very likely
supervisor question.

| mode | cost | best when |
|---|---|---|
| reverse (`jacrev`, standard backprop) | one pass **per output row** | few outputs, many inputs |
| forward (`jacfwd`) | one pass **per input column** | many outputs, few inputs |

Here: **5704 outputs, 20 inputs.** Reverse mode would need 5704 backward passes.
Forward mode needs 20 JVPs. That is a **285× difference**, and it is why LM on a lens is
tractable at all.

This is the mirror image of the usual deep-learning situation (millions of parameters,
one scalar loss), where reverse mode wins by the same argument in the other direction.
Neural networks made backprop synonymous with autodiff; least-squares optical design is
exactly the regime where the opposite is true.

`randomness="same"` forces every JVP to use the same random draw. If the pupil sampler
re-randomized per column, each Jacobian column would be a derivative of a *different*
function and the matrix would be meaningless. (Chapter 4 established the jitter is
deterministic anyway; this is belt and braces.)

The `functional_call` wrapper is what lets `jacfwd` work on an `nn.Module` — parameters
are passed as an explicit argument rather than read from module state, so the
transformation can substitute dual numbers.

**Measured:** `jacobian: [5704, 20]`, values spanning −485.83 to +383.87. Rows =
residuals, columns = trainable variables — exactly the 20 from Chapter 3.

## 9.3 Splitting objectives from constraints

`optimizers.py:73-78`:

```python
residual_jacobian = jacobian[~constraint_mask]
constraint_jacobian = jacobian[constraint_mask]
residuals = loss[~constraint_mask]
constraints = loss[constraint_mask]
n_variables = jacobian.shape[1]
```

The mask built in Chapter 7 slices the system into an objective part (minimize) and a
constraint part (drive to zero exactly).

**Measured: `n_constraints = 0`.** As established in Chapter 7, no shipped residual sets
`constraint = True`, so `constraint_mask` is all-`False`, `constraints.numel() == 0`,
and the branch at `optimizers.py:93` is never taken. Everything below about the KKT
branch is code reading, not observed behaviour — be honest about that if asked.

## 9.4 The damping terms

`optimizers.py:80-91`:

```python
damping_terms = residual_jacobian.norm(dim=0)   # sqrt of diagonal of pseudo-Hessian
if group["damping_terms"] is None:
    damping_terms = damping_terms.clamp(min=min_damp)
else:
    beta = group["beta"]
    damping_terms = (beta * torch.max(damping_terms, group["damping_terms"])
                     + (1 - beta) * damping_terms)
```

`J.norm(dim=0)` is the column norms — and `‖J[:,i]‖² = (JᵀJ)ᵢᵢ`, so these are the square
roots of the pseudo-Hessian diagonal, as the comment says.

**Why scale the damping per variable.** The naive choice `D = I` damps every variable
equally, which is only sensible if the variables are commensurate. They are not: a
curvature in mm⁻¹ and a spacing in mm have different units and utterly different
sensitivities. Using `D = diag(JᵀJ)` makes the damping **scale-equivariant** — this is
Marquardt's original 1963 refinement over Levenberg's `λI`, and it is why the method
carries both names.

**Measured:** `damping_terms: [20]`, from **0.3719** to **1501.38**. Four orders of
magnitude between the least and most sensitive variable. Uniform damping would be
hopeless here.

Two guards:

**`clamp(min=min_damp)`** on the first iteration only. Toy config sets
`damped_term_min: 1e-8`. A variable with a near-zero column would otherwise get
near-zero damping and take an unbounded step. Note the constructor default is `1e-4`
(`optimizers.py:17`) — the toy config deliberately loosens it by four orders of
magnitude, which is safe here because the design is already near-optimal.

**The running maximum with `beta = 0.99`** on every later iteration:

```
D ← 0.99 · max(D_new, D_old) + 0.01 · D_new
```

Read the `max` carefully — this is a **ratchet**. If a column norm suddenly collapses,
`max` holds the old larger value and the `(1−β)` term lets it decay only 1% per
iteration. Damping can rise instantly but falls slowly. This is a stability device: a
variable that momentarily looks insensitive does not immediately get permission to take
a huge step. The asymmetry is intentional and worth pointing out — it is not a plain
exponential moving average.

## 9.5 Solving for the step — the branch that runs

`optimizers.py:114-127`:

```python
matrix = torch.cat((residual_jacobian, np.sqrt(lam) * damping_terms.diag_embed()), dim=0)
bb = torch.cat((residuals, torch.zeros_like(damping_terms)), dim=0)
try:
    lstsq = torch.linalg.lstsq(matrix.cpu(), -bb.cpu(), driver="gelsd")
    step = lstsq[0].to(matrix.device)
    logs["rank"] = float(lstsq.rank.item())
except RuntimeError:
    warnings.warn("Least-squares solver failed; step ignored")
    step = None
```

Note the comment on `optimizers.py:115`: *"Pseudo-Hessian is not computed explicitly; we
solve the least-squares problem directly."*

**This is the numerically important detail of the whole optimizer.** Instead of forming
`JᵀJ + λD` and solving that `20×20` system, the code stacks

```
        ⎡      J       ⎤            ⎡  l  ⎤
  A  =  ⎢              ⎥ ,     b = −⎢     ⎥
        ⎣ √λ · diag(d) ⎦            ⎣  0  ⎦
```

and solves `min ‖AΔ + b‖²` directly. The normal equations of this stacked system are

```
(JᵀJ + λ D²) Δ = −Jᵀl
```

— identical mathematically. But **the condition number of `A` is the square root of the
condition number of `AᵀA`.** Forming `JᵀJ` squares the conditioning and can destroy
half the available precision. In float32 with `κ(J) ≈ 10³`, `κ(JᵀJ) ≈ 10⁶` and you are
close to losing everything. Solving the stacked system avoids that entirely.

Confirm with the measured numbers: `pseudo_hessian_JTJ` spans **−2.13e6 to +2.25e6**
while `jacobian` spans **−486 to +384**. The squaring is visible in the data. The code
never forms that matrix in the branch that runs.

**Measured:** `augmented_matrix: [5724, 20]` — that is `5704 + 20`, the Jacobian with
the `20×20` damping block appended.

Three more details:

**`.cpu()` on both arguments** (`optimizers.py:122`) with the comment *"Least-squares
solver on CPU is more reliable"*. The `gelsd` driver's GPU implementations have known
robustness issues on rank-deficient systems. A `[5724, 20]` solve is microseconds on
CPU; the transfer costs more than the arithmetic. Pragmatic, and honest about why.

**`driver="gelsd"`** — the SVD-based LAPACK driver with divide-and-conquer. Unlike the
default QR driver, `gelsd` handles rank-deficient systems by returning the
**minimum-norm** solution rather than failing. Which brings us to:

**`lstsq.rank`** is logged. This is a diagnostic you should know how to read, and §9.6
is about what it says.

**A caught `RuntimeError` sets `step = None`** and warns rather than crashing. Same
philosophy as Chapter 5's ray handling: a numerical failure in one iteration must not
kill a long optimization run. The iteration is simply skipped, and note that `λ` is
*not* updated in that case (the update at `optimizers.py:152` is inside
`if step is not None`).

## 9.6 The rank diagnostic — a real finding

Measured at the shipped optimum:

```
lstsq_rank : 14.0        (out of 20 variables)
```

Measured after perturbing the design by 5%:

```
rank : 20                (every iteration of the 12-step run)
```

**Six of twenty directions are numerically degenerate at the converged design, but the
system is full-rank away from it.** This is worth being precise about, because it is
easy to misread as a bug.

It is not a bug. It is the signature of a converged, well-balanced design. Near an
optimum the residual surface flattens along directions where different parameters trade
off against each other — bend one surface, compensate with the next, and the image is
unchanged. Those directions are genuine **null directions of `J`**, and their existence
means the design is not uniquely determined: a six-parameter family of nearby designs
gives essentially the same image quality.

This is exactly why `gelsd` was chosen. A rank-deficient system has infinitely many
solutions; `gelsd` returns the minimum-norm one, i.e. it takes **no** step along the
degenerate directions rather than an arbitrary large one. Any QR-based driver would
either fail or produce a wild step.

Three things follow that you can say in a defense:

1. The shipped `optimized_spot.yml` really is at a local optimum, and I can show it:
   rank drops from 20 to 14 there.
2. The degeneracy is a property of the *design*, not of the code.
3. Chapter 6's warning matters here — leaving a solved variable trainable adds a
   *spurious* rank deficiency on top of the genuine one, and you would no longer be able
   to tell them apart.

## 9.7 The unconstrained/constrained asymmetry

For completeness, the branch that never runs (`optimizers.py:93-113`):

```python
pseudo_hessian = residual_jacobian.T @ residual_jacobian
damping_matrix = (damping_terms ** 2).diag_embed()
mat = pseudo_hessian + lam * damping_matrix
b = -residual_jacobian.T @ residuals
mat = torch.nn.functional.pad(mat, (0, m, 0, m))
mat[n_variables:, :n_variables] = constraint_jacobian
mat[:n_variables, n_variables:] = constraint_jacobian.T
b = torch.cat((b, -constraints), dim=0)
lstsq = torch.linalg.lstsq(mat, b)
step = lstsq[0][:n_variables]
```

This builds the **KKT system**

```
⎡ JᵀJ + λD²   Cᵀ ⎤ ⎡ Δ ⎤   ⎡ −Jᵀl ⎤
⎢               ⎥ ⎢   ⎥ = ⎢      ⎥
⎣    C        0  ⎦ ⎣ ν ⎦   ⎣  −c  ⎦
```

with `ν` the Lagrange multipliers, discarded by the `[:n_variables]` slice. The
constraint rows enforce `c + CΔ = 0` — the constraint is driven to zero to first order
*exactly*, not penalized.

Note it **must** form `JᵀJ` explicitly here, because the bordered structure cannot be
written as a stacked least-squares problem. So this branch pays the squared-conditioning
cost the other branch avoids. It also uses the default `lstsq` driver, not `gelsd`.

Since no shipped residual sets `constraint = True`, this path is untested by any
configuration in the repository. If you ever enable it, that is the code to scrutinize
first.

---

*Previous: [Chapter 8 — Generalized transverse ray aberrations](08_gtra.md)*
*Continue: [Chapter 9 part 2 — accept/reject and measured convergence](09_lm_optimizer_part2.md)*
