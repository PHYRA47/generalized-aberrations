# Chapter 9 (part 2) — Accept, reject, and the measured convergence run

> **What this chapter answers.** How does LM decide whether to keep a step? Why is
> `tolerance = 1.0` in the toy config a strict setting? What does a real 12-iteration run
> look like, iteration by iteration? And why does a step get *rejected* at the shipped
> design?

---

## 9.8 Trial-and-revert

`optimizers.py:129-153`. This is the adaptive part of the algorithm.

```python
if step is not None:
    trainable_params = [v for v in group["params"] if v.requires_grad]
    assert len(trainable_params) == 1, "LMOptimizer does not support multiple parameter tensors"
    p = trainable_params[0]
    p_copy = p.data.clone()          # 1. save
    p.data.add_(step)                # 2. apply

    with torch.inference_mode(True):
        updated_loss = closure.evaluate_least_squares_loss()   # 3. re-evaluate
    loss_ratio = (updated_loss / scalar_loss).item()
    logs["loss_ratio"] = loss_ratio
    if not math.isfinite(loss_ratio) or loss_ratio > group["tolerance"]:
        p.data = p_copy              # 4. revert if worse
    if loss_ratio <= 1.0:
        lam = lam / group["lam_decrease_factor"]
    else:
        lam = lam * group["lam_increase_factor"]
    group["lm_parameter"] = np.clip(lam, group["lam_eps"], 1 / group["lam_eps"])
    group["damping_terms"] = damping_terms
```

Save, apply, re-evaluate, revert if worse. That structure is why the whole codebase
insists on never raising and never producing non-finite values (Chapter 5): the
optimizer *deliberately* proposes steps that may be bad, and it needs a finite number
back to judge them.

Points worth noting:

**`torch.inference_mode(True)` around the re-evaluation.** This is a trial evaluation
only — no graph is needed. Inference mode is stronger than `no_grad`: it skips version
counting and view tracking, so the trial forward pass is meaningfully cheaper. This is
also why Chapter 8's GTRA code has to explicitly re-enable it
(`imaging_system.py:475`) — it may be called from inside this block.

**`not math.isfinite(loss_ratio)`** is checked first. A step producing NaN or Inf is
rejected on that ground alone. Combined with `reset_bad_rays`, a catastrophic trial
design costs one wasted evaluation and nothing more.

**Only `p.data` is touched, never `p`.** Assigning to `.data` bypasses autograd
bookkeeping entirely, which is correct — the optimizer's own updates should not appear
in any graph. It is also why `@torch.no_grad()` decorates `step` (`optimizers.py:49`).

**Reverting is `p.data = p_copy`**, a rebind rather than a copy into the existing
storage. Slightly wasteful, functionally identical.

## 9.9 The two thresholds are different, and that matters

Read lines 143 and 146 side by side:

```python
if not math.isfinite(loss_ratio) or loss_ratio > group["tolerance"]:
    p.data = p_copy          # revert uses `tolerance`
if loss_ratio <= 1.0:
    lam = lam / group["lam_decrease_factor"]    # damping uses 1.0
else:
    lam = lam * group["lam_increase_factor"]
```

**Acceptance** is governed by `tolerance`; **damping** is governed by `1.0`. With the
constructor default `tolerance = 2.0`, there is a window:

```
1.0 < loss_ratio ≤ 2.0   →   step is KEPT (loss got worse!) but λ is INCREASED
```

That is deliberate non-monotonicity: allowing the loss to rise by up to 2× lets the
optimizer escape a shallow local basin, while the simultaneous damping increase makes
the next step shorter and safer. It is a mild trust-region relaxation.

**The toy config sets `tolerance: 1.` (`configs/toy/defaults.yml`)**, which closes the
window entirely. Accept only if the loss did not increase; monotone descent. This is the
strict, conservative setting, and it is appropriate for a design already near its
optimum. If you were optimizing from a poor starting point, the looser default would be
the better choice.

Toy config also sets `lam_eps: 1e-12` (vs. the `1e-6` default), permitting `λ` to range
over `[1e−12, 1e12]` — twelve orders of magnitude in each direction, so the method can
go essentially pure Gauss–Newton near convergence.

## 9.10 One step at the shipped optimum

Measured (`5_lm_step`) at `configs/toy/designs/optimized_spot.yml`:

| quantity | value |
|---|---|
| `residual_vector` | `[5704]`, range ±1.4e−3 |
| `least_squares_loss` (= ½‖l‖²) | **1.418987e−4** |
| `jacobian` | `[5704, 20]`, range −485.8 … +383.9 |
| `n_constraints` | **0** |
| `damping_terms` | `[20]`, 0.3719 … 1501.38 |
| `augmented_matrix` | `[5724, 20]` |
| `lstsq_rank` | **14** |
| `step_delta_x` | `[20]`, range −2.7e−8 … +5.6e−8 |
| `gradient_JT_r` | `[20]`, range −3.8e−4 … +4.0e−4 |
| `loss_before` | 1.418987e−4 |
| `loss_after_step` | 1.418987e−4 |
| `loss_ratio` | **1.0000001192** |
| `step_accepted` | **False** |
| `lam_next` | **2.0** |

Read this carefully, because it is initially surprising.

**The step is ~1e−8 in magnitude** — utterly negligible. **The loss ratio is
1.0000001** — the loss changed in the eighth significant figure. **The step is
rejected** because `1.0000001 > tolerance = 1.0`, and `λ` doubles from 1.0 to 2.0.

This is not a failure. **This is what convergence looks like.** The design is at a local
optimum; the gradient `Jᵀl` is ~1e−4 against Jacobian entries of ~500, i.e. essentially
zero; the proposed step is at the level of float32 noise; and the resulting loss change
is pure rounding, which under the strict `tolerance = 1.0` registers as "not an
improvement" and is rejected.

The `step_dot_gradient` is logged as `−0.0` — negative (a descent direction) but
rounded to zero at float32. Consistent with everything else.

**If you show your supervisor a single LM step on the shipped config, it will be
rejected, and you should be able to explain immediately that this is the correct
behaviour of a converged design under a strict acceptance rule.**

## 9.11 A real convergence run

A single rejected step is not a demonstration of the algorithm. So I perturbed all 20
trainable variables by 5% (fixed seed) and ran 12 iterations
(`instrument_trace.py`, section `9_lm_convergence`). Every number below is measured.

| it | loss before | loss after | ratio | λ | rank | ‖Δx‖ | accepted |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 1.609285 | 1.384984 | 0.8606 | 1.000000 | 20 | 0.0859 | ✓ |
| 1 | 1.384984 | 1.265769 | 0.9139 | 0.500000 | 20 | 0.0793 | ✓ |
| 2 | 1.265769 | 1.077678 | 0.8514 | 0.250000 | 20 | 0.4443 | ✓ |
| 3 | 1.077678 | 1.045608 | 0.9702 | 0.125000 | 20 | 0.1556 | ✓ |
| 4 | 1.045608 | 0.698053 | 0.6676 | 0.062500 | 20 | 1.0527 | ✓ |
| 5 | 0.698053 | 0.444370 | 0.6366 | 0.031250 | 20 | 1.4287 | ✓ |
| 6 | 0.444370 | 0.369469 | 0.8314 | 0.015625 | 20 | 0.2967 | ✓ |
| 7 | 0.369469 | 0.350848 | 0.9496 | 0.007813 | 20 | 0.4189 | ✓ |
| 8 | 0.350848 | 0.173532 | 0.4946 | 0.003906 | 20 | 0.3540 | ✓ |
| 9 | 0.173532 | 0.089286 | 0.5145 | 0.001953 | 20 | 0.3449 | ✓ |
| 10 | 0.089286 | 0.085794 | 0.9609 | 0.000977 | 20 | 0.4671 | ✓ |
| 11 | 0.085794 | 0.095251 | 1.1102 | 0.000488 | 20 | 0.4593 | ✗ |

Summary: **11 of 12 steps accepted; loss 1.609285 → 0.095251, a 16.9× reduction in
twelve iterations.**

Five things this table teaches.

**1. λ halves on every success.** 1 → ½ → ¼ → … → 1/2048. Each acceptance is evidence
that the quadratic model is trustworthy, so the method reduces damping and moves closer
to pure Gauss–Newton. By iteration 10, `λ ≈ 1e−3` and the step is essentially
Gauss–Newton.

**2. Step length grows as damping falls.** ‖Δx‖ goes 0.086 → 0.079 → 0.444 → … → 1.43.
Larger steps are permitted precisely because the model has proven reliable. Compare the
1e−8 step at the converged design in §9.10 — that is small because the *gradient* is
zero, not because damping is high. Two different reasons for a small step.

**3. Convergence is not monotone in rate.** Ratios wander: 0.86, 0.91, 0.85, 0.97, then
a sudden 0.67, 0.64, then 0.83, 0.95, then 0.49, 0.51. The near-1.0 ratios at
iterations 3, 7, 10 are the method probing a flat direction; the sharp drops follow once
damping is low enough to move decisively. **This is the characteristic rhythm of LM** —
periods of little progress punctuated by large drops.

**4. Rank is 20 throughout the run.** Away from the optimum every variable has an
independent effect. Compare rank 14 at the shipped optimum (§9.6): the degeneracy
appears **only at convergence**. This is the clean confirmation that the rank drop is a
property of the converged design rather than a defect in the parameterization.

**5. Iteration 11 rejects.** Ratio 1.1102 > 1.0, so the parameters revert and λ doubles.
The run has essentially converged: from 0.0893 to 0.0858 to a rejected 0.0953. Given
more iterations, λ would climb until the steps became small enough to make progress
again, or the run would sit at the optimum as in §9.10.

## 9.12 What the optimizer does *not* handle

Both assertions in `step` are worth knowing:

```python
assert len(self.param_groups) == 1, "LMOptimizer does not support per-parameter options"
assert len(trainable_params) == 1, "LMOptimizer does not support multiple parameter tensors"
```

`optimizers.py:59-61, 132-134`. **Exactly one flat parameter tensor.** This is why
Chapter 3's parameterization goes to the trouble of packing everything into a single
1-D `_lens_variables` vector with a scatter/gather scheme. The optimizer's design forces
the parameterization's design.

It also means the LM optimizer **cannot** optimize the image restoration network. A
NAFNet has hundreds of parameter tensors. That is a structural reason — not just a
convenience — for the two-optimizer split in Chapter 12: LM for the lens, Adam for the
network.

Also note `closure = closure.args[-1]` at `optimizers.py:56`, with the comment *"Remove
wrapper"*. Lightning wraps the closure in a `functools.partial`; this unwraps it to
reach the real `CustomClosure` with its `get_least_squares_quantities` method. A
compatibility shim, and a fragile one — it depends on Lightning's internal wrapping
convention.

## 9.13 Cost per iteration

Worth being able to quantify:

| operation | cost |
|---|---|
| Jacobian (`jacfwd`) | **20** forward traces |
| `lstsq` on `[5724, 20]` | negligible (µs on CPU) |
| trial evaluation | **1** forward trace |
| **total** | **~21 forward passes per iteration** |

Compare Adam: 1 forward + 1 backward ≈ 2 passes per iteration. So one LM iteration costs
about 10× one Adam iteration — but the measured run reduced the loss 16.9× in **12**
iterations. Adam on a problem this stiff would need hundreds to thousands. The
arithmetic favours LM by a wide margin, and that is the practical case for the paper's
whole approach.

When the end-to-end loss is active (Chapter 8), each of those 21 passes also drives the
PSF and image pipeline — which is exactly why the GTRA construction linearizes it once
per iteration rather than re-evaluating it inside every trial.

## 9.14 What to take away

1. 20 variables, 5704 residuals: heavily overdetermined, badly scaled — the regime where
   Gauss–Newton wins and first-order methods struggle.
2. The Jacobian uses **forward mode** (`jacfwd`): 20 JVPs instead of 5704 VJPs, a 285×
   saving. The opposite of the usual deep-learning choice, for the opposite reason.
3. Damping is scaled by **column norms of `J`**, giving scale-equivariance
   (Marquardt's refinement). Measured spread 0.37 … 1501.
4. The damping update is a **ratchet**, not an EMA: rises instantly, decays 1% per
   iteration.
5. The step is solved as a **stacked least-squares** problem, never by forming `JᵀJ` —
   avoiding squared conditioning. `gelsd` on CPU, returning minimum-norm solutions.
6. **Rank 14 of 20** at the shipped optimum, **20** throughout a perturbed run: genuine
   degeneracy that appears only at convergence.
7. Save–apply–re-evaluate–revert. Acceptance uses `tolerance`; damping uses `1.0`. The
   toy config's `tolerance: 1.` enforces monotone descent.
8. **Measured convergence: 11/12 steps accepted, loss 1.609 → 0.0953 (16.9×), λ halving
   on every success.** At the shipped optimum a step is *rejected* with ratio
   1.0000001 — the correct behaviour of a converged design.
9. Exactly one parameter tensor is supported, which is why the parameterization packs a
   flat vector and why the restoration network needs a separate optimizer.

---

*Previous: [Chapter 9 part 1 — the Jacobian and the step](09_lm_optimizer_part1.md)*
*Next: [Chapter 10 — PSF simulation and image formation](10_psf_simulation.md)*
