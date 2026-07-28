# Appendix A — Supervisor defense bank, part 1 (Q1–Q25)

> **How to use this.** Each answer is short enough to say out loud and carries the file and
> line that backs it. Nothing here is a paraphrase of a docstring — every number was measured
> by running the code (Chapters 14–15) or read off the source.
>
> Part 1 covers the method and the optics. Part 2 (Q26–Q50) covers optimization, imaging,
> training, and your own implementation.
>
> **The pattern that works:** state the answer, name the file:line, give the measured number.
> Three sentences. If you can do that, you have understood the code.

---

## The method

**Q1. In one sentence, what problem does this paper solve?**

Levenberg–Marquardt needs a *vector* of residuals with `k ≫ n` to work, but an end-to-end
image-quality loss is a *scalar* — so LM cannot be used for task-driven lens design. GTRA
converts the scalar into a vector of per-ray residuals with the same value and gradient at the
current iterate, restoring the structure LM needs. `imaging_system.py:474–513`.

**Q2. Why not just use Adam on the scalar loss?**

You can, and the paper's Table 1 shows it converging to a worse design. The reason is
structural: a lens has ~20 heterogeneous variables (curvatures in mm⁻¹, spacings in mm,
aspheric coefficients spanning orders of magnitude) and LM's Marquardt damping is
scale-invariant by construction while Adam's per-parameter scaling is learned. Measured: the
reference's damping terms span `0.3719` to `1501.38` across 20 variables — four orders of
magnitude, which LM handles by design.

**Q3. What exactly is `k ≫ n` in your run?**

`k = 5704`, `n = 20`. Ratio 285. Measured, Chapter 14 §14.3. The 5704 is 5632 GTRA residuals
(2 × 11 fields × 1 wavelength × 256 pupil points) plus 72 geometric constraints.

**Q4. Write down the GTRA lift.**

```
L_TD(x) ≈ L₀ + ∇L₀ᵀ(ε − ε₀) ≈ ½ w ‖ε − ε′‖²
w  = ‖∇L₀‖²/(2L₀)
ε′ = ε₀ − 2L₀∇L₀/‖∇L₀‖²
ℓ_GTRA = √w (ε − ε′)
```

`imaging_system.py:484` is `‖∇L₀‖²`, `:487` is `ε′`, `:510` is `√w`, `:513` is `ℓ_GTRA`.

**Q5. Prove the lift preserves the loss value.**

Substitute: `½w‖ε₀ − ε′‖² = ½ · (‖∇L₀‖²/2L₀) · ‖2L₀∇L₀/‖∇L₀‖²‖² = ½ · (‖∇L₀‖²/2L₀) ·
4L₀²/‖∇L₀‖² = L₀`. Measured in Chapter 15: `½‖ℓ‖² = 0.004496567` against `L₀ =
0.0044965702`, relative error **7.249e−07**.

**Q6. And the gradient?**

`∂/∂ε [½w‖ε−ε′‖²] = w(ε−ε′)`, and at `ε = ε₀` that is `w · 2L₀∇L₀/‖∇L₀‖² = ∇L₀`, since
`w · 2L₀/‖∇L₀‖² = 1` by construction. So the surrogate is a **first-order match** — same value,
same gradient, at the current iterate only.

**Q7. Where does the approximation break down?**

As `ε` moves away from `ε₀`. Quantitatively: Chapter 15 measured `xy_control_shift` reaching
`0.0445 mm` against a corner RMS spot of `0.0300 mm` — the virtual target sits about **1.5 spot
radii** from the current spot. That is why the residual is rebuilt every LM iteration
(`imaging_system.py` `training_step`, and see Q41).

**Q8. Is `ε′` a physically meaningful target?**

No, and that is the point. It is a *virtual* spot diagram — where the rays would have to land
for the image loss to be zero under a first-order model. It can lie outside the PSF grid, which
is why the code clips it (`imaging_system.py:489-507`).

**Q9. What does the clipping compensation factor do?**

Clipping `ε′` changes `‖ε − ε′‖`, which would break Q5's identity. So the code computes
`compensation_factor = ‖ε−ε′‖²/‖ε−ε′_clipped‖²` and folds it into the weight at `:510`, so
`½‖ℓ‖² = L₀` still holds exactly. **This is in the code and not in the paper** — a good answer
to "what did the implementation add?"

**Q10. How is `∇L₀` computed, and how many passes?**

**One** backward pass. `imaging_system.py:477-479` uses `torch.func.grad_and_value` on the
scalar loss with respect to the spot diagram — so the expensive part (PSF build, convolution,
restoration network) is traversed once per LM iteration, not once per variable.

**Q11. And the Jacobian `J`?**

**Forward-mode** AD, and only through the ray tracer. Because `w` and `ε′` are constants
during the LM solve, `J = √w · ∂ε/∂x`, and forward-mode costs `n` tangent passes — `n = 20`
here, not `k = 5704`. The paper (§3.5, "Derivatives") says explicitly that forward-mode is
*not* applied to the image pipeline. **This asymmetry — backward through the image pipeline
once, forward through the ray tracer `n` times — is the computational heart of the method.**

**Q12. If someone says "this is just a Gauss-Newton approximation of the image loss," are they
right?**

Partly. The lift *is* a quadratic model of the scalar loss, but not the Gauss-Newton model of
it — a true Gauss-Newton on `L_TD` would need the Jacobian of the whole image pipeline with
respect to `x`. GTRA instead builds a quadratic in `ε` (the ray coordinates) that matches
`L_TD` to first order, then applies Gauss-Newton in `x` to *that*. The saving is that the
image pipeline is differentiated once, w.r.t. `ε`, not `n` times w.r.t. `x`.

**Q13. What is the relationship between GTRA and classical TRA?**

TRA is the special case `w = 1/(fwp)` and `ε′ = ε̄` (the centroid). Paper §3.3, and
`residuals.py:82` is `xy - xy_centroid`. Your own implementation makes this executable:
`tra_control_values` (`gtra.py:99`) returns exactly that `(w, ε′)` pair so TRA runs through
the GTRA code path.

**Q14. Why does the code variable `weight_w` not equal the paper's `w`?**

It stores `√w`, because it multiplies the residual while the paper's `w` multiplies the squared
norm. Verified: `√0.02474414833 = 0.15730273` against measured `weight_w = 0.1573027223`,
relative error 3.81e−08. **Say which one you mean.** In your own code the variable `w` holds
`w` and you take the square root separately (`gtra.py:73`, `:166`) — your naming matches the
paper.

---

## Optics and ray tracing

**Q15. What is the shape of a spot diagram in this code, and why that shape?**

`[2, f, w, p, 1]` — measured `[2, 11, 1, 256, 1]`. Leading `2` is `(x, y)`; the trailing `1`
is a broadcast slot for the lens-batch dimension. Chapter 4. It flattens to `2fwp = 5632`,
which is the residual length.

**Q16. How is the lens represented?**

A **sequence string** parsed into a list of events (`optics.py`). Chapter 2. Each event is a
surface, an aperture, or a gap; the string is the single source of truth for the lens topology,
which is why the YAML can be short.

**Q17. Which dispersion model, and why not a Sellmeier fit?**

Hartmann three-parameter, `n(λ) = A + C/(λ − B)`, `optics.py:735`. The reason is
differentiability in the *design variables*: `A`, `B`, `C` are derived from `(nd, vd, dpgf)`
at `:771-774`, so the optimizer can move a glass continuously through index/Abbe space.
Sellmeier coefficients have no such smooth relationship to catalog glass position.

**Q18. What are the glass design variables and what happens at the end?**

`(nd, vd, dpgf)` — index, Abbe number, deviation from normal partial dispersion. They are
optimized continuously, penalized toward the catalog mesh by `GlassVariableResiduals`
(`residuals.py:340`) with a ramped weight, then snapped to real catalog glasses **one at a
time** by `BindMaterialsCallback` (`callbacks.py:151`), with the optimizer state reset after each snap
because LM damping is meaningless after a discontinuous jump. Chapter 13 §13.6.

**Q19. Why one at a time?**

So the remaining free glasses can compensate for each snap. Snapping all of them
simultaneously would land on a design nobody optimized.

**Q20. Where is the conic constant stored?**

`a[..., 0]` — the *first* "aspheric coefficient" is the conic constant, documented at
`optics.py:199`. So a `[4, 1, 4]` aspheric tensor is 4 surfaces × (1 conic + 3 coefficients).
That is why the paper says "12 aspheric coefficients" for 16 stored values.

**Q21. How does the tracer find the ray–surface intersection on an aspheric surface?**

Iterative marching. `ray_tracing.py` `approximate_marching_distance` (`:160`) gives a starting
guess from the spherical base curve, then `refine_marching_distance` (`:221`) Newton-iterates
on the sag residual. A closed form exists only for spheres (`:42`).

**Q22. How many ray-trace events in your configuration, and what happens at each?**

Measured 21 events, Chapter 14 §14.5. The per-event table there shows `r.max` growing from the
entrance to 4.98 mm at the sensor, with `ray_status` staying 0 (all rays valid) throughout.
Each refractive event applies the marching solve then Snell (`apply_snell_aspherical`,
`:383`).

**Q23. What is `ray_status` for?**

A per-ray integer flag marking failures — total internal reflection, missed surface, outside
aperture. Failed rays get `inf` coordinates and are excluded from centroids and residuals. In
the toy trace it stays 0 everywhere; on a bad design it does not, and that is the mechanism
that keeps a diverging optimizer from producing NaN residuals.

**Q24. Explain the solves. Why not just optimize the last curvature directly?**

Because the focal length is a hard constraint, not a soft one. The last curvature is *solved*
each trace to hold EFL (or image height), so it is a function of the other variables rather
than a variable. `optics.py` `curvature_solve`, Chapter 6. The solve is differentiable and
applied inside the trace, so the Jacobian includes the `(∂ε/∂c_L)(∂c_L/∂x_i)` term — detaching
it would give LM a Jacobian that disagrees with the trace it predicts.

**Q25. Why are the solves closed-form and not iterative?**

Because they are paraxial. `paraxial_ray_tracing.py` (97 lines) reduces the system to a
cumulative ABCD product (`reduce_abcd_cumulative`), and the EFL constraint becomes a linear
equation in the last curvature. Chapter 6. That is also why the solve costs nothing per trace.

---

*Next: [Appendix A part 2 — Q26–Q50](A_defense_bank_part2.md)*
*Back to: [Chapter 17](17_your_implementation.md) · [Index](README.md)*
