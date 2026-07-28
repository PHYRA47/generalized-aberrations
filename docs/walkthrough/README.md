# `eisoptx` — a line-level walkthrough

A teaching walkthrough of the reference implementation of Côté et al. (2026), *Generalized
Aberrations for Processing-Aware Optical Design* (ACM TOG).

**Who this is for.** Someone who needs to sit in front of a supervisor, put a finger on a line
of Python, and say *this line is Eq. 5 of the paper, here is why it is written this way and not
the obvious way, and here is what breaks if you change it.* That is the bar throughout.

**What makes it different from a docstring tour.** Every shape, value, and number quoted in
these chapters was **measured by running the code**, not paraphrased from a comment. The
instrumentation script is `instrument_trace.py`, its output is `trace_dump.json`, and
`trace_tables.md` is the human-readable rendering. Where a comment in the source is wrong —
and one is — the chapter says so.

**Total: 21 chapter files, ~7,100 lines.**

---

## Reading order

Read **0 → 1 → 2 → 4 → 5** in order once. Everything after that can be read by lookup, with
two exceptions: Chapter 8 assumes Chapter 7, and Chapter 9 assumes Chapter 8.

If you have one hour: **0, 8, 14, 16**. That gets you the orientation, the paper's core
contribution, a full forward trace with real numbers, and the equation-to-code map.

If you have a meeting tomorrow: **Appendix A**, then Chapter 16 §16.12 (the divergence table),
then Chapter 17 §17.11–17.12.

### Part I — Orientation and setup

| # | file | what it covers |
|---|---|---|
| 0 | [00_orientation.md](00_orientation.md) | the one sentence the codebase implements; file map; vocabulary; how to run it |
| 1 | [01_configuration.md](01_configuration.md) | config-as-program: jsonargparse, class paths, `link_arguments`, the five shipped configs |

### Part II — The optical model

| # | file | what it covers |
|---|---|---|
| 2 | [02_lens_object.md](02_lens_object.md) | the sequence string, event parsing, the glass model (Eq. 8) |
| 3 | [03_parameterization.md](03_parameterization.md) | which numbers are variables, packing, scaling, the trainable mask |
| 4 | [04_ray_initialization.md](04_ray_initialization.md) | field and pupil sampling, the `[2,f,w,p,1]` shape, ray aiming |
| 5 | [05_ray_tracing.md](05_ray_tracing.md) | the event loop, marching solves, Snell (Eq. 9), `ray_status` |
| 6 | [06_solves.md](06_solves.md) | paraxial ABCD, EFL/image-height/TTL solves, why they are closed-form |

### Part III — Objective and optimization

| # | file | what it covers |
|---|---|---|
| 7 | [07_residuals.md](07_residuals.md) | the residual library, TRA (Eq. 3), geometric constraints, glass mesh |
| 8 | [08_gtra.md](08_gtra.md) | **the paper's core**: the lift (Eq. 5–6), clipping, the compensation factor |
| 9a | [09_lm_optimizer_part1.md](09_lm_optimizer_part1.md) | LM structure, the stacked solve, Marquardt damping (Eq. 11) |
| 9b | [09_lm_optimizer_part2.md](09_lm_optimizer_part2.md) | acceptance, λ adaptation, the closure protocol, failure modes |

### Part IV — Imaging and restoration

| # | file | what it covers |
|---|---|---|
| 10 | [10_psf_simulation.md](10_psf_simulation.md) | KDE PSFs, rotation, diffraction (Eq. 12), SVOLA convolution (Eq. 13) |
| 11 | [11_image_restoration.md](11_image_restoration.md) | NAFNet, the parametrized Wiener stage, chaining, `detach_psfs` |

### Part V — Putting it together

| # | file | what it covers |
|---|---|---|
| 12 | [12_training_loop.md](12_training_loop.md) | manual optimization, two optimizers, the closure, caching, the CLI |
| 13 | [13_periphery.md](13_periphery.md) | what you can skip and why; datasets, callbacks, the glass strategy, DOEs |

### Part VI — Verification and comparison

| # | file | what it covers |
|---|---|---|
| 14 | [14_trace_forward.md](14_trace_forward.md) | **YAML → CLI → lens → rays → sensor**, every shape measured |
| 15 | [15_trace_training.md](15_trace_training.md) | **one full LM/joint iteration**, every intermediate value measured |
| 16 | [16_paper_map.md](16_paper_map.md) | all 13 equations → file:line:variable; the 10 paper/code divergences |
| 17 | [17_your_implementation.md](17_your_implementation.md) | `eisoptx` vs your `e2e-gtra-optics`: correspondences, divergences, gaps |

### Appendix

| # | file | what it covers |
|---|---|---|
| A1 | [A_defense_bank_part1.md](A_defense_bank_part1.md) | Q1–Q25: the method, the optics |
| A2 | [A_defense_bank_part2.md](A_defense_bank_part2.md) | Q26–Q50: optimization, imaging, training, your code |

### Supporting files

| file | what it is |
|---|---|
| [instrument_trace.py](instrument_trace.py) | the instrumentation harness — re-run it to regenerate every number |
| [trace_dump.json](trace_dump.json) | machine-readable trace: shapes, values, tables, 9 sections |
| [trace_tables.md](trace_tables.md) | the trace rendered as readable tables |

---

## Which chapter answers which question

| your question | go to |
|---|---|
| What does this codebase *do*, in one sentence? | 0 §0.2 |
| Where do I start reading the source? | 0 §0.4, 13 §13.8 |
| How do I run it? Which config? | 0 §0.5, 1 §1.6 |
| Why is there so little in the YAML? | 1 §1.3 (`link_arguments`) |
| Where does `sensor_diagonal = 11.547` come from? It is in no YAML. | 12 §12.9 |
| How is the lens described? What is that string? | 2 §2.2 |
| How does glass become a differentiable variable? | 2 §2.5, A1 Q17 |
| Which numbers does the optimizer actually move? | 3 §3.4 |
| Why is the conic constant the first "aspheric coefficient"? | 2 §2.4, A1 Q20 |
| What is the shape of a ray tensor, and why? | 4 §4.4, A1 Q15 |
| How does a ray find an aspheric surface? | 5 §5.4, A1 Q21 |
| What happens to a ray that fails? | 5 §5.6, A1 Q23 |
| Why is the last curvature not a variable? | 6 §6.2, A1 Q24 |
| Why are the solves closed-form? | 6 §6.4, A1 Q25 |
| What is a residual, and what is in the vector? | 7 §7.2 |
| **What is the paper's actual contribution?** | **8** |
| Prove the lift preserves value and gradient. | 8 §8.4, A1 Q5–Q6 |
| Why is `weight_w` not the paper's `w`? | 16 §16.5, A1 Q14 |
| What does the clipping compensation factor do? | 8 §8.6, A1 Q9 |
| How good is the linearization? When does it break? | 8 §8.7, A1 Q7 |
| Why LM and not Adam? | 9a §9.1, A1 Q2 |
| Why is `JᵀJ` never formed? | 9a §9.4, A2 Q26 |
| What is `D²` and why not the identity? | 9a §9.5, A2 Q27–Q28 |
| When is a step rejected? | 9b §9.8, A2 Q29 |
| **Is the paper's monotonicity claim true?** | **16 §16.10, A2 Q30** |
| Forward-mode or backward-mode AD, and where? | 8 §8.3, A1 Q10–Q11 |
| How does a spot diagram become a PSF? | 10 §10.3, A2 Q34 |
| Why only 11 fields for a 2D sensor? | 10 §10.5, A2 Q39 |
| Intensity or amplitude convolution for diffraction? | 16 §16.11, A2 Q37 |
| How does spatially varying blur work? | 10 §10.7, A2 Q38 |
| Does the restoration network see the PSF? | 11 §11.6, A2 Q40 |
| What is `detach_psfs` for? | 11 §11.8, A2 Q45 |
| Walk me through one training step. | 12 §12.4, 15, A2 Q41 |
| Why `automatic_optimization = False`? | 12 §12.2, A2 Q42 |
| Does the lens gradient flow during the network step? | 12 §12.7, A2 Q44 |
| Classical vs network-only vs co-design — same code? | 12 §12.8, A2 Q46 |
| What can I skip reading? | 13 §13.2 |
| How are glasses snapped to a real catalog? | 13 §13.6, A1 Q18–Q19 |
| Where are DOEs and metasurfaces? | 13 §13.4, 16 §16.9 |
| **What are the real tensor shapes end to end?** | **14** |
| **What are the real values in one LM iteration?** | **15** |
| Where is Eq. *N* implemented? | 16 §16.1 |
| **Where do the paper and the code disagree?** | **16 §16.12** |
| Can I reproduce Table 1? | 16 §16.14, A2 Q47 |
| What is in the paper but not the code? | 16 §16.4, A2 Q48 |
| How does my own implementation compare? | 17 §17.2 |
| **What should I fix in my code first?** | **17 §17.11, A2 Q49** |
| Is my implementation just a transcription? | 17 §17.12, A2 Q50 |
| I have a meeting in an hour. | A1 + A2, then 16 §16.12 |

---

## The results that authenticate this walkthrough

Two independent checks that the traced configuration **is** the design in the paper's Table 1,
not merely a similar one:

| quantity | paper Table 1 | measured here |
|---|---|---|
| effective spot radius (LM + `ℓ_TRA`) | 16.6 µm | **16.6029 µm** |
| image MSE (Proposed, LM + `ℓ_GTRA`) | 0.0033 | **0.0033178618** |

And the lift's defining identity, verified numerically (Chapter 15):

| identity | result |
|---|---|
| `½‖ℓ_GTRA‖² = L₀` | `0.004496567` vs `0.0044965702`, rel. err **7.249e−07** |
| `weight_w = √w` | `√0.02474414833 = 0.15730273` vs `0.1573027223`, rel. err **3.81e−08** |

## Corrections this walkthrough makes to the source and the paper

Kept together because these are the highest-value things to know.

| # | claim | reality | where |
|---|---|---|---|
| 1 | YAML comment says PSF is 37×37 | measured **35×35** | 10 |
| 2 | trace note says the Wiener stage "helped" | MSE went `0.0033 → 0.0481`; it **hurt** at the default SNR | 11 |
| 3 | paper: `D² = diag(JᵀJ)` | `beta=0.99` ratcheted running **max** | 16 §16.10 |
| 4 | paper: steps accepted only if the loss decreases | `tolerance` defaults to **2.0**; 3 of 5 configs allow increases | 16 §16.10 |
| 5 | paper: "monotonic decrease of LM curves" | **config-dependent** | 16 §16.10 |
| 6 | paper: λ ÷ 3 on success | code default **÷ 2**; the toy config diverges from the paper | 16 §16.10 |
| 7 | paper: `ℓ_ESR` as a baseline | **not implemented** | 16 §16.4 |
| 8 | paper Eq. 12 reads as intensity convolution | default mode convolves **amplitude**, then squares | 16 §16.11 |
| 9 | `compute_rms_size` lives in `visualization.py` | a diagnostic in the wrong module; cite `ray_analysis.py` | 13 §13.7 |
| 10 | `constraints.py` in your repo | 321 lines, tested, and **never called** | 17 §17.8 |

---

## Regenerating the numbers

```bash
cd /home/rcp756/code/generalized-aberrations
python docs/walkthrough/instrument_trace.py
```

Writes `trace_dump.json` and `trace_tables.md`. Every measured value in Chapters 14–17 and
Appendix A comes from that JSON; if you change a config, re-run it and the chapters' numbers
become checkable against the new output.
