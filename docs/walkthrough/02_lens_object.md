# Chapter 2 — The `Lens` object and sequence parsing

> **What this chapter answers.** What does `s-aRa-aRa-` actually mean? How does a
> 10-character string become a ray-tracing program? Why is the lens a plain object and
> not an `nn.Module`? Where do the paraxial quantities (EFL, BFL) come from?

---

## 2.1 The central idea: a lens is a compiled instruction list

`optics.py` contains two classes that do very different jobs:

- **`LensSequence`** (`optics.py:11`) — the **topology**. Parses the sequence string
  into a list of events. Pure structure, no numbers, no tensors.
- **`Lens`** (`optics.py:170`) — the **numbers**. Holds `c`, `s`, `nd`, `vd`, `a` as
  tensors and knows how to trace rays through them.

The separation matters. Topology is fixed for a whole optimization run; the numbers
change every iteration. Parsing happens once; tracing happens thousands of times.

Think of `LensSequence.events` as **bytecode**. The string is source, `parse_sequence`
is the compiler, and `trace_rays` is the interpreter. Every event carries indices into
the parameter tensors — so at trace time there is no string handling at all, only
tensor indexing.

## 2.2 The alphabet

From `parse_sequence` (`optics.py:77-167`):

| char | meaning | handled at |
|---|---|---|
| `R` | refractive element (glass) begins/ends | `optics.py:91` |
| `-` | air gap / propagation | `optics.py:118` |
| `a` | make the *next* surface aspheric | `optics.py:130` |
| `d` | diffractive surface | `optics.py:134` |
| `m` | miscellaneous surface (metasurface) | `optics.py:145` |
| `s` | aperture stop | `optics.py:156` |

One hard rule, asserted at construction (`optics.py:22`):

```python
assert sequence.count("s") == 1
```

Exactly one stop. Not zero, not two. An optical system without a defined stop has no
well-defined pupil, and ray initialization (Chapter 4) would have nothing to aim at.

Note that `a` is a **modifier**, not a surface. It attaches to the refraction that
follows it. That is why `s-aRa-aRa-` has four `a` characters and four aspheric
surfaces but only two `R` characters: each `R` is bracketed by two `a`s, one for its
front face and one for its back.

## 2.3 Compiling `s-aRa-aRa-`, character by character

Here is the actual event list, dumped from the parser:

```
 0  {'type': 'p', 's': None}
 1  {'type': 's'}
 2  {'type': 'p', 's': 0, 'a': 0, 'c': 0}
 3  {'type': 'r', 'n1': None, 'n2': 0,    'c': 0, 's': 1, 'a': 0}
 4  {'type': 'p', 's': 1, 'c': 1, 'a': 1, 'refractive': True}
 5  {'type': 'r', 'n1': 0,    'n2': None, 'c': 1, 's': 2, 'a': 1}
 6  {'type': 'p', 's': 2, 'a': 2, 'c': 2}
 7  {'type': 'r', 'n1': None, 'n2': 1,    'c': 2, 's': 3, 'a': 2}
 8  {'type': 'p', 's': 3, 'c': 3, 'a': 3, 'refractive': True}
 9  {'type': 'r', 'n1': 1,    'n2': None, 'c': 3, 's': 4, 'a': 3}
10  {'type': 'p', 's': 4}
```

**11 events** — matching the measured `sequence_events` in
`1_config_and_lens`: `['p','s','p','r','p','r','p','r','p','r','p']`.

Walk the string against the list:

| char | index | what the parser does | events emitted |
|---|---|---|---|
| `s` | 0 | stop; flush the pending empty propagation first | 0, 1 |
| `-` | 1 | air gap, consumes spacing `s[0]` | 2 |
| `a` | 2 | tag next refraction as aspheric, `a[0]` | — |
| `R` | 3 | enter glass: refraction into `nd[0]`, then propagation *inside* glass | 3, 4 |
| `a` | 4 | tag next refraction aspheric, `a[1]` | — |
| `-` | 5 | exit glass: close refraction (`n2=None` = into air), then air gap | 5, 6 |
| `a` | 6 | `a[2]` | — |
| `R` | 7 | enter glass 2 | 7, 8 |
| `a` | 8 | `a[3]` | — |
| `-` | 9 | exit glass 2, final gap to sensor | 9, 10 |

Two subtleties worth being able to explain:

**Event 0 is an empty propagation.** `{'type': 'p', 's': None}` — a propagation with no
spacing index. The parser initializes `current_propagation` before the loop
(`optics.py:86`) and flushes it when it hits the stop (`optics.py:157-159`). At trace
time this is a no-op placeholder. It exists so that the event list always begins with a
propagation, keeping the interpreter's alternating rhythm intact.

**`n1` and `n2` encode which side is glass.** Look at event 3: `n1: None, n2: 0`. The
ray goes *from* air (`None`) *into* material 0. Event 5: `n1: 0, n2: None` — from
material 0 back into air. The parser assigns these by tracking whether it is inside an
`R`. This is how Snell's law knows the index ratio without any lookup at trace time.

**`refractive: True` on events 4 and 8** marks propagations *inside glass*. This is
what `n_refractive` counts (`optics.py:30-34`) — it counts propagations, not
refractions, because a glass element is defined by its interior. Measured:
`n_refractive = 2`. Two glass elements. ✓

## 2.4 The derived counts, and why they are what they are

All from the event list, all measured (`1_config_and_lens`):

| property | code | value | reasoning |
|---|---|---|---|
| `n_interfaces` | `optics.py:52` | **4** | count of `type == 'r'` — events 3, 5, 7, 9 |
| `n_propagations` | `optics.py:57` | **5** | `max(s index) + 1` = 4 + 1 |
| `n_refractive` | `optics.py:30` | **2** | propagations with `refractive` |
| `n_aspherical` | `optics.py:37` | **4** | refractions carrying `a` |
| `n_diffractive` | `optics.py:42` | **0** | no `d` in the string |
| `stop_idx` | `optics.py:71` | **1** | event index of the stop |

`n_propagations = 5` deserves a second look. It is computed as `max(s) + 1` over
events, *not* as a count of propagation events. There are 6 events of type `p`
(indices 0, 2, 4, 6, 8, 10) but only 5 spacings, because event 0 has `s: None` and is
excluded by the `isinstance(item["s"], int)` guard (`optics.py:62`). This matches the
config: `s: [.5, 1., .5, 2., 9.]` — five numbers.

Physically the five spacings are: stop→surface 1, inside element 1, element 1→element
2, inside element 2, element 2→sensor.

## 2.5 The `Lens` constructor: shape discipline

`Lens.__init__` (`optics.py:176-218`) takes nine tensors and immediately reshapes every
one against the parsed topology:

```python
n_lens = s.shape[1] if len(s.shape) > 1 else 1
self.s  = s.view(self.sequence.n_propagations, n_lens)      # [5, 1]
self.c  = c.view(self.sequence.n_interfaces,   n_lens)      # [4, 1]
self.nd = nd.view(self.sequence.n_refractive,  n_lens)      # [2, 1]
self.vd = vd.view(self.sequence.n_refractive,  n_lens)      # [2, 1]
self.a  = a.view(self.sequence.n_aspherical,   n_lens, a.shape[-1])  # [4, 1, 4]
```

Note what this does for you: **a mismatch between the sequence string and the parameter
arrays is caught immediately**, at construction, by `view` raising. If you write
`s-aRa-aRa-` but supply three curvatures, you find out now rather than getting silently
wrong rays.

The trailing `n_lens` dimension is **batched optics** — the same topology with several
different parameter sets traced simultaneously. The toy config has `n_lens = 1`, which
is why so many measured shapes have a trailing `1`
(e.g. `lens_variables[c]` is `[4, 1]`). An assertion at `optics.py:220-231` verifies
every array agrees on `n_lens`.

> **Defense note.** If asked "could this optimize several designs at once?" — yes,
> that is what the `n_lens` axis is for, provided they share a sequence string. The
> shipped configs never use it.

## 2.6 Why `Lens` is not an `nn.Module`

This surprises people. `Lens` is a plain Python object. It holds tensors that require
gradients, but it does not register them as parameters, has no `state_dict`, and is not
part of any module tree.

That is deliberate, and it follows from a design decision made one level up. The
`Lens` is **rebuilt from scratch on every forward pass** by
`LensParameterization.forward` (Chapter 3). The parameters that persist across
iterations are the flat variable vector inside the parameterization; the `Lens` is a
short-lived *view* of those variables after unpacking, scaling and solving.

Consequences:

1. Gradients flow *through* the `Lens` into the parameterization's variables — the
   `Lens` is a waypoint on the autograd graph, not a leaf.
2. Checkpointing saves the parameterization, not the lens. The lens is reconstructible.
3. Solves (Chapter 6) can overwrite `c[-1]` freely, because the write happens during
   construction and is part of the differentiable graph.

If `Lens` were an `nn.Module` with registered parameters, the solve would have to
mutate a leaf tensor in place every step — the classic way to break autograd.

## 2.7 Paraxial properties: EFL, BFL, and the ABCD matrix

`Lens.get_abcd` (in `optics.py`, backed by `paraxial_ray_tracing.py`) builds the
system's **ray transfer matrix** by multiplying 2×2 matrices along the event list.

The two primitives (`paraxial_ray_tracing.py`, 97 lines total):

```
translation by distance t:      [[1, t], [0, 1]]
refraction at curvature c
   between indices n1, n2:      [[1, 0], [-(n2-n1)c/n2, n1/n2]]
```

Multiply them in reverse order along the event list and you get the system matrix
`[[A, B], [C, D]]`. Then:

```
EFL = −1/C          BFL follows from A, C
```

Measured on the toy system (`1_config_and_lens`):

| quantity | value |
|---|---|
| `lens.efl` | **10.0** (exactly) |
| `lens.abcd` | 2×2, see `trace_dump.json` |

The EFL is exactly 10.0 — not 9.9998 — because it is not measured, it is **enforced**.
`solve_type: focal_length` computes `c[-1]` from the requirement `EFL = target_efl`
before the lens is finished being built. Chapter 6 derives that solve.

**Why keep a paraxial model at all when you have exact ray tracing?** Three reasons,
all of which show up in the code:

1. **Solves need it.** Inverting "what curvature gives EFL = 10?" is closed-form in
   the paraxial model and would need Newton iteration in the exact one.
2. **Ray aiming needs a starting guess** (Chapter 4) — the paraxial pupil position is
   where ray aiming begins.
3. **Sanity/logging.** EFL and BFL are the quantities an optical engineer reads first.

## 2.8 The trace loop is a generator

`Lens.trace_rays` is a **generator**, and the call sites look like this
(`imaging_system.py:543-545`):

```python
r, d, ray_status, event_info = next(
    lens.trace_rays(r0, d0, wavelengths, yield_on="end")
)
```

`yield_on` controls how often it yields:

- `"end"` — yield once, at the sensor. Used for the spot diagram.
- per-event modes — yield intermediate states. Used by the residuals that need to know
  what happened *between* surfaces (ray path lengths, incidence angles) and by the
  layout visualization, which needs every vertex to draw the ray fan.

This is why `RayPathResiduals` and `SurfaceNormalResiduals` can exist at all: the
generator exposes the interior of the trace. A monolithic `trace()` returning only
final coordinates could not support them.

Measured (`3_ray_trace.per_event_steps`): the generator produces state at each of the
**11 events**, and `ray_status` is carried alongside `r` and `d` throughout — the trace
never drops a ray silently, it *marks* it (Chapter 5).

## 2.9 What to take away

1. `LensSequence` compiles a string into 11 events, each carrying tensor indices;
   `Lens` holds the tensors. Parse once, trace thousands of times.
2. `a` is a modifier on the following surface, not a surface itself.
3. `n1`/`n2` in each refraction event encode the air/glass direction, so Snell needs no
   lookups.
4. `view()` against parsed counts turns a config/sequence mismatch into an immediate
   error.
5. `Lens` is deliberately not an `nn.Module` — it is rebuilt each forward pass and is a
   waypoint on the autograd graph.
6. EFL is exactly 10.0 because a solve enforces it, not because optimization achieved
   it.
7. `trace_rays` is a generator; the interior states it yields are what make the
   manufacturability residuals possible.

---

*Previous: [Chapter 1 — The configuration is the program](01_configuration.md)*
*Next: [Chapter 3 — Parameterization and the glass model](03_parameterization.md)*
