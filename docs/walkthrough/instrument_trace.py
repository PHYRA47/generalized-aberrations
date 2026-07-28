"""Instrumentation harness for the eisoptx walkthrough.

Runs the toy configuration end to end and records real tensor shapes, dtypes and
intermediate values at every stage the walkthrough chapters describe. Everything the
chapters quote as a number comes from `trace_dump.json`, produced by this script.

Run from the repository root:

    export KMP_AFFINITY=disabled OMP_PROC_BIND=false KMP_INIT_AT_FORK=FALSE
    export PYTHONPATH=$PWD
    python docs/walkthrough/instrument_trace.py

Outputs (written next to this script):
    trace_dump.json   - machine-readable record of every probe
    trace_tables.md   - the same content rendered as markdown tables
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from eisoptx.main import CustomCLI  # noqa: E402
from eisoptx.imaging_system import ImagingSystemModule  # noqa: E402
from eisoptx.modeling import optics, ray_initialization as ri, ray_tracing as rt  # noqa: E402
from eisoptx.modeling import ray_analysis as ra, paraxial_ray_tracing as prt  # noqa: E402

TRACE = {}
OUT_DIR = Path(__file__).resolve().parent


def probe(section, key, value, note=None):
    """Record a probe entry under a named section."""
    TRACE.setdefault(section, {})[key] = _describe(value, note)
    return value


def _describe(value, note=None):
    """Turn a tensor/array/scalar into a JSON-safe description."""
    entry = {}
    if isinstance(value, torch.Tensor):
        entry["type"] = "Tensor"
        entry["shape"] = list(value.shape)
        entry["dtype"] = str(value.dtype)
        entry["requires_grad"] = bool(value.requires_grad)
        flat = value.detach().reshape(-1)
        if value.dtype in (torch.bool,):
            entry["true_fraction"] = float(flat.float().mean())
            entry["numel"] = int(flat.numel())
        elif flat.numel() <= 12:
            entry["values"] = [_num(v) for v in flat.tolist()]
        else:
            entry["numel"] = int(flat.numel())
            f = flat.float()
            entry["min"] = _num(f.min())
            entry["max"] = _num(f.max())
            entry["mean"] = _num(f.mean())
            entry["first4"] = [_num(v) for v in flat[:4].tolist()]
    elif isinstance(value, np.ndarray):
        entry["type"] = "ndarray"
        entry["shape"] = list(value.shape)
        entry["dtype"] = str(value.dtype)
        entry["values"] = value.reshape(-1).tolist()[:12]
    elif isinstance(value, (int, float, bool, str)) or value is None:
        entry["type"] = type(value).__name__
        entry["value"] = _num(value) if isinstance(value, float) else value
    elif isinstance(value, (list, tuple)):
        entry["type"] = type(value).__name__
        entry["len"] = len(value)
        entry["repr"] = repr(value)[:400]
    elif isinstance(value, dict):
        entry["type"] = "dict"
        entry["keys"] = list(value.keys())
    else:
        entry["type"] = type(value).__name__
        entry["repr"] = repr(value)[:400]
    if note:
        entry["note"] = note
    return entry


def _num(v):
    if isinstance(v, torch.Tensor):
        v = v.item()
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return str(v)
        return round(v, 10)
    return v


def build_module(configs, extra_args=()):
    """Instantiate ImagingSystemModule exactly as the CLI does, from YAML configs.

    Uses the repository's own CustomCLI so that the argument links defined in
    `eisoptx/main.py` (sensor diagonal from EFL and HFOV, wavelength propagation,
    logger name from the lens sequence) are applied identically to a real run.
    """
    args = []
    for c in configs:
        args += ["-c", c]
    args += list(extra_args)
    torch.manual_seed(0)
    cli_obj = CustomCLI(
        ImagingSystemModule,
        auto_configure_optimizers=False,
        run=False,
        args=args,
        save_config_callback=None,
    )
    return cli_obj.model, cli_obj


# ----------------------------------------------------------------------------
# Stage 1 - configuration, parameterization, lens construction
# ----------------------------------------------------------------------------
def stage_config_and_lens(module):
    """Probe the parameterization state and the Lens object it builds."""
    S = "1_config_and_lens"
    p = module.parameterization

    probe(S, "sequence_string", p.sequence.sequence)
    probe(S, "n_interfaces", int(p.sequence.n_interfaces))
    probe(S, "n_propagations", int(p.sequence.n_propagations))
    probe(S, "n_refractive", int(p.sequence.n_refractive))
    probe(S, "n_aspherical", int(p.sequence.n_aspherical))
    probe(S, "n_diffractive", int(p.sequence.n_diffractive))
    probe(S, "stop_idx", int(p.sequence.stop_idx))
    probe(S, "sequence_events", [e["type"] for e in p.sequence.events])
    probe(S, "scale_factor", float(p.scale_factor))
    probe(S, "target_efl", p.target_efl)
    probe(S, "solve_idx", p.solve_idx)
    probe(S, "solve_type", p.solve_type)
    probe(S, "paraxial_image_solve", bool(p.paraxial_image_solve))
    probe(S, "total_track_length_solve", p.total_track_length_solve)

    # The flat variable vector actually handed to the optimizer
    probe(S, "_lens_variables_trainable", p._lens_variables,
          "the flat vector of trainable values; this is what LM updates")
    probe(S, "initial_variables_full", p.initial_variables,
          "full variable vector including frozen entries")
    probe(S, "optimization_mask", p.optimization_mask,
          "True where a variable is trainable")

    # Per-key breakdown of the packed variable vector
    layout = {}
    i = 0
    for k in p.variable_keys:
        shape = p.variable_shape_dict[k]
        n = int(shape.numel())
        layout[k] = {
            "shape": list(shape),
            "numel": n,
            "slice": [i, i + n],
            "n_trainable": int(p.optimization_mask[i:i + n].sum()),
        }
        i += n
    TRACE.setdefault(S, {})["variable_layout"] = {
        "type": "layout", "note": "packing order of the flat variable vector",
        "total_numel": i,
        "total_trainable": int(p.optimization_mask.sum()),
        "keys": layout,
    }

    lv = p.lens_variables
    for k, v in lv.items():
        probe(S, f"lens_variables[{k}]", v)

    scaled = p.scale_lens_parameters()
    for k, v in scaled.items():
        probe(S, f"scaled_parameters[{k}]", v)

    # The Lens object, after all in-place solves have been applied
    lens = p.lens
    probe(S, "lens.c_after_solves", lens.c, "curvatures after curvature solve")
    probe(S, "lens.s_after_solves", lens.s, "spacings after image/TTL solve")
    probe(S, "lens.nd", lens.nd)
    probe(S, "lens.vd", lens.vd)
    probe(S, "lens.efl", lens.efl)
    probe(S, "lens.bfl", lens.bfl)
    probe(S, "lens.pupil_position", lens.pupil_position)
    probe(S, "lens.get_abcd(reduce=False)", lens.get_abcd(reduce=False),
          "per-event 2x2 ABCD matrices before reduction")
    probe(S, "lens.get_abcd(reduce=True)", lens.get_abcd(reduce=True),
          "system ABCD matrix; efl = -1/C")
    return lens


# ----------------------------------------------------------------------------
# Stage 2 - ray initialization at the entrance pupil
# ----------------------------------------------------------------------------
def stage_ray_initialization(module, lens):
    """Probe the rays launched at the entrance pupil."""
    S = "2_ray_initialization"
    ri_obj = module.ray_initialization

    probe(S, "hfov_deg", float(ri_obj.hfov))
    probe(S, "n_fields", int(ri_obj.n_fields))
    probe(S, "wavelengths_nm", list(ri_obj.wavelengths))
    probe(S, "pupil_sampling_mode", ri_obj.pupil_sampling_mode)
    probe(S, "pupil_sampling_kwargs", ri_obj.pupil_sampling_kwargs)
    probe(S, "ray_aiming_steps", int(ri_obj.ray_aiming_steps))
    probe(S, "field_stop_position", ri_obj.field_stop_position)

    epd = ri_obj.epd(lens.efl) if callable(ri_obj.epd) else ri_obj.epd
    probe(S, "epd_computed", epd, "entrance pupil diameter = efl / f_number")
    probe(S, "pupil_position_z", lens.pupil_position)

    # Directions alone, before pupil sampling
    d_only = ri.initialize_ray_directions(
        ri_obj.n_fields, torch.tensor(float(ri_obj.hfov)).to(lens.pupil_position)
    )
    probe(S, "ray_directions_only", d_only,
          "unit direction per field, before pupil sampling")

    r, d = ri_obj(lens)
    probe(S, "r_initial", r, "ray positions at entrance pupil")
    probe(S, "d_initial", d, "ray direction cosines")
    probe(S, "r_norm_check", (d ** 2).sum(-1).sqrt(),
          "direction cosines must be unit length")
    return r, d


# ----------------------------------------------------------------------------
# Stage 3 - surface-by-surface ray trace to the sensor
# ----------------------------------------------------------------------------
def stage_ray_trace(module, lens, r, d):
    """Probe the ray trace surface by surface.

    `Lens.trace_rays` is a generator. With yield_on='all' it yields
    (r, d, ray_status, event_info) after every event in the sequence, which is
    exactly the surface-by-surface view the walkthrough needs.
    """
    S = "3_ray_trace"
    wavelengths = torch.tensor(module.ray_initialization.wavelengths).to(r)
    probe(S, "wavelengths_tensor", wavelengths)
    probe(S, "w0_reference", lens.w0)
    probe(S, "wavelength_ratios", wavelengths / lens.w0)

    steps = []
    for i, (r_i, d_i, status_i, info_i) in enumerate(
        lens.trace_rays(r, d, wavelengths, yield_on="all")
    ):
        ev = lens.sequence.events[i]
        entry = {
            "event_index": i,
            "event_type": ev["type"],
            "event_keys": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                               else str(v)) for k, v in ev.items()},
            "r": _describe(r_i),
            "d": _describe(d_i),
            "ray_status": _describe(status_i),
            "status_counts": {
                "ok_0": int((status_i == 0).sum()),
                "backtrack_1": int((status_i == 1).sum()),
                "tir_2": int((status_i == 2).sum()),
                "backward_3": int((status_i == 3).sum()),
                "miss_4": int((status_i == 4).sum()),
            },
            "event_info": {k: _describe(v) for k, v in info_i.items()},
        }
        steps.append(entry)
        r_last, d_last, status_last = r_i, d_i, status_i

    TRACE.setdefault(S, {})["per_event_steps"] = {
        "type": "step_list",
        "n_events": len(steps),
        "note": "one entry per yielded ray-tracing event, in sequence order",
        "steps": steps,
    }
    probe(S, "r_at_sensor", r_last, "ray coordinates at the image plane")
    probe(S, "d_at_sensor", d_last)
    probe(S, "ray_status_final", status_last,
          "0=ok 1=backtrack 2=TIR 3=backward 4=miss")
    ray_valid = status_last == 0
    probe(S, "ray_valid_final", ray_valid)

    # Downstream ray statistics used by the residuals
    x, y = r_last[0], r_last[1]
    probe(S, "x_at_sensor", x)
    probe(S, "y_at_sensor", y)
    reduce_dims = tuple(range(1, y.dim()))
    y_mean = ra.evaluate_mean_ray_height(y, ray_valid, reduce_dims)
    probe(S, "y_centroid", y_mean, "weighted mean ray height per field")
    tra_x, tra_y = ra.evaluate_transverse_ray_aberrations(x, y, ray_valid, reduce_dims)
    probe(S, "transverse_ray_aberrations_x", tra_x,
          "per-ray x deviation from the centroid (epsilon_x in the paper)")
    probe(S, "transverse_ray_aberrations_y", tra_y,
          "per-ray y deviation from the centroid (epsilon_y in the paper)")
    probe(S, "rms_spot_size", ra.compute_rms_spot_size(x, y, ray_valid, reduce_dims))
    return r_last, d_last, ray_valid


# ----------------------------------------------------------------------------
# Stage 4 - residual assembly (the least-squares objective)
# ----------------------------------------------------------------------------
def stage_residuals(module, batch=None):
    """Probe the residual dictionary that becomes the LM objective."""
    S = "4_residuals"
    probe(S, "registered_residual_names", [r.name for r in module.residuals])
    probe(S, "data_collection_keys", sorted(module.data_collection_keys),
          "intermediate ray-tracing quantities the residuals need")
    probe(S, "weight_dict", {k: v for k, v in module.weight_dict.items()})
    probe(S, "constraint_dict", {k: v for k, v in module.constraint_dict.items()})
    probe(S, "e2e_enabled", bool(module.e2e_enabled))
    probe(S, "e2e_vector_mode", module.e2e_vector_mode)
    probe(S, "e2e_loss_weight", module.e2e_loss_weight)

    TRACE[S]["weights_by_name"] = {
        "type": "table",
        "note": "weight and constraint flag per residual term",
        "rows": [
            {"name": k,
             "weight": module.weight_dict.get(k),
             "constraint": module.constraint_dict.get(k)}
            for k in module.weight_dict
        ],
    }

    residuals_dict, weight_dict, constraint_dict, logs = (
        module.compute_residuals_dict_and_logs(batch)
    )
    rows = []
    for name, vec in residuals_dict.items():
        v = vec.detach().reshape(-1)
        rows.append({
            "name": name,
            "numel": int(v.numel()),
            "weight": weight_dict.get(name),
            "constraint": constraint_dict.get(name),
            "l2_norm": _num(v.norm()) if v.numel() else 0.0,
            "sum_of_squares": _num((v ** 2).sum()) if v.numel() else 0.0,
        })
        probe(S, f"residual[{name}]", vec)
    TRACE[S]["residual_vector_table"] = {
        "type": "table",
        "note": "one row per residual term actually produced this call",
        "total_residual_entries": int(sum(r["numel"] for r in rows)),
        "rows": rows,
    }
    TRACE[S]["logs"] = {"type": "dict_values",
                        "values": {k: _num(v) for k, v in logs.items()}}
    return residuals_dict, logs


# ----------------------------------------------------------------------------
# Stage 5 - Jacobian and one Levenberg-Marquardt step
# ----------------------------------------------------------------------------
def stage_lm_step(module, batch=(None,)):
    """Probe the forward-mode Jacobian and a full LM iteration.

    `batch` is passed straight to `torch.func.functional_call` as the argument
    tuple, so it must be a tuple even when there is no image. The repository's own
    test dataloader collates to `(None,)` for exactly this reason
    (eisoptx/data/datasets.py:142); pass a real `(images, field_limits)` tuple to
    exercise the end-to-end path.
    """
    import math as _math
    from eisoptx.imaging_system import CustomClosure

    S = "5_lm_step"
    # module.get_optimizers() needs an attached Trainer; build the optimizer
    # directly from the same callable the config supplies.
    lens_optimizer = module.lens_optimizer(
        [p for p in module.parameterization.parameters() if p.requires_grad]
    )
    probe(S, "lens_optimizer_class", type(lens_optimizer).__name__)

    group = lens_optimizer.param_groups[0]
    for k in ("lm_parameter", "damped_term_min", "tolerance",
              "lam_increase_factor", "lam_decrease_factor", "beta"):
        if k in group:
            probe(S, f"hyperparameter[{k}]", group[k])

    closure = CustomClosure(module, lens_optimizer, batch)
    with torch.enable_grad():
        residual_vector, ls_loss, jacobian, constraint_mask = (
            closure.get_least_squares_quantities()
        )

    probe(S, "residual_vector", residual_vector,
          "stacked residuals from every term; this is l(x)")
    probe(S, "least_squares_loss", ls_loss, "0.5 * sum of squared residuals")
    probe(S, "jacobian", jacobian,
          "forward-mode Jacobian dl/dx, shape [n_residuals, n_variables]")
    probe(S, "constraint_mask", constraint_mask)
    probe(S, "n_residuals", int(residual_vector.numel()))
    probe(S, "n_variables", int(jacobian.shape[1]))
    probe(S, "n_constraints", int(constraint_mask.sum()),
          "zero in every shipped config: no residual sets constraint=True")

    # Reproduce the optimizer's own linear algebra so the walkthrough can show it
    lam = group["lm_parameter"]
    rj = jacobian[~constraint_mask]
    res = residual_vector[~constraint_mask]
    damping_terms = rj.norm(dim=0).clamp(min=group["damped_term_min"])
    probe(S, "damping_terms", damping_terms,
          "column norms of J, clamped; the diagonal of the damping matrix")

    matrix = torch.cat((rj, np.sqrt(lam) * damping_terms.diag_embed()), dim=0)
    bb = torch.cat((res, torch.zeros_like(damping_terms)), dim=0)
    probe(S, "augmented_matrix", matrix,
          "[J; sqrt(lam)*D] - the augmented LM system")
    probe(S, "augmented_rhs", bb)
    lstsq = torch.linalg.lstsq(matrix.cpu(), -bb.cpu(), driver="gelsd")
    step = lstsq[0]
    probe(S, "lstsq_rank", float(lstsq.rank.item()))
    probe(S, "step_delta_x", step, "the parameter update LM proposes")

    # Gauss-Newton / gradient cross-checks
    gradient = rj.T @ res
    probe(S, "gradient_JT_r", gradient, "J^T l, the gradient of the scalar loss")
    probe(S, "pseudo_hessian_JTJ", rj.T @ rj)
    probe(S, "step_dot_gradient", float(step @ gradient.cpu()),
          "negative value confirms the step is a descent direction")

    # Apply the step exactly as LMOptimizer.step does, and measure acceptance
    p = [v for v in group["params"] if v.requires_grad][0]
    p_copy = p.data.clone()
    p.data.add_(step.to(p.device))
    with torch.inference_mode(True):
        updated_loss = closure.evaluate_least_squares_loss()
    loss_ratio = (updated_loss / ls_loss).item()
    probe(S, "loss_before", ls_loss)
    probe(S, "loss_after_step", updated_loss)
    probe(S, "loss_ratio", loss_ratio, "accepted when <= tolerance")
    accepted = _math.isfinite(loss_ratio) and loss_ratio <= group["tolerance"]
    probe(S, "step_accepted", bool(accepted))
    probe(S, "lam_next", lam / group["lam_decrease_factor"] if loss_ratio <= 1.0
          else lam * group["lam_increase_factor"])
    p.data = p_copy  # restore; instrumentation must not mutate state
    return jacobian, step


# ----------------------------------------------------------------------------
# Stage 6 - the generalized aberration lift (the paper's core contribution)
# ----------------------------------------------------------------------------
def stage_generalized_aberrations(module, batch):
    """Probe the scalar-loss -> least-squares conversion, term by term.

    This reproduces `ImagingSystemModule.evaluate_generalized_transverse_ray_aberrations`
    (eisoptx/imaging_system.py:445) step by step so the walkthrough can show every
    intermediate of

        xy' = xy0 - grad0 * 2*L0 / ||grad0||^2
        w   = ||grad0|| / sqrt(2*L0)
        l(xy) = w * (xy - xy')
    """
    S = "6_generalized_aberrations"
    xy = module.compute_spot_diagrams()
    probe(S, "xy_spot_diagrams", xy, "spot diagrams entering the lift")

    with torch.inference_mode(False):
        with torch.no_grad():
            grad, (scalar_loss, _) = torch.func.grad_and_value(
                module.scalar_e2e_loss, 0, has_aux=True
            )(xy.detach(), batch, module.lens_e2e_loss_fn)

    probe(S, "L0_scalar_e2e_loss", scalar_loss,
          "L0: the scalar end-to-end (image) loss at the current spot diagram")
    probe(S, "grad0", grad, "dL/d(xy) evaluated at xy0, treated as a constant")

    ray_valid = xy.isfinite().all(dim=0)
    probe(S, "ray_valid", ray_valid)
    grad_valid = grad.where(ray_valid, 0.0).view(-1)
    grad_norm_squared = grad_valid @ grad_valid
    probe(S, "grad_norm_squared", grad_norm_squared, "||grad0||^2, failures excluded")

    xy_control = xy.detach() - grad * 2 * scalar_loss / grad_norm_squared
    probe(S, "xy_control", xy_control,
          "xy': the virtual target the least-squares problem drives rays toward")
    probe(S, "xy_control_shift", (xy_control - xy.detach()),
          "how far the lift displaces each ray's target")

    weight = (grad_norm_squared / (2 * scalar_loss)).sqrt()
    probe(S, "weight_w", weight, "w = ||grad0|| / sqrt(2*L0)")

    residual_vector = weight * (xy - xy_control)
    residual_vector = residual_vector[ray_valid.broadcast_to(xy.shape)]
    probe(S, "e2e_residual_vector", residual_vector,
          "l(xy): the generalized transverse ray aberrations")

    # The identity that makes the lift valid: 0.5*||l||^2 must equal L0
    recovered = 0.5 * (residual_vector ** 2).sum()
    probe(S, "half_sum_sq_residuals", recovered,
          "0.5*||l||^2, which should reproduce L0 exactly")
    probe(S, "identity_abs_error", float((recovered - scalar_loss).abs()),
          "residual of the equivalence 0.5*||l(xy0)||^2 == L0")
    probe(S, "identity_rel_error",
          float(((recovered - scalar_loss) / scalar_loss).abs()))

    # Compare against the module's own implementation
    rv_ref, sl_ref = module.evaluate_generalized_transverse_ray_aberrations(xy, batch)
    probe(S, "reference_residual_vector", rv_ref)
    probe(S, "reference_matches_manual",
          bool(torch.allclose(rv_ref, residual_vector, atol=1e-6)),
          "manual reproduction agrees with the shipped method")
    return residual_vector, scalar_loss


# ----------------------------------------------------------------------------
# Stage 7 - PSF simulation and image formation
# ----------------------------------------------------------------------------
def stage_psf_and_image(module, batch):
    """Probe the spot-diagram -> PSF -> blurred image -> restoration chain."""
    S = "7_psf_and_image"
    sim = module.optics_simulator
    if sim is None:
        TRACE[S] = {"note": _describe("no optics simulator in this config")}
        return None
    images, field_limits = batch
    probe(S, "simulator_class", type(sim).__name__)
    probe(S, "psf_sampler_class", type(sim.psf_sampler).__name__)
    probe(S, "convolution_class", type(sim.convolution).__name__)
    probe(S, "psf_abs_size_mm", sim.psf_abs_size,
          "physical extent of the PSF window on the sensor")
    probe(S, "sensor_diagonal_mm", sim.sensor_diagonal)
    probe(S, "psf_grid_shape", sim.psf_grid_shape)
    probe(S, "default_image_size", sim.default_image_size)

    xy = module.compute_spot_diagrams()
    probe(S, "xy_spot_diagrams", xy)

    # The module's own entry point: spot diagrams -> RGB PSFs (1 lens assumed)
    psfs = module.try_build_simulation_model(xy=xy)
    probe(S, "rgb_psfs", psfs,
          "per-field RGB PSFs from kernel density estimation of the spot diagram")
    probe(S, "psf_energy_per_field", psfs.sum(dim=(-2, -1)),
          "each PSF integrates to ~1 unless rays were lost off the grid")

    im_h, im_w = images.shape[-2:]
    psf_grid = sim.compute_psf_grid(psfs, im_h, im_w, field_limits)
    probe(S, "psf_grid", psf_grid,
          "PSFs interpolated onto the patch grid for spatially-varying convolution")

    blurred = sim.apply_optics_model(images, psf_grid)
    probe(S, "blurred_image", blurred, "output of the SVOLA patch convolution")
    probe(S, "blur_mse_vs_sharp",
          float(torch.nn.functional.mse_loss(blurred, images)))

    irm = module.image_restoration_model
    probe(S, "restoration_model_class",
          type(irm).__name__ if irm is not None else None)
    if irm is not None:
        restored = irm(blurred, psf_grid)
        probe(S, "restored_image", restored)
        probe(S, "restored_mse_vs_sharp",
              float(torch.nn.functional.mse_loss(restored, images)))
    return psf_grid


# ----------------------------------------------------------------------------
# Stage 8 - image restoration
# ----------------------------------------------------------------------------
def stage_restoration(images, blurred, psf_grid):
    """Probe the restoration chain.

    `configs/toy/defaults_e2e.yml:9` sets `image_restoration_model:` to null, so the
    toy end-to-end run optimizes the lens against the sharp image with no restoration
    at all. To give the walkthrough real numbers for the restoration stage, this
    builds the same chain the c_mount config uses
    (configs/c_mount/defaults_e2e_restored.yml:10-29): NAFNet -> Wiener -> NAFNet.
    The NAFNets are untrained, so only the Wiener stage is meaningful numerically.
    """
    from eisoptx.image_restoration.models import ImageRestorationChainer
    from eisoptx.image_restoration.wiener import WienerDeconvolution
    from eisoptx.image_restoration.nafnet import NAFNet

    S = "8_restoration"
    torch.manual_seed(0)

    wiener = WienerDeconvolution(detach_psfs=False, snr_initial_value=1000.0)
    probe(S, "wiener_class", type(wiener).__name__)
    probe(S, "wiener_learned_snr", getattr(wiener, "snr", None))
    for attr in ("detach_psfs", "snr_initial_value"):
        if hasattr(wiener, attr):
            probe(S, f"wiener.{attr}", getattr(wiener, attr))

    wiener_out = wiener(blurred, psf_grid)
    probe(S, "wiener_output", wiener_out)
    probe(S, "mse_blurred_vs_sharp",
          float(torch.nn.functional.mse_loss(blurred, images)))
    probe(S, "mse_wiener_vs_sharp",
          float(torch.nn.functional.mse_loss(wiener_out, images)),
          "lower than the blurred MSE means the Wiener stage helped")

    nafnet = NAFNet(width=16, middle_blk_num=1,
                    enc_blk_nums=[1, 1, 1, 4], dec_blk_nums=[1, 1, 1, 1])
    probe(S, "nafnet_n_parameters",
          int(sum(p.numel() for p in nafnet.parameters())))
    chainer = ImageRestorationChainer(models=[nafnet, wiener, nafnet])
    probe(S, "chainer_class", type(chainer).__name__)
    probe(S, "chainer_n_stages", len(chainer.models))
    with torch.no_grad():
        chained = chainer(blurred, psf_grid)
    probe(S, "chainer_output", chained)
    probe(S, "mse_chainer_vs_sharp",
          float(torch.nn.functional.mse_loss(chained, images)),
          "NAFNets are randomly initialized here, so this is not a trained result")
    return wiener_out


# ----------------------------------------------------------------------------
# Stage 9 - LM convergence from a perturbed start
# ----------------------------------------------------------------------------
def stage_lm_convergence(configs, n_iterations=12, perturbation=0.05):
    """Run several LM iterations from a deliberately detuned design.

    Stage 5 probes a single step at the shipped optimum, where the step is rejected
    because there is nothing left to gain. To show the optimizer actually working -
    accepted steps, the damping parameter adapting - this perturbs the trainable
    variables and iterates. `configs/toy/designs/starting_point.yml` is 0 bytes in the
    repository, so a perturbed optimum is the only reproducible detuned start.
    """
    import math as _math
    from eisoptx.imaging_system import CustomClosure

    S = "9_lm_convergence"
    module, _ = build_module(configs)
    optimizer = module.lens_optimizer(
        [p for p in module.parameterization.parameters() if p.requires_grad]
    )
    group = optimizer.param_groups[0]
    p = [v for v in group["params"] if v.requires_grad][0]

    torch.manual_seed(0)
    with torch.no_grad():
        noise = torch.randn_like(p) * perturbation
        p.add_(noise)
    probe(S, "perturbation_scale", perturbation)
    probe(S, "perturbation_applied", noise)
    probe(S, "n_iterations", n_iterations)

    closure = CustomClosure(module, optimizer, (None,))
    lam = group["lm_parameter"]
    history = []
    for it in range(n_iterations):
        with torch.enable_grad():
            residual_vector, ls_loss, jacobian, cmask = (
                closure.get_least_squares_quantities()
            )
        rj = jacobian[~cmask]
        res = residual_vector[~cmask]
        damping = rj.norm(dim=0).clamp(min=group["damped_term_min"])
        matrix = torch.cat((rj, np.sqrt(lam) * damping.diag_embed()), dim=0)
        bb = torch.cat((res, torch.zeros_like(damping)), dim=0)
        lstsq = torch.linalg.lstsq(matrix.cpu(), -bb.cpu(), driver="gelsd")
        step = lstsq[0].to(p.device)

        p_copy = p.data.clone()
        p.data.add_(step)
        with torch.inference_mode(True):
            updated_loss = closure.evaluate_least_squares_loss()
        loss_ratio = (updated_loss / ls_loss).item()
        accepted = _math.isfinite(loss_ratio) and loss_ratio <= group["tolerance"]
        if not accepted:
            p.data = p_copy
        lam_before = lam
        lam = (lam / group["lam_decrease_factor"] if loss_ratio <= 1.0
               else lam * group["lam_increase_factor"])

        history.append({
            "iteration": it,
            "loss_before": _num(ls_loss),
            "loss_after": _num(updated_loss),
            "loss_ratio": _num(loss_ratio),
            "lm_parameter": _num(lam_before),
            "lm_parameter_next": _num(lam),
            "rank": int(lstsq.rank.item()),
            "step_norm": _num(step.norm()),
            "grad_norm": _num((rj.T @ res).norm()),
            "accepted": bool(accepted),
        })

    TRACE[S] = TRACE.get(S, {})
    TRACE[S]["iteration_history"] = {
        "type": "table",
        "note": "one row per LM iteration from the perturbed start",
        "rows": history,
    }
    probe(S, "loss_first_iteration", history[0]["loss_before"])
    probe(S, "loss_last_iteration", history[-1]["loss_after"])
    probe(S, "n_accepted_steps", sum(h["accepted"] for h in history))
    probe(S, "loss_reduction_factor",
          history[0]["loss_before"] / max(history[-1]["loss_after"], 1e-30))
    return history


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
def render_markdown():
    """Render TRACE as markdown tables, one section per stage."""
    lines = ["# Trace tables", "",
             "Generated by `docs/walkthrough/instrument_trace.py`. Every number the",
             "walkthrough chapters quote comes from here.", ""]
    for section in sorted(TRACE):
        lines += [f"## {section}", ""]
        simple = {k: v for k, v in TRACE[section].items()
                  if v.get("type") not in ("layout", "table", "step_list",
                                           "event_list", "dict_values")}
        if simple:
            lines += ["| probe | type | shape | dtype | grad | value / stats |",
                      "|---|---|---|---|---|---|"]
            for k, v in simple.items():
                shape = v.get("shape", "")
                if v.get("type") == "Tensor" and "values" in v:
                    val = ", ".join(str(x) for x in v["values"])
                elif v.get("type") == "Tensor" and "min" in v:
                    val = (f"n={v['numel']} min={v['min']} max={v['max']} "
                           f"mean={v['mean']}")
                elif v.get("type") == "Tensor" and "true_fraction" in v:
                    val = f"n={v['numel']} true_fraction={v['true_fraction']}"
                else:
                    val = v.get("value", v.get("repr", v.get("keys", "")))
                val = str(val).replace("|", "\\|").replace("\n", " ")[:180]
                lines.append(
                    f"| `{k}` | {v.get('type','')} | {shape} | "
                    f"{v.get('dtype','')} | {v.get('requires_grad','')} | {val} |"
                )
            lines.append("")
        for k, v in TRACE[section].items():
            if v.get("type") == "layout":
                lines += [f"### {k}", "",
                          f"Total elements: {v['total_numel']}, "
                          f"trainable: {v['total_trainable']}", "",
                          "| key | shape | numel | slice | n_trainable |",
                          "|---|---|---|---|---|"]
                for kk, vv in v["keys"].items():
                    lines.append(f"| `{kk}` | {vv['shape']} | {vv['numel']} | "
                                 f"{vv['slice']} | {vv['n_trainable']} |")
                lines.append("")
            elif v.get("type") == "table":
                lines += [f"### {k}", ""]
                if v.get("note"):
                    lines += [v["note"], ""]
                rows = v["rows"]
                if rows:
                    cols = list(rows[0])
                    lines += ["| " + " | ".join(cols) + " |",
                              "|" + "---|" * len(cols)]
                    for row in rows:
                        lines.append("| " + " | ".join(
                            str(row.get(c, "")) for c in cols) + " |")
                lines.append("")
            elif v.get("type") == "step_list":
                lines += [f"### {k}", "", v.get("note", ""), "",
                          "| # | event | r shape | ok | backtrack | TIR | "
                          "backward | miss | event_info |", "|---|" + "---|" * 8]
                for s in v["steps"]:
                    c = s["status_counts"]
                    lines.append(
                        f"| {s['event_index']} | `{s['event_type']}` | "
                        f"{s['r']['shape']} | {c['ok_0']} | {c['backtrack_1']} | "
                        f"{c['tir_2']} | {c['backward_3']} | {c['miss_4']} | "
                        f"{', '.join(s['event_info']) or '-'} |")
                lines.append("")
            elif v.get("type") == "dict_values":
                lines += [f"### {k}", "", "| key | value |", "|---|---|"]
                for kk, vv in v["values"].items():
                    lines.append(f"| `{kk}` | {vv} |")
                lines.append("")
    return "\n".join(lines)


def get_real_batch(cli_obj):
    """Return one real (images, field_limits) batch from the configured data module.

    NOTE: configs/toy/defaults_e2e.yml ships train_folder/val_folder pointing at
    `configs/ablation/sample_image`, which does not exist in the repository. The
    driver below overrides both to `configs/toy/sample_image`, which does.
    """
    dm = cli_obj.datamodule
    dm.setup("fit")
    return next(iter(dm.train_dataloader()))


def main():
    configs = ["configs/toy/defaults.yml", "configs/toy/designs/optimized_spot.yml"]
    e2e_configs = [
        "configs/toy/defaults.yml",
        "configs/toy/defaults_e2e.yml",
        "configs/toy/designs/optimized_e2e.yml",
    ]
    e2e_overrides = (
        "--data.train_folder=configs/toy/sample_image",
        "--data.val_folder=configs/toy/sample_image",
    )
    TRACE["0_meta"] = {
        "configs_spot": _describe(configs),
        "configs_e2e": _describe(list(e2e_configs)),
        "e2e_path_override": _describe(list(e2e_overrides),
            "configs/ablation/sample_image does not exist in the repo; "
            "overridden to configs/toy/sample_image"),
        "torch_version": _describe(torch.__version__),
        "seed": _describe(0),
        "note": _describe("all values produced by docs/walkthrough/instrument_trace.py"),
    }
    module, _ = build_module(configs)
    lens = stage_config_and_lens(module)
    r, d = stage_ray_initialization(module, lens)
    stage_ray_trace(module, lens, r, d)
    stage_residuals(module, None)
    stage_lm_step(module)
    stage_lm_convergence(configs)

    # End-to-end path: needs the e2e config and a real image batch
    e2e_module, e2e_cli = build_module(e2e_configs, e2e_overrides)
    batch = get_real_batch(e2e_cli)
    images, field_limits = batch
    probe("6_generalized_aberrations", "batch_images", images)
    probe("6_generalized_aberrations", "batch_field_limits", field_limits)
    stage_generalized_aberrations(e2e_module, batch)
    psf_grid = stage_psf_and_image(e2e_module, batch)
    if psf_grid is not None:
        blurred = e2e_module.optics_simulator.apply_optics_model(images, psf_grid)
        stage_restoration(images, blurred.detach(), psf_grid.detach())

    (OUT_DIR / "trace_dump.json").write_text(json.dumps(TRACE, indent=1))
    (OUT_DIR / "trace_tables.md").write_text(render_markdown())
    print(f"sections: {sorted(TRACE)}")
    print(f"wrote {OUT_DIR/'trace_dump.json'} and {OUT_DIR/'trace_tables.md'}")


if __name__ == "__main__":
    main()
