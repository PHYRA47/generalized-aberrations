#!/usr/bin/env python3
"""Emit a cellphone design config pinned to a given total track length.

Mirrors the convention in configs/telephoto/designs/, where every grid cell is a
standalone design file carrying (a) a converged prescription and (b) its own
`total_track_length_solve` value. Those files cannot be hand-authored for the
cellphone because the prescription at each TTL level is the OUTPUT of the
previous stage's optimization -- so this script extracts it.

Usage
-----
Seed the first stage from the as-filed patent (the only TTL directly reachable
from it is >= 4.5; see the feasibility check below):

    python tools/emit_ttl_design.py \
      --source configs/cellphone/designs/param_patent_native.yml \
      --ttl 4.50 --arm spot \
      --out configs/cellphone/designs/5p_ttl450_spot.yml

Seed a later stage from a finished run's log directory (reads the highest-numbered
lens_parameters dump):

    python tools/emit_ttl_design.py \
      --source logs/logs_cellphone/R-s-aRa-aRa-aRa-aRa-aRa-R-/5p_ttl450_spot \
      --ttl 4.25 --arm spot \
      --out configs/cellphone/designs/5p_ttl425_spot.yml

Why the feasibility check matters
---------------------------------
`total_track_length_solve` pins TTL by COMPUTING the last spacing:
s[-1] = ttl - sum(s[:-1]). It does this unconditionally. Pinning the as-filed
patent (sum(s) = 4.802, s[-1] = 0.475) to 4.0 yields s[-1] = -0.327 -- the image
plane behind the last surface -- and the LM solve fails with
"Intel oneMKL ERROR: Parameter 4 was incorrect on entry to DGELSD".
This script refuses to emit such a file, which is the whole reason the staircase
has to proceed in steps small enough that each stage leaves enough back focal
distance for the next.
"""

import argparse
import glob
import os
import numpy as np
import yaml

# Template supplying every init_args key that is NOT part of the prescription.
# Read from the baseline so the grid inherits one definition of freeze, EFL
# solve, and asphere parameterization rather than duplicating them per cell.
BASELINE = "configs/cellphone/designs/param_patent_native.yml"
PRESCRIPTION_KEYS = ("a", "c", "s", "nd", "vd")
MIN_BFD = 0.05  # matches RayPathResiduals min_cutoff in configs/cellphone/defaults.yml


def load_prescription(source):
    """Return (prescription_dict, provenance_string) from a design file or a log dir."""
    if os.path.isdir(source):
        dumps = sorted(glob.glob(os.path.join(source, "lens_parameters", "*.yml")))
        if not dumps:
            raise SystemExit(f"no lens_parameters/*.yml under {source}")
        path = dumps[-1]
        raw = yaml.safe_load(open(path))
        return raw, f"{source} (dump {os.path.basename(path)})"
    raw = yaml.safe_load(open(source))
    if "model" in raw:
        raw = raw["model"]["lens_parameterization"]["init_args"]
    return raw, source


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="design .yml, or a run log dir")
    p.add_argument("--ttl", type=float, required=True, help="total track length to pin, mm")
    p.add_argument("--arm", choices=["spot", "e2e"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--baseline", default=BASELINE)
    args = p.parse_args()

    src, provenance = load_prescription(args.source)
    base = yaml.safe_load(open(args.baseline))["model"]["lens_parameterization"]["init_args"]

    init = dict(base)
    for k in PRESCRIPTION_KEYS:
        if k not in src:
            raise SystemExit(f"source is missing prescription key '{k}'")
        init[k] = src[k]
    init["total_track_length_solve"] = args.ttl
    init["paraxial_image_solve"] = False  # mutually exclusive with the TTL solve

    # --- stop gap (s[2]): clamp non-negative, then freeze -----------------------
    # defaults.yml exempts propagation index 2 from the minimum-gap floor
    # (`other_min_cutoffs: [[2, -.inf]]`, see README.md:84-88), so s[2] is a free
    # variable with NO lower bound. Pinning a shorter TTL forces the optimizer to
    # find axial room, and s[2] is the cheapest place to take it. Observed twice:
    #   rung4_min_ttl   s[2] 0.0000 -> -0.1330, ray_valid 1.00 -> 0.136
    #   5p_ttl425_spot  s[2] +0.0136 -> -0.0829, ray_valid 1.00 -> 0.254
    # Both put L1's front vertex inside the cover glass. In both cases
    # loss/ray_path stayed at ~1e-6 and reported the geometry as healthy, so this
    # cannot be caught by watching the loss -- it has to be prevented.
    # Freezing is safe here only BECAUSE the exemption stays in place: with no
    # hinge violation to relieve, nothing pushes L1's front surface to bend
    # instead (the failure mode README.md:84-88 warns about).
    s = np.asarray(init["s"], dtype=float)
    STOP_IDX = 2
    if s[STOP_IDX] < 0.0:
        print(f"note        : clamping s[{STOP_IDX}] {s[STOP_IDX]:+.5f} -> 0.0")
        s[STOP_IDX] = 0.0
    init["s"] = list(map(float, s))

    freeze = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base["freeze"].items()}
    s_toggle = list(freeze["s"]["toggle_row_col_list"])
    if STOP_IDX not in s_toggle:
        s_toggle.insert(1, STOP_IDX)
    freeze["s"] = {"default": freeze["s"]["default"], "toggle_row_col_list": s_toggle}
    init["freeze"] = freeze

    required_bfd = args.ttl - s[:-1].sum()
    neg = [(i, round(float(v), 5)) for i, v in enumerate(s) if v < 0]

    print(f"source      : {provenance}")
    print(f"source TTL  : {s.sum():.4f} mm   s[-1] = {s[-1]:.4f}")
    print(f"target TTL  : {args.ttl:.4f} mm")
    print(f"solved s[-1]: {required_bfd:+.4f} mm  (floor {MIN_BFD})")
    if neg:
        raise SystemExit(f"REFUSING: source prescription already has negative spacings {neg}")
    # Tolerance is set by the PRECISION OF THE DUMPS, not by binary float error.
    # lens_parameters/*.yml stores ~7 significant figures (e.g. s[-1] as
    # 0.2999999), so a stage that parked exactly on the 0.30 BFD floor reads back
    # as 3.70000013 for the front group. The next 0.25 mm step then computes
    # s[-1] = 0.04999987 -- short of 0.05 by 1.3e-7, which is rounding noise, not
    # geometry. Without this tolerance the staircase would refuse a step that is
    # physically identical to the TTL 4.00 stage, which ran to ray_valid 1.0.
    if required_bfd < MIN_BFD - 1e-6:
        raise SystemExit(
            f"REFUSING: target TTL {args.ttl} needs s[-1] = {required_bfd:+.4f} < {MIN_BFD}.\n"
            f"  Take a smaller step, or re-seed from a stage whose front group is shorter."
        )

    # Write the SOLVED last spacing into the stored prescription so the file is
    # self-consistent: sum(s) == total_track_length_solve, matching every file in
    # configs/telephoto/designs/ (e.g. 5p_tr70_*.yml store sum(s) == 18.13 == their
    # own pinned TTL). Functionally redundant -- the solve recomputes s[-1] at
    # runtime regardless -- but it means the file can be read for its true geometry
    # instead of the pre-solve source geometry, and makes the feasibility of the
    # emitted cell visible from the file itself.
    s_out = list(map(float, s))
    s_out[-1] = float(required_bfd)
    init["s"] = s_out
    assert abs(sum(s_out) - args.ttl) < 1e-9, f"sum(s)={sum(s_out)} != ttl={args.ttl}"

    header = (
        f"# 5-element cellphone, TTL pinned to {args.ttl:.2f} mm, {args.arm} arm.\n"
        f"# EFL stays solved at {init['target_efl']} (solve_idx {init['solve_idx']}), as in every\n"
        f"# cell of the grid -- that is what makes a shorter TTL a real result rather than\n"
        f"# just a smaller lens. Compare against configs/telephoto/designs/, where all 70\n"
        f"# files likewise fix EFL at 25.9 and vary only the pinned track length.\n"
        f"#\n"
        f"# TTL is pinned by total_track_length_solve, NOT by TotalTrackLengthResiduals.\n"
        f"# The solve satisfies TTL by construction at every step, so nothing rewards\n"
        f"# collapsing a gap and the unbounded s[2] stop exemption cannot be exploited.\n"
        f"#\n"
        f"# Prescription extracted by tools/emit_ttl_design.py from:\n"
        f"#   {provenance}\n"
        f"# Non-prescription init_args inherited from {args.baseline}.\n"
        f"# Solved back focal distance at this TTL: {required_bfd:+.4f} mm.\n"
        f"# GENERATED FILE -- regenerate rather than editing by hand.\n"
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(header)
        yaml.safe_dump(
            {"model": {"lens_parameterization": {"init_args": init}}},
            f, default_flow_style=None, sort_keys=False, width=88,
        )
    print(f"wrote       : {args.out}")


if __name__ == "__main__":
    main()
