#!/usr/bin/env python3
"""Gate check for a finished TR-grid run. Exit 0 = pass, 1 = fail.

Pass = final ray_valid is 100%. Prints the final spot error and PSNR (if logged)
so the run summary has the numbers.
"""
import glob
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run_dir = sys.argv[1]
files = sorted(glob.glob(f"{run_dir}/events*"))
if not files:
    print(f"FAIL no event file in {run_dir}")
    sys.exit(1)
ea = EventAccumulator(files[-1], size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags()["scalars"]


def last(tag):
    return ea.Scalars(tag)[-1] if tag in tags else None


rv = last("ray_tracing/ray_valid")
tra = last("loss/transverse_ray_aberration")
psnr = [t for t in tags if "psnr" in t.lower() and "val" in t.lower()]
msg = f"step={rv.step if rv else '?'} ray_valid={rv.value if rv else 'n/a'}"
msg += f" spot={tra.value:.3e}" if tra else ""
msg += "".join(f" {t}={last(t).value:.2f}" for t in psnr)
ok = rv is not None and rv.value >= 0.999
print(("PASS " if ok else "FAIL ") + msg)
sys.exit(0 if ok else 1)
