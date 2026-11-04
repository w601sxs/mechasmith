"""`mechasmith build <design.py>`: one command -> meshes, MJCF, checks, renders.

Prints a JSON report on stdout so an agent can parse it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

from . import checks, mjcf, render


def load_design(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mechasmith")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build a design file")
    b.add_argument("design", type=Path)
    b.add_argument("--out", type=Path, default=None)
    b.add_argument("--no-sweep", action="store_true", help="skip the joint sweep interference check")
    b.add_argument("--steps", type=int, default=24)
    args = ap.parse_args(argv)

    t0 = time.time()
    mod = load_design(args.design)
    robot = mod.build()
    out = args.out or Path("build") / args.design.stem
    out.mkdir(parents=True, exist_ok=True)
    base_height = getattr(mod, "BASE_HEIGHT", "auto")
    scene = mjcf.export(robot, out, base_height=base_height)
    t_export = time.time() - t0

    manifest = json.loads((out / "manifest.json").read_text())
    report = {"design": str(args.design), "out": str(out), "scene": str(scene),
              "mass": checks.mass_report(manifest)}
    home = getattr(mod, "HOME", {})
    if args.no_sweep:
        pg = checks.PosedGeometry(scene)
        pg.set_pose(home)
        report["interference"] = {"home_collisions": pg.collisions()}
    else:
        report["interference"] = checks.interference_report(scene, steps=args.steps, base_pose=home)
    report["renders"] = [str(render.contact_sheet(scene, out / "views.png", qpos=home, title=robot.name))]
    for name, pose in getattr(mod, "POSES", {}).items():
        report["renders"].append(str(render.contact_sheet(scene, out / f"pose_{name}.png", qpos=pose,
                                                          title=f"{robot.name}: {name}")))
    report["timing_s"] = {"export": round(t_export, 1), "total": round(time.time() - t0, 1)}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
