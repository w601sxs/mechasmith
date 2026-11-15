"""mechasmith command line.

    mechasmith describe <design.py>             parameters, ranges, poses (JSON)
    mechasmith build <design.py> [--set k=v]    meshes, MJCF, URDF, checks, renders, viewer (JSON report)

Output is JSON on stdout so an agent can parse it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

from . import checks, mjcf, render, sim, urdf, viewer
from .design import Design, find_design


def load_module(path: Path):
    path = Path(path)
    if str(path.parent.resolve()) not in sys.path:
        sys.path.insert(0, str(path.parent.resolve()))  # designs can import each other
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def load_design(path) -> Design:
    """Import a design file and return its @design object."""
    return find_design(load_module(path))


def _parse_sets(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        k, _, v = pair.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = {"True": True, "False": False, "None": None}.get(v, v)     # Python spellings too
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mechasmith")
    sub = ap.add_subparsers(dest="cmd", required=True)
    dsc = sub.add_parser("describe", help="list a design's parameters, ranges and poses")
    dsc.add_argument("design", type=Path)
    b = sub.add_parser("build", help="build a design file")
    b.add_argument("design", type=Path)
    b.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                   help="override a design parameter (repeatable)")
    b.add_argument("--out", type=Path, default=None)
    b.add_argument("--no-sweep", action="store_true", help="skip the joint sweep interference check")
    b.add_argument("--steps", type=int, default=24)
    b.add_argument("--sim", choices=["mjx"], default=None,
                   help="also write a training model, robot_<sim>.xml (primitive collisions, no meshes, floor)")
    args = ap.parse_args(argv)

    if args.cmd == "describe":
        print(json.dumps(load_design(args.design).describe(), indent=2))
        return

    # Keep stdout for the JSON report: anything else (geometry-kernel warnings printed from C++,
    # library prints) goes to stderr while the build runs.
    import os
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    try:
        report = _build(args)
    finally:
        sys.stdout.flush()
        try:                                  # C-level stdio buffers too (OpenCascade prints there)
            import ctypes
            ctypes.CDLL(None).fflush(None)
        except Exception:
            pass
        os.dup2(saved, 1)
        os.close(saved)
    print(json.dumps(report, indent=2))


def _build(args) -> dict:
    t0 = time.time()
    dz = load_design(args.design)
    overrides = _parse_sets(args.set)
    robot = dz.build(**overrides)
    out = args.out or Path("build") / args.design.stem
    out.mkdir(parents=True, exist_ok=True)
    scene = mjcf.export(robot, out, base_height=dz.base_height, keyframes={"home": dz.home, **dz.poses},
                        sim=args.sim)
    lift = 0.0 if robot.floating else (-mjcf.lowest_point_mm(robot) * 1e-3 if dz.base_height == "auto"
                                       else float(dz.base_height))
    urdf_path = urdf.export(robot, out, base_height=lift)
    t_export = time.time() - t0

    manifest = json.loads((out / "manifest.json").read_text())
    report = {"design": str(args.design), "params": dz.resolve(**overrides).as_dict(), "out": str(out),
              "scene": str(scene), "urdf": str(urdf_path), "mass": checks.mass_report(manifest)}
    if args.sim:
        report["sim_model"] = str(out / f"robot_{args.sim}.xml")
    home = dz.home
    if args.no_sweep:
        pg = checks.PosedGeometry(scene)
        pg.set_pose(home)
        report["interference"] = {"home_collisions": pg.collisions()}
    else:
        report["interference"] = checks.interference_report(scene, steps=args.steps, base_pose=home)
    report["renders"] = [str(render.contact_sheet(scene, out / "views.png", qpos=home, title=robot.name))]
    for name, pose in dz.poses.items():
        report["renders"].append(str(render.contact_sheet(scene, out / f"pose_{name}.png", qpos=pose,
                                                          title=f"{robot.name}: {name}")))
    if robot.rotors:
        report["hover"] = sim.hover_test(scene, pose=home)
    elif robot.floating:
        st = sim.stand_test(scene, home)
        report["stand"] = st
        if "final_qpos" in st:
            final = st.pop("final_qpos")
            report["renders"].append(str(render.contact_sheet(
                scene, out / "stand.png", full_qpos=final,
                title=f"{robot.name}: after {st['seconds']} s holding HOME ({'standing' if st['stands'] else 'FELL'})")))
    if not robot.floating and dz.tool_site:
        report["arm"] = sim.arm_test(scene, home, dz.tool_site, dz.payload_kg)
    report["viewer"] = str(viewer.write_viewer(out, {"home": home, **dz.poses}, report))
    report["timing_s"] = {"export": round(t_export, 1), "total": round(time.time() - t0, 1)}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
