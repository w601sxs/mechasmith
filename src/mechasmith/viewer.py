"""Self-contained browser viewer: the robot's meshes, its joint tree and a slider per joint.

One HTML file with the meshes embedded, so it opens from disk or publishes as-is.
Forward kinematics runs in the page: every body frame has identity orientation at
the home pose and sits at its parent joint's anchor, the same convention as the MJCF.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("viewer_template.html")


def write_viewer(out_dir: Path, poses: dict[str, dict], report: dict | None = None) -> Path:
    out_dir = Path(out_dir)
    man = json.loads((out_dir / "manifest.json").read_text())
    bodies = []
    for name, b in man["bodies"].items():
        parts = []
        for p in b["parts"]:
            stl = (out_dir / p["mesh"]).read_bytes()
            parts.append({"name": p["name"], "color": p.get("color", [0.8, 0.8, 0.8, 1]),
                          "stl": base64.b64encode(stl).decode()})
        bodies.append({"name": name, "parent": b.get("parent"), "origin": b["origin_mm"],
                       "joint": b.get("joint"), "mass_g": round(b["mass"] * 1000, 1), "parts": parts})
    summary = {}
    if report:
        summary["mass_g"] = report.get("mass", {}).get("total_g")
        st = report.get("stand")
        if st and "stands" in st:
            summary["stand"] = {k: st.get(k) for k in ("stands", "settled_pitch_deg", "worst_servo",
                                                       "pd_stiffness_ratio")}
        inter = report.get("interference", {})
        if "joints" in inter:
            summary["limited"] = {j: v["usable"] for j, v in inter["joints"].items() if v["blocked"]}
        summary["home_collisions"] = len(inter.get("home_collisions", []))
    data = {"robot": man["robot"], "bodies": bodies, "poses": poses, "summary": summary}
    html = TEMPLATE.read_text().replace("__ROBOT_NAME__", man["robot"]).replace(
        "__DATA__", json.dumps(data, separators=(",", ":")))
    path = out_dir / "viewer.html"
    path.write_text(html)
    return path
