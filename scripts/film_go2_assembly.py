"""Film the Go2 being made: its torso designed step by step as a hologram, with the lines of
designs/go2.py that make it, then the other 40 parts flying in and locking into place, and a
half turn round the finished robot as it fades out. 25 s, 1080x1080, 24 fps.

usage: python scripts/film_go2_assembly.py build go2_assembly.mp4 [--stills 1.2,8,14 stills.png] [--workers 4]

Build the design first (`mechasmith build designs/go2.py`). The torso stages repeat the CSG in
go2.py with its default parameters (`torso_stages` below), so keep the two in step. Needs ffmpeg,
and an OpenGL context: on a headless machine run under Xvfb with MUJOCO_GL=glfw.
"""
import shutil
import subprocess
import sys
import tempfile
import warnings
from multiprocessing import get_context
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mechasmith import GO_M8010_6, box, cyl, swept_link
from mechasmith.render import _rest_on_floor

FPS, SIZE, SS = 24, 1080, 2
DURATION = 25.0
TURNTABLE = (15.0, 25.0)  # half a turn round the finished robot
FADE_OUT = 5.0  # seconds of fade to black at the end
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_MB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_B = "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"
FONT_S = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"

HOLO, ADD, CUT = (0.30, 0.85, 1.0), (0.35, 1.0, 0.55), (1.0, 0.32, 0.30)

# ---- design phase: (start s, tool meshes, tool colour, tool slide-in offset m, result mesh, code line)
STEPS = [
    (0.6, ["s1_outer"], ADD, (0, 0, 0.0), "s1_outer", 0),
    (1.75, ["t_inner"], CUT, (0, 0, 0.0), "s2_shell", 1),
    (2.9, ["t_bridge"], ADD, (0, 0, 0.18), "s3_bridge", 2),
    (4.05, ["t_head"], ADD, (0.22, 0, 0), "s4_torso", 3),
    (5.2, ["t_room0", "t_room1", "t_room2", "t_room3"], CUT, (0, 0, 0.0), "s5_carved", 5),
]
SOLIDIFY = (6.6, 7.3)
CODE = [
    "shell = box((-188, -47, -57), (188, 47, 57)) + box(...)",
    "      - box((-185, -44, -54), (185, 44, 54)) - box(...)",
    "bridge = box((-250, -30, 55), (250, 30, 62))",
    "head = cyl((285, 0, 10), Z, 50, 90) + cyl((280, 0, -50), X, 40, 50)",
    "torso = shell + bridge + head",
    "for r in rooms:   # what each hip sweeps through +-60 deg",
    "    torso = torso - r",
    'base = Component.world("base", torso, BOUGHT, BLACK)',
]
CODE_AT = [0.6, 1.75, 2.9, 4.05, 4.4, 5.2, 5.5, 6.6]  # when each line types in

# ---- assembly phase: groups fly in leg by leg, joint by joint
ASSEMBLY_START = 8.4
GROUPS = [  # (geom name suffixes, the call that places them)
    (["abduction_motor_case", "abduction_motor_flange"], 'asm.fasten(base.c("FL_abd_seat"), abd.case.c("back"))'),
    (["hip_housing"], "asm.attach(abd.horn, hip)"),
    (["thigh_motor_case", "thigh_motor_flange"], 'asm.fasten(hip.c("thigh_motor_seat"), thigh_motor.case.c("back"))'),
    (["thigh_link"], "asm.attach(thigh_motor.horn, thigh)"),
    (["knee_motor_case", "knee_motor_flange"], 'asm.fasten(thigh.c("knee_motor_seat"), knee_motor.case.c("back"))'),
    (["calf_link"], 'asm.hinge("FL_calf_joint", thigh, calf, knee, Y, ...)'),
    (["foot"], "asm.attach(calf, foot)"),
]
GROUP_GAP, FLIGHT, SNAP = 0.5, 1.05, 0.16
END_CARD = 13.2

WORLD = """
<mujoco model="clip2">
  <compiler meshdir="{meshdir}"/>
  <visual>
    <headlight diffuse="0.45 0.45 0.47" ambient="0.25 0.25 0.27" specular="0.1 0.1 0.1"/>
    <global fovy="36"/>
    <quality shadowsize="8192" offsamples="8"/>
    <map shadowclip="1.0" znear="0.005" fogstart="0.9" fogend="2.2"/>
    <rgba fog="0.10 0.11 0.13 1"/>
  </visual>
  <statistic center="0 0 0.3" extent="3"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.10 0.11 0.13" rgb2="0.10 0.11 0.13" width="512" height="512"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.13 0.14 0.16" rgb2="0.11 0.12 0.14"
             markrgb="0.18 0.30 0.36" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="30 30" reflectance="0"/>
    {meshes}
    {materials}
  </asset>
  <worldbody>
    <light pos="1.2 -0.8 2.5" dir="-0.45 0.3 -1" directional="true" castshadow="true" diffuse="0.75 0.75 0.75"/>
    <light pos="-1.5 1.0 1.2" dir="1 -0.6 -0.35" directional="true" castshadow="false" diffuse="0.35 0.5 0.65" specular="0.6 0.7 0.8"/>
    <light pos="-1.5 -1.0 1.2" dir="1 0.8 -0.3" directional="true" castshadow="false" diffuse="0.2 0.28 0.36" specular="0.5 0.6 0.7"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="grid"/>
    <body name="stages" pos="{pos}">
      {geoms}
    </body>
  </worldbody>
</mujoco>
"""
STAGE_NAMES = ["s1_outer", "s2_shell", "s3_bridge", "s4_torso", "s5_carved", "t_inner", "t_bridge", "t_head",
               "t_room0", "t_room1", "t_room2", "t_room3"]


def font(path, size):
    """The named font, or Pillow's built-in one where it isn't installed."""
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size)


def torso_stages(out: Path):
    """Each step of the torso CSG in designs/go2.py (default parameters) as an STL in `out`."""
    from build123d import export_stl
    M, X, Z = GO_M8010_6, (1.0, 0, 0), (0, 0, 1.0)
    hx, hy, my, hw = 193.4, 46.5, 72.0, 47.0
    hip_reach = my + M.length / 2 + 2
    rooms = [swept_link((sx * hx, sy * hy, 0), X, [(sx * hx, sy * hy, 0), (sx * hx, sy * hip_reach, 0)],
                        M.diameter + 6, (-1.0472, 1.0472), (-58, 58), gap=2.0)
             for sx in (1, -1) for sy in (1, -1)]
    outer = box((-188, -47, -57), (188, 47, 57)) + box((-135, -hw, -57), (135, hw, 57))
    inner = box((-185, -44, -54), (185, 44, 54)) + box((-132, -hw + 3, -54), (132, hw - 3, 54))
    shell = outer - inner
    bridge = box((-250, -30, 55), (250, 30, 62))
    head = cyl((285, 0, 10), Z, 50, 90) + cyl((280, 0, -50), X, 40, 50)
    torso = shell + bridge + head
    carved = torso
    for r in rooms:
        carved = carved - r
    shapes = dict(s1_outer=outer, s2_shell=shell, s3_bridge=shell + bridge, s4_torso=torso, s5_carved=carved,
                  t_inner=inner, t_bridge=bridge, t_head=head, **{f"t_room{i}": r for i, r in enumerate(rooms)})
    for name, shape in shapes.items():
        export_stl(shape, str(out / f"{name}.stl"), tolerance=0.05, angular_tolerance=0.15)


def ease_io(x):
    x = np.clip(x, 0, 1)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out(x):
    return 1 - (1 - np.clip(x, 0, 1)) ** 3


def ramp(t, t0, t1):
    return float(np.clip((t - t0) / (t1 - t0), 0, 1))


def axis_angle_mat(axis, ang):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), ang)
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


# ---------------------------------------------------------------------------------------------
class Film:
    def __init__(self, build: Path, stages: Path):
        alone = mujoco.MjModel.from_xml_path(str(build / "go2" / "robot.xml"))
        d = mujoco.MjData(alone)
        d.qpos[:] = alone.key("home").qpos
        mujoco.mj_forward(alone, d)
        _rest_on_floor(alone, d)
        self.home_qpos = d.qpos.copy()
        base_pos = d.xpos[alone.body("base").id]

        meshes = "\n    ".join(f'<mesh name="{n}" file="{(stages / (n + ".stl")).resolve()}" scale="0.001 0.001 0.001"/>'
                               for n in STAGE_NAMES)
        materials = "\n    ".join(f'<material name="m_{n}" emission="0.55" specular="0.6" shininess="0.6"/>'
                                   for n in STAGE_NAMES)
        geoms = "\n      ".join(f'<geom name="{n}" type="mesh" mesh="{n}" material="m_{n}" group="2" '
                                f'contype="0" conaffinity="0" rgba="0.3 0.85 1 0.4"/>' for n in STAGE_NAMES)
        world = mujoco.MjSpec.from_string(WORLD.format(
            meshdir=(build / "go2" / "meshes").resolve(), meshes=meshes, materials=materials, geoms=geoms,
            pos=" ".join(map(str, base_pos))))
        child = mujoco.MjSpec.from_file(str(build / "go2" / "robot.xml"))
        child.meshdir = str((build / "go2" / "meshes").resolve())
        world.attach(child, frame=world.worldbody.add_frame(), prefix="go2/")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m = world.compile()
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = self.home_qpos  # the only joints are the robot's, in the same order
        mujoco.mj_forward(self.m, self.d)
        m = self.m
        self.base_rgba = m.geom_rgba.copy()
        self.stage = {n: m.geom(n).id for n in STAGE_NAMES}
        self.stage_pos0 = {n: m.geom_pos[g].copy() for n, g in self.stage.items()}
        self.go2 = [g for g in range(m.ngeom) if m.geom(g).name.startswith("go2/") and m.geom_group[g] == 2]
        self.base_geom = m.geom("go2/base_visual").id
        self.inside = [m.geom("go2/battery_visual").id, m.geom("go2/computer_visual").id]
        self.torso_center = self.d.xpos[m.body("go2/base").id] + np.array([0.04, 0, 0])

        # Flight plan for every leg part: final world pose, a far start pose and a start time.
        rng = np.random.default_rng(7)
        self.flights = {}
        for gi, (suffixes, _) in enumerate(GROUPS):
            for leg_i, leg in enumerate(("FL", "FR", "RL", "RR")):
                for suf in suffixes:
                    g = m.geom(f"go2/{leg}_{suf}_visual").id
                    pos, mat = self.d.geom_xpos[g].copy(), self.d.geom_xmat[g].reshape(3, 3).copy()
                    out = pos - self.torso_center
                    out[2] = 0
                    out = out / (np.linalg.norm(out) + 1e-9)
                    far = out * rng.uniform(0.9, 1.3) + np.array([0, 0, rng.uniform(0.25, 0.7)]) \
                        + rng.normal(0, 0.12, 3)
                    axis = rng.normal(size=3)
                    self.flights[g] = dict(
                        target=pos, mat=mat, start=pos + far, out=out, axis=axis / np.linalg.norm(axis),
                        angle=rng.uniform(1.2, 2.4) * np.pi * rng.choice([-1, 1]),
                        t0=ASSEMBLY_START + gi * GROUP_GAP + leg_i * 0.07 + rng.uniform(0, 0.04), group=gi)
        pts = np.array([self.d.geom_xpos[g] for g in self.go2])
        self.robot_center = (pts.min(0) + pts.max(0)) / 2
        self.landings = sorted(f["t0"] + FLIGHT for f in self.flights.values())

    # ---- per-frame state ---------------------------------------------------------------------
    def part_pose(self, f, t):
        u = (t - f["t0"]) / FLIGHT
        if u <= 0:
            return None
        if u >= 1:
            return f["target"], f["mat"]
        approach = 1 - SNAP
        staging = f["target"] + f["out"] * 0.07 + np.array([0, 0, 0.03])
        if u < approach:
            s = ease_io(u / approach)
            pos = f["start"] + (staging - f["start"]) * s + np.array([0, 0, 0.18 * np.sin(np.pi * s)])
        else:
            s = ((u - approach) / SNAP) ** 2
            pos = staging + (f["target"] - staging) * s
        spin = f["angle"] * (1 - ease_out(u / 0.78))
        return pos, axis_angle_mat(f["axis"], spin) @ f["mat"]

    def apply(self, t):
        m = self.m
        m.geom_rgba[:] = self.base_rgba
        m.geom_group[self.go2] = 2
        # design phase: the hologram stages and tools
        for g in self.stage.values():
            m.geom_group[g] = 5
        current = None
        for k, (t0, tools, col, slide, result, _) in enumerate(STEPS):
            t_end = STEPS[k + 1][0] if k + 1 < len(STEPS) else SOLIDIFY[1]
            if t >= t0:
                current = (k, t0, result)
            merge = ramp(t, t0 + 0.55, t0 + 0.85)
            if t0 <= t < t_end:
                for i, n in enumerate(tools):
                    a = ramp(t, t0 + 0.08 * i, t0 + 0.08 * i + 0.4)
                    if a <= 0 or merge >= 1:
                        continue
                    g = self.stage[n]
                    m.geom_group[g] = 2
                    m.geom_pos[g] = self.stage_pos0[n] + np.array(slide) * (1 - ease_out(a))
                    m.geom_rgba[g] = (*col, 0.45 * a * (1 - merge))
                    m.mat_emission[m.geom_matid[g]] = 0.7
        if current is not None:
            k, t0, result = current
            merged = ramp(t, t0 + 0.55, t0 + 0.85)
            prev = STEPS[k - 1][4] if k > 0 else None
            fade = 1 - ramp(t, *SOLIDIFY)
            flash = np.exp(-((t - (t0 + 0.7)) / 0.12) ** 2)  # brightens as the step merges
            for n, w in ((prev, 1 - merged), (result, merged if k > 0 else ramp(t, t0, t0 + 0.6))):
                if n is None or w <= 0 or fade <= 0:
                    continue
                g = self.stage[n]
                m.geom_group[g] = 2
                m.geom_pos[g] = self.stage_pos0[n]
                c = np.array(HOLO) + (1 - np.array(HOLO)) * 0.5 * flash
                m.geom_rgba[g] = (*c, 0.42 * w * fade)
                m.mat_emission[m.geom_matid[g]] = 0.55 + 0.6 * flash
        # the real robot: the torso solidifies out of the hologram, the rest flies in
        solid = ramp(t, *SOLIDIFY)
        b = self.base_geom
        if solid <= 0:
            m.geom_group[b] = 5
        else:
            glow = 1 - solid
            m.geom_rgba[b] = (*(np.array(HOLO) * glow * 0.6), 0.25 + 0.75 * solid)
        for g in self.inside:
            m.geom_group[g] = 2 if solid >= 1 else 5
        for g, f in self.flights.items():
            if t < f["t0"]:
                m.geom_group[g] = 5
            land = t - (f["t0"] + FLIGHT)
            if 0 <= land < 0.35:  # a flash where the part locks in
                w = np.exp(-land / 0.09)
                m.geom_rgba[g, :3] = self.base_rgba[g, :3] + (np.array([0.55, 0.9, 1.0]) - self.base_rgba[g, :3]) * w

    def overrides(self, scene, t):
        """Move the in-flight parts in the built scene (their bodies stay put)."""
        for i in range(scene.ngeom):
            sg = scene.geoms[i]
            if sg.objtype != int(mujoco.mjtObj.mjOBJ_GEOM) or sg.objid not in self.flights:
                continue
            pose = self.part_pose(self.flights[sg.objid], t)
            if pose is not None:
                sg.pos[:] = pose[0]
                sg.mat[:] = pose[1]

    def camera(self, t):
        c = self.torso_center
        design = dict(lookat=c, distance=1.08 - 0.1 * ramp(t, 0, 7), azimuth=140 + 55 * ease_io(t / 7.4),
                      elevation=-24 + 4 * ease_io(t / 7.4))
        rc = self.robot_center
        wide = dict(lookat=rc + [0, 0, 0.05], distance=2.0, azimuth=212, elevation=-12)
        build_end = dict(lookat=rc + [0, 0, 0.02], distance=1.8, azimuth=236, elevation=-12)
        hero = dict(lookat=rc + [0, 0, -0.02], distance=1.62, azimuth=244, elevation=-8)

        def mix(a, b, s):
            return {k: np.asarray(a[k]) + (np.asarray(b[k]) - np.asarray(a[k])) * s for k in a}
        if t < 7.3:
            cam = design
        elif t < 8.7:
            cam = mix(design | dict(distance=0.98, azimuth=195, elevation=-20), wide, ease_io((t - 7.3) / 1.4))
        elif t < 12.6:
            cam = mix(wide, build_end, ease_io((t - 8.7) / 3.9))
        else:
            cam = mix(build_end, hero, ease_io((t - 12.6) / 2.4))
            cam["azimuth"] = cam["azimuth"] + 180 * ease_io(ramp(t, *TURNTABLE))
        # a small kick each time a part locks in
        shake = np.zeros(3)
        for tl in self.landings:
            dt = t - tl
            if 0 <= dt < 0.3:
                shake += 0.0012 * np.exp(-dt / 0.08) * np.array([np.sin(97 * dt), np.cos(71 * dt), np.sin(53 * dt)])
        cam["lookat"] = np.asarray(cam["lookat"], float) + shake
        return cam


# ---------------------------------------------------------------------------------------------
def overlay(img, t):
    d = ImageDraw.Draw(img, "RGBA")
    mono, monob = font(FONT_M, 17), font(FONT_MB, 17)
    small = font(FONT_S, 15)
    # code panel during the design
    vis = ramp(t, 0.3, 0.8) * (1 - ramp(t, 7.4, 8.0))
    if vis > 0:
        lh = 26
        x0, y0 = 36, SIZE - 36 - 24 - lh * len(CODE)
        d.rounded_rectangle((x0 - 18, y0 - 44, x0 + 712, SIZE - 36), 14, fill=(10, 12, 16, int(215 * vis)))
        d.text((x0, y0 - 34), "designs/go2.py  ·  the torso", font=small, fill=(150, 160, 175, int(255 * vis)))
        active = max([i for i, ta in enumerate(CODE_AT) if t >= ta], default=-1)
        for i, (line, ta) in enumerate(zip(CODE, CODE_AT)):
            n = int(len(line) * ramp(t, ta, ta + 0.5))
            if n <= 0:
                continue
            y = y0 + i * lh
            if i == active:
                d.rectangle((x0 - 8, y - 3, x0 + 700, y + lh - 5), fill=(40, 120, 160, int(70 * vis)))
            col = (235, 240, 245) if i == active else (150, 158, 170)
            d.text((x0, y), line[:n], font=monob if i == active else mono, fill=(*col, int(255 * vis)))
            if n < len(line) and int(t * 6) % 2 == 0:
                w = d.textlength(line[:n], font=mono)
                d.rectangle((x0 + w + 1, y, x0 + w + 10, y + 19), fill=(255, 200, 60, int(255 * vis)))
    # the call that places each group, while it lands
    starts = [ASSEMBLY_START + gi * GROUP_GAP + 0.3 for gi in range(len(GROUPS))]
    for gi, (_, call) in enumerate(GROUPS):
        t0 = starts[gi]
        t1 = starts[gi + 1] if gi + 1 < len(GROUPS) else t0 + 1.4
        a = ramp(t, t0, t0 + 0.08) * (1 - ramp(t, t1 - 0.08, t1))  # one call at a time
        if a > 0:
            w = d.textlength(call, font=mono)
            d.rounded_rectangle(((SIZE - w) / 2 - 16, SIZE - 92, (SIZE + w) / 2 + 16, SIZE - 52), 10,
                                fill=(10, 12, 16, int(200 * a)))
            d.text(((SIZE - w) / 2, SIZE - 82), call, font=mono, fill=(255, 200, 60, int(255 * a)))
    # end card
    a = ramp(t, END_CARD, END_CARD + 0.6)
    if a > 0:
        big, mid = font(FONT_B, 54), font(FONT_M, 21)
        d.text((56, SIZE - 150), "Unitree Go2", font=big, fill=(245, 246, 248, int(255 * a)))
        d.text((58, SIZE - 82), "43 parts  ·  12 joints  ·  147 lines of Python", font=mid,
               fill=(255, 200, 60, int(255 * a)))
    # fade in from black, fade out to black
    fade = min(ramp(t, 0, 0.5), 1 - ramp(t, DURATION - FADE_OUT, DURATION))
    if fade < 1:
        d.rectangle((0, 0, SIZE, SIZE), fill=(0, 0, 0, int(255 * (1 - fade))))
    return img


def render_frames(args):
    build, stages, times, outdir = args
    film = Film(Path(build), Path(stages))
    m = film.m
    px = SIZE * SS
    m.vis.global_.offwidth = m.vis.global_.offheight = px
    r = mujoco.Renderer(m, px, px)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = opt.geomgroup[2] = 1
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
    for idx, t in times:
        # motion blur while parts are flying: average sub-frames across the shutter
        flying = any(f["t0"] < t < f["t0"] + FLIGHT for f in film.flights.values())
        subs = [t + dt for dt in np.linspace(-0.4, 0.4, 5) / FPS] if flying else [t]
        acc = None
        for ts in subs:
            film.apply(ts)
            c = film.camera(ts)
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = c["lookat"], c["distance"], c["azimuth"], c["elevation"]
            r.update_scene(film.d, camera=cam, scene_option=opt)
            film.overrides(r.scene, ts)
            img = r.render().astype(np.float32)
            acc = img if acc is None else acc + img
        img = Image.fromarray((acc / len(subs)).astype(np.uint8)).resize((SIZE, SIZE), Image.LANCZOS)
        overlay(img, t).save(Path(outdir) / f"{idx:05d}.png")
    r.close()


def main(build, out, *rest):
    rest = list(rest)
    stages = tempfile.mkdtemp()
    torso_stages(Path(stages))
    workers = 4
    if "--workers" in rest:
        workers = int(rest[rest.index("--workers") + 1])
    if "--stills" in rest:
        ts = [float(x) for x in rest[rest.index("--stills") + 1].split(",")]
        tmp = Path(tempfile.mkdtemp())
        jobs = [(build, stages, [(i, t)], tmp) for i, t in enumerate(ts)]
        with get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            pool.map(render_frames, jobs)
        ims = [np.asarray(Image.open(tmp / f"{i:05d}.png").resize((540, 540))) for i in range(len(ts))]
        while len(ims) % 3:
            ims.append(np.zeros_like(ims[0]))
        Image.fromarray(np.vstack([np.hstack(ims[i:i + 3]) for i in range(0, len(ims), 3)])).save(out)
        shutil.rmtree(tmp)
        shutil.rmtree(stages)
        return
    n = int(DURATION * FPS)
    tmp = Path(tempfile.mkdtemp())
    frames = [(i, i / FPS) for i in range(n)]
    jobs = [(build, stages, frames[k::workers], tmp) for k in range(workers)]
    with get_context("spawn").Pool(workers) as pool:
        pool.map(render_frames, jobs)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(tmp / "%05d.png"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(out)], check=True)
    shutil.rmtree(tmp)
    shutil.rmtree(stages)


if __name__ == "__main__":
    main(*sys.argv[1:])
