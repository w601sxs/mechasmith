"""Film every example robot posed side by side in one scene: a 6 s, 1080x1080 looping camera move.

usage: python scripts/film_lineup.py build lineup.mp4 [--stills stills.png]

Build the designs first (`mechasmith build designs/<name>.py` for each name in LINEUP). Joints hold
the named keyframe; only the camera moves. Needs ffmpeg, and an OpenGL context: on a headless
machine run under Xvfb with MUJOCO_GL=glfw (or MUJOCO_GL=egl where EGL is available).
"""
import subprocess
import sys
import warnings
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mechasmith.render import _rest_on_floor

FPS, SECONDS, SIZE = 24, 6, 1080
N = FPS * SECONDS
SS = 2  # supersampling
FONT_B = "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# name: (keyframe or None for the zero pose, (x, y) in metres, yaw in degrees, fixed height or None)
# Robots face +x (towards the camera); the big ones stand at the back, the small ones in front.
LINEUP = {
    "g1": ("home", (-0.30, 0.00), 0, None),
    "go2": ("home", (-0.20, -0.62), 25, None),
    "so101": ("rest", (-0.05, 0.58), 60, None),
    "microduck": ("look_left", (0.42, 0.16), -15, None),
    "microduck_jetpack": ("tilt_forward", (0.42, -0.20), 15, None),
    "duck_leg": (None, (0.50, 0.46), -30, None),
    "robotiq_2f85": (None, (0.52, -0.50), 20, None),
    "crazyflie2": ("hover", (0.30, -0.02), 30, 0.55),
}

WORLD = """
<mujoco model="lineup">
  <visual>
    <headlight diffuse="0.4 0.4 0.4" ambient="0.32 0.32 0.32" specular="0 0 0"/>
    <global fovy="40"/>
    <quality shadowsize="8192" offsamples="8"/>
    <map shadowclip="1.2"/>
  </visual>
  <statistic center="0 0 0.3" extent="3"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.96 0.96 0.97" rgb2="0.85 0.87 0.9" width="512" height="512"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.92 0.92 0.93" rgb2="0.88 0.88 0.9"
             markrgb="0.8 0.8 0.82" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="40 40" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="1 -0.6 3" dir="-0.3 0.2 -1" directional="true" castshadow="true" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="grid"/>
  </worldbody>
</mujoco>
"""


def font(path, size):
    """The named font, or Pillow's built-in one where it isn't installed."""
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size)


def yaw_quat(deg):
    a = np.radians(deg) / 2
    return np.array([np.cos(a), 0, 0, np.sin(a)])


def standalone_pose(build, name, key, height):
    """Full qpos of the robot alone, posed and resting on the floor (or held at `height`)."""
    m = mujoco.MjModel.from_xml_path(str(build / name / "robot.xml"))
    d = mujoco.MjData(m)
    if key:
        d.qpos[:] = m.key(key).qpos
    mujoco.mj_forward(m, d)
    floating = m.njnt and int(m.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
    if floating:
        if height is None:
            _rest_on_floor(m, d)
        else:
            d.qpos[2] = height
    return m, d.qpos.copy(), floating


def build_scene(build: Path):
    world = mujoco.MjSpec.from_string(WORLD)
    poses = []
    for name, (key, (x, y), yaw, height) in LINEUP.items():
        child = mujoco.MjSpec.from_file(str(build / name / "robot.xml"))
        child.meshdir = str((build / name / "meshes").resolve())
        frame = world.worldbody.add_frame(pos=[x, y, 0], quat=yaw_quat(yaw))
        world.attach(child, frame=frame, prefix=f"{name}/")
        poses.append((name, key, (x, y), yaw, height))
    with warnings.catch_warnings():  # children's keyframes are dropped on attach; poses are set below
        warnings.simplefilter("ignore")
        model = world.compile()
    data = mujoco.MjData(model)
    for name, key, (x, y), yaw, height in poses:
        m, q, floating = standalone_pose(build, name, key, height)
        for j in range(m.njnt):
            src, dst = m.jnt_qposadr[j], model.jnt_qposadr[model.joint(f"{name}/{m.joint(j).name}").id]
            if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                rot = np.zeros(9)
                mujoco.mju_quat2Mat(rot, yaw_quat(yaw))
                data.qpos[dst:dst + 3] = np.array([x, y, 0]) + rot.reshape(3, 3) @ q[src:src + 3]
                quat = np.zeros(4)
                mujoco.mju_mulQuat(quat, yaw_quat(yaw), q[src + 3:src + 7])
                data.qpos[dst + 3:dst + 7] = quat
            else:
                n = 1 if int(m.jnt_type[j]) in (2, 3) else 4
                data.qpos[dst:dst + n] = q[src:src + n]
    mujoco.mj_forward(model, data)
    return model, data


def camera_path(t):
    """A closed drone move: sweeps across the front of the group, rising as it pulls back,
    dropping and pushing in on the way round."""
    w = 2 * np.pi * t
    azimuth = 180 + 32 * np.sin(w)
    elevation = -14 - 7 * np.cos(w)
    distance = 2.7 + 0.25 * np.cos(w)
    lookat = np.array([0.08, -0.02 + 0.08 * np.sin(w), 0.47 + 0.03 * np.cos(w)])
    return lookat, distance, azimuth, elevation


def title_overlay(img):
    d = ImageDraw.Draw(img, "RGBA")
    big = font(FONT_B, 46 * SS)
    small = font(FONT_M, 21 * SS)
    x, y = 44 * SS, 40 * SS
    w = max(d.textlength("mechasmith", font=big), d.textlength("/ robots as Python", font=small))
    d.rounded_rectangle((x - 22 * SS, y - 18 * SS, x + w + 22 * SS, y + 98 * SS), 14 * SS,
                        fill=(24, 26, 32, 235))
    d.text((x, y + 44 * SS), "mechasmith", font=big, fill=(245, 246, 248), anchor="ls")
    d.text((x, y + 78 * SS), "/ robots as Python", font=small, fill=(255, 200, 60), anchor="ls")
    return img


def main(build, out_mp4, *rest):
    build = Path(build)
    model, data = build_scene(build)
    px = SIZE * SS
    model.vis.global_.offwidth = model.vis.global_.offheight = px
    renderer = mujoco.Renderer(model, px, px)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = opt.geomgroup[2] = 1
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def frame(t):
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = camera_path(t)
        renderer.update_scene(data, camera=cam, scene_option=opt)
        img = title_overlay(Image.fromarray(renderer.render()))
        return img.resize((SIZE, SIZE), Image.LANCZOS)

    if rest and rest[0] == "--stills":
        tiles = [np.asarray(frame(t).resize((540, 540))) for t in (0, 0.25, 0.5, 0.75)]
        Image.fromarray(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])).save(rest[1])
        return
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{SIZE}x{SIZE}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
           "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_mp4)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N):
        ff.stdin.write(np.asarray(frame(i / N)).tobytes())
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    main(*sys.argv[1:])
