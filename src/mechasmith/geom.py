"""Everyday geometry for designs: world directions, boxes between corners,
cylinders along any axis, and colours from hex."""
from __future__ import annotations

import numpy as np
from build123d import Box, Cylinder, Pos

from .frames import frame, local_axes, unit

UP, DOWN = np.array([0.0, 0, 1]), np.array([0.0, 0, -1])
FWD, BACK = np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])
LEFT, RIGHT = np.array([0.0, 1, 0]), np.array([0.0, -1, 0])


def box(lo, hi):
    """Axis-aligned box spanning two opposite corners (any order), mm."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    a, b = np.minimum(lo, hi), np.maximum(lo, hi)
    return Pos(*((a + b) / 2)) * Box(*(b - a))


def cyl(center, axis, radius: float, length: float):
    """Cylinder centred at `center` with its axis along `axis`, mm."""
    z = unit(axis)
    x = local_axes(np.cross(z, [1.0, 0, 0]) if abs(z[0]) < 0.9 else np.cross(z, [0, 1.0, 0]), z)[0]
    return Cylinder(radius, length).moved(frame(center, x, z))


def color(hex_rgb: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """'#ff6620' -> (1.0, 0.4, 0.125, 1.0)."""
    h = hex_rgb.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"expected #rrggbb, got {hex_rgb!r}")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (float(alpha),)


# ---- 2D outlines and extrusion (so designs never need build123d directly) -----------

def rect(w: float, h: float, r: float = 0.0):
    """Rectangle outline centred at the origin, optional corner radius."""
    from build123d import Rectangle, RectangleRounded
    return RectangleRounded(w, h, r) if r > 0 else Rectangle(w, h)


def slot(length: float, width: float):
    """Stadium (pill) outline centred at the origin."""
    from build123d import SlotOverall
    return SlotOverall(length, width)


def circle(r: float):
    from build123d import Circle
    return Circle(r)


_PLANES = {  # name -> (origin offset axis, in-plane u, in-plane v, extrusion direction)
    "XY": (2, (1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "YZ": (0, (0, 1, 0), (0, 0, 1), (1, 0, 0)),
    "XZ": (1, (1, 0, 0), (0, 0, 1), (0, 1, 0)),
}


def extrude(outline, plane: str = "XY", at: float = 0.0, center=(0.0, 0.0), depth: float = 1.0):
    """Place a 2D outline on a world plane and extrude it along the plane's positive normal.

    plane "XY": outline (x, y) at z = at, grows +z.  "YZ": (y, z) at x = at, grows +x.
    "XZ": (x, z) at y = at, grows +y.  `center` moves the outline within the plane.
    """
    from build123d import Plane, Pos
    from build123d import extrude as _ex
    ax, u, v, n = _PLANES[plane]
    origin = [0.0, 0.0, 0.0]
    origin[ax] = at
    pl = Plane(origin=tuple(origin), x_dir=u, z_dir=tuple(np.cross(u, v)))
    return _ex(pl * (Pos(center[0], center[1]) * outline), amount=depth)


def move(shape, offset):
    """Translate a shape by (dx, dy, dz) mm."""
    return Pos(*offset) * shape
