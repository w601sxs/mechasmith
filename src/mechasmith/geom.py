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


def polygon(points, round: float = 0.0):
    """Outline through 2D points, optionally grown by `round` mm with rounded corners.
    Two points make a stadium (a rounded bar between them), the usual shape of a link."""
    from build123d import Kind, Polygon, offset
    pts = [tuple(map(float, p)) for p in points]
    if len(pts) == 2:
        (x0, y0), (x1, y1) = pts
        d = np.hypot(x1 - x0, y1 - y0)
        if d == 0 or round <= 0:
            raise ValueError("polygon: two points need distinct points and round > 0")
        nx, ny = -(y1 - y0) / d * 0.01, (x1 - x0) / d * 0.01
        pts = [(x0 + nx, y0 + ny), (x1 + nx, y1 + ny), (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)]
    face = Polygon(*pts, align=None)
    return offset(face, amount=round, kind=Kind.ARC) if round > 0 else face


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
    # explicit direction: the result never depends on the outline's winding
    return _ex(pl * (Pos(center[0], center[1]) * outline), amount=depth, dir=n)


def move(shape, offset):
    """Translate a shape by (dx, dy, dz) mm."""
    return Pos(*offset) * shape


def diamonds(center, normal, along, rows: int, cols: int, size: float = 6.0, pitch: float = 9.0,
             depth: float = 100.0):
    """A grid of diamond holes (rows x cols) for `cut=`: centred at `center` on a face with
    outward `normal`, rows stepping along `along`; each diamond is `size` mm across its
    points and is cut `depth` mm straight through (half each way from the face)."""
    from build123d import Plane, Polygon, extrude as _ex
    n, a = unit(normal), unit(along)
    b = np.cross(n, a)
    out = None
    h = size / 2
    for i in range(rows):
        for j in range(cols):
            c = np.asarray(center, float) + a * (i - (rows - 1) / 2) * pitch + b * (j - (cols - 1) / 2) * pitch
            pl = Plane(origin=tuple(c - n * depth / 2), x_dir=tuple(a), z_dir=tuple(n))
            d = _ex(pl * Polygon((h, 0), (0, h), (-h, 0), (0, -h), align=None), amount=depth, dir=tuple(n))
            out = d if out is None else out + d
    return out


def tray(points, plane: str, at: float, depth: float, wall: float, round: float = 6.0):
    """A plate with a skirt round its edge (covers, caps, foot tubs). The outline is the
    polygon through 2D `points` grown by `round`; the plate is `wall` thick at `at` and the
    skirt runs `depth` mm along the plane normal (negative: the other way), open at its end."""
    if round <= wall:
        raise ValueError("tray: round must exceed wall")
    outer, inner = polygon(points, round=round), polygon(points, round=round - wall)
    d = abs(depth)
    if depth > 0:
        return extrude(outer, plane, at, depth=d) - extrude(inner, plane, at + wall, depth=d)
    return extrude(outer, plane, at - d, depth=d) - extrude(inner, plane, at - d - 1, depth=d + 1 - wall)
