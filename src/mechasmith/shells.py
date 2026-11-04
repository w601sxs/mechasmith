"""Smooth shell primitives for the parts people look at (heads, bodies, covers)."""
from __future__ import annotations

import numpy as np
from build123d import Box, Pos, Sphere, fillet, scale


def ellipsoid(center, radii):
    return Pos(*center) * scale(Sphere(1.0), by=tuple(float(r) for r in radii))


def ellipsoid_shell(center, radii, wall: float):
    inner = [r - wall for r in radii]
    return ellipsoid(center, radii) - ellipsoid(center, inner)


def rounded_box(lo, hi, radius: float):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    b = Box(*(hi - lo))
    b = fillet(b.edges(), radius)
    return Pos(*((lo + hi) / 2)) * b


def rounded_box_shell(lo, hi, radius: float, wall: float):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return rounded_box(lo, hi, radius) - rounded_box(lo + wall, hi - wall, max(radius - wall, 0.5))
