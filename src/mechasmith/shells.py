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


def arch(width: float, height: float, r_top: float, r_bottom: float = 2.0):
    """Face centred at the origin: a rectangle with large top corners and small bottom
    ones (the front outline of helmets, heads, arched windows)."""
    from build123d import RectangleRounded
    r_top = min(r_top, width / 2 - 0.5, height / 2 - 0.5)
    r_bottom = min(r_bottom, width / 2 - 0.5, height / 4 - 0.5)
    upper = RectangleRounded(width, height, r_top)
    lower = Pos(0, -height / 4) * RectangleRounded(width, height / 2, max(r_bottom, 0.5))
    return upper + lower


def loft_body(stations, bottom: float, radius: float = 12.0, bottom_radius: float | None = None):
    """Solid lofted along x through rounded-rectangle sections.

    stations: [(x, half_width, top), ...] in mm; every section spans z from `bottom`
    to `top`. Good for heads, noses and fairings that are wider and taller in places.
    With `bottom_radius`, sections are arches: `radius` on the top corners only.
    """
    from build123d import Plane, RectangleRounded, loft
    secs = []
    for x, hw, top in stations:
        h = top - bottom
        if bottom_radius is None:
            r = min(radius, hw - 0.5, h / 2 - 0.5)
            f = RectangleRounded(2 * hw, h, r)
        else:
            f = arch(2 * hw, h, radius, bottom_radius)
        secs.append(Plane.YZ.offset(x) * (Pos(0, (top + bottom) / 2) * f))
    return loft(secs)


def loft_shell(stations, bottom: float, wall: float, radius: float = 12.0, open_bottom: bool = True,
               bottom_radius: float | None = None):
    """Hollow version of loft_body with walls of `wall` mm (open underneath by default)."""
    outer = loft_body(stations, bottom, radius, bottom_radius)
    inner_st = [(x, hw - wall, top - wall) for x, hw, top in stations]
    # pull the end sections inward so the front and back stay closed
    (x0, hw0, t0), (x1, hw1, t1) = inner_st[0], inner_st[-1]
    inner_st[0] = (x0 + wall, hw0, t0)
    inner_st[-1] = (x1 - wall, hw1, t1)
    inner = loft_body(inner_st, bottom - 50 if open_bottom else bottom + wall, max(radius - wall, 1.0),
                      None if bottom_radius is None else max(bottom_radius - wall, 0.5))
    return outer - inner


def profile_plate(points_xz, y0: float, y1: float, round_r: float = 0.0):
    """Flat plate whose outline is a polygon in the XZ plane (rounded by `round_r`),
    spanning y0..y1. Thigh covers, ear lobes, side panels."""
    from build123d import Kind, Plane, Polygon, extrude, offset
    face = Plane.XZ * Polygon(*[(x, z) for x, z in points_xz], align=None)
    if round_r > 0:
        face = offset(face, amount=round_r, kind=Kind.ARC)
    # Plane.XZ extrudes toward -y; place it at y1 and extrude by the thickness
    return Pos(0, y1, 0) * extrude(face, amount=y1 - y0)
