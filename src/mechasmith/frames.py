"""Small helpers for placing local geometry into the world frame."""
from __future__ import annotations

import numpy as np
from build123d import Location, Plane


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero-length direction")
    return v / n


def local_axes(x_dir, z_dir) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed (x, y, z) world vectors for a frame with the given x and z."""
    x, z = unit(x_dir), unit(z_dir)
    if abs(np.dot(x, z)) > 1e-9:
        raise ValueError(f"x_dir {x} and z_dir {z} are not perpendicular")
    return x, np.cross(z, x), z


def frame(origin, x_dir, z_dir) -> Location:
    """Location whose local X maps to x_dir and local Z to z_dir."""
    x, _, z = local_axes(x_dir, z_dir)
    return Location(Plane(origin=tuple(map(float, origin)), x_dir=tuple(x), z_dir=tuple(z)))
