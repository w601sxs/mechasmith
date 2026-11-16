"""Exhaust smoke and ground dust for rendering (drawn only; it does not push on anything).

A small particle system: each fan sheds puffs from its nozzle exit at a rate and speed set by its
thrust, along its thrust axis (so the jet tilts with the nacelle). Puffs slow down in the air,
rise a little, grow and fade. A puff that reaches the ground (the floor, or the top of a box)
loses its downward speed and is thrown outward instead, so a jet close to the ground spreads into
a ring of dust, as a real jet does. MuJoCo has no fluids, so this is a look, not a simulation of
the flow: speeds are scaled down from a real EDF's ~40 m/s exhaust so the puffs stay on screen.

    smoke = Smoke()
    ...each rendered frame:
    smoke.step(frame_dt, nozzles=[(exit_pos, thrust_axis, throttle), ...], ground=fn(x, y) -> z)
    renderer.update_scene(...); img = render(renderer, smoke)
"""
from __future__ import annotations

import mujoco
import numpy as np

EXHAUST = (0.80, 0.81, 0.84)        # fresh exhaust, light grey
DUST = (0.72, 0.69, 0.64)           # after it has scoured the ground


class Smoke:
    def __init__(self, rate=420.0, life=(0.6, 1.2), speed=(1.2, 3.0), drag=3.5, lift=0.25,
                 nozzle_r=0.028, max_n=1500, alpha=0.18, seed=0):
        self.rate, self.life, self.speed, self.drag, self.lift = rate, life, speed, drag, lift
        self.nozzle_r, self.max_n, self.alpha = nozzle_r, max_n, alpha
        self.rng = np.random.default_rng(seed)
        self.p = np.zeros((0, 3))
        self.v = np.zeros((0, 3))
        self.age = np.zeros(0)
        self.tmax = np.zeros(0)
        self.dust = np.zeros(0)       # 0 = fresh exhaust .. 1 = dust
        self.carry = 0.0

    def _emit(self, dt, exit_pos, axis, u):
        n_f = self.rate * u * dt + self.carry
        n = int(n_f)
        self.carry = n_f - n
        if n == 0:
            return
        axis = axis / np.linalg.norm(axis)
        a = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(axis, a)
        th = self.rng.uniform(0, 2 * np.pi, n)
        rr = self.nozzle_r * np.sqrt(self.rng.uniform(0, 1, n))
        off = (np.cos(th) * rr)[:, None] * a + (np.sin(th) * rr)[:, None] * b
        s = self.speed[0] + (self.speed[1] - self.speed[0]) * u
        jet = -axis                                        # exhaust leaves opposite to the thrust
        v = s * (jet + 0.12 * self.rng.normal(size=(n, 3)))
        # spread the puffs along the frame's travel so they don't come out in clumps
        lag = self.rng.uniform(0, dt, n)[:, None]
        self.p = np.vstack([self.p, exit_pos + off + v * lag])
        self.v = np.vstack([self.v, v])
        self.age = np.r_[self.age, lag[:, 0]]
        self.tmax = np.r_[self.tmax, self.rng.uniform(*self.life, n)]
        self.dust = np.r_[self.dust, np.zeros(n)]

    def step(self, dt, nozzles, ground=lambda x, y: np.zeros_like(x), substeps=4):
        """nozzles: [(exit position, thrust axis, throttle 0..1), ...]."""
        h = float(dt) / substeps
        for _ in range(substeps):
            for exit_pos, axis, u in nozzles:
                if u > 0.02:
                    self._emit(h, np.asarray(exit_pos, float), np.asarray(axis, float), float(u))
            if not len(self.age):
                continue
            self.v *= np.exp(-self.drag * h)
            self.v[:, 2] += self.lift * h
            self.p += self.v * h
            self.age += h
            # the ground: kill the downward speed and throw it outward (a wall jet)
            g = ground(self.p[:, 0], self.p[:, 1])
            r = self.radius()
            hit = (self.p[:, 2] < g + 0.4 * r) & (self.v[:, 2] < 0)
            if hit.any():
                vn = -self.v[hit, 2]
                vh = self.v[hit, :2]
                sh = np.linalg.norm(vh, axis=1, keepdims=True)
                rand = self.rng.uniform(0, 2 * np.pi, hit.sum())
                d = np.where(sh > 0.15 * vn[:, None], vh / np.maximum(sh, 1e-9),
                             np.c_[np.cos(rand), np.sin(rand)])
                self.v[hit, :2] = vh + d * (0.65 * vn)[:, None]
                self.v[hit, 2] = 0.06 * vn
                self.p[hit, 2] = g[hit] + 0.4 * r[hit]
                self.dust[hit] = np.minimum(1.0, self.dust[hit] + 0.6)
            keep = self.age < self.tmax
            if len(keep) > self.max_n:                    # oldest go first
                keep &= np.arange(len(keep)) >= len(keep) - self.max_n
            self.p, self.v, self.age = self.p[keep], self.v[keep], self.age[keep]
            self.tmax, self.dust = self.tmax[keep], self.dust[keep]

    def radius(self):
        f = self.age / self.tmax
        return 0.008 + 0.035 * np.sqrt(f) + 0.015 * self.dust * f

    def composite(self, img, depth, cam):
        """Splat the puffs onto a rendered frame as soft Gaussian blobs, hidden behind anything
        nearer in the depth buffer. img: HxWx3 uint8, depth: HxW metres (Renderer depth
        rendering), cam: the scene's mjvGLCamera (renderer.scene.camera[0])."""
        if not len(self.age):
            return img
        H, W = depth.shape
        pos, fwd, up = np.array(cam.pos), np.array(cam.forward), np.array(cam.up)
        right = np.cross(fwd, up)
        rel = self.p - pos
        z = rel @ fwd
        ok = z > cam.frustum_near
        f_px = (H / 2) / ((cam.frustum_top - cam.frustum_bottom) / 2 / cam.frustum_near)
        u = W / 2 + (rel @ right) / np.maximum(z, 1e-6) * f_px
        v = H / 2 - (rel @ up) / np.maximum(z, 1e-6) * f_px
        f = self.age / self.tmax
        sig = np.maximum(self.radius() * f_px / np.maximum(z, 1e-6), 0.8)
        a = self.alpha * np.minimum(1.0, self.age / 0.04) * (1 - f) ** 1.4
        dens = -np.log1p(-np.clip(a, 0, 0.95))             # optical depth at the blob centre
        col = np.array(EXHAUST) + self.dust[:, None] * (np.array(DUST) - np.array(EXHAUST))
        D = np.zeros((H, W))
        C = np.zeros((H, W, 3))
        for i in np.nonzero(ok & (a > 0.005))[0]:
            R = int(np.ceil(2.5 * sig[i]))
            x0, x1 = max(0, int(u[i]) - R), min(W, int(u[i]) + R + 1)
            y0, y1 = max(0, int(v[i]) - R), min(H, int(v[i]) + R + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            xs, ys = np.arange(x0, x1) - u[i], np.arange(y0, y1) - v[i]
            g = np.exp(-(ys[:, None] ** 2 + xs[None, :] ** 2) / (2 * sig[i] ** 2)) * dens[i]
            g *= depth[y0:y1, x0:x1] > z[i]                 # behind the duck or the box: hidden
            D[y0:y1, x0:x1] += g
            C[y0:y1, x0:x1] += g[..., None] * col[i]
        alpha = 1 - np.exp(-D)
        smoke = C / np.maximum(D, 1e-9)[..., None]
        smoke *= (1 - 0.3 * alpha)[..., None]                # thick smoke is darker (self-shadow)
        out = img.astype(np.float32) / 255 * (1 - alpha[..., None]) + smoke * alpha[..., None]
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def render(renderer, smoke):
    """renderer.render() with the smoke composited (call after update_scene)."""
    img = renderer.render()
    renderer.enable_depth_rendering()
    depth = renderer.render()
    renderer.disable_depth_rendering()
    return smoke.composite(img, depth, renderer.scene.camera[0])


def nozzles(d, sites, throttle, exit_offset=0.047):
    """(exit position, thrust axis, throttle) for each fan site (z axis = thrust)."""
    out = []
    for s, u in zip(sites, throttle):
        axis = d.site_xmat[s].reshape(3, 3)[:, 2]
        out.append((d.site_xpos[s] - axis * exit_offset, axis, float(u)))
    return out
