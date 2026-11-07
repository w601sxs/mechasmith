"""The design contract: typed parameters, a build function, a home pose and named poses.

    from mechasmith import design, param

    @design(
        params=dict(
            shin_len=param(42.0, (36, 60), "knee axis -> ankle axis, mm"),
            battery_g=param(100.0, (20, 200), "battery mass, g"),
        ),
        home={"left_knee": -0.2},
        poses={"crouch": {...}},
    )
    def build(p):
        ...           # p.shin_len, p.battery_g
        return robot

Every tool (build, sweeps, balance, compare, viewer) reads this one object, so an
agent always knows which knobs exist, what they mean and how far they may go.
The ranges are also the natural bounds for domain randomization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .core import Robot


@dataclass(frozen=True)
class Param:
    default: Any
    range: tuple[float, float] | None = None   # inclusive bounds for numeric params
    doc: str = ""

    def check(self, name: str, value: Any) -> None:
        if self.range is None or not isinstance(value, (int, float)):
            return
        lo, hi = self.range
        if not lo <= value <= hi:
            raise ValueError(f"param {name}={value} is outside its range [{lo}, {hi}]")


def param(default: Any, range: tuple[float, float] | None = None, doc: str = "") -> Param:
    return Param(default, None if range is None else (float(range[0]), float(range[1])), doc)


class Params:
    """Read-only parameter values with attribute access (p.shin_len)."""

    __slots__ = ("_values",)

    def __init__(self, values: dict[str, Any]):
        object.__setattr__(self, "_values", dict(values))

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(f"no design parameter {name!r}") from None

    def __setattr__(self, name, value):
        raise AttributeError("Params are read-only; use design.build(name=value) to override")

    def __getitem__(self, name: str) -> Any:
        return self._values[name]

    def as_dict(self) -> dict[str, Any]:
        return dict(self._values)

    def __repr__(self) -> str:
        return "Params(" + ", ".join(f"{k}={v!r}" for k, v in self._values.items()) + ")"


@dataclass
class Design:
    name: str
    build_fn: Callable[[Params], Robot]
    params: dict[str, Param] = field(default_factory=dict)
    home: dict[str, float] = field(default_factory=dict)
    poses: dict[str, dict[str, float]] = field(default_factory=dict)
    base_height: float | str = "auto"   # metres, or "auto" to rest the lowest geometry on z = 0
    tool_site: str | None = None         # fixed-base arms: the end-effector site
    payload_kg: float = 0.0              # arms: payload held at the tool site in the arm check
    doc: str = ""

    def resolve(self, **overrides) -> Params:
        unknown = set(overrides) - set(self.params)
        if unknown:
            raise KeyError(f"unknown design parameter(s) {sorted(unknown)}; known: {sorted(self.params)}")
        values = {k: p.default for k, p in self.params.items()}
        values.update(overrides)
        for k, v in values.items():
            self.params[k].check(k, v)
        return Params(values)

    def build(self, **overrides) -> Robot:
        robot = self.build_fn(self.resolve(**overrides))
        if not isinstance(robot, Robot):
            raise TypeError(f"design {self.name!r} build function returned {type(robot).__name__}, not Robot")
        return robot

    def describe(self) -> dict:
        """Plain-dict summary (what an agent reads to learn the knobs)."""
        return {
            "name": self.name,
            "doc": self.doc,
            "params": {k: {"default": p.default, "range": p.range, "doc": p.doc} for k, p in self.params.items()},
            "home": self.home,
            "poses": sorted(self.poses),
            "tool_site": self.tool_site,
            "payload_kg": self.payload_kg,
        }


def design(*, name: str | None = None, params: dict[str, Param] | None = None, home: dict | None = None,
           poses: dict | None = None, base_height: float | str = "auto", tool_site: str | None = None,
           payload_kg: float = 0.0) -> Callable[[Callable], Design]:
    """Decorator turning `build(p) -> Robot` into a Design."""
    def wrap(fn: Callable[[Params], Robot]) -> Design:
        import sys
        mod = fn.__module__.rsplit(".", 1)[-1]
        doc = fn.__doc__ or getattr(sys.modules.get(fn.__module__), "__doc__", None) or ""
        return Design(name=name or mod, build_fn=fn, params=dict(params or {}), home=dict(home or {}),
                      poses=dict(poses or {}), base_height=base_height, tool_site=tool_site,
                      payload_kg=payload_kg, doc=doc.strip())
    return wrap


def find_design(module) -> Design:
    found = [v for v in vars(module).values() if isinstance(v, Design)]
    if len(found) != 1:
        raise LookupError(f"{module.__name__}: expected exactly one @design, found {len(found)}")
    return found[0]


def symmetric(pose: dict[str, float], src: str = "left_", dst: str = "right_", axes: str = "vector") -> dict:
    """Complete a pose written for one side: {"left_knee": 0.4} -> also "right_knee".

    Use the same `axes` convention as `Robot.mirror`: with "vector" the mirror joint
    gets the opposite value, with "pseudo" the same value. Joints without the prefix
    (neck, head) pass through unchanged.
    """
    out = dict(pose)
    for k, v in pose.items():
        if k.startswith(src):
            out[dst + k[len(src):]] = -v if axes == "vector" else v
    return out
