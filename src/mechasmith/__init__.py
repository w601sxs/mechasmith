"""mechasmith: agent-first robot design, from CAD code to a simulatable MJCF.

Everything a design needs is importable from the top level:

    from mechasmith import Assembly, Component, servo, XL330, design, param, UP, box, color
"""
from .assembly import (Assembly, AttachmentError, Component, ServoUnit, horn_plate, l_bracket, servo, spacer,
                       u_bracket)
from .components.servo import XL330, MountFace, ServoSpec
from .core import ActuatorModel, Robot
from .design import Design, Param, Params, design, param, symmetric
from .geom import BACK, DOWN, FWD, LEFT, RIGHT, UP, box, circle, color, cyl, extrude, move, rect, slot
from .materials import ALUMINIUM, BOUGHT, PLA, PLA_SOLID, STEEL, TPU, Material
from .shells import (arch, loft_body, loft_cavity, loft_shell, profile_plate, rounded_box, rounded_box_cavity,
                     rounded_box_shell)

__all__ = [
    # designs
    "design", "param", "symmetric", "Design", "Param", "Params",
    # assembling parts
    "Assembly", "Component", "AttachmentError", "servo", "ServoUnit", "XL330", "ServoSpec", "MountFace",
    "u_bracket", "horn_plate", "spacer", "l_bracket",
    # geometry
    "UP", "DOWN", "FWD", "BACK", "LEFT", "RIGHT", "box", "cyl", "rect", "slot", "circle", "extrude", "move",
    "color", "arch", "loft_body", "loft_shell", "loft_cavity", "profile_plate", "rounded_box",
    "rounded_box_shell", "rounded_box_cavity",
    # resolved model and materials
    "Robot", "ActuatorModel",
    "Material", "PLA", "PLA_SOLID", "TPU", "ALUMINIUM", "STEEL", "BOUGHT",
]
