"""mechasmith: agent-first robot design, from CAD code to a simulatable MJCF.

Everything a design needs is importable from the top level:

    from mechasmith import Assembly, Component, servo, XL330, design, param, UP, box, color
"""
from .assembly import (Assembly, AttachmentError, Component, MotorUnit, ServoUnit, horn_plate, l_bracket, motor, servo,
                       servo_housing,
                       spacer, u_bracket)
from .components.motor import GO_M8010_6, MotorSpec
from .components.servo import STS3215, XL330, MountFace, ServoSpec
from .core import ActuatorModel, Robot, Rotor
from .design import Design, Param, Params, design, param, symmetric
from .geom import BACK, DOWN, FWD, LEFT, RIGHT, UP, box, circle, color, cyl, diamonds, tray, extrude, move, polygon, rect, slot
from .links import Hold, bearing, printed, swept_link
from .materials import ALUMINIUM, BOUGHT, PLA, PLA_SOLID, STEEL, TPU, Material
from .shells import arch, loft_body, loft_cavity, loft_shell, rounded_box, rounded_box_cavity, rounded_box_shell

__all__ = [
    # designs
    "design", "param", "symmetric", "Design", "Param", "Params",
    # assembling parts
    "Assembly", "Component", "AttachmentError", "servo", "ServoUnit", "motor", "MotorUnit", "MotorSpec", "GO_M8010_6", "XL330", "STS3215", "ServoSpec", "MountFace",
    "u_bracket", "horn_plate", "spacer", "l_bracket", "servo_housing", "Hold", "printed", "swept_link", "bearing",
    # geometry
    "UP", "DOWN", "FWD", "BACK", "LEFT", "RIGHT", "box", "diamonds", "tray", "cyl", "rect", "slot", "circle", "polygon", "extrude", "move",
    "color", "arch", "loft_body", "loft_shell", "loft_cavity", "rounded_box",
    "rounded_box_shell", "rounded_box_cavity",
    # resolved model and materials
    "Robot", "ActuatorModel",
    "Material", "PLA", "PLA_SOLID", "TPU", "ALUMINIUM", "STEEL", "BOUGHT",
]
