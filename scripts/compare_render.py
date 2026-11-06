"""Side-by-side renders of a reference MJCF and a built design, same poses and views.

usage: python scripts/compare_render.py <reference scene.xml> build/<design> designs/<design>.py out.png
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from mechasmith.cli import load_design
from mechasmith.render import contact_sheet

ref_xml, build, design, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
home = load_design(design).home
V = {"side": (270, -5), "front-iso": (200, -15), "front": (180, -5), "back-iso": (30, -30)}
W, H = 360, 420


def row(png):
    im = Image.open(png)
    top = im.height - 2 * H
    r = Image.new("RGB", (4 * W, H))
    for i in range(4):
        r.paste(im.crop(((i % 2) * W, top + (i // 2) * H, (i % 2 + 1) * W, top + (i // 2 + 1) * H)), (i * W, 0))
    return r


ra = row(contact_sheet(ref_xml, out.with_suffix(".ref.png"), qpos=home, size=(W, H), views=V, zoom=1.5))
rb = row(contact_sheet(build / "scene.xml", out.with_suffix(".ours.png"), qpos=home, size=(W, H), views=V, zoom=1.5))
img = Image.new("RGB", (ra.width, 2 * H + 60), (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((10, 8), f"reference: {ref_xml}", fill=(20, 20, 20))
img.paste(ra, (0, 28))
d.text((10, H + 36), f"mechasmith: {design}", fill=(20, 20, 20))
img.paste(rb, (0, H + 58))
img.save(out)
out.with_suffix(".ref.png").unlink()
out.with_suffix(".ours.png").unlink()
print(out)
