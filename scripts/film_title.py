"""Opening title card: the logo, the name, the tagline and the pip install line.

Usage: uv run python scripts/film_title.py out.mp4 [--still t out.png]
Needs ffmpeg. Pure 2-D (Pillow), no display needed.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS, SIZE, SS = 24, 1080, 2  # SS: supersampling
DURATION = 7.0  # the last 1.2 s are overlapped by the crossfade into the next clip
ROOT = Path(__file__).resolve().parent.parent
LOGO = ROOT / "docs" / "media" / "logo.png"
FONT_B = "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"
FONT_R = "/usr/share/fonts/opentype/inter/Inter-Regular.otf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BG, ORANGE = (19, 26, 42), (255, 106, 36)  # the logo's background and accent
NAME, TAGLINE = "mechasmith", "Robots as Python code, ready to simulate."
PIP = "pip install mechasmith"


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size)


def ramp(t, a, b):
    return float(np.clip((t - a) / (b - a), 0, 1))


def ease(x):
    return 1 - (1 - x) ** 3


class Card:
    def __init__(self):
        px = SIZE * SS
        self.px = px
        self.logo = Image.open(LOGO).convert("RGBA")
        self.big, self.mid, self.mono = font(FONT_B, 112 * SS), font(FONT_R, 34 * SS), font(FONT_M, 36 * SS)
        # Background: the logo's navy with a soft vignette.
        yy, xx = np.mgrid[0:px, 0:px] / px - 0.5
        v = 1 + 0.35 * (0.5 - np.hypot(xx, yy * 1.1))
        self.bg = np.clip(np.array(BG)[None, None] * v[..., None], 0, 255).astype(np.uint8)

    def layer(self, draw_fn, alpha, dy):
        """Draw onto a transparent layer, fade it by alpha and lift it by dy (output px)."""
        lay = Image.new("RGBA", (self.px, self.px))
        draw_fn(ImageDraw.Draw(lay), lay)
        if alpha < 1:
            a = lay.getchannel("A").point(lambda v: int(v * alpha))
            lay.putalpha(a)
        return lay, int(dy * SS)

    def frame(self, t):
        c = self.px // 2
        img = Image.fromarray(self.bg).convert("RGBA")
        layers = []

        a = ease(ramp(t, 0.0, 1.3))  # logo: fades in, grows from 88 %
        side = int(250 * SS * (0.88 + 0.12 * a))
        logo = self.logo.resize((side, side), Image.LANCZOS)
        layers.append(self.layer(lambda d, lay: lay.alpha_composite(logo, (c - side // 2, int(340 * SS) - side // 2)),
                                 a, 0))

        a = ease(ramp(t, 0.7, 1.9))  # name
        layers.append(self.layer(lambda d, lay: d.text((c, 590 * SS), NAME, font=self.big, fill=(245, 246, 248),
                                                       anchor="ms"), a, 18 * (1 - a)))
        a = ease(ramp(t, 1.1, 2.3))  # tagline
        layers.append(self.layer(lambda d, lay: d.text((c, 650 * SS), TAGLINE, font=self.mid, fill=(160, 170, 188),
                                                       anchor="ms"), a, 18 * (1 - a)))

        a = ease(ramp(t, 1.9, 2.5))  # pip line: the box appears, then the command types out
        typed = PIP[: int(len(PIP) * ramp(t, 2.5, 3.8))]
        cursor = t < 3.8 or int((t - 3.8) * 2.4) % 2 == 0

        def pip(d, lay):
            full = "$ " + PIP
            w = d.textlength(full, font=self.mono)
            x0, y0 = c - w / 2, 790 * SS
            d.rounded_rectangle((x0 - 34 * SS, y0 - 54 * SS, x0 + w + 58 * SS, y0 + 26 * SS), 16 * SS,
                                fill=(11, 15, 25, 255), outline=(52, 62, 84, 255), width=2 * SS)
            d.text((x0, y0), "$", font=self.mono, fill=ORANGE + (255,), anchor="ls")
            x = x0 + d.textlength("$ ", font=self.mono)
            d.text((x, y0), typed, font=self.mono, fill=(232, 236, 242, 255), anchor="ls")
            if cursor:
                cx = x + d.textlength(typed, font=self.mono) + 3 * SS
                d.rectangle((cx, y0 - 34 * SS, cx + 18 * SS, y0 + 6 * SS), fill=ORANGE + (255,))
        layers.append(self.layer(pip, a, 14 * (1 - a)))

        for lay, dy in layers:
            img.alpha_composite(lay, (0, -dy))
        return img.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)


def main(out, *rest):
    card = Card()
    if rest and rest[0] == "--still":
        card.frame(float(rest[1])).save(rest[2])
        return
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{SIZE}x{SIZE}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow",
           "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(round(DURATION * FPS)):
        ff.stdin.write(np.asarray(card.frame(i / FPS)).tobytes())
    ff.stdin.close()
    ff.wait()


if __name__ == "__main__":
    main(*sys.argv[1:])
