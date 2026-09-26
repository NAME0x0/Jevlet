"""Render the Jevlet icon: a J whose dot has fallen into its hook (one option, chosen).

python -m scripts.make_icon            # writes app/src/Jevlet.App/Assets/jevlet.ico
python -m scripts.make_icon --preview  # also writes temp/icon_options.png (both variants)
python -m scripts.make_icon --variant ring  # the plain ring-and-dot mark instead
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw

SCALE = 4  # draw large, then downsample for clean anti-aliasing
TOP, BOTTOM = (99, 102, 241), (168, 85, 247)


def _tile(size: int) -> Image.Image:
    """Rounded-square gradient background."""
    big = size * SCALE
    gradient = Image.new("RGBA", (big, big))
    draw = ImageDraw.Draw(gradient)
    for y in range(big):
        t = y / (big - 1)
        draw.line(
            [(0, y), (big, y)],
            fill=tuple(int(a + (b - a) * t) for a, b in zip(TOP, BOTTOM, strict=True)) + (255,),
        )
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, big - 1, big - 1], radius=int(big * 0.23), fill=255
    )
    tile = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    tile.paste(gradient, (0, 0), mask)
    return tile


def _stroke(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], width: float) -> None:
    """One continuous stroke with round caps and joins: stamp a disc along the path."""
    radius = width / 2
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        steps = max(1, int(math.hypot(x1 - x0, y1 - y0) / (radius / 4)))
        for step in range(steps + 1):
            x, y = x0 + (x1 - x0) * step / steps, y0 + (y1 - y0) * step / steps
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius], fill=(255, 255, 255, 255)
            )


def j_ring(size: int = 256) -> Image.Image:
    """A J whose dot has fallen into its hook: the stem comes down, the hook cups the dot."""
    tile = _tile(size)
    draw = ImageDraw.Draw(tile)
    s = size * SCALE
    w = s * 0.11
    cx, cy, r = s * 0.465, s * 0.58, s * 0.2
    stem_x, top = cx + r, s * 0.2
    path = [(stem_x, top), (stem_x, cy)]
    # Hook: clockwise from 3 o'clock, under the dot, rising to the upper left like a J's tail.
    path += [
        (cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
        for a in range(0, 206, 2)
    ]
    _stroke(draw, path, w)
    d = s * 0.075
    draw.ellipse([cx - d, cy - d, cx + d, cy + d], fill=(255, 255, 255, 255))
    return tile.resize((size, size), Image.LANCZOS)


def ring_dot(size: int = 256) -> Image.Image:
    """The earlier mark: a ring with a dot at its centre."""
    tile = _tile(size)
    draw = ImageDraw.Draw(tile)
    s = size * SCALE
    c, r, w = s / 2, s * 0.29, s * 0.1
    draw.ellipse(
        [c - r - w / 2, c - r - w / 2, c + r + w / 2, c + r + w / 2],
        outline=(255, 255, 255, 255),
        width=int(w),
    )
    d = s * 0.1
    draw.ellipse([c - d, c - d, c + d, c + d], fill=(255, 255, 255, 255))
    return tile.resize((size, size), Image.LANCZOS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--variant", choices=("j", "ring"), default="j")
    args = parser.parse_args()
    icon = j_ring() if args.variant == "j" else ring_dot()
    target = Path("app/src/Jevlet.App/Assets/jevlet.ico")
    target.parent.mkdir(parents=True, exist_ok=True)
    icon.save(target, sizes=[(n, n) for n in (16, 20, 24, 32, 40, 48, 64, 128, 256)])
    if args.preview:
        sheet = Image.new("RGBA", (256 * 2 + 48 + 96, 256 + 32), (32, 32, 38, 255))
        sheet.paste(j_ring(), (16, 16), j_ring())
        sheet.paste(ring_dot(), (256 + 32, 16), ring_dot())
        for n, x in ((32, 256 * 2 + 64), (16, 256 * 2 + 64 + 48)):
            small = j_ring(n)
            sheet.paste(small, (x, 16), small)
        Path("temp").mkdir(exist_ok=True)
        sheet.save("temp/icon_options.png")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
