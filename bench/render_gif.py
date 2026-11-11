"""Render frames recorded by `bench.demo --record` into an animated GIF (for the README).

    python -m bench.render_gif demo_frames.jsonl docs/demo.gif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG, FG, DIM, ACCENT = (24, 24, 23), (240, 239, 236), (155, 154, 148), (57, 135, 229)


def _font(size: int) -> ImageFont.FreeTypeFont:
    import matplotlib

    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSansMono.ttf"
    return ImageFont.truetype(str(path), size)


def render(frames: list[dict], out: Path, size: int = 14, max_frames: int = 160) -> None:
    font = _font(size)
    step = max(1, len(frames) // max_frames)
    picked = frames[::step]
    if picked[-1] is not frames[-1]:
        picked.append(frames[-1])
    width = max(len(line) for f in picked for line in f["lines"])
    height = max(len(f["lines"]) for f in picked)
    cw, ch = font.getbbox("M")[2], int(size * 1.35)
    W, H = cw * width + 32, ch * height + 32
    images, durations = [], []
    for k, f in enumerate(picked):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        for row, line in enumerate(f["lines"]):
            y = 16 + row * ch
            if row == 0:
                d.text((16, y), line, font=font, fill=ACCENT)
                continue
            head, sep, rest = line.partition("[")
            if sep:                              # stream line: dim index, bright bar, dim text
                bar, _, tail = rest.partition("]")
                d.text((16, y), head + "[", font=font, fill=DIM)
                x = 16 + cw * len(head + "[")
                d.text((x, y), bar, font=font, fill=FG)
                d.text((x + cw * len(bar), y), "]" + tail, font=font, fill=DIM)
            else:
                d.text((16, y), line, font=font, fill=FG)
        images.append(img.convert("P", palette=Image.ADAPTIVE, colors=32))
        nxt = picked[k + 1]["t"] if k + 1 < len(picked) else f["t"] + 2.5
        durations.append(max(60, int(1000 * (nxt - f["t"]))))
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {out} ({len(images)} frames, {out.stat().st_size / 1024:.0f} KiB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--font-size", type=int, default=14)
    args = ap.parse_args()
    frames = [json.loads(line) for line in open(args.frames, encoding="utf-8") if line.strip()]
    render(frames, Path(args.out), args.font_size)


if __name__ == "__main__":
    main()
