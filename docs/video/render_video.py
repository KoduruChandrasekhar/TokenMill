"""Render the narrated demo video: 1280x720, 24 fps, H.264 + AAC.

    python render_video.py <video_dir> <project_root> <out.mp4>

Inputs in <video_dir>: narration.json, voice/sNN.wav, session.json (recorded locally), shots/*.png
(showcase sections). The T4 recording comes from <project_root>/bench/results/demo.gif."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence

V, ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
W, H, FPS = 1280, 720, 24
TOP, SUB = 54, 112                      # header band, subtitle band
BOX = (40, TOP + 8, W - 40, H - SUB - 10)  # content area
BG, INK, MUTED, ACCENT, TERM_BG = (14, 15, 16), (240, 239, 236), (150, 148, 142), (57, 135, 229), (11, 13, 16)
F = "C:/Windows/Fonts/"
font = lambda name, size: ImageFont.truetype(F + name, size)  # noqa: E731
F_SUB, F_LABEL, F_MONO, F_BADGE = font("segoeui.ttf", 26), font("CascadiaMono.ttf", 15), font("CascadiaMono.ttf", 15), font("CascadiaMono.ttf", 14)
F_TITLE, F_H2, F_BODY = font("segoeuib.ttf", 150), font("seguisb.ttf", 34), font("segoeui.ttf", 24)

narr = json.loads((V / "narration.json").read_text(encoding="utf-8"))
for i, n in enumerate(narr):
    with wave.open(str(V / "voice" / f"s{i:02d}.wav")) as w:
        n["dur"], n["rate"] = w.getnframes() / w.getframerate(), w.getframerate()
        n["pcm"] = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
session = json.loads((V / "session.json").read_text(encoding="utf-8"))
def shot(name: str, pad: int = 48) -> Image.Image:
    """A showcase screenshot with a margin, so the slow push-in never clips its edges."""
    src = Image.open(V / "shots" / f"{name}.png").convert("RGB")
    out = Image.new("RGB", (src.width + 2 * pad, src.height + 2 * pad), BG)
    out.paste(src, (pad, pad))
    return out

# ----------------------------------------------------------------------------------- drawing helpers

def base(label: str) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((40, 18), label, font=F_LABEL, fill=MUTED)
    d.text((W - 40, 18), "infer", font=font("segoeuib.ttf", 18), fill=INK, anchor="ra")
    d.line((40, TOP - 6, W - 40, TOP - 6), fill=(44, 44, 42), width=1)
    return im


def subtitle(im: Image.Image, text: str | None) -> None:
    if not text:
        return
    d = ImageDraw.Draw(im, "RGBA")
    lines = textwrap.wrap(text, 88)[:3]
    y0 = H - SUB + (SUB - 34 * len(lines)) // 2
    d.rectangle((0, H - SUB, W, H), fill=(0, 0, 0, 200))
    for k, line in enumerate(lines):
        d.text((W // 2, y0 + 34 * k), line, font=F_SUB, fill=(255, 255, 255), anchor="ma")


def badge(im: Image.Image, text: str) -> None:
    d = ImageDraw.Draw(im)
    x1, y = BOX[2], BOX[1] + 8
    w = d.textlength(text, font=F_BADGE) + 20
    d.rounded_rectangle((x1 - w - 10, y, x1 - 10, y + 26), radius=6, fill=(57, 135, 229))
    d.text((x1 - 10 - w / 2, y + 13), text, font=F_BADGE, fill=(255, 255, 255), anchor="mm")


def place(im: Image.Image, src: Image.Image, t: float, dur: float, mode: str = "zoom") -> None:
    """Fit src into BOX; 'zoom' = slow push-in, 'pan' = fit width and scroll top to bottom."""
    bw, bh = BOX[2] - BOX[0], BOX[3] - BOX[1]
    u = 0.0 if dur <= 0 else min(1.0, max(0.0, t / dur))
    ease = u * u * (3 - 2 * u)
    if mode == "pan" and src.height * bw / src.width > bh:
        s = bw / src.width
        scaled = src.resize((bw, round(src.height * s)), Image.LANCZOS)
        off = round(ease * (scaled.height - bh))
        im.paste(scaled.crop((0, off, bw, off + bh)), (BOX[0], BOX[1]))
        return
    s = min(bw / src.width, bh / src.height) * (1.0 + 0.035 * ease)
    scaled = src.resize((round(src.width * s), round(src.height * s)), Image.LANCZOS)
    x, y = BOX[0] + (bw - scaled.width) // 2, BOX[1] + (bh - scaled.height) // 2
    if scaled.width > bw or scaled.height > bh:
        cx, cy = max(0, -x + BOX[0]), max(0, -y + BOX[1])
        scaled = scaled.crop((cx, cy, cx + min(bw, scaled.width), cy + min(bh, scaled.height)))
        x, y = max(x, BOX[0]), max(y, BOX[1])
    im.paste(scaled, (x, y))


class Terminal:
    """Draws a terminal window with wrapped, scrolling text."""
    PAD, LH = 16, 21

    def __init__(self, title: str):
        self.title = title
        self.cw = F_MONO.getlength("M")
        self.cols = int((BOX[2] - BOX[0] - 2 * self.PAD) // self.cw)
        self.rows = int((BOX[3] - BOX[1] - 36 - 2 * self.PAD) // self.LH)

    def draw(self, im: Image.Image, text: str, cursor: bool = True) -> None:
        d = ImageDraw.Draw(im)
        d.rounded_rectangle(BOX, radius=10, fill=TERM_BG)
        d.rounded_rectangle((BOX[0], BOX[1], BOX[2], BOX[1] + 30), radius=10, fill=(21, 24, 29))
        d.rectangle((BOX[0], BOX[1] + 20, BOX[2], BOX[1] + 30), fill=(21, 24, 29))
        for k, c in enumerate(((255, 95, 87), (254, 188, 46), (40, 200, 64))):
            d.ellipse((BOX[0] + 14 + 18 * k, BOX[1] + 10, BOX[0] + 24 + 18 * k, BOX[1] + 20), fill=c)
        d.text(((BOX[0] + BOX[2]) / 2, BOX[1] + 15), self.title, font=F_BADGE, fill=(139, 146, 156), anchor="mm")
        lines = []
        for raw in text.split("\n"):
            lines += textwrap.wrap(raw, self.cols, replace_whitespace=False, drop_whitespace=False) or [""]
        lines = lines[-self.rows:]
        y = BOX[1] + 30 + self.PAD
        for line in lines:
            x = BOX[0] + self.PAD
            if line.startswith("$ "):
                d.text((x, y), "$", font=F_MONO, fill=(57, 200, 120))
                d.text((x + 2 * self.cw, y), line[2:], font=F_MONO, fill=(230, 232, 235))
            else:
                d.text((x, y), line, font=F_MONO, fill=(196, 201, 208))
            y += self.LH
        if cursor:
            last = lines[-1] if lines else ""
            cx = BOX[0] + self.PAD + self.cw * len(last)
            d.rectangle((cx, y - self.LH + 3, cx + self.cw - 2, y - 3), fill=(200, 205, 212))


def clean_server(events):
    text = "".join(e[1] for e in events)
    keep = [l for l in text.splitlines() if l.strip() and "httpx" not in l and "Fetching" not in l and "numexpr" not in l]
    # reveal each kept line at the time it was printed
    out, acc = [], ""
    for t, chunk in events:
        acc += chunk
        for l in keep:
            if l in acc and l not in [o[1] for o in out]:
                out.append((t, l))
    return out

# ----------------------------------------------------------------------------------- segments

segments = []          # each: dict(label, dur, draw(t)->Image, sentences=[(start, idx)])
GAP, LEAD, TAIL = 0.35, 0.45, 0.7


def seg_from_narration(name, label, draw, min_dur=0.0):
    idx = [i for i, n in enumerate(narr) if n["seg"] == name]
    t, sents = LEAD, []
    for i in idx:
        sents.append((t, i)); t += narr[i]["dur"] + GAP
    segments.append({"label": label, "dur": max(min_dur, t - GAP + TAIL), "draw": draw, "sents": sents})


def title_draw(t, dur):
    im = base("")
    d = ImageDraw.Draw(im)
    a = min(1.0, t / 0.8)
    col = tuple(int(BG[k] + (INK[k] - BG[k]) * a) for k in range(3))
    d.text((90, 150), "infer", font=F_TITLE, fill=col)
    x_end = 90 + d.textlength("infer", font=F_TITLE)
    if int(t * 2) % 2 == 0:
        d.rectangle((x_end + 10, 300, x_end + 70, 312), fill=ACCENT)
    d.text((94, 345), "An LLM inference server built from scratch in PyTorch", font=F_H2, fill=col)
    d.text((94, 400), "Qwen2.5-0.5B  ·  continuous batching  ·  paged KV cache  ·  CUDA graphs  ·  OpenAI API",
           font=F_BODY, fill=tuple(int(BG[k] + (MUTED[k] - BG[k]) * a) for k in range(3)))
    d.text((94, 440), "measured against vLLM on one Kaggle T4", font=F_BODY,
           fill=tuple(int(BG[k] + (MUTED[k] - BG[k]) * a) for k in range(3)))
    return im


seg_from_narration("title", "", title_draw)
arch = shot("arch")
seg_from_narration("arch", "How a request flows", lambda t, dur: (lambda im: (place(im, arch, t, dur), im)[1])(base("How a request flows")))

# --- live segment, driven by the recorded blocks
live_sents = {n.get("block"): [] for n in narr if n["seg"] == "live"}
for i, n in enumerate(narr):
    if n["seg"] == "live":
        live_sents[n["block"]].append(i)
server_lines = clean_server(session[0]["events"])
srv_cmd = "$ " + session[0]["cmd"]
TYPE_CPS = 55.0
plan = []   # (start, dur, kind, data)


def speech(idx):
    return sum(narr[i]["dur"] for i in idx) + GAP * max(0, len(idx) - 1)


cursor = 0.0
# block 0: server start-up, compressed
d0 = max(speech(live_sents[0]) + LEAD + TAIL, 8.0)
plan.append((cursor, d0, "server", None)); cursor += d0
# block 1: raw SSE at real speed
curl_cmd = "$ " + session[1]["cmd"]
d1 = max(speech(live_sents[1]) + LEAD + TAIL, len(curl_cmd) / TYPE_CPS + session[1]["events"][-1][0] + 1.5)
plan.append((cursor, d1, "curl", None)); cursor += d1
# blocks 2+3: the OpenAI SDK at 3x
SDK_SPEED = 3.0
sdk_stream = session[3]["events"][-1][0] / SDK_SPEED
d3 = max(speech(live_sents[3]) + LEAD + TAIL, 0.4 + 0.3 + 0.5 + sdk_stream + 1.5)
plan.append((cursor, d3, "sdk", None)); cursor += d3
# block 4: metrics
met_cmd = "$ " + session[4]["cmd"]
d4 = max(speech(live_sents[4]) + LEAD + TAIL, len(met_cmd) / TYPE_CPS + 2.5)
plan.append((cursor, d4, "metrics", None)); cursor += d4
# block 5: eight streams at 5x
DEMO_SPEED = 5.0
frames5 = session[5]["frames"]
demo_play = frames5[-1]["t"] / DEMO_SPEED
d5 = max(speech(live_sents[5]) + LEAD + TAIL, 0.8 + demo_play + 1.5)
plan.append((cursor, d5, "demo", None)); cursor += d5
live_total = cursor
term_srv, term_cli = Terminal("terminal 1 · server"), Terminal("terminal 2 · client")
sdk_file = "".join(e[1] for e in session[2]["events"]).rstrip("\n")


def typed(cmd, t):
    n = int(max(0.0, t) * TYPE_CPS)
    return cmd[:n], n >= len(cmd)


def replay(events, t, speed=1.0):
    return "".join(c for (tt, c) in events if tt / speed <= t)


def live_draw(t, dur):
    for start, d, kind, _ in plan:
        if start <= t < start + d or (kind == "demo" and t >= start):
            u = t - start
            break
    im = base("Live on this laptop's CPU  ·  real model, real server, recorded")
    if kind == "server":
        shown, done = typed(srv_cmd, u)
        body = shown
        if done:
            # weights load silently for most of the real minute and the log lines arrive together
            # at the end, so reveal them from a third of the block on, in order, timestamps intact
            t_cmd = len(srv_cmd) / TYPE_CPS
            first = t_cmd + 0.33 * max(0.0, d - t_cmd - 1.5)
            for k, (tt, line) in enumerate(server_lines):
                if first + 0.35 * k <= u:
                    body += "\n" + line
        term_srv.draw(im, body)
        badge(im, f"start-up compressed ({session[0]['events'][-1][0]:.0f} s real)")
    elif kind == "curl":
        shown, done = typed(curl_cmd, u)
        body = shown
        if done:
            body += "\n" + replay(session[1]["events"], u - len(curl_cmd) / TYPE_CPS)
        term_cli.draw(im, body)
        badge(im, "real time")
    elif kind == "sdk":
        a = "$ cat client.py"
        shown, done = typed(a, u)
        body = shown
        if done:
            body += "\n" + sdk_file
            t2 = len(a) / TYPE_CPS + 0.4
            b, done2 = typed("$ python client.py", u - t2)
            body += "\n" + b
            if done2:
                body += "\n" + replay(session[3]["events"], u - t2 - len("$ python client.py") / TYPE_CPS, SDK_SPEED)
        term_cli.draw(im, body)
        badge(im, f"{SDK_SPEED:g}x speed")
    elif kind == "metrics":
        shown, done = typed(met_cmd, u)
        body = shown + ("\n" + "".join(e[1] for e in session[4]["events"]).rstrip("\n") if done and u > len(met_cmd) / TYPE_CPS + 0.3 else "")
        term_cli.draw(im, body)
    else:
        cmd = "$ python -m bench.demo --streams 8"
        shown, done = typed(cmd, u)
        body = shown
        if done:
            tt = (u - len(cmd) / TYPE_CPS) * DEMO_SPEED
            fr = frames5[0]
            for f in frames5:
                if f["t"] <= tt:
                    fr = f
            body += "\n" + "\n".join(fr["lines"])
        term_cli.draw(im, body, cursor=False)
        badge(im, f"{DEMO_SPEED:g}x speed")
    return im


live_sent_times = []
for (start, d, kind, _), blk in zip(plan, (0, 1, 3, 4, 5)):
    t = start + LEAD
    for i in live_sents[blk]:
        live_sent_times.append((t, i)); t += narr[i]["dur"] + GAP
segments.append({"label": "live", "dur": live_total, "draw": live_draw, "sents": live_sent_times})

# --- the T4 recording
gif = Image.open(ROOT / "bench" / "results" / "demo.gif")
gif_frames = [(fr.convert("RGB"), fr.info.get("duration", 250) / 1000) for fr in ImageSequence.Iterator(gif)]


def t4_draw(t, dur):
    im = base("Recorded on a Kaggle T4  ·  50 concurrent streams")
    acc, cur = 0.0, gif_frames[-1][0]
    tt = t - 0.6
    for fr, fd in gif_frames:
        if tt < acc + fd:
            cur = fr
            break
        acc += fd
    if tt < 0:
        cur = gif_frames[0][0]
    place(im, cur, 0, 1, "zoom")
    badge(im, "real time")
    return im


seg_from_narration("t4", "t4", t4_draw)

# --- results from the showcase page
hero, vllm, prof, graphs, cards, facts, stories = (shot(n) for n in ("hero", "vllm", "profile", "graphs", "cards", "facts", "stories"))


def still(img, label, mode="zoom"):
    return lambda t, dur: (lambda im: (place(im, img, t, dur, mode), im)[1])(base(label))


seg_from_narration("scoreboard", "Results", still(hero, "Results  ·  ~17 GPU hours on Kaggle, three runs per point", "pan"))
seg_from_narration("vllm", "", still(vllm, "Results  ·  against vLLM, same users in flight"))
seg_from_narration("profile", "", still(prof, "Where a decode step goes  ·  batch 32"))
seg_from_narration("graphs", "", still(graphs, "CUDA graphs  ·  three versions"))
seg_from_narration("cards", "", still(cards, "Tradeoffs, measured"))
seg_from_narration("facts", "", still(facts, "Correctness"))
seg_from_narration("stories", "", still(stories, "Bugs worth telling", "pan"))


def close_draw(t, dur):
    im = base("")
    d = ImageDraw.Draw(im)
    d.text((90, 200), "infer", font=F_TITLE, fill=INK)
    rows = ["src/infer        the engine, scheduler, paged KV cache, CUDA graphs, API",
            "tests/           97 CPU tests, 5 GPU tests, fp32 golden set",
            "bench/results    every raw run, figures, results.md",
            "docs/showcase    the interactive results page"]
    for k, r in enumerate(rows):
        d.text((96, 400 + 34 * k), r, font=font("CascadiaMono.ttf", 20), fill=MUTED if k else INK)
    return im


seg_from_narration("close", "", close_draw, min_dur=5.0)

# ----------------------------------------------------------------------------------- timeline + audio
t0 = 0.0
subs, audio_marks = [], []
for s in segments:
    s["t0"] = t0
    for st, i in s["sents"]:
        subs.append((t0 + st, t0 + st + narr[i]["dur"], narr[i]["text"]))
        audio_marks.append((t0 + st, i))
    t0 += s["dur"]
total = t0
(V / "timeline.json").write_text(json.dumps([[s["label"], round(s["t0"], 2), round(s["dur"], 2)] for s in segments]))
rate = narr[0]["rate"]
audio = np.zeros(int((total + 0.5) * rate), dtype=np.int32)
for st, i in audio_marks:
    a = int(st * rate); pcm = narr[i]["pcm"]
    audio[a:a + len(pcm)] += pcm
audio = np.clip(audio, -32768, 32767).astype(np.int16)
wav_path = V / "narration_track.wav"
with wave.open(str(wav_path), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(audio.tobytes())
def _ts(x: float) -> str:
    ms = int(round(x * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


(OUT.with_suffix(".srt")).write_text("".join(f"{k + 1}\n{_ts(a)} --> {_ts(b)}\n{txt}\n\n"
                                             for k, (a, b, txt) in enumerate(subs)), encoding="utf-8")

# ----------------------------------------------------------------------------------- frames
silent = V / "video_silent.mp4"
writer = imageio_ffmpeg.write_frames(str(silent), (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                                     output_params=["-crf", "20", "-preset", "medium"], macro_block_size=8)
writer.send(None)
n_frames = int(total * FPS)
prev_last = None
for f in range(n_frames):
    t = f / FPS
    seg = next(s for s in segments if s["t0"] <= t < s["t0"] + s["dur"]) if t < total else segments[-1]
    lt = t - seg["t0"]
    im = seg["draw"](lt, seg["dur"])
    if lt < 0.3 and seg is not segments[0] and seg.get("first") is not None:
        im = Image.blend(seg["first"], im, lt / 0.3)
    sub = next((txt for a, b, txt in subs if a <= t < b + 0.25), None)
    subtitle(im, sub)
    # crossfade source for the next segment
    nxt = segments.index(seg) + 1
    if nxt < len(segments) and t + 1 / FPS >= seg["t0"] + seg["dur"]:
        segments[nxt]["first"] = im.copy()
    writer.send(np.asarray(im, dtype=np.uint8).tobytes())
    if f % (FPS * 20) == 0:
        print(f"  {t:6.1f} / {total:.1f} s", flush=True)
writer.close()

ff = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(silent), "-i", str(wav_path), "-c:v", "copy", "-c:a", "aac",
                "-b:a", "160k", "-ar", "48000", "-shortest", "-movflags", "+faststart", str(OUT)], check=True)
print(f"wrote {OUT} ({total:.1f} s, {n_frames} frames)")
