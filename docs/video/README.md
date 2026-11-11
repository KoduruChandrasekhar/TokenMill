# The narrated walkthrough video

`docs/showcase/infer_demo.mp4` (3 min 23 s, 1280×720, subtitles burned in, plus `infer_demo.srt`)
was made with the scripts in this folder, on a Windows laptop, with no online services:

1. `python record_local.py <repo> <this folder>` starts the real server on the CPU, runs a raw curl
   SSE request, the OpenAI SDK (`client.py`), a `/metrics` query and 8 concurrent streams
   (`bench.demo`), and timestamps every chunk of output into `session.json`.
2. `powershell -File speak.ps1 -NarrationJson narration.json -OutDir voice` reads each sentence of
   `narration.json` (its `say` field) with the Windows "Microsoft Mark" voice into `voice/sNN.wav`.
   To use your own voice instead, record `voice/s00.wav` … `voice/s24.wav` (16-bit mono WAV, any
   rate) one sentence each; the video timing follows their lengths.
3. `python section_shots.py <repo>/docs/showcase/index.html shots` screenshots the showcase page's
   sections (dark theme) with Edge through Playwright.
4. `python render_video.py <this folder> <repo> out.mp4` draws every frame (title, architecture,
   the terminal replay, the T4 recording from `bench/results/demo.gif`, the result sections),
   lays each sentence on the audio track at its scene, and encodes H.264 + AAC with the ffmpeg
   that ships in `imageio-ffmpeg`. Replays faster than real time are labelled on screen.
