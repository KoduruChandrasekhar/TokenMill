import sys, asyncio
from pathlib import Path
from playwright.async_api import async_playwright
page_path, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
TARGETS = {"hero": "header.hero", "scoreboard": "#scoreboard", "arch": "#architecture figure", "vllm": "#vs-vllm .panel",
           "profile": "#profile .panel", "graphs": "#graphs .two", "cards": "#cards", "facts": "#facts",
           "stories": "#stories .stories", "batching": "#batching .two", "kv": "#kv .two"}
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="msedge")
        ctx = await b.new_context(viewport={"width": 1200, "height": 900}, color_scheme="dark", device_scale_factor=1.6)
        pg = await ctx.new_page()
        await pg.goto(page_path.as_uri()); await pg.wait_for_timeout(1500)
        for name, sel in TARGETS.items():
            await pg.locator(sel).first.screenshot(path=str(out / f"{name}.png"))
        await b.close()
asyncio.run(main())
