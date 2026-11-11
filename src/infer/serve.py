"""One CLI entry point for the server.

    python -m infer.serve --model Qwen/Qwen2.5-0.5B-Instruct --kv-cache-mb 512 --block-size 16 \
        --max-num-seqs 128 --max-batched-tokens 2048 --port 8000
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from infer.config import add_engine_args, engine_config_from_args


def build_server(cfg, served_model_name: str | None = None):
    from infer.api.server import create_app
    from infer.engine.engine import AsyncEngine, LLMEngine
    from infer.engine.static_engine import StaticBatchEngine
    from infer.metrics import Metrics

    metrics = Metrics()
    engine_cls = StaticBatchEngine if cfg.engine == "static" else LLMEngine
    engine = engine_cls(cfg, metrics=metrics)
    async_engine = AsyncEngine(engine, max_waiting=cfg.max_waiting)
    return create_app(async_engine, engine.tokenizer, served_model_name or cfg.model, cfg.max_model_len, metrics)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="infer: continuous-batching LLM server")
    add_engine_args(parser)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--no-request-log", action="store_true", help="skip the JSON line per finished request")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.no_request_log:
        logging.getLogger("infer.requests").setLevel(logging.WARNING)
    cfg = engine_config_from_args(args)
    app = build_server(cfg, args.served_model_name)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, access_log=False)


if __name__ == "__main__":
    main()
