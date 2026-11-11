# GPU image for the server and the benchmarks (CUDA 12.6 runtime; T4 = compute capability 7.5).
#   docker build -t infer .
#   docker run --gpus all -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface infer
#   docker run --gpus all infer python -m bench.load_gen --system D --base-url http://host:8000 ...
FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl git && rm -rf /var/lib/apt/lists/*

COPY requirements.lock .
RUN pip install -r requirements.lock

COPY pyproject.toml README.md ./
COPY src ./src
COPY baselines ./baselines
COPY bench ./bench
COPY tests ./tests
RUN pip install -e . --no-deps

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -sf localhost:8000/health || exit 1
CMD ["python", "-m", "infer.serve", "--model", "Qwen/Qwen2.5-0.5B-Instruct", "--port", "8000", \
     "--dtype", "float16", "--max-model-len", "2048", "--max-num-seqs", "128", "--max-batched-tokens", "2048"]
