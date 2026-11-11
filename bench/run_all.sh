#!/usr/bin/env bash
# Every system at its budgeted load points (plan, Part 7 "GPU budget"), three runs each, then
# the offline experiments and every plot. Run on the GPU you report numbers for (Kaggle T4).
#
#   bash bench/run_all.sh                      # everything
#   ONLY="D F" bash bench/run_all.sh           # a subset of systems
#   ONLY="A B C D" SKIP_EXPERIMENTS=0 bash bench/run_all.sh     # e.g. part 1 of a split run
#
# Systems: A naive HF | B static batching | C continuous + contiguous cache | D continuous + paged
#          E = D + CUDA graphs (stretch) | F vLLM (same model, fp16, same max length and memory share)
#          X = Tier 2 extras on D: decode-first, recompute vs swap under overload, prefix caching on/off,
#              and the vllm-bench-serve cross-check of D
#          K = chunked prefill before/after on the mixed workload
#          Dc = D at 8/32/128 users in flight only
# Each system's load points run in one server session, so model loading happens once.
# Unattended runs: a system whose server fails to start is skipped and the rest still run.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
PY=${PY:-python}
VLLM=${VLLM:-vllm}                 # vLLM CLI; on Kaggle install it in its own venv
RUNS=${RUNS:-3}
PORT=${PORT:-8000}
UTIL=${UTIL:-0.90}                 # same GPU memory share for our server and vLLM
MAXLEN=${MAXLEN:-2048}
ONLY=${ONLY:-"A B C D E F X"}
SKIP_EXPERIMENTS=${SKIP_EXPERIMENTS:-0}
SHAREGPT=${SHAREGPT:-}             # optional local path to ShareGPT_V3_unfiltered_cleaned_split.json
FULL_RATES=1,2,4,8,16,32,inf
CONC=1,8,32,128
LOGS=bench/results/logs
mkdir -p "$LOGS" bench/results/raw

SERVER_PID=""
has() { [[ " $ONLY " == *" $1 "* ]]; }
wait_ready() {
  for _ in $(seq 1 300); do
    if curl -sf "localhost:$PORT/health" >/dev/null; then return 0; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "server died; see $LOGS"; return 1; fi
    sleep 2
  done
  echo "server did not become ready"; return 1
}
start() { local name=$1; shift; echo "=== starting $name: $*"; "$@" >"$LOGS/$name.log" 2>&1 & SERVER_PID=$!; wait_ready; }
stop() { if [[ -n "$SERVER_PID" ]]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi; SERVER_PID=""; sleep 5; }
trap stop EXIT
lg() {
  local extra=()
  [[ -n "$SHAREGPT" ]] && extra+=(--sharegpt-path "$SHAREGPT")
  $PY -m bench.load_gen --base-url "http://localhost:$PORT" --model "$MODEL" --tokenizer "$MODEL" --runs "$RUNS" "${extra[@]}" "$@"
}
# exec: the background job IS the server process, so stop() kills the server itself
ours() { exec $PY -m infer.serve --model "$MODEL" --port "$PORT" --max-model-len "$MAXLEN" --dtype float16 \
           --gpu-memory-utilization "$UTIL" --no-request-log "$@"; }

if has A; then
  if start A $PY -m baselines.naive_hf --model "$MODEL" --port "$PORT" --dtype float16 --max-model-len "$MAXLEN"; then
    lg --system A --workload fixed --num-requests 100 --rates inf        # saturates at any rate
  fi
  stop
fi

if has B; then
  if start B ours --engine static --static-batch-size 16 --static-batch-wait-ms 50; then
    lg --system B --workload fixed --num-requests 200 --rates 1,4,inf
    lg --system B --workload mixed --num-requests 200 --rates 1,4,inf
  fi
  stop
fi

if has C; then
  if start C ours --cache-type contiguous; then
    lg --system C --workload fixed --num-requests 300 --rates 1,4,inf
    lg --system C --workload mixed --num-requests 300 --rates 1,4,inf
  fi
  stop
fi

if has D; then
  if start D ours --cache-type paged --block-size 16; then
    lg --system D --workload fixed --num-requests 300 --rates "$FULL_RATES" --concurrency "$CONC"
    lg --system D --workload sharegpt --num-requests 300 --rates "$FULL_RATES" --concurrency "$CONC"
    lg --system D --workload mixed --num-requests 300 --rates 1,4,inf
  fi
  stop
fi

if has Dc; then
  # D at 8/32/128 users in flight only (a rerun of points an older load_gen skipped)
  if start D ours --cache-type paged --block-size 16; then
    lg --system D --workload fixed --num-requests 300 --concurrency 8,32,128
    lg --system D --workload sharegpt --num-requests 300 --concurrency 8,32,128
  fi
  stop
fi

# E_FLAGS/E_TAG/E_RATES/E_CONC rerun E on a subset or another layout, e.g. the power-of-two
# context buckets it was first measured with:  E_FLAGS="--cuda-graph-context-step 0" E_TAG=pow2
E_RATES=${E_RATES:-$FULL_RATES}
E_CONC=${E_CONC:-$CONC}
if has E; then
  if start E ours --cache-type paged --cuda-graphs ${E_FLAGS:-}; then
    lg --system E ${E_TAG:+--tag "$E_TAG"} --workload fixed --num-requests 300 --rates "$E_RATES" --concurrency "$E_CONC"
    lg --system E ${E_TAG:+--tag "$E_TAG"} --workload sharegpt --num-requests 300 --rates "$E_RATES" --concurrency "$E_CONC"
  fi
  stop
fi

vllm_bench() {   # independent client: vLLM's own benchmark tool against whatever server is up
  "$VLLM" bench serve --backend openai --base-url "http://localhost:$PORT" --model "$MODEL" \
    --dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos --num-prompts 300 \
    --request-rate 4 --save-result --result-dir bench/results/vllm_bench --result-filename "$1" \
    || echo "vllm bench serve ($1) failed (non-fatal)"
}

if has F; then
  if command -v "$VLLM" >/dev/null; then
    if start F "$VLLM" serve "$MODEL" --dtype float16 --max-model-len "$MAXLEN" --gpu-memory-utilization "$UTIL" \
         --port "$PORT"; then
      lg --system F --workload fixed --num-requests 300 --rates "$FULL_RATES" --concurrency "$CONC"
      lg --system F --workload sharegpt --num-requests 300 --rates "$FULL_RATES" --concurrency "$CONC"
      vllm_bench F_fixed_rate4.json
    fi
    stop
  else
    echo "vLLM CLI '$VLLM' not found: skipping system F (set VLLM=/path/to/venv/bin/vllm)"
  fi
fi

if has X; then
  if command -v "$VLLM" >/dev/null && start D-crosscheck ours; then
    vllm_bench D_fixed_rate4.json                  # compare with load_gen's D fixed rate=4 numbers
  fi
  stop
  # prefill-first vs decode-first on the mixed workload (prefill-first = system D's mixed runs)
  if start D-decode_first ours --policy decode_first --max-waiting-steps 20; then
    lg --system D --tag decode_first --workload mixed --num-requests 300 --rates 4,8
  fi
  stop
  if start D-prefill_first ours --policy prefill_first; then
    lg --system D --tag prefill_first --workload mixed --num-requests 300 --rates 4,8
  fi
  stop
  # recompute vs swap under overload: a deliberately small KV pool forces preemption
  for mode in recompute swap; do
    if start "D-$mode" ours --kv-cache-mb 64 --preemption-mode "$mode" --swap-space-mb 2048; then
      lg --system D --tag "overload_$mode" --workload sharegpt --num-requests 300 --rates 8,16 --no-stop-on-saturation
    fi
    stop
  done
  # prefix caching on the shared-prefix workload, on and off
  for pc in on off; do
    flag=(); [[ "$pc" == on ]] && flag=(--enable-prefix-caching)
    if start "D-prefix_$pc" ours "${flag[@]}"; then
      lg --system D --tag "prefix_$pc" --workload shared_prefix --num-requests 200 --rates 4,inf
    fi
    stop
  done
fi

if has K; then
  # chunked prefill before/after on the mixed workload: does splitting prompts cut p99 TPOT?
  if start D-unchunked ours; then
    lg --system D --tag unchunked --workload mixed --num-requests 300 --rates 4,8,16 --no-stop-on-saturation
  fi
  stop
  if start D-chunked ours --enable-chunked-prefill --max-batched-tokens 512; then
    lg --system D --tag chunked512 --workload mixed --num-requests 300 --rates 4,8,16 --no-stop-on-saturation
  fi
  stop
fi

if [[ "$SKIP_EXPERIMENTS" != "1" ]]; then
  $PY -m bench.batch1 --model "$MODEL" || echo "batch1 failed"
  $PY -m bench.static_experiments --model "$MODEL" || echo "static_experiments failed"
  $PY -m bench.kv_experiments --model "$MODEL" ${SHAREGPT:+--sharegpt-path "$SHAREGPT"} || echo "kv_experiments failed"
  $PY -m bench.profile_decode --model "$MODEL" --batch 32 || echo "profile (eager) failed"
  $PY -m bench.profile_decode --model "$MODEL" --batch 32 --cuda-graphs --tag graphs || echo "profile (graphs) failed"
fi

$PY -m bench.plot || echo "plot failed"
echo "done: bench/results/results.md, bench/results/figures/"
