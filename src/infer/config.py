"""Model and engine configuration, plus the CLI flags that build them."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class ModelConfig:
    """The subset of a HuggingFace Qwen2 config.json the runtime needs."""

    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    eos_token_ids: list[int] = field(default_factory=lambda: [151645, 151643])

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def kv_bytes_per_token(self, dtype_bytes: int) -> int:
        # 2 (K and V) x layers x kv_heads x head_dim x bytes
        return 2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim * dtype_bytes

    @classmethod
    def from_pretrained_dir(cls, path: str | Path) -> ModelConfig:
        path = Path(path)
        raw = json.loads((path / "config.json").read_text())
        eos: list[int] = []
        gen_path = path / "generation_config.json"
        for source in ([json.loads(gen_path.read_text())] if gen_path.exists() else []) + [raw]:
            val = source.get("eos_token_id")
            if val is None:
                continue
            for tok in val if isinstance(val, list) else [val]:
                if tok not in eos:
                    eos.append(int(tok))
        rope_theta = raw.get("rope_theta")
        if rope_theta is None:  # transformers 5 writes rope_parameters instead
            rope_theta = (raw.get("rope_parameters") or {}).get("rope_theta", 10000.0)
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw.get("num_key_value_heads", raw["num_attention_heads"]),
            max_position_embeddings=raw.get("max_position_embeddings", 32768),
            rope_theta=float(rope_theta),
            rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            eos_token_ids=eos,
        )


@dataclass
class EngineConfig:
    """Every knob of the engine. Field names match the CLI flags (dashes become underscores)."""

    model: str = DEFAULT_MODEL
    dtype: str = "auto"                  # auto = float16 on CUDA, float32 on CPU
    device: str = "auto"                 # auto = cuda if available
    engine: str = "continuous"           # continuous | static
    cache_type: str = "paged"            # paged | contiguous
    block_size: int = 16
    kv_cache_mb: float | None = None     # cap on the KV pool; None = size from free memory
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 2048
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 2048
    watermark: float = 0.01              # fraction of blocks kept back when admitting
    policy: str = "prefill_first"        # prefill_first | decode_first
    max_waiting_steps: int = 20          # decode_first: force a prefill after this many decode steps
    preemption_mode: str = "recompute"   # recompute | swap
    swap_space_mb: float = 1024.0
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = False # mixed steps: all decodes + prompt chunks in the leftover budget
    cuda_graphs: bool = False
    cuda_graph_context_step: int = 128   # context buckets every N tokens up to 2048; 0 = powers of two
    cuda_graph_reserve_mb: float = 2048.0  # GPU memory kept out of the KV pool for the graphs
    cuda_graph_max_batch: int = 32       # larger decode batches run eagerly (faster there on a T4)
    max_waiting: int = 1000              # waiting-queue cap; beyond it the API returns 503
    request_timeout: float | None = 600.0
    # static batching (baseline B)
    static_batch_size: int = 16
    static_batch_wait_ms: float = 50.0
    debug_steps: bool = False
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def add_engine_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    d = EngineConfig()
    parser.add_argument("--model", default=d.model, help="HF repo id or local directory")
    parser.add_argument("--dtype", default=d.dtype, choices=["auto", "float16", "float32", "bfloat16"])
    parser.add_argument("--device", default=d.device)
    parser.add_argument("--engine", default=d.engine, choices=["continuous", "static"])
    parser.add_argument("--cache-type", default=d.cache_type, choices=["paged", "contiguous"])
    parser.add_argument("--block-size", type=int, default=d.block_size)
    parser.add_argument("--kv-cache-mb", type=float, default=d.kv_cache_mb)
    parser.add_argument("--gpu-memory-utilization", type=float, default=d.gpu_memory_utilization)
    parser.add_argument("--max-model-len", type=int, default=d.max_model_len)
    parser.add_argument("--max-num-seqs", type=int, default=d.max_num_seqs)
    parser.add_argument("--max-batched-tokens", "--max-num-batched-tokens", dest="max_num_batched_tokens",
                        type=int, default=d.max_num_batched_tokens)
    parser.add_argument("--watermark", type=float, default=d.watermark)
    parser.add_argument("--policy", default=d.policy, choices=["prefill_first", "decode_first"])
    parser.add_argument("--max-waiting-steps", type=int, default=d.max_waiting_steps)
    parser.add_argument("--preemption-mode", default=d.preemption_mode, choices=["recompute", "swap"])
    parser.add_argument("--swap-space-mb", type=float, default=d.swap_space_mb)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true",
                        help="split long prompts into chunks that share steps with running decodes")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--cuda-graph-context-step", type=int, default=d.cuda_graph_context_step,
                        help="graph context buckets every N tokens up to 2048 (0: powers of two)")
    parser.add_argument("--cuda-graph-reserve-mb", type=float, default=d.cuda_graph_reserve_mb,
                        help="GPU memory kept out of the KV pool for CUDA graphs")
    parser.add_argument("--cuda-graph-max-batch", type=int, default=d.cuda_graph_max_batch,
                        help="largest decode batch run as a graph; larger batches run eagerly")
    parser.add_argument("--max-waiting", type=int, default=d.max_waiting)
    parser.add_argument("--request-timeout", type=float, default=d.request_timeout)
    parser.add_argument("--static-batch-size", type=int, default=d.static_batch_size)
    parser.add_argument("--static-batch-wait-ms", type=float, default=d.static_batch_wait_ms)
    parser.add_argument("--debug-steps", action="store_true", help="log one line per engine step")
    parser.add_argument("--seed", type=int, default=d.seed)
    return parser


def engine_config_from_args(args: argparse.Namespace) -> EngineConfig:
    names = EngineConfig.__dataclass_fields__.keys()
    return EngineConfig(**{k: getattr(args, k) for k in names if hasattr(args, k)})
