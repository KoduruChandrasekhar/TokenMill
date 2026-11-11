"""CUDA graphs for decode steps (stretch goal).

For a 0.5B model a decode step launches hundreds of small kernels; at small batch the GPU waits on
Python and launch overhead more than on math. A CUDA graph records the whole decode forward once
and replays it with a single launch.

Everything inside a graph must have fixed shapes, so:
* static input buffers (token ids, positions, slot mapping, context lengths, block tables) are
  allocated once at the largest batch size;
* one graph per (batch size, context bucket): the batch is padded up to the next captured size,
  and the block-table width is fixed at the bucket's length and masked per sequence. Bucketing
  the context keeps a short-context batch from gathering a full max_model_len of KV every step.
  Buckets are every 128 tokens up to 2048 (power-of-two buckets made a 640-token batch attend
  over 1024 positions);
* graphs only up to batch 32 (--cuda-graph-max-batch). Small batches are launch-bound and graphs
  win 1.6-2.4x on a T4; at 128 concurrent sequences the step is bound by the KV gather and
  attention, and replaying a padded graph was 5-12% slower than eager with either bucketing;
* padded rows write their K/V into a reserved scratch block that is never handed to a sequence.
"""

from __future__ import annotations

import bisect
import logging

import torch

from infer.kv.context import StepContext

log = logging.getLogger("infer.cuda_graphs")

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
CONTEXT_BUCKETS = [256, 512, 1024, 2048, 4096, 8192]
FINE_BUCKETS_UP_TO = 2048


def context_buckets(max_len: int, step: int) -> list[int]:
    """Every `step` tokens up to FINE_BUCKETS_UP_TO, powers of two beyond; step 0: powers of two
    only. The last bucket is always max_len."""
    fine = list(range(step, min(max_len, FINE_BUCKETS_UP_TO) + 1, step)) if step > 0 else []
    coarse = [b for b in CONTEXT_BUCKETS if b > (fine[-1] if fine else 0)]
    return sorted({b for b in fine + coarse if b < max_len} | {max_len})


class CudaGraphRunner:
    def __init__(self, model, cache, cfg):
        self.model = model
        self.cache = cache
        self.pool = cache.pool
        self.block_size = cache.block_size
        self.device = model.embed_tokens.weight.device
        self.scale = model.cfg.head_dim ** -0.5
        limit = min(cfg.max_num_seqs, cfg.cuda_graph_max_batch)    # above it: eager (can_run is False)
        self.batch_sizes = [b for b in BATCH_SIZES if b < limit] + [limit]
        self.batch_sizes = sorted(set(b for b in self.batch_sizes if b <= max(BATCH_SIZES)))
        self.buckets = context_buckets(cfg.max_model_len, cfg.cuda_graph_context_step)
        self.sink = self.pool.num_blocks - 1                  # reserved: the manager never allocates it
        assert cache.num_blocks <= self.sink, "the scratch block must be outside the allocatable range"
        max_bs = self.batch_sizes[-1]
        self.max_width = -(-self.buckets[-1] // self.block_size)
        dev = self.device
        self.input_ids = torch.zeros(max_bs, dtype=torch.long, device=dev)
        self.positions = torch.zeros(max_bs, dtype=torch.long, device=dev)
        self.slot_mapping = torch.zeros(max_bs, dtype=torch.long, device=dev)
        self.context_lens = torch.ones(max_bs, dtype=torch.long, device=dev)
        self.block_tables = torch.full((max_bs, self.max_width), self.sink, dtype=torch.long, device=dev)
        # Graphs write final hidden states here; the logits matmul (one kernel) runs after replay,
        # so no graph keeps a [batch, 151936] output alive in the shared memory pool.
        self.hidden_out = torch.zeros(max_bs, model.cfg.hidden_size, dtype=model.embed_tokens.weight.dtype,
                                      device=dev)
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.mempool = None

    def _reset_buffers(self) -> None:
        self.input_ids.zero_()
        self.positions.zero_()
        self.slot_mapping.fill_(self.sink * self.block_size)
        self.context_lens.fill_(1)
        self.block_tables.fill_(self.sink)

    def _forward(self, bs: int, bucket: int) -> torch.Tensor:
        width = -(-bucket // self.block_size)
        ctx = StepContext(self.pool, self.slot_mapping[:bs], self.scale, [], 0, bs,
                          self.block_tables[:bs, :width], self.context_lens[:bs], bucket)
        return self.model(self.input_ids[:bs], self.positions[:bs], ctx)

    @torch.inference_mode()
    def capture(self) -> None:
        """Capture every (batch, bucket) graph that fits. Out of memory part-way, the graphs
        already captured are kept and the remaining shapes decode eagerly (run() falls back)."""
        self._reset_buffers()
        torch.cuda.synchronize(self.device)
        free_before = torch.cuda.mem_get_info(self.device)[0]
        total = len(self.buckets) * len(self.batch_sizes)
        # Largest first, so the shared memory pool is sized by the biggest graph.
        try:
            for bucket in reversed(self.buckets):
                for bs in reversed(self.batch_sizes):
                    self._forward(bs, bucket)                     # warm-up (cuBLAS handles, workspaces)
                    torch.cuda.synchronize(self.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=self.mempool):
                        self.hidden_out[:bs].copy_(self._forward(bs, bucket))
                    if self.mempool is None:
                        self.mempool = graph.pool()
                    self.graphs[(bs, bucket)] = graph
        except (torch.OutOfMemoryError, RuntimeError) as e:     # AcceleratorError is a RuntimeError
            if "out of memory" not in str(e):
                raise
            log.warning("out of GPU memory after %d of %d graphs; the rest decode eagerly "
                        "(raise --cuda-graph-reserve-mb or the context step)", len(self.graphs), total)
        torch.cuda.synchronize(self.device)
        used = free_before - torch.cuda.mem_get_info(self.device)[0]
        log.info("captured %d of %d decode graphs (batch %s x context %s), %.0f MiB of device memory",
                 len(self.graphs), total, self.batch_sizes, self.buckets, used / 2**20)

    def can_run(self, seqs) -> bool:
        return (0 < len(seqs) <= self.batch_sizes[-1]
                and max(s.length for s in seqs) <= self.buckets[-1])

    @torch.inference_mode()
    def run(self, seqs) -> torch.Tensor:
        n = len(seqs)
        bs = self.batch_sizes[bisect.bisect_left(self.batch_sizes, n)]
        max_len = max(s.length for s in seqs)
        bucket = self.buckets[bisect.bisect_left(self.buckets, max_len)]
        width = -(-bucket // self.block_size)
        bsz = self.block_size
        ids, pos, slots, lens, tables = [], [], [], [], []
        for s in seqs:
            p = s.length - 1
            ids.append(s.last_token)
            pos.append(p)
            slots.append(s.block_table[p // bsz] * bsz + p % bsz)
            lens.append(s.length)
            row = s.block_table[:width]
            tables.append(row + [self.sink] * (width - len(row)))
        pad = bs - n
        ids += [0] * pad
        pos += [0] * pad
        slots += [self.sink * bsz] * pad
        lens += [1] * pad
        tables += [[self.sink] * width] * pad
        pin = self.device.type == "cuda"
        host = torch.tensor([ids, pos, slots, lens], dtype=torch.long, pin_memory=pin)
        dev = host.to(self.device, non_blocking=True)
        self.input_ids[:bs].copy_(dev[0])
        self.positions[:bs].copy_(dev[1])
        self.slot_mapping[:bs].copy_(dev[2])
        self.context_lens[:bs].copy_(dev[3])
        self.block_tables[:bs, :width].copy_(torch.tensor(tables, dtype=torch.long, pin_memory=pin)
                                             .to(self.device, non_blocking=True))
        graph = self.graphs.get((bs, bucket))
        if graph is not None:
            graph.replay()
        else:       # not captured (CPU tests): the same padded static inputs, run eagerly
            self.hidden_out[:bs].copy_(self._forward(bs, bucket))
        return self.model.compute_logits(self.hidden_out[:n])
