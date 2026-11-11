# Runs kept out of the tables

`part4_E_graph_capture_oom/` — system E (`--cuda-graphs`, 128-token context buckets) from the part 4
Kaggle run. Capturing 128 graphs ran out of GPU memory next to a KV pool sized at 90% of the T4
(`logs/E-part4-graph-capture-oom.log`), the engine fell back to eager decoding as designed, and these
runs therefore measure eager D, not graphs. They agree with D within 1% (e.g. fixed, all at once:
1,682 vs 1,686 tok/s), which makes them a useful repeatability check, but they are not E.
The fix: `--cuda-graph-reserve-mb` keeps memory out of the KV pool for the graphs, and capture keeps
whatever graphs fit instead of dropping all of them.
