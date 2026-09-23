# Jevlet night-one research program

Objective: identify a small, calibrated System-One decision architecture that improves decision
quality per activated compute on an RTX A2000 4 GB laptop.

The benchmark, generated split files, manifest hashes, metric code, compute budgets, and vault are
immutable during a search. The runner can vary only the explicit candidate dimensions in
`research/search_space.py`: one state representation, option representation, head, loss, or
attention topology at a time. The hidden vault is never referenced by the autoresearch runner.

Each run records its hypothesis and expected mechanism before training. It writes the exact config,
checkpoint, raw/calibrated metrics, hardware measurements, and outcome to a run directory plus the
append-only JSONL experiment log. Promotion peels Pareto layers across quality, calibration,
robustness, throughput, latency, and VRAM; a tie-break is used only when a stage has more Pareto
winners than promotion slots.

Stages use many short proxy runs, then promote approximately eight and three candidates. The
morning result is exploratory evidence and a useful checkpoint, not statistical proof or a claim to
have reproduced TypeSafe's proprietary Jev implementation.

