# Phase 2 protocol (written 2026-11-07, before any phase 0 or phase 2 result)

Fixed before seeing results, so selection cannot follow the data. Screening uses seed 1337 only; phase 3 uses
new seeds 2001 to 2005 and never 1337. Every number below is the validation loss train.py prints at the
final eval ("step N: ... val loss X"), which is also what selects the saved checkpoint.

## Phase 0 check

- s_short (5K iters, the README short budget of 327.68M tokens): val loss within 0.01 of 1.511, else stop
  before phase 2 and report.
- s_full (100K iters): within 0.005 of 1.2905. It runs beside phase 2 if cards allow; a miss stops the work
  before phase 3, not phase 2 results already in hand.

## Phase 2a: 7 configs x {5e-4, 1e-3, 2e-3}, 5K iters, warmup 50 (plan file phase2a.txt)

An LR is unstable for a config if any logged loss is not finite, or its final val loss is more than 0.02 above
the lowest val loss of the same run. 2a ranks nothing else.

Phase 2b pair per config: 1e-3 plus the stable other LR with the lower 2a final val loss. If 1e-3 is unstable,
the two stable LRs. If only one LR is stable, 2b runs that one.

## Phase 2b: 7 configs x their pair, 20K iters, warmup 200, eval every 2000

- LR per config: the one with the lower 20K final val loss, but 1e-3 if it is within 0.01 of the other.
- P-eq split: the lowest 20K final val loss among the three P-eq configs at their chosen LRs.
- Frozen for phase 3: S, D, P-free, the chosen P-eq, A, each at its chosen LR.

Phase 3 does not start until Gary has seen the frozen choices and the measured cost.
