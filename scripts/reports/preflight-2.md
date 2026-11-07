# Preflight 2 (2026-11-07, A40 bak-hpc1, source 208d9e1, the real Job environment)

First run inside the Job image: apt gcc, the venv tarball, the tok512 checksum and the repo tests all worked.
Raw output: preflight-2-check_kernel.txt.

## Result: FAILED, 2 of 54. No training started.

Both failures are rule 3 (whole-gradient error at most 1.10 x S at the same d, T and mode), in bf16 eager:

- P(140,140), d=64, T=512: kernel 5.76e-3, control 5.65e-3, S 5.12e-3, ratio 1.13.
- P(93,93), d=64, T=512: kernel 5.77e-3, control 5.73e-3, S 5.12e-3, ratio 1.13.

In both, the dense control with no Triton is just as far from float64, so the extra error belongs to how
these architectures round in bf16, not to the kernel. In bf16 compiled, the training mode, both pass
(5.52e-3 and 5.51e-3 against S 5.10e-3, ratio 1.08). Rules 1 and 2 pass in all 54 cases.

scripts/preflight.py did not run, because preflight.sh stops at the first failure. It passed on the dev pod
in preflight 1 with the same code.

## Other findings

- The compiler reset works: compiled results now differ from eager in every case, including from
  `D d=288` on, where preflight 1 printed identical numbers.
- The 2.61e+01 for log_eps_s of P(105,70) at T=200 reproduced with the same tokens. That parameter holds
  9.4e-9 of the total gradient norm, so it is the ill-conditioned case, not a kernel fault.
- log_rate, not gated: the largest kernel-specific errors where its share is at least 0.1% are fp32 S
  1.05e-1 (share 1.3e-3), D 8.6e-2, P(93,93) 8.7e-2 in fp32 and 9.1e-2 in bf16 (share 4.4e-3), against
  controls below 4e-2 and mostly below 1e-3. With these tokens it is not one-sided against P: S is worse
  than P(105,70) and P(140,140).

## log_rate diagnosis

Hypothesis: d/dg in _bwd_stats_kernel is sum_t t (q dqf - k dk + ...). dqf and dk are built from bf16 q and k,
but are paired with fp32 q and k in that sum, so the rounding difference survives weighted by absolute
position t instead of cancelling to (t - s). scripts/diag_log_rate.py tests this at operator level, with
scripts/kernel_dgfix.py as a diagnosis-only copy that pairs them consistently. Results: see below.
