# What a causal mask is worth: a rank-logit law for prefix averaging — artifacts

Complete manuscript source and executable audit. The layout mirrors the paths
referenced in the paper, so every `\path{...}` in the manuscript resolves
against this repository root.

## Contents

- `one_head_paper.tex`, `audited_references.bib`, `one_head_paper.bbl` —
  manuscript source. Build with `latexmk -pdf one_head_paper.tex`
  (Figure 1 reads `experiments/output/one_head_bracket.csv` at compile time;
  needs pgfplots 1.18, natbib, microtype).
- `one_head_paper.pdf` — compiled manuscript.
- `experiments/run_one_head_experiments.py` — offline executable audit
  (standard library only). Regenerates every CSV in `experiments/output/`.
- `experiments/certify_one_head_thresholds.py` — certifies every reported
  representability crossing and the n = 8192 anchor bound with exact
  big-integer Stirling numbers and outward-rounded interval arithmetic
  (requires `mpmath`).
- `experiments/verify_finite_inequalities.py` — exhaustive verification of
  the permitted permutation count through m = 8 and direct checks of
  inequalities (5), (6), the Hadamard companion, and the all-rank floor on
  every bracket instance (requires `mpmath`).
- `experiments/output/` — committed outputs of all three programs, including
  the interval certificates and inequality checks.
- `C17/OneHead.lean` — the machine-checked centered-rank-one,
  constant-profile specialization of Theorem 3 (Lean 4, imports Mathlib
  only; builds in any project with a current Mathlib dependency).

## Reproduce

    python3 experiments/run_one_head_experiments.py
    pip install mpmath
    python3 experiments/certify_one_head_thresholds.py
    python3 experiments/verify_finite_inequalities.py

License: to be added by the author.
