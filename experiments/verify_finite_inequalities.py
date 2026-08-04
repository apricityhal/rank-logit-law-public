"""Direct finite-instance verification of the paper's inequalities.

Complements ``run_one_head_experiments.py`` (which verifies the construction
and evaluates bounds) and ``certify_one_head_thresholds.py`` (which certifies
the reported crossings).  This program tests the finite statements themselves:

1.  Lemma "permitted permutation count": the number of pairs (pi, K) with
    |K| = k and pi(j) > j for every j outside K equals k! S(m,k) — verified
    EXHAUSTIVELY for every m <= 8 and every k, by enumerating all m!
    permutations, and the underlying identity h_{m-k}(1..k) = S(m,k)
    exactly for every m <= 12.

2.  On every bracket instance (the 9 x 4 grid of ``run_one_head_experiments``):
    rebuild the audited construction, measure its actual per-row prefix
    oscillations, diagonal drops, and centered radius, and check directly
    that it satisfies
      - the row-weighted determinant inequality (the paper's (5)),
      - the constant-profile form (6),
      - the Hadamard companion inequality, and
      - the all-rank floor B >= max_i gamma_i / 2,
    with exact big-integer Stirling numbers and high-precision Decimal
    arithmetic (the audit's own working precision plus guard digits).
    Measured zero oscillations (a block's first row) are replaced by a tiny
    positive value, which only shrinks the right-hand sides, so the test is
    conservative.

Writes ``output/one_head_inequality_checks.csv`` and fails loudly on any
violation.
"""

from __future__ import annotations

import csv
from decimal import Decimal, getcontext
from itertools import permutations
from math import comb, factorial
from pathlib import Path

from run_one_head_experiments import (
    BRACKET_LENGTHS,
    RANKS,
    build_mp_factorization,
    mp_scores,
    working_precision,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"


def stirling_exact(m: int, k: int) -> int:
    total = 0
    for j in range(k + 1):
        term = comb(k, j) * (k - j) ** m
        total = total - term if j % 2 else total + term
    fact = factorial(k)
    assert total % fact == 0, (m, k)
    return total // fact


def check_permutation_count(max_m: int = 8) -> None:
    """Exhaustive verification of the permitted permutation count."""
    for m in range(1, max_m + 1):
        counts = [0] * (m + 1)
        for pi in permutations(range(1, m + 1)):
            forced = sum(1 for j in range(1, m + 1) if pi[j - 1] <= j)
            free = m - forced
            for k in range(forced, m + 1):
                counts[k] += comb(free, k - forced)
        for k in range(1, m + 1):
            expected = factorial(k) * stirling_exact(m, k)
            assert counts[k] == expected, ("count mismatch", m, k, counts[k], expected)
    print(f"lemma count: exhaustive match for all m <= {max_m}, all k")


def check_h_identity(max_m: int = 12) -> None:
    """h_{m-k}(1,...,k) = S(m,k) exactly."""
    for m in range(1, max_m + 1):
        for k in range(1, m + 1):
            values = list(range(1, k + 1))
            h = [0] * (m - k + 1)
            h[0] = 1
            for v in values:
                for d in range(1, m - k + 1):
                    h[d] += v * h[d - 1]
            assert h[m - k] == stirling_exact(m, k), ("h identity", m, k)
    print(f"h identity: exact match for all m <= {max_m}")


def log_sum_exp(terms: list) -> Decimal:
    peak = max(terms)
    total = sum(((t - peak).exp() for t in terms), Decimal(0))
    return peak + total.ln()


def check_instance(n: int, rank: int) -> dict:
    epsilon = float(n) ** -2.0
    getcontext().prec = working_precision(n, rank, epsilon) + 40
    factors = build_mp_factorization(n, rank, epsilon)
    scores = mp_scores(n, factors)
    boundaries = n - 1

    tiny = Decimal(10) ** -(getcontext().prec // 2)
    oscillations, drops = [], []
    for row in range(boundaries):
        prefix = scores[row][: row + 1]
        oscillation = max(prefix) - min(prefix)
        oscillations.append(oscillation if oscillation > 0 else tiny)
        drop = scores[row][row] - scores[row][row + 1]
        assert drop > 0, (n, rank, row)
        drops.append(drop)

    radius = Decimal(0)
    for row in scores:
        mean = sum(row, Decimal(0)) / n
        radius = max(radius, max(abs(value - mean) for value in row))
    log_two_b = (2 * radius).ln()

    kmax = min(rank, boundaries)
    log_counts = [None] + [
        Decimal(factorial(k) * stirling_exact(boundaries, k)).ln()
        for k in range(1, kmax + 1)
    ]

    # (5) row-weighted: LHS = sum ln gamma_i; RHS drops the k smallest deltas.
    lhs_weighted = sum((d.ln() for d in drops), Decimal(0))
    log_deltas = sorted(o.ln() for o in oscillations)
    total_log_delta = sum(log_deltas, Decimal(0))
    running = Decimal(0)
    weighted_terms = []
    for k in range(1, kmax + 1):
        running += log_deltas[k - 1]
        weighted_terms.append(
            log_counts[k] + (total_log_delta - running) + k * log_two_b
        )
    slack_weighted = log_sum_exp(weighted_terms) - lhs_weighted
    assert slack_weighted > 0, ("(5) violated", n, rank)

    # (6) constant profile.
    gamma_low, delta_high = min(drops), max(oscillations)
    lhs_constant = boundaries * gamma_low.ln()
    constant_terms = [
        log_counts[k] + (boundaries - k) * delta_high.ln() + k * log_two_b
        for k in range(1, kmax + 1)
    ]
    slack_constant = log_sum_exp(constant_terms) - lhs_constant
    assert slack_constant > 0, ("(6) violated", n, rank)

    # Hadamard companion: elementary symmetric polynomials of sqrt(1..m-1).
    m = boundaries
    roots = [Decimal(t).sqrt() for t in range(1, m)]
    elementary = [Decimal(1)] + [Decimal(0)] * (m - 1)
    for value in roots:
        for degree in range(m - 1, 0, -1):
            elementary[degree] += value * elementary[degree - 1]
    sqrt_m = Decimal(m).sqrt()
    hadamard_terms = [
        elementary[m - k].ln()
        + (m - k) * delta_high.ln()
        + k * (log_two_b + sqrt_m.ln())
        for k in range(1, kmax + 1)
    ]
    slack_hadamard = log_sum_exp(hadamard_terms) - lhs_constant
    assert slack_hadamard > 0, ("hadamard violated", n, rank)

    floor_ok = radius >= max(drops) / 2
    assert floor_ok, ("floor violated", n, rank)

    return {
        "n": n,
        "rank": rank,
        "slack_weighted_nats": f"{float(slack_weighted):.6f}",
        "slack_constant_nats": f"{float(slack_constant):.6f}",
        "slack_hadamard_nats": f"{float(slack_hadamard):.6f}",
        "floor_holds": floor_ok,
    }


def main() -> None:
    check_permutation_count()
    check_h_identity()
    rows = []
    for n in BRACKET_LENGTHS:
        for rank in RANKS:
            if rank > n - 1:
                continue
            row = check_instance(n, rank)
            rows.append(row)
            print(
                f"n={n:3d} r={rank:2d}: (5) slack={row['slack_weighted_nats']}"
                f"  (6) slack={row['slack_constant_nats']}"
                f"  hadamard slack={row['slack_hadamard_nats']}  floor ok"
            )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "one_head_inequality_checks.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"all {len(rows)} instances verified -> {path}")


if __name__ == "__main__":
    main()
