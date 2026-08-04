"""Interval certification of the one-head representability crossings.

Recomputes every reported format/length crossing, every ambient-dimension
crossing at n = 8192, and the n = 8192, r = 128 anchor bound, using

  * exact big-integer Stirling numbers S(N,k) (alternating-sum formula,
    exactness asserted by divisibility), so no floating-point rounding
    enters the permutation counts; and
  * outward-rounded interval arithmetic (mpmath's ``iv`` context) for every
    logarithm, exponential, and sum, so each reported decision is certified
    rather than approximated.

Decision logic.  With row profiles from the output lemma at eps = n^-2 and
the weighted determinant inequality, the minimal admissible log B is the
root of  logRHS(x) = logLHS,  where logRHS is nondecreasing in x.  Hence

  admitted  at cap  <=>  logRHS(cap) >= logLHS   (certified with cap's lower
                                                  endpoint and interval
                                                  lower bound of RHS), and
  excluded  at cap  <=>  logRHS(cap) <  logLHS   (certified with cap's upper
                                                  endpoint and interval
                                                  upper bound of RHS).

The all-rank floor log(gamma_1/2) is certified below every admitted cap so
that the combined bound cannot flip an admitted decision.

Writes ``output/one_head_interval_certificates.csv`` and fails loudly if any
reported decision is not certified.
"""

from __future__ import annotations

import csv
from math import comb, factorial
from pathlib import Path

from mpmath import iv

iv.dps = 60

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"

LOG2 = iv.log(iv.mpf(2))

FORMAT_SPECS = (
    ("float16", (2 ** 15) * (2 ** 11 - 1) // 2 ** 10),      # 65504, exact
    ("bfloat16", (2 ** 127) * (2 ** 8 - 1) // 2 ** 7),       # exact integer
    ("float32", (2 ** 127) * (2 ** 24 - 1) // 2 ** 23),      # exact integer
)
LENGTH_CROSSINGS = {
    "float16": {8: (65, 66), 32: (262, 263), 64: (531, 532), 128: (1075, 1076)},
    "bfloat16": {8: (268, 269), 32: (1053, 1054), 64: (2093, 2094), 128: (4165, 4166)},
    "float32": {8: (268, 269), 32: (1053, 1054), 64: (2093, 2094), 128: (4165, 4166)},
}
AMBIENT_CROSSINGS = {"float16": (934, 935), "bfloat16": (252, 253), "float32": (252, 253)}
ANCHOR = (8192, 128, "418.72")


def ilog(value: int):
    """Rigorous interval enclosure of log(value) for a positive integer."""
    if value <= 0:
        raise ValueError(value)
    bits = value.bit_length()
    if bits <= 50:
        return iv.log(iv.mpf(value))
    shift = bits - 50
    mantissa = value >> shift              # value in [mantissa 2^s, (mantissa+1) 2^s)
    bracket = iv.mpf([mantissa, mantissa + 1])
    return iv.log(bracket) + shift * LOG2


def stirling_table(n_tokens: int, max_k: int) -> list:
    """log S(N,k) + log k! as intervals, k = 1..max_k, N = n_tokens - 1, exact core."""
    boundaries = n_tokens - 1
    table = [None]
    for k in range(1, min(max_k, boundaries) + 1):
        total = 0
        for j in range(k + 1):
            term = comb(k, j) * (k - j) ** boundaries
            total = total - term if j % 2 else total + term
        fact = factorial(k)
        if total % fact:
            raise AssertionError(("stirling divisibility", n_tokens, k))
        stirling = total // fact
        table.append(ilog(stirling) + ilog(fact))
    return table


def profile_data(n_tokens: int, max_k: int):
    """logLHS, suffix sums of log delta, and the all-rank floor, as intervals.

    eps = n^-2 exactly; delta_i = log((n^2+i)/(n^2-i)); gamma_i = log((n^2-i)/i).
    """
    boundaries = n_tokens - 1
    square = n_tokens * n_tokens
    log_target = iv.mpf(0)
    log_deltas = [None] * (boundaries + 1)
    for i in range(1, boundaries + 1):
        gamma = ilog(square - i) - ilog(i)
        log_target += iv.log(gamma)
        delta = ilog(square + i) - ilog(square - i)
        log_deltas[i] = iv.log(delta)
    kmax = min(max_k, boundaries)
    suffix = [iv.mpf(0)] * (kmax + 1)
    running = iv.mpf(0)
    for i in range(boundaries, kmax, -1):
        running += log_deltas[i]
    suffix[kmax] = running
    for k in range(kmax - 1, -1, -1):
        suffix[k] = suffix[k + 1] + log_deltas[k + 1]
    floor = iv.log((ilog(square - 1) - ilog(1)) / 2)
    return log_target, suffix, floor


def log_rhs(counts, suffix, kmax: int, x):
    """Interval enclosure of log sum_k k! S(N,k) (2 e^x)^k max_R prod delta."""
    terms = [counts[k] + suffix[k] + k * (LOG2 + x) for k in range(1, kmax + 1)]
    peak = max(t.b for t in terms)
    total = iv.mpf(0)
    for t in terms:
        total += iv.exp(t - peak)
    return peak + iv.log(total)


def cap_interval(dimension: int, finite_maximum: int):
    return LOG2 + ilog(dimension) + 2 * ilog(finite_maximum)


def certify(n_tokens, rank, cap, expect_admitted, shared=None):
    kmax = min(rank, n_tokens - 1)
    counts = shared["counts"] if shared else stirling_table(n_tokens, kmax)
    if shared:
        target, suffix_full, floor = shared["profile"]
        suffix = suffix_full
    else:
        target, suffix, floor = profile_data(n_tokens, kmax)
    if expect_admitted:
        value = log_rhs(counts, suffix, kmax, iv.mpf(cap.a))
        ok = value.a >= target.b and floor.b <= cap.a
        margin = float(value.a - target.b)
    else:
        value = log_rhs(counts, suffix, kmax, iv.mpf(cap.b))
        ok = value.b < target.a
        margin = float(target.a - value.b)
    return ok, margin


def main() -> None:
    rows = []

    def record(kind, fmt, dim, n_tokens, decision, ok, margin):
        rows.append({
            "kind": kind, "format": fmt, "ambient_dimension": dim, "n": n_tokens,
            "decision": decision, "certified": ok,
            "certified_log_margin_nats": f"{margin:.6e}",
        })
        flag = "OK " if ok else "FAIL"
        print(f"[{flag}] {kind} {fmt} d={dim} n={n_tokens} {decision} margin={margin:.3e}")
        if not ok:
            raise AssertionError((kind, fmt, dim, n_tokens, decision))

    for fmt, fmax in FORMAT_SPECS:
        for dim, (admitted_n, excluded_n) in LENGTH_CROSSINGS[fmt].items():
            cap = cap_interval(dim, fmax)
            ok, margin = certify(admitted_n, dim, cap, True)
            record("format_length", fmt, dim, admitted_n, "admitted", ok, margin)
            ok, margin = certify(excluded_n, dim, cap, False)
            record("format_length", fmt, dim, excluded_n, "excluded", ok, margin)

    # n = 8192 work shares one exact Stirling table up to the largest rank.
    top_rank = max(max(AMBIENT_CROSSINGS[f]) for f, _ in FORMAT_SPECS)
    counts = stirling_table(8192, top_rank)
    profile = profile_data(8192, top_rank)

    def shared_for(rank):
        kmax = min(rank, 8191)
        target, suffix, floor = profile
        return {"counts": counts[: kmax + 1],
                "profile": (target, suffix[: kmax + 1], floor)}

    for fmt, fmax in FORMAT_SPECS:
        excluded_d, admitted_d = AMBIENT_CROSSINGS[fmt]
        ok, margin = certify(8192, excluded_d, cap_interval(excluded_d, fmax),
                             False, shared_for(excluded_d))
        record("ambient_dimension", fmt, excluded_d, 8192, "excluded", ok, margin)
        ok, margin = certify(8192, admitted_d, cap_interval(admitted_d, fmax),
                             True, shared_for(admitted_d))
        record("ambient_dimension", fmt, admitted_d, 8192, "admitted", ok, margin)

    anchor_n, anchor_rank, anchor_bound = ANCHOR
    shared = shared_for(anchor_rank)
    bound = iv.mpf(anchor_bound)
    value = log_rhs(shared["counts"], shared["profile"][1], anchor_rank, iv.mpf(bound.b))
    ok = value.b < shared["profile"][0].a
    margin = float(shared["profile"][0].a - value.b)
    record("anchor", "-", anchor_rank, anchor_n, f"logB>={anchor_bound}", ok, margin)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "one_head_interval_certificates.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"all {len(rows)} decisions certified -> {path}")


if __name__ == "__main__":
    main()
