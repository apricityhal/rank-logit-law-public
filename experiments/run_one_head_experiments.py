"""Reproducible numerical checks for the one-head rank--logit law.

The script produces three theorem-aligned data sets:

1. ``one_head_bracket.csv`` compares the finite row-weighted determinant
   lower bound, the realized cost of the explicit rank-r construction, and
   its proved closed-form upper budget.
2. ``one_head_anchor.csv`` evaluates the finite determinant bound at one
   realistic sequence-length/head-rank scale without substituting an
   uncontrolled asymptotic remainder.
3. ``one_head_thresholds.csv`` records both sides of every finite-format
   crossing quoted in the paper.

The script also writes ``environment.txt`` with the interpreter, platform, and
decimal settings used for the audit.

The code is offline and has no data-download or application-specific path.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from decimal import Decimal, Subnormal, Underflow, getcontext
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
P = 2.0
BRACKET_LENGTHS = (12, 16, 20, 24, 32, 40, 48, 56, 64)
RANKS = (1, 2, 4, 8)
ANCHOR_N = 8192
ANCHOR_RANK = 128
ANCHOR_P = 2.0
FORMAT_SPECS = (
    ("float16", (2.0 - 2.0**-10) * 2.0**15),
    ("bfloat16", (2.0 - 2.0**-7) * 2.0**127),
    ("float32", (2.0 - 2.0**-23) * 2.0**127),
)
FORMAT_DIMENSIONS = (8, 32, 64, 128)


@dataclass
class RankOneBlock:
    start: int
    stop: int
    queries: list[Decimal]
    keys: list[Decimal]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def row_budgets(n: int, epsilon: float) -> tuple[list[float], list[float], list[float]]:
    deltas: list[float] = []
    gammas: list[float] = []
    weights: list[float] = []
    for i in range(1, n):
        delta = i * epsilon / 16.0
        gamma = math.log(8.0 * (n - i) / (i * epsilon))
        deltas.append(delta)
        gammas.append(gamma)
        weights.append(math.log1p(gamma / delta))
    return deltas, gammas, weights


def weighted_blocks(weights: list[float], count: int) -> list[tuple[int, int]]:
    """Proof construction's greedy partition, using half-open row blocks."""
    if not 1 <= count <= len(weights):
        raise ValueError((count, len(weights)))
    capacity = math.fsum(weights) / count
    blocks: list[tuple[int, int]] = []
    start = 0
    load = 0.0
    for position in range(1, len(weights)):
        if load + weights[position] <= capacity:
            load += weights[position]
        else:
            blocks.append((start, position))
            start = position
            load = 0.0
    blocks.append((start, len(weights)))
    while len(blocks) < count:
        index = next(
            i for i, (left, right) in enumerate(blocks) if right - left >= 2
        )
        left, right = blocks[index]
        blocks[index : index + 1] = [(left, left + 1), (left + 1, right)]
    if len(blocks) != count:
        raise AssertionError((blocks, count))
    return blocks


def upper_log_budget(n: int, rank: int, epsilon: float) -> float:
    n_boundaries = n - 1
    log_scale = math.log(8.0 * n / epsilon)
    return (
        (n_boundaries / rank)
        * math.log(64.0 * math.e * log_scale / (n * epsilon))
        + math.log(log_scale)
    )


def working_precision(n: int, rank: int, epsilon: float) -> int:
    decimal_exponent = upper_log_budget(n, rank, epsilon) / math.log(10.0)
    return max(100, math.ceil(decimal_exponent) + 80)


def build_mp_factorization(n: int, rank: int, epsilon: float) -> list[RankOneBlock]:
    deltas_float, gammas_float, weights = row_budgets(n, epsilon)
    blocks = weighted_blocks(weights, rank)
    factors: list[RankOneBlock] = []
    for start, stop in blocks:
        scales: list[Decimal] = []
        cumulative = Decimal(0)
        for offset, row in enumerate(range(start, stop)):
            delta = Decimal(str(deltas_float[row]))
            gamma = Decimal(str(gammas_float[row]))
            scale = Decimal(1) if offset == 0 else cumulative / delta
            scales.append(scale)
            cumulative += gamma * scale

        key_profile: list[Decimal] = []
        cumulative = Decimal(0)
        for key in range(n):
            diagonal = key - 1
            if start <= diagonal < stop:
                offset = diagonal - start
                cumulative += Decimal(str(gammas_float[diagonal])) * scales[offset]
            key_profile.append(-cumulative)
        factors.append(
            RankOneBlock(
                start=start,
                stop=stop,
                queries=[Decimal(1) / scale for scale in scales],
                keys=key_profile,
            )
        )
    return factors


def mp_scores(n: int, factors: list[RankOneBlock]) -> list[list[Decimal]]:
    scores = [[Decimal(0) for _ in range(n)] for _ in range(n)]
    for factor in factors:
        for offset, row in enumerate(range(factor.start, factor.stop)):
            query = factor.queries[offset]
            scores[row] = [query * key for key in factor.keys]
    return scores


# Log-weights below -LOG_WEIGHT_FLOOR are dropped from the softmax sum rather
# than exponentiated.  Decimal's exponent range (Emin ~ -1e6 here) is far too
# narrow for this construction: at n=64, r=1 the smallest score gap is around
# -1e275, so a direct ``.exp()`` returns exact zero and sets the Underflow,
# Subnormal and Clamped signals -- which are not trapped by default, so the
# loss is silent.  Raising Emin cannot fix this: CPython caps it near -1e18.
# We therefore truncate explicitly and bound the discarded mass.
LOG_WEIGHT_FLOOR = Decimal(10) ** 5


def mp_operator_error(scores: list[list[Decimal]]) -> tuple[Decimal, Decimal]:
    """Worst-row l1 error of the finite-score softmax, plus a truncation slack.

    Returns ``(error, slack)`` where ``slack`` bounds the effect of the
    truncation alone; ordinary rounding at the working decimal precision is a
    separate and much larger (though still negligible) effect.  Writing
    ``l_j`` for the log-weights
    shifted by the row maximum, terms with ``l_j < -LOG_WEIGHT_FLOOR`` are
    omitted.  Since the shifted total is at least 1, the omitted probability
    mass is at most ``n*exp(-LOG_WEIGHT_FLOOR)``, and renormalizing the kept
    terms perturbs the row distribution by at most the same amount again; the
    row-l1 error therefore moves by at most ``2n*exp(-LOG_WEIGHT_FLOOR)``.
    """
    n = len(scores)
    worst = Decimal(0)
    tiny = (-LOG_WEIGHT_FLOOR).exp()
    slack = 2 * n * tiny
    for row_index, row in enumerate(scores):
        row_max = max(row)
        kept: list[tuple[int, Decimal]] = []
        for key, value in enumerate(row):
            shifted = value - row_max
            if shifted >= -LOG_WEIGHT_FLOOR:
                weight = shifted.exp()
                if weight == 0:
                    raise AssertionError((row_index, key, shifted))
                kept.append((key, weight))
        total = sum((weight for _, weight in kept), Decimal(0))
        prefix_size = row_index + 1
        target = Decimal(1) / prefix_size
        probabilities = {key: weight / total for key, weight in kept}
        error = sum(
            (
                abs(probabilities.get(key, Decimal(0))
                    - (target if key < prefix_size else 0))
                for key in range(n)
            ),
            Decimal(0),
        )
        worst = max(worst, error)
    return worst, slack


def mp_log_centered_radius(scores: list[list[Decimal]]) -> Decimal:
    n = len(scores)
    radius = Decimal(0)
    for row in scores:
        mean = sum(row, Decimal(0)) / n
        radius = max(radius, max(abs(value - mean) for value in row))
    if radius <= 0:
        raise AssertionError(radius)
    return radius.ln()


def reference_construction(
    n: int,
    rank: int,
    epsilon: float,
) -> tuple[list[RankOneBlock], float, float]:
    getcontext().prec = working_precision(n, rank, epsilon)
    getcontext().clear_flags()
    factors = build_mp_factorization(n, rank, epsilon)
    scores = mp_scores(n, factors)
    error, slack = mp_operator_error(scores)
    # The truncation slack must be irrelevant at the reported precision.
    if slack >= Decimal(str(epsilon)) / Decimal(10) ** 6:
        raise AssertionError((n, rank, float(slack)))
    if getcontext().flags[Underflow] or getcontext().flags[Subnormal]:
        raise AssertionError(("decimal underflow", n, rank))
    log_radius = mp_log_centered_radius(scores)
    return factors, float(error), float(log_radius)


def logsumexp(values: Iterable[float]) -> float:
    values = list(values)
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def log_stirling_second_row(n: int, max_k: int) -> list[float]:
    """Return log Stirling numbers S(n,k), 0 <= k <= max_k."""
    negative_infinity = -math.inf
    row = [negative_infinity] * (max_k + 1)
    row[0] = 0.0
    for current_n in range(1, n + 1):
        next_row = [negative_infinity] * (max_k + 1)
        for current_k in range(1, min(max_k, current_n) + 1):
            terms = [row[current_k - 1]]
            if row[current_k] != negative_infinity:
                terms.append(math.log(current_k) + row[current_k])
            next_row[current_k] = logsumexp(terms)
        row = next_row
    return row


def solve_log_polynomial_lower(
    log_coefficients: list[tuple[int, float]],
    target: float,
    initial_high: float,
) -> float:
    """Solve sum_k exp(c_k) (2B)^k = exp(target) for log B."""

    def residual(log_b: float) -> float:
        return logsumexp(
            constant + k * (math.log(2.0) + log_b)
            for k, constant in log_coefficients
        ) - target

    low = -1000.0
    high = max(1.0, initial_high)
    while residual(low) >= 0:
        low -= 1000.0
    while residual(high) < 0:
        high *= 2.0
    for _ in range(160):
        middle = (low + high) / 2.0
        if residual(middle) >= 0:
            high = middle
        else:
            low = middle
    return high


def finite_determinant_log_lowers(
    n: int,
    rank: int,
    epsilon: float,
) -> dict[str, float]:
    """Evaluate the uniform and row-weighted finite determinant bounds."""
    boundaries = n - 1
    max_k = min(rank, boundaries)
    deltas = [
        math.log((1.0 + i * epsilon) / (1.0 - i * epsilon))
        for i in range(1, boundaries + 1)
    ]
    gammas = [
        math.log((1.0 - i * epsilon) / (i * epsilon))
        for i in range(1, boundaries + 1)
    ]
    if not min(gammas) > max(deltas) > 0:
        raise ValueError((n, rank, epsilon, min(gammas), max(deltas)))

    log_counts = log_stirling_second_row(boundaries, max_k)
    uniform_delta = deltas[-1]
    uniform_gamma = gammas[-1]
    uniform_coefficients = []
    for k in range(1, max_k + 1):
        uniform_coefficients.append(
            (
                k,
                math.lgamma(k + 1)
                + log_counts[k]
                + (boundaries - k) * math.log(uniform_delta),
            )
        )
    initial_high = upper_log_budget(n, rank, epsilon)
    uniform_root = solve_log_polynomial_lower(
        uniform_coefficients,
        boundaries * math.log(uniform_gamma),
        initial_high,
    )

    log_deltas = sorted(math.log(delta) for delta in deltas)
    total_log_delta = math.fsum(log_deltas)
    omitted_log_delta = 0.0
    weighted_coefficients = []
    for k in range(1, max_k + 1):
        # Deltas increase with the row index.  The maximum product left after
        # omitting k rows therefore omits the k smallest deltas.
        omitted_log_delta += log_deltas[k - 1]
        weighted_coefficients.append(
            (
                k,
                math.lgamma(k + 1)
                + log_counts[k]
                + total_log_delta
                - omitted_log_delta,
            )
        )
    weighted_root = solve_log_polynomial_lower(
        weighted_coefficients,
        math.fsum(math.log(gamma) for gamma in gammas),
        initial_high,
    )
    log_floor = math.log(max(gammas) / 2.0)
    return {
        "uniform_root": uniform_root,
        "uniform_combined": max(uniform_root, log_floor),
        "weighted_root": weighted_root,
        "weighted_combined": max(weighted_root, log_floor),
        "floor": log_floor,
    }


def normalized_remainder(n: int, rank: int, log_radius: float, p: float = P) -> float:
    leading_scale = (p - 1.0) * math.log(n) + math.log(math.log(n))
    return rank * log_radius / (n - 1) - leading_scale


def bracket_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n in BRACKET_LENGTHS:
        epsilon = n ** (-P)
        for rank in RANKS:
            if rank > n - 1:
                continue
            _, reference_error, log_constructed = reference_construction(n, rank, epsilon)
            lowers = finite_determinant_log_lowers(n, rank, epsilon)
            log_lower = lowers["weighted_combined"]
            log_upper = upper_log_budget(n, rank, epsilon)
            if reference_error > epsilon * (1.0 + 1e-10):
                raise AssertionError(("reference error", n, rank, reference_error, epsilon))
            if log_lower > log_constructed + 1e-9:
                raise AssertionError(("lower exceeds construction", n, rank, log_lower, log_constructed))
            if log_constructed > log_upper + 1e-9:
                raise AssertionError(("construction exceeds budget", n, rank, log_constructed, log_upper))
            rows.append(
                {
                    "n": n,
                    "rank": rank,
                    "p": P,
                    "epsilon": epsilon,
                    "reference_operator_error": reference_error,
                    "error_over_epsilon": reference_error / epsilon,
                    "log_B_lower_weighted": log_lower,
                    "log_B_lower_uniform": lowers["uniform_combined"],
                    "log_B_floor": lowers["floor"],
                    "log_B_constructed": log_constructed,
                    "log_B_upper_budget": log_upper,
                    "lower_remainder": normalized_remainder(n, rank, log_lower),
                    "constructed_remainder": normalized_remainder(n, rank, log_constructed),
                    "upper_budget_remainder": normalized_remainder(n, rank, log_upper),
                }
            )
    return rows


def anchor_rows() -> list[dict[str, object]]:
    epsilon = ANCHOR_N ** (-ANCHOR_P)
    lowers = finite_determinant_log_lowers(ANCHOR_N, ANCHOR_RANK, epsilon)
    return [
        {
            "n": ANCHOR_N,
            "rank": ANCHOR_RANK,
            "p": ANCHOR_P,
            "epsilon": epsilon,
            "log_B_lower_weighted": lowers["weighted_combined"],
            "log_B_lower_uniform": lowers["uniform_combined"],
            "log_B_floor": lowers["floor"],
        }
    ]


def factor_log_ceiling(dimension: int, finite_maximum: float) -> float:
    return math.log(2.0) + math.log(dimension) + 2.0 * math.log(finite_maximum)


def finite_bound_admits(
    n: int,
    dimension: int,
    p: float,
    finite_maximum: float,
) -> tuple[bool, float, float]:
    epsilon = n ** (-p)
    log_lower = finite_determinant_log_lowers(n, dimension, epsilon)[
        "weighted_combined"
    ]
    ceiling = factor_log_ceiling(dimension, finite_maximum)
    return log_lower <= ceiling, log_lower, ceiling


def length_crossing(
    dimension: int,
    p: float,
    finite_maximum: float,
) -> tuple[int, float, float, int, float, float]:
    admitted_n = 2
    is_admitted, admitted_lower, ceiling = finite_bound_admits(
        admitted_n, dimension, p, finite_maximum
    )
    if not is_admitted:
        raise AssertionError((dimension, p, finite_maximum, admitted_n))

    excluded_n = 4
    while True:
        is_admitted, excluded_lower, _ = finite_bound_admits(
            excluded_n, dimension, p, finite_maximum
        )
        if not is_admitted:
            break
        admitted_n = excluded_n
        admitted_lower = excluded_lower
        excluded_n *= 2

    while excluded_n - admitted_n > 1:
        middle = (admitted_n + excluded_n) // 2
        is_admitted, log_lower, _ = finite_bound_admits(
            middle, dimension, p, finite_maximum
        )
        if is_admitted:
            admitted_n = middle
            admitted_lower = log_lower
        else:
            excluded_n = middle
            excluded_lower = log_lower
    return (
        admitted_n,
        admitted_lower,
        ceiling,
        excluded_n,
        excluded_lower,
        ceiling,
    )


def dimension_crossing(
    n: int,
    p: float,
    finite_maximum: float,
) -> tuple[int, float, float, int, float, float]:
    excluded_dimension = 1
    is_admitted, excluded_lower, excluded_ceiling = finite_bound_admits(
        n, excluded_dimension, p, finite_maximum
    )
    if is_admitted:
        raise AssertionError((n, p, finite_maximum, excluded_dimension))

    admitted_dimension = 2
    while True:
        is_admitted, admitted_lower, admitted_ceiling = finite_bound_admits(
            n, admitted_dimension, p, finite_maximum
        )
        if is_admitted:
            break
        excluded_dimension = admitted_dimension
        excluded_lower = admitted_lower
        excluded_ceiling = admitted_ceiling
        admitted_dimension *= 2

    while admitted_dimension - excluded_dimension > 1:
        middle = (excluded_dimension + admitted_dimension) // 2
        is_admitted, log_lower, ceiling = finite_bound_admits(
            n, middle, p, finite_maximum
        )
        if is_admitted:
            admitted_dimension = middle
            admitted_lower = log_lower
            admitted_ceiling = ceiling
        else:
            excluded_dimension = middle
            excluded_lower = log_lower
            excluded_ceiling = ceiling
    return (
        excluded_dimension,
        excluded_lower,
        excluded_ceiling,
        admitted_dimension,
        admitted_lower,
        admitted_ceiling,
    )


def empty_threshold_row() -> dict[str, object]:
    return {
        "kind": "",
        "format": "",
        "log_F": "",
        "n": "",
        "p": "",
        "ambient_dimension": "",
        "factor_log_ceiling": "",
        "last_admitted_n": "",
        "last_admitted_log_lower": "",
        "last_admitted_slack": "",
        "first_excluded_n": "",
        "first_excluded_log_lower": "",
        "first_excluded_overshoot": "",
        "last_excluded_dimension": "",
        "last_excluded_log_lower": "",
        "last_excluded_margin": "",
        "first_admitted_dimension": "",
        "first_admitted_log_lower": "",
        "first_admitted_slack": "",
    }


def threshold_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for format_name, finite_maximum in FORMAT_SPECS:
        for dimension in FORMAT_DIMENSIONS:
            (
                admitted_n,
                admitted_lower,
                ceiling,
                excluded_n,
                excluded_lower,
                _,
            ) = length_crossing(dimension, P, finite_maximum)
            row = empty_threshold_row()
            row.update(
                {
                    "kind": "format_length",
                    "format": format_name,
                    "log_F": math.log(finite_maximum),
                    "p": P,
                    "ambient_dimension": dimension,
                    "factor_log_ceiling": ceiling,
                    "last_admitted_n": admitted_n,
                    "last_admitted_log_lower": admitted_lower,
                    "last_admitted_slack": ceiling - admitted_lower,
                    "first_excluded_n": excluded_n,
                    "first_excluded_log_lower": excluded_lower,
                    "first_excluded_overshoot": excluded_lower - ceiling,
                }
            )
            rows.append(row)

        (
            excluded_dimension,
            excluded_lower,
            excluded_ceiling,
            admitted_dimension,
            admitted_lower,
            admitted_ceiling,
        ) = dimension_crossing(ANCHOR_N, ANCHOR_P, finite_maximum)
        row = empty_threshold_row()
        row.update(
            {
                "kind": "ambient_dimension",
                "format": format_name,
                "log_F": math.log(finite_maximum),
                "n": ANCHOR_N,
                "p": ANCHOR_P,
                "last_excluded_dimension": excluded_dimension,
                "last_excluded_log_lower": excluded_lower,
                "last_excluded_margin": excluded_lower - excluded_ceiling,
                "first_admitted_dimension": admitted_dimension,
                "first_admitted_log_lower": admitted_lower,
                "first_admitted_slack": admitted_ceiling - admitted_lower,
            }
        )
        rows.append(row)

    return rows


def summarize(
    bracket: list[dict[str, object]],
    anchor: list[dict[str, object]],
    thresholds: list[dict[str, object]],
) -> None:
    constructed = [float(row["constructed_remainder"]) for row in bracket]
    print(
        "bracket rows=",
        len(bracket),
        "constructed remainder range=",
        (min(constructed), max(constructed)),
    )
    print("finite anchor=", anchor[0])
    print("threshold rows=", len(thresholds))


def write_environment(path: Path) -> None:
    """Record the interpreter and platform used for the audit."""
    import platform
    import sys

    lines = [
        f"python           {sys.version.split()[0]} ({platform.python_implementation()})",
        f"platform         {platform.platform()}",
        f"machine          {platform.machine()}",
        f"processor        {platform.processor() or 'unknown'}",
        f"decimal Emin     {getcontext().Emin}",
        f"log weight floor {LOG_WEIGHT_FLOOR}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    bracket = bracket_rows()
    anchor = anchor_rows()
    thresholds = threshold_rows()
    write_csv(OUTPUT / "one_head_bracket.csv", bracket)
    write_csv(OUTPUT / "one_head_anchor.csv", anchor)
    write_csv(OUTPUT / "one_head_thresholds.csv", thresholds)
    write_environment(OUTPUT / "environment.txt")
    summarize(bracket, anchor, thresholds)


if __name__ == "__main__":
    main()
