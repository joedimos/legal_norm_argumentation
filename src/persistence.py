"""Small Vietoris–Rips H0/H1 over F2, including triangle boundaries."""

from __future__ import annotations

from itertools import combinations


def rips(counts, denominator, budget):
    n = len(counts)
    if n > 16:
        raise ValueError("Rips reduction is limited to16 points")
    simplices = [(0, 0, (i,)) for i in range(n)]
    simplices += [(counts[i][j], 1, (i, j)) for i, j in combinations(range(n), 2)]
    simplices += [
        (max(counts[i][j], counts[i][k], counts[j][k]), 2, (i, j, k)) for i, j, k in combinations(range(n), 3)
    ]
    simplices.sort()
    indices = {s[2]: i for i, s in enumerate(simplices)}
    columns = {}
    births = []
    deaths = {}
    for j, (_scale, dim, vertices) in enumerate(simplices):
        budget.tick()
        column = {indices[face] for face in combinations(vertices, len(vertices) - 1)} if dim else set()
        while column and max(column) in columns:
            budget.tick()
            column ^= columns[max(column)]
        if column:
            pivot = max(column)
            columns[pivot] = column
            deaths[pivot] = j
        else:
            births.append(j)
    bars = []
    for i in births:
        scale, dim, vertices = simplices[i]
        if dim > 1:
            continue
        end = simplices[deaths[i]][0] if i in deaths else None
        bars.append(
            {
                "dimension": dim,
                "birth": scale / denominator,
                "death": end / denominator if end is not None else None,
                "right_censored": end is None,
                "lifetime": (end - scale) / denominator if end is not None else None,
            }
        )
    return {
        "coefficient_field": "F2",
        "max_simplex_dimension": 2,
        "bars": bars,
        "largest_finite_H0_lifetime": max(
            (b["lifetime"] for b in bars if b["dimension"] == 0 and b["lifetime"] is not None), default=0
        ),
        "legal_interpretation_claimed": False,
    }
