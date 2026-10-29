"""Predeclared synthetic annotation pipeline and leave-one-unit-out comparison."""

from __future__ import annotations

from .geometry import geometry
from .legal_annotation import DIMENSIONS, assess_annotations, prepare_annotation


def loocv_mse(x, y):
    errors = []
    for omitted in range(len(x)):
        indices = [i for i in range(len(x)) if i != omitted]
        mx = sum(x[i] for i in indices) / len(indices)
        my = sum(y[i] for i in indices) / len(indices)
        variance = sum((x[i] - mx) ** 2 for i in indices)
        slope = sum((x[i] - mx) * (y[i] - my) for i in indices) / variance if variance else 0
        errors.append((my + slope * (x[omitted] - mx) - y[omitted]) ** 2)
    return sum(errors) / len(errors)


def run_study(corpus):
    if corpus.get("synthetic") is not True or corpus.get("review_status") != "unreviewed":
        raise ValueError("synthetic study only")
    if len(corpus["study"]["units"]) != 8:
        raise ValueError("the frozen study requires exactly8 units; no silent exclusions")
    packet = prepare_annotation(corpus["study"])
    submissions = []
    for reviewer in range(2):
        annotations = []
        for i, unit in enumerate(packet["units"]):
            labels = dict.fromkeys(DIMENSIONS, "not_applicable")
            labels["faithfulness"] = "unsupported" if reviewer == 1 and i % 2 else "supported"
            annotations.append(
                {
                    "unit_id": unit["id"],
                    "unit_sha256": unit["unit_sha256"],
                    "judgment": {
                        "labels": labels,
                        "rationale": "SIMULATED_NONEXPERT_ORACLE; no person or expert judgment.",
                        "alternative_interpretations": [],
                    },
                }
            )
        submissions.append(
            {
                "reviewer": {
                    "id": "SIMULATED_ORACLE_" + str(reviewer),
                    "qualification": "SIMULATED_NONEXPERT_NOT_A_PERSON",
                    "independent": True,
                    "conflicts_disclosed": False,
                },
                "packet_sha256": packet["packet_sha256"],
                "annotations": annotations,
            }
        )
    assessment = assess_annotations(
        {"schema": "vle-annotation-assessment/v1", "packet": packet, "submissions": submissions}
    )
    rows = []
    for unit in packet["units"]:
        r = geometry(corpus["geometry_inputs"][unit["id"]])
        if r["status"] != "reported":
            raise ValueError("geometry study incomplete; do not silently exclude unit")
        distances = [r["distances"][i][j] for i in range(len(r["ids"])) for j in range(i + 1, len(r["ids"]))]
        rows.append(
            {
                "id": unit["id"],
                "h0": r["persistence"]["largest_finite_H0_lifetime"],
                "alternatives": len(r["ids"]),
                "mean_distance": sum(distances) / len(distances),
                "disagreement": int("faithfulness" in assessment["disagreements"].get(unit["id"], [])),
            }
        )
    mse = {
        key: loocv_mse([r[key] for r in rows], [r["disagreement"] for r in rows])
        for key in ("h0", "alternatives", "mean_distance")
    }
    return {
        "schema": "vle-geometry-study-result/v1",
        "synthetic": True,
        "review_status": "unreviewed",
        "real_reviewers": 0,
        "simulated_only": True,
        "rows": rows,
        "LOOCV_MSE": mse,
        "H3_simulation_supported": mse["h0"] < min(mse["alternatives"], mse["mean_distance"]),
        "real_predictive_validity": "unmeasured",
        "assessment": assessment,
        "decision_authorized": False,
        "legal_certification": False,
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    print(json.dumps(run_study(json.loads(Path("benchmarks/geometry-study-v1.json").read_text())), indent=2))
