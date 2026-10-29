"""Frozen synthetic certificate-mutation experiment, including trust-boundary escapes."""

from copy import deepcopy

from .certificates import certify
from .checker import check_certificate, sha


def run_study():
    base = certify({"property": {"op": "atom", "name": "p"}, "horizon": 1})["certificate"]
    unknown = deepcopy(base)
    unknown["evidence"]["trace"][0]["p"] = None
    horizon = certify({"property": {"op": "true"}, "horizon": 2})["certificate"]
    horizon["query"]["horizon"] = 1
    horizon["query_sha256"] = sha(horizon["query"])
    priority = certify({"property": {"op": "true"}, "horizon": 1, "binding": {"priority": "reversed"}})["certificate"]
    intended = {**priority["query"], "binding": {"priority": "original"}}
    rows = []
    for name, c, expected in [
        ("unknown_promotion", unknown, base["query_sha256"]),
        ("off_by_one_horizon", horizon, horizon["query_sha256"]),
        ("priority_binding", priority, sha(intended)),
        ("no_certificate", None, None),
    ]:
        produced = c is not None
        rows.append(
            {
                "mutant": name,
                "attempts": 1,
                "certificate_produced": produced,
                "rejected_unpinned": not check_certificate(c)["valid"] if produced else None,
                "rejected_pinned": not check_certificate(c, expected_query_sha256=expected)["valid"]
                if produced
                else None,
            }
        )
    return {
        "schema": "vle-certificate-study/v1",
        "synthetic": True,
        "review_status": "unreviewed",
        "rows": rows,
        "attempts": 4,
        "certificates_produced": 3,
        "no_certificate": 1,
        "unpinned_escape_rate": sum(r["certificate_produced"] and not r["rejected_unpinned"] for r in rows) / 3,
        "pinned_escape_rate": sum(r["certificate_produced"] and not r["rejected_pinned"] for r in rows) / 3,
        "real_reviewer_count": 0,
        "decision_authorized": False,
        "legal_certification": False,
        "limitation": "A self-consistent certificate of the wrong model can escape without an independently trusted query binding.",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_study(), indent=2))
