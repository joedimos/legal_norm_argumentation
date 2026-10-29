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


def runtime_mutations():
    """Fault injection at actual producer/reasoner call sites; no code changes persist."""
    from unittest.mock import patch

    from . import certificates as producer

    rows = []
    original = producer.verify_request
    with patch.object(producer, "verify_request", return_value={"status": "verified_bounded"}):
        result = producer.certify(
            {"property": {"op": "atom", "name": "p"}, "horizon": 1, "observations": [{"p": None}]}
        )
    rows.append(
        {
            "mutant": "forced_unknown_to_positive",
            "certificate_type": result["certificate"]["type"],
            "affirmative_escape": result["status"] == "compliant_within_bound",
        }
    )

    def shortened(r):
        h = r["horizon"] - 1
        return original(
            {**r, "horizon": h, "observations": r.get("observations", [])[:h] if r.get("observations") else None}
        )

    with patch.object(producer, "verify_request", side_effect=shortened):
        result = producer.certify({"property": {"op": "next", "args": [{"op": "true"}]}, "horizon": 2})
    rows.append(
        {
            "mutant": "shortened_reasoning_horizon",
            "certificate_type": result["certificate"]["type"],
            "affirmative_escape": result["status"] == "violation",
        }
    )
    request = {"property": {"op": "true"}, "horizon": 1, "binding": {"priority": "original"}}
    intended = producer.query_from_request(request)
    wrong = {**intended, "binding": {"priority": "reversed"}}
    with patch.object(producer, "query_from_request", return_value=wrong):
        result = producer.certify(request)
    c = result["certificate"]
    rows.append(
        {
            "mutant": "priority_translation_binding",
            "certificate_type": c["type"],
            "unpinned_escape": check_certificate(c)["valid"],
            "pinned_escape": check_certificate(c, expected_query_sha256=sha(intended))["valid"],
        }
    )
    with patch.object(producer, "verify_request", side_effect=ValueError("synthetic_no_certificate_mutant")):
        try:
            producer.certify(request)
        except ValueError:
            rows.append(
                {
                    "mutant": "no_certificate",
                    "certificate_type": None,
                    "reason": "mutant_emitted_no_verdict_certificate",
                }
            )
    return {
        "schema": "vle-runtime-certificate-mutation/v1",
        "synthetic": True,
        "review_status": "unreviewed",
        "attempts": 4,
        "diagnostic_unknown_certificates": sum(r["certificate_type"] == "unknown" for r in rows),
        "affirmative_certificates": sum(
            r["certificate_type"] in {"violation", "compliant_within_bound", "conflict"} for r in rows
        ),
        "no_certificate": sum(r["certificate_type"] is None for r in rows),
        "rows": rows,
        "affirmative_unpinned_escape_rate": sum(r.get("unpinned_escape", False) for r in rows)
        / max(1, sum("unpinned_escape" in r for r in rows)),
        "affirmative_pinned_escape_rate": sum(r.get("pinned_escape", False) for r in rows)
        / max(1, sum("pinned_escape" in r for r in rows)),
        "decision_authorized": False,
        "legal_certification": False,
    }
