"""Three-way faithfulness disagreements; synthetic agreement is not legal accuracy."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import UTC, datetime

from .aspic import analyze_aspic, contrary
from .audit_bundle import canonical_json, digest
from .phase4 import BoundHit, Budget, envelope, fields
from .research import run_research

VERDICTS = {
    "compliant",
    "non_compliant",
    "indeterminate",
    "conflict",
    "empty",
    "supported",
    "refuted",
    "undecided",
    "unknown",
}
CORPUS_SCHEMA = {
    "schema": "vle-faithfulness-corpus/v1",
    "max_cases": 128,
    "required_case_fields": [
        "id",
        "family",
        "modality",
        "has_exception",
        "source_text",
        "direct_solver",
        "formalization",
        "formalization_provenance",
        "gold_verdict",
        "synthetic",
        "review_status",
    ],
    "verdicts": sorted(VERDICTS),
    "review_status": "unreviewed",
    "synthetic": True,
}


def symbolic_verdict(formalization):
    if not isinstance(formalization, dict) or set(formalization) - {"kind", "request", "query"}:
        raise ValueError("invalid recorded formalization")
    kind, request = formalization.get("kind"), formalization.get("request")
    if kind == "research":
        if not isinstance(request, dict) or len(request.get("norms", [])) > 8:
            raise ValueError("faithfulness research cases require <=8 norms")
        r = run_research(request)
        if any(f["status"] == "conflicted" for f in r["findings"]):
            return "conflict", r
        if all(f["status"] in {"inactive", "defeated"} for f in r["findings"]):
            return "empty", r
        return r["status"], r
    if kind == "aspic":
        r = analyze_aspic(request)
        query = formalization.get("query")
        if not isinstance(query, str) or not query:
            raise ValueError("ASPIC query is required")
        if r["status"] == "unknown":
            return "unknown", r
        # Evaluate the query using grounded labels, never a chosen preferred extension.
        accepted = {a["conclusion"] for a in r["arguments"] if r["grounded_labels"][a["id"]] == "in"}
        if query in accepted and contrary(query) in accepted:
            return "conflict", r
        if query in accepted:
            return "supported", r
        if contrary(query) in accepted:
            return "refuted", r
        conclusions = {a["conclusion"] for a in r["arguments"]}
        return ("undecided" if query in conclusions or contrary(query) in conclusions else "empty"), r
    return "unknown", {"reason": "unsupported_formalization_fragment"}


def validate_provenance(p):
    if (
        not isinstance(p, dict)
        or not {"model_id", "prompt_sha256", "seed", "params", "timestamp"} <= set(p)
        or not isinstance(p["model_id"], str)
        or not p["model_id"]
        or not isinstance(p["prompt_sha256"], str)
        or len(p["prompt_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in p["prompt_sha256"])
        or type(p["seed"]) is not int
        or not isinstance(p["params"], dict)
    ):
        raise ValueError("invalid model provenance")
    datetime.fromisoformat(p["timestamp"])


def faithfulness_report(corpus):
    fields(
        corpus,
        "vle-faithfulness-corpus/v1",
        {"version", "synthetic", "review_status", "cases", "max_states", "timeout_ms"},
        {"version", "synthetic", "review_status", "cases"},
    )
    if corpus["synthetic"] is not True or corpus["review_status"] != "unreviewed":
        raise ValueError("this version evaluates explicitly synthetic/unreviewed recordings only")
    cases = corpus["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 128 or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("requires 1-128 unique cases")
    budget = Budget(max_states=corpus.get("max_states", 65536), timeout_ms=corpus.get("timeout_ms", 5000))
    rows = []
    reason, status = "complete_recorded_evaluation", "reported"
    try:
        for c in cases:
            budget.tick()
            if (
                not isinstance(c, dict)
                or set(c) != set(CORPUS_SCHEMA["required_case_fields"])
                or c["synthetic"] is not True
                or c["review_status"] != "unreviewed"
                or any(
                    not isinstance(c[k], str) or not c[k].strip() for k in ("id", "family", "modality", "source_text")
                )
                or len(c["source_text"]) > 32000
                or type(c["has_exception"]) is not bool
                or c["gold_verdict"] not in VERDICTS
            ):
                raise ValueError("invalid faithfulness case")
            direct = c["direct_solver"]
            if (
                not isinstance(direct, dict)
                or set(direct) != {"verdict", "provenance"}
                or direct["verdict"] not in VERDICTS
            ):
                raise ValueError("invalid direct-solver recording")
            validate_provenance(direct["provenance"])
            validate_provenance(c["formalization_provenance"])
            symbolic, evidence = symbolic_verdict(c["formalization"])
            budget.tick()
            rows.append(
                {
                    "id": c["id"],
                    "family": c["family"],
                    "modality": c["modality"],
                    "has_exception": c["has_exception"],
                    "direct": direct["verdict"],
                    "symbolic": symbolic,
                    "gold": c["gold_verdict"],
                    "symbolic_evidence": evidence,
                }
            )
    except BoundHit as exc:
        status, reason, rows = "unknown", str(exc), []

    def rates(group):
        result = {
            "rows": len(group),
            "unknown_rows": sum("unknown" in (r["direct"], r["symbolic"], r["gold"]) for r in group),
        }
        for a, b in [("direct", "symbolic"), ("direct", "gold"), ("symbolic", "gold")]:
            determinate = [
                r
                for r in group
                if r[a] not in {"unknown", "indeterminate", "undecided"}
                and r[b] not in {"unknown", "indeterminate", "undecided"}
            ]
            n = sum(r[a] != r[b] for r in group)
            result[a + "_vs_" + b] = {
                "disagreements": n,
                "total": len(group),
                "rate": n / len(group) if group else None,
                "determinate_total": len(determinate),
                "determinate_disagreement_rate": sum(r[a] != r[b] for r in determinate) / len(determinate)
                if determinate
                else None,
            }
        return result

    groups = {}
    for field in ("family", "modality", "has_exception"):
        grouped = defaultdict(list)
        for row in rows:
            grouped[str(row[field])].append(row)
        groups["by_" + field] = {key: rates(group) for key, group in sorted(grouped.items())}
    return envelope(
        "vle-faithfulness-report/v1",
        corpus,
        status=status,
        reason=reason,
        corpus_schema=corpus["schema"],
        corpus_version=corpus["version"],
        corpus_sha256=digest(canonical_json(corpus)),
        corpus_schema_sha256=digest(canonical_json(CORPUS_SCHEMA)),
        synthetic=True,
        case_count=len(cases),
        evaluated_count=len(rows),
        rows=rows,
        disagreement_rates=rates(rows),
        **groups,
        bounds={**budget.to_dict(), "max_cases": 128, "max_norms_per_research_case": 8},
        accuracy_is_not_faithfulness=True,
        scope="synthetic_unreviewed_recorded_regression_not_legal_accuracy",
    )


def live_recordings(corpus, solver, formalizer, *, timestamp=None, max_cases=8):
    """Explicit provider calls; never sends gold labels or recorded answers.

    Returned model drafts are recorded unreviewed, then evaluated as their own
    encodings. Model claims about facts/reviews are not made trusted evidence.
    """
    from .phase4 import integer

    integer(max_cases, 1, 8, "live max_cases")
    if not isinstance(corpus.get("cases"), list) or not 1 <= len(corpus["cases"]) <= max_cases:
        raise ValueError("live input requires 1-8 cases within the explicit cap; no silent sampling")
    if faithfulness_report(corpus)["status"] != "reported":
        raise ValueError("live preflight incomplete; no provider calls made")
    result = deepcopy(corpus)
    timestamp = timestamp or datetime.now(UTC).isoformat()
    for i, c in enumerate(result["cases"]):
        data = {
            "source_text": c["source_text"],
            "family": c["family"],
            "observations": c["formalization"].get("request", {}).get("trace"),
            "query": c["formalization"].get("query"),
        }
        for role, provider in [("direct", solver), ("formalization", formalizer)]:
            prompt = canonical_json({"task": role, **data}).decode()
            provenance = {
                "model_id": provider.model,
                "prompt_sha256": digest(prompt.encode()),
                "seed": i,
                "params": {"temperature": 0},
                "timestamp": timestamp,
            }
            try:
                raw = provider.generate(prompt, seed=i, temperature=0)
                if getattr(provider, "last_request", None) is not None:
                    provenance["prompt_sha256"] = digest(canonical_json(provider.last_request))
                    provenance["generation_request"] = provider.last_request
                if role == "direct":
                    if not isinstance(raw, dict) or set(raw) != {"verdict"} or raw["verdict"] not in VERDICTS:
                        raise ValueError("invalid live verdict")
                    c["direct_solver"] = {"verdict": raw["verdict"], "provenance": provenance}
                else:
                    if not isinstance(raw, dict) or raw.get("kind") != "research":
                        raise ValueError(
                            "live formalization supports typed research norms, not model-generated ASPIC theories"
                        )
                    symbolic_verdict(raw)  # reject malformed model ASTs, never select a candidate
                    c["formalization"], c["formalization_provenance"] = raw, provenance
            except (ValueError, TypeError, KeyError, RecursionError):
                provenance["generation_failed"] = True
                if role == "direct":
                    c["direct_solver"] = {"verdict": "unknown", "provenance": provenance}
                else:
                    c["formalization"] = {"kind": "unsupported", "request": {}}
                    c["formalization_provenance"] = provenance
    return result
