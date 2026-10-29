"""Controlled source transformations and bounded, disagreement-preserving testing."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import replace
from datetime import UTC, datetime

from .audit_bundle import canonical_json, digest
from .experiments import run_experiment
from .formalization import candidate_to_interpretations
from .norm_codec import decode_norm
from .norm_extraction import extract_norms
from .normative_model import Predicate
from .phase4 import BoundHit, Budget, envelope, fields, integer
from .redundancy.crosscheck import consensus_report
from .redundancy.generate import FakeProvider, RuntimeProvider
from .temporal_deontic import TemporalNorm

EQUIVALENT = {
    "voice",
    "condition_order",
    "exception_synonym",
    "modality_synonym",
    "double_negation",
    "list_order",
    "parallel",
    "exception_position",
}
EXPECTED = {
    **dict.fromkeys(EQUIVALENT, "equivalent_bounded"),
    "weaken_modality": "strict_weakening",
    "add_exception": "strict_weakening",
    "remove_condition": "strict_strengthening",
    "invert_modality": "incomparable",
}


def transform(text, relation, case):
    """Return only mechanically applicable proposals; no parser/LLM rewrites."""
    if relation not in EXPECTED:
        raise ValueError("unsupported metamorphic relation")
    if relation == "parallel":
        return [case["parallel"]] if text == case["text"] and case.get("parallel") else []
    if relation == "voice":
        return ["The report must be filed by the operator."] if text == "The operator must file the report." else []
    if relation == "condition_order":
        m = re.fullmatch(r"If (.+?), (.+?)\.", text)
        return [m[2] + " if " + m[1] + "."] if m else []
    if relation == "exception_position":
        m = re.fullmatch(r"(.+?) unless (.+?)\.", text)
        return ["Unless " + m[2] + ", " + m[1] + "."] if m else []
    if relation == "exception_synonym":
        return (
            [text.replace(" unless ", " except if "), text.replace(" unless ", " save where ")]
            if " unless " in text
            else []
        )
    if relation == "modality_synonym":
        for old, new in [(" must ", " shall "), (" doit ", " est tenu de ")]:
            if old in text:
                return [text.replace(old, new, 1)]
        return []
    if relation == "double_negation":
        return (
            [text.replace(" must ", " must not not ", 1)]
            if " must " in text and not re.search(r"\bmust\s+not\b", text)
            else []
        )
    if relation == "list_order":
        m = re.fullmatch(r"If (\w+) and (\w+), (.+)", text)
        return [f"If {m[2]} and {m[1]}, {m[3]}"] if m else []
    if relation == "weaken_modality":
        return (
            [text.replace(" must ", " may ", 1)] if " must " in text and not re.search(r"\bmust\s+not\b", text) else []
        )
    if relation == "add_exception":
        return (
            [text.rstrip(".") + " unless waived."]
            if "unless" not in text and ("must " in text or "shall " in text)
            else []
        )
    if relation == "remove_condition":
        m = re.fullmatch(r"If .+?, (.+)", text)
        return [m[1]] if m else []
    if relation == "invert_modality":
        return (
            [text.replace(" must ", " must not ", 1)]
            if " must " in text and not re.search(r"\bmust\s+not\b", text)
            else []
        )
    return []


def extract_one(text):
    candidates = extract_norms(text, source_id="synthetic-mr")
    if len(candidates) != 1:
        raise ValueError("one complete norm required")
    return candidate_to_interpretations(candidates[0]).interpretations[0]


def normalize(norm, case, id):
    aliases = case.get("aliases", {})
    if not isinstance(aliases, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in aliases.items()
    ):
        raise ValueError("aliases must be explicit string mappings")
    return replace(
        norm,
        id=id,
        source_norm_id=case["id"],
        subject=aliases.get(norm.subject, norm.subject),
        action=aliases.get(norm.action, norm.action),
        conditions=tuple(Predicate(aliases.get(p.name, p.name), p.negated) for p in norm.conditions),
        exceptions=tuple(Predicate(aliases.get(p.name, p.name), p.negated) for p in norm.exceptions),
    )


def metamorphic(request, *, provider="extractor", runtime=None):
    fields(
        request,
        "vle-metamorphic-corpus/v1",
        {"cases", "synthetic", "review_status", "max_depth", "max_states", "timeout_ms", "fault"},
        {"cases", "synthetic", "review_status"},
    )
    if request["synthetic"] is not True or request["review_status"] != "unreviewed":
        raise ValueError("synthetic unreviewed corpus required")
    cases = request["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 64 or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("1-64 unique cases")
    depth = integer(request.get("max_depth", 3), 1, 3, "max_depth")
    budget = Budget(max_states=request.get("max_states", 65536), timeout_ms=request.get("timeout_ms", 5000))
    if provider not in {"extractor", "fake", "ollama", "vllm"}:
        raise ValueError("invalid provider")
    if provider in {"ollama", "vllm"} and runtime is None:
        runtime = RuntimeProvider({"provider": provider, "timeout_seconds": max(1, budget.timeout_ms // 1000)})
    rows = []
    bases = []
    radii = []
    calls = 0

    def formalize(text, case, is_changed=False):
        nonlocal calls
        budget.tick()
        if len(text) > 32000:
            raise ValueError("source exceeds32k")
        if provider == "extractor":
            norm = extract_one(text)
            provenance = {
                "model_id": "deterministic-extractor/v1",
                "prompt_sha256": digest(text.encode()),
                "seed": 0,
                "params": {},
                "timestamp": "2000-01-01T00:00:00+00:00",
            }
        else:
            prompt = canonical_json({"source": text, "task": "propose_unreviewed_typed_norm"}).decode()
            if provider == "fake":
                if request.get("fault") == "ignore_transform" and is_changed:
                    value = extract_one(case["text"]).to_dict()
                else:
                    value = extract_one(text).to_dict()
                used = FakeProvider([value])
            else:
                used = runtime
            calls += 1
            raw = used.generate(prompt, seed=0, temperature=0)
            norm = decode_norm(raw)
            provenance = {
                "model_id": used.model,
                "prompt_sha256": digest(prompt.encode()),
                "seed": 0,
                "params": {"temperature": 0},
                "timestamp": "2000-01-01T00:00:00+00:00" if provider == "fake" else datetime.now(UTC).isoformat(),
            }
            if getattr(used, "last_request", None) is not None:
                provenance["prompt_sha256"] = digest(canonical_json(used.last_request))
                provenance["generation_request"] = used.last_request
        budget.tick()
        return norm, provenance

    def compare(original, text, case, relation, level):
        expected = EXPECTED[relation]
        try:
            raw, p = formalize(text, case, True)
            a, b = normalize(original, case, "left"), normalize(raw, case, "right")
            triage = consensus_report(
                {
                    "schema": "vle-redundancy/v1",
                    "horizon": 1,
                    "candidates": [{"norm": a.to_dict()}, {"norm": b.to_dict()}],
                    "max_traces": 4096,
                    "timeout_ms": budget.timeout_ms,
                }
            )
            actual = "unknown"
            evidence = {}
            if not triage["alignment"]["unresolved"] and not (a.assumptions or b.assumptions):
                la, lb = TemporalNorm(a, 0, 0), TemporalNorm(b, 0, 0)
                evidence = run_experiment(
                    {
                        "schema": "vle-experiment/v1",
                        "kind": "amendment",
                        "horizon": 1,
                        "before": la.constraint().to_dict(),
                        "after": lb.constraint().to_dict(),
                        "max_traces": 4096,
                        "timeout_ms": budget.timeout_ms,
                    }
                )
                actual = evidence["status"]
                if relation in EQUIVALENT and triage["pairs"][0]["classification"] != "equivalent":
                    actual = "unknown" if triage["pairs"][0]["classification"] == "unknown" else "incomparable"
            outcome = (
                "unknown"
                if actual in {"unknown", "unavailable", "inconsistent_evidence"}
                else "passed"
                if actual == expected
                else "violated"
            )
            return {
                "case": case["id"],
                "language": case["language"],
                "modality": case["modality"],
                "relation": relation,
                "depth": level,
                "text": text,
                "expected": expected,
                "actual": actual,
                "outcome": outcome,
                "normative_output_relation": triage["pairs"][0]["classification"],
                "evidence": evidence,
                "provenance": p,
                "interpretation": raw.to_dict(),
            }
        except (ValueError, KeyError, TypeError):
            return {
                "case": case["id"],
                "language": case["language"],
                "modality": case["modality"],
                "relation": relation,
                "depth": level,
                "text": text,
                "expected": expected,
                "actual": "unknown",
                "outcome": "unknown",
                "reason": "unsupported_or_unresolved_formalization",
            }

    status = "reported"
    reason = "complete_in_declared_scope"
    try:
        for case in cases:
            if (
                not isinstance(case, dict)
                or not {"id", "text", "language", "modality", "relations"} <= set(case)
                or not isinstance(case["relations"], list)
                or len(case["relations"]) > 12
                or set(case["relations"]) - set(EXPECTED)
            ):
                raise ValueError("invalid metamorphic case")
            original, provenance = formalize(case["text"], case)
            bases.append(
                {
                    "case": case["id"],
                    "interpretation": original.to_dict(),
                    "provenance": provenance,
                    "review_status": "unreviewed",
                }
            )
            queue = deque([(case["text"], 0)])
            seen = {case["text"]}
            fail_depth = None
            while queue:
                text, level = queue.popleft()
                budget.tick()
                if level >= depth:
                    continue
                for relation in case["relations"]:
                    if level and relation not in EQUIVALENT:
                        continue
                    variants = transform(text, relation, case)
                    if not variants and level == 0:
                        rows.append(
                            {
                                "case": case["id"],
                                "relation": relation,
                                "language": case["language"],
                                "modality": case["modality"],
                                "outcome": "inapplicable",
                                "depth": 1,
                            }
                        )
                    for variant in variants:
                        budget.tick()
                        # Directional chains are excluded; only equivalences compose.
                        if level == 0 or relation in EQUIVALENT:
                            row = compare(original, variant, case, relation, level + 1)
                            rows.append(row)
                            if relation in EQUIVALENT and row["outcome"] == "violated":
                                fail_depth = min(fail_depth or depth + 1, level + 1)
                        if relation in EQUIVALENT and variant not in seen:
                            seen.add(variant)
                            queue.append((variant, level + 1))
            uncertain_depths = [
                r["depth"]
                for r in rows
                if r["case"] == case["id"] and r["relation"] in EQUIVALENT and r["outcome"] == "unknown"
            ]
            lower_inconclusive = any(d < (fail_depth or depth + 1) for d in uncertain_depths)
            radii.append(
                {
                    "case": case["id"],
                    "minimum_observed_failure_depth": fail_depth,
                    "tested_depth": depth,
                    "right_censored": fail_depth is None,
                    "infinite_robustness_claimed": False,
                    "lower_depth_inconclusive": lower_inconclusive,
                    "exact_radius_established": fail_depth is not None and not lower_inconclusive,
                    "right_censored_at_depth": min(uncertain_depths) - 1
                    if fail_depth is None and uncertain_depths
                    else depth
                    if fail_depth is None
                    else None,
                }
            )
    except BoundHit as exc:
        status, reason, rows, radii, bases = "unknown", str(exc), [], [], []

    def aggregate(group):
        counts = {k: sum(r["outcome"] == k for r in group) for k in ("passed", "violated", "unknown", "inapplicable")}
        counts["complete_comparisons"] = counts["passed"] + counts["violated"]
        counts["MVR"] = counts["violated"] / counts["complete_comparisons"] if counts["complete_comparisons"] else None
        return counts

    groups = {}
    for key in ("relation", "language", "modality"):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row[key]].append(row)
        groups["by_" + key] = {k: aggregate(v) for k, v in sorted(grouped.items())}
    return envelope(
        "vle-metamorphic-report/v1",
        request,
        status=status,
        reason=reason,
        provider=provider,
        rows=rows,
        robustness_radius=radii,
        base_formalizations=bases,
        metrics=aggregate(rows),
        **groups,
        bounds={**budget.to_dict(), "max_depth": depth, "max_cases": 64},
        model_calls=calls,
        H2="descriptive_only_ordering_unestablished_when_comparisons_inconclusive",
    )
