"""Versioned, supervised French/U.S. IP release evidence; no legal certification.

These checks validate an externally pinned attestation and its accounting. They
cannot establish that reported experiments, expert credentials or sources are
truthful. Source probes measure bounded access, never corpus completeness.
"""

from datetime import UTC, datetime
from math import isfinite

from src.audit_bundle import canonical_json, digest
from src.courtlistener import IP_QUERIES

SCOPE = "fr-us-ip-supervised/v1"
REVIEW_CHECKS = (
    "holdings_vs_dicta", "quotations_vs_court_reasoning", "majority_vs_separate_opinions",
    "precedential_status", "jurisdiction_and_date", "conditions_exceptions_cross_references",
    "source_change_reassessment", "independent_review_and_adjudication",
)
DIMENSIONS = {
    "language": {"fr", "en"},
    "modality": {"obligation", "prohibition", "permission"},
    "exceptions": {"present", "absent"},
    "source_kind": {"code_article", "judicial_opinion"},
}


def natural(value, minimum=0):
    return type(value) is int and value >= minimum


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def recent(value, max_age, now):
    stamp = datetime.fromisoformat(value)
    return stamp.tzinfo is not None and 0 <= (now - stamp).total_seconds() <= max_age


def validate_ip_sources(report, settings, manifest, *, now=None):
    now = now or datetime.now(UTC)
    if (report.get("schema") != "vle-operational-evidence/v2" or report.get("kind") != "ip_sources"
            or report.get("passed") is not True or report.get("scope") != SCOPE
            or report.get("build_revision") != settings.build_revision
            or report.get("deployment_id") != manifest["deployment_id"]
            or report.get("evidence_origin") != "live_provider_probe"
            or not recent(report["observed_at"], settings.release_source_max_age_seconds, now)
            or set(report["providers"]) != {"legifrance_ip", "courtlistener_ip"}):
        raise ValueError("current live French/U.S. IP source evidence required")
    for provider, value in report["providers"].items():
        expected = {"ip"} if provider == "legifrance_ip" else set(IP_QUERIES)
        if set(value) != expected:
            raise ValueError("source evidence does not cover every declared IP topic")
        for receipt in value.values():
            prefix = "legifrance:ip:" if provider == "legifrance_ip" else "courtlistener:opinion:"
            if (not isinstance(receipt["source_key"], str) or not receipt["source_key"].startswith(prefix)
                    or not sha(receipt["content_sha256"]) or not sha(receipt["raw_payload_sha256"])
                    or receipt.get("full_text") is not True):
                raise ValueError("source receipt must bind actual full text and upstream provenance")


def validate_ip_quality(benchmark, protocol, settings):
    if (benchmark.get("schema") != "vle-production-benchmark/v2"
            or not sha(benchmark.get("predictions_sha256")) or not sha(benchmark.get("protocol_sha256"))
            or benchmark["protocol_sha256"] != digest(canonical_json(protocol))
            or protocol.get("schema") != "vle-ip-review-protocol/v1" or protocol.get("scope") != SCOPE
            or set(protocol.get("checks", {})) != set(REVIEW_CHECKS)
            or any(v is not True for v in protocol["checks"].values())
            or not sha(protocol.get("instructions_sha256"))):
        raise ValueError("IP benchmark requires predictions and an explicit judicial review protocol")
    if set(benchmark["breakdowns"]) != set(DIMENSIONS):
        raise ValueError("benchmark dimensions differ from the declared IP protocol")
    for dimension, labels in DIMENSIONS.items():
        strata = benchmark["breakdowns"][dimension]
        if set(strata) != labels or sum(s["case_count"] for s in strata.values()) != benchmark["case_count"]:
            raise ValueError("each benchmark dimension must partition every held-out case")
        for item in strata.values():
            n = item["case_count"]
            f1 = item["extraction_f1"]
            if (not natural(n, 1) or not natural(item["abstentions"]) or not natural(item["correct"])
                    or not natural(item["false_compliance"]) or item["correct"] + item["abstentions"] > n
                    or item["false_compliance"] > n - item["correct"] - item["abstentions"]):
                raise ValueError("invalid counts or excluded abstentions in benchmark stratum")
            if (item["correct"] / n < settings.release_min_compliance_accuracy
                    or type(f1) not in {int, float} or not settings.release_min_extraction_f1 <= f1 <= 1):
                raise ValueError("quality threshold failed in an IP benchmark stratum")
            if item["false_compliance"] / n > settings.release_max_false_compliance_rate:
                raise ValueError("declared false-compliance budget exceeded")
    for pair in ("direct_vs_symbolic", "direct_vs_gold", "symbolic_vs_gold"):
        item = benchmark["faithfulness"][pair]
        if (not natural(item["disagreements"]) or type(item["total"]) is not int
                or item["total"] != benchmark["case_count"] or item["disagreements"] > item["total"]):
            raise ValueError("all three faithfulness gaps must retain every held-out case")
    score = benchmark["compliance_accuracy"]
    total_correct = sum(v["correct"] for v in benchmark["breakdowns"]["language"].values())
    if not isfinite(score) or abs(score - total_correct / benchmark["case_count"]) > 1e-12:
        raise ValueError("aggregate accuracy contradicts stratum counts")
    # Dimensions describe the same predictions; counts must agree across them.
    for name in ("correct", "abstentions", "false_compliance"):
        if len({sum(v[name] for v in strata.values()) for strata in benchmark["breakdowns"].values()}) != 1:
            raise ValueError("benchmark dimensions contradict one another")


def source_scope_gate(settings, source, context):
    if settings.release_profile == "legacy":
        return {"passed": True, "basis": "legacy_scope_policy"}
    p = source.extraction_audit.get("provenance", {})
    expected = {"FR": ("code_article", "legifrance:ip:"), "US": ("judicial_opinion", "courtlistener:opinion:")}
    kind, prefix = expected.get(p.get("jurisdiction"), (None, "!unsupported!"))
    passed = bool(kind and p.get("document_type") == kind and source.source_key.startswith(prefix)
                  and context.get("jurisdiction") == p["jurisdiction"] and sha(p.get("raw_payload_sha256")))
    return {"passed": passed, "reason": "within_declared_IP_scope" if passed else "unsupported_or_unconfirmed_source_scope",
            "judicial_authority": "requires_independent_case_review" if kind == "judicial_opinion" else "not_assessed_here"}


def probe_ip_sources(settings, *, client_factory=None, now=None):
    """At most one page and one full opinion per topic; no imports or reviews."""
    from .automatic_imports import build_client

    now = now or datetime.now(UTC)
    make = client_factory or build_client
    french = make(settings, "legifrance")
    day = now.date().isoformat()
    manifest = french.manifest(day)
    provision = french.article(manifest["articles"][0], effective_date=day, manifest_sha256=manifest["manifest_sha256"])

    def receipt(item):
        return {"source_key": item.source_key, "content_sha256": digest(item.source_text.encode()),
                "raw_payload_sha256": item.provenance["raw_payload_sha256"], "full_text": True}

    providers = {"legifrance_ip": {"ip": receipt(provision)}, "courtlistener_ip": {}}
    for topic in sorted(IP_QUERIES):
        # A new bounded transport gives each topic its own recorded deadline.
        american = make(settings, "courtlistener")
        page = american.search_page(topic=topic, since=settings.import_start_date, until=day, cursor=None)
        if not page["pending"]:
            raise ValueError("no full-opinion evidence for a required IP topic")
        opinion = american.opinion(page["pending"][0])
        if opinion is None:
            raise ValueError("first opinion outside supported scope; inspect provider evidence")
        providers["courtlistener_ip"][topic] = receipt(opinion)
    return {"providers": providers, "scope": SCOPE, "evidence_origin": "live_provider_probe",
            "coverage": "one_full_text_per_declared_topic_not_corpus_completeness",
            "bounds": {"max_pages_per_topic": 1, "max_texts": 5,
                       "max_seconds_per_topic": settings.import_max_run_seconds,
                       "max_total_seconds": 5 * settings.import_max_run_seconds}}
