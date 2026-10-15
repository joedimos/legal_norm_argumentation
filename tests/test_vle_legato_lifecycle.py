from copy import deepcopy
from dataclasses import replace

import pytest

from src.benchmark import evaluate_compliance, evaluate_extraction, propagation_report
from src.certification import readiness_report
from src.compliance import evaluate, generate_counterexamples
from src.defeasible import activation, reason
from src.formalization import add_alternative, behaviorally_equivalent, candidate_to_interpretations, select_interpretation
from src.knowledge_base import NormKnowledgeBase
from src.lifecycle import run_lifecycle
from src.monitoring import ComplianceMonitor
from src.normative_model import FactContext, Modality, NormInterpretation, Predicate, SourceAuthority, SourceKind
from src.priority import resolve_pair
from src.references import ReferenceResolver
from src.traceability import EvidenceGraph


def candidate(**overrides):
    base = {
        "id": "norm-1", "source_id": "law-1", "source_span": [0, 42],
        "sentence": "If risk is high, operator must notify unless barred.",
        "modality": "obligation", "actor": "operator", "action": "notify",
        "condition": "If risk is high", "exception": "unless barred",
        "references": ["Article 7"], "ambiguity_flags": [], "confidence": 0.9,
    }
    base.update(overrides)
    return base


def make_norm(id="n", modality=Modality.OBLIGATION, action="notify", *, rank=None, enacted=None, conditions=(), exceptions=()):
    return NormInterpretation(
        id=id, source_norm_id=id, modality=modality, subject="system", action=action,
        conditions=tuple(conditions), exceptions=tuple(exceptions),
        authority=SourceAuthority(authority_rank=rank, enacted_on=enacted),
    )


def test_formalization_preserves_provenance_conditions_and_exceptions():
    result = candidate_to_interpretations(candidate(), SourceAuthority(kind=SourceKind.LAW, authority_rank=10))
    norm = result.selected()
    assert norm is not None
    assert norm.conditions[0].name == "risk is high"
    assert norm.exceptions[0].name == "barred"
    assert norm.provenance.quote.startswith("If risk")
    assert "OBLIGATION[operator:notify]" in norm.canonical()


def test_ambiguous_candidate_fails_closed_until_explicit_selection():
    result = candidate_to_interpretations(candidate(ambiguity_flags=["possible_anaphora"]))
    assert result.unresolved
    assert result.selected() is None
    chosen = select_interpretation(result, result.interpretations[0].id, "Human reviewer resolved antecedent.")
    assert chosen.selected() is not None


def test_alternative_interpretations_and_behavioral_equivalence():
    result = candidate_to_interpretations(candidate())
    expanded = add_alternative(result, label="narrow", action="notify authority", assumptions=("scope:narrow",))
    assert expanded.unresolved
    contexts = [{"risk is high": True}, {"risk is high": False}]
    assert not behaviorally_equivalent(expanded.interpretations[0], expanded.interpretations[1], contexts)


def test_unknown_modality_cannot_be_formalized():
    with pytest.raises(ValueError):
        candidate_to_interpretations(candidate(modality="unknown"))


def test_unknown_condition_is_indeterminate_and_exception_defeats():
    conditional = make_norm(conditions=[Predicate("breach")])
    assert activation(conditional, FactContext(facts={})).status == "undetermined"
    excepted = make_norm(exceptions=[Predicate("court order")])
    assert activation(excepted, FactContext(facts={"court order": True})).status == "defeated"


def test_priority_rules_require_evidence_and_surface_unresolved_conflict():
    low = make_norm("low", Modality.PROHIBITION, "release", rank=1)
    high = make_norm("high", Modality.OBLIGATION, "release", rank=5)
    assert resolve_pair(low, high).winner_id == "high"
    a = make_norm("a", Modality.PROHIBITION, "release")
    b = make_norm("b", Modality.OBLIGATION, "release")
    assert reason([a, b], FactContext(facts={})) ["unresolved_conflicts"]


def test_lex_posterior_equal_rank():
    old = make_norm("old", Modality.PROHIBITION, "release", rank=5, enacted="2024-01-01")
    new = make_norm("new", Modality.OBLIGATION, "release", rank=5, enacted="2025-01-01")
    assert resolve_pair(old, new).winner_id == "new"


def test_obligation_prohibition_permission_compliance_semantics():
    obligation = make_norm("o", Modality.OBLIGATION, "log")
    assert evaluate([obligation], FactContext(facts={}, observed_actions=()))["status"] == "non_compliant"
    assert evaluate([obligation], FactContext(facts={}, observed_actions=("log",)))["status"] == "compliant"
    prohibition = make_norm("p", Modality.PROHIBITION, "leak")
    assert evaluate([prohibition], FactContext(facts={}, observed_actions=("leak",)))["status"] == "non_compliant"
    permission = make_norm("perm", Modality.PERMISSION, "appeal")
    report = evaluate([permission], FactContext(facts={}, observed_actions=()))
    assert report["status"] == "compliant"
    assert report["findings"][0]["status"] == "not_exercised"


def test_counterexample_generator_finds_missing_required_action():
    rule = make_norm("o", Modality.OBLIGATION, "notify", conditions=[Predicate("breach")])
    examples = generate_counterexamples([rule], ["breach"])
    assert any(item["facts"]["breach"] and item["observed_actions"] == [] for item in examples)


def test_evidence_graph_immutable_and_lineage_complete():
    graph = EvidenceGraph()
    graph.add("source", "source", {"x": 1})
    graph.add("formal", "formal", {"x": 2})
    graph.add("decision", "decision", {"x": 3})
    graph.link("source", "formal", "formalizes")
    graph.link("formal", "decision", "supports")
    assert len(graph.lineage("decision")) == 2
    with pytest.raises(ValueError):
        graph.add("source", "source", {"x": 999})


def test_metrics_include_field_accuracy_and_error_propagation():
    gold = [{"source_span": [0, 4], "modality": "obligation", "actor": "a", "action": "x", "condition": None, "exception": None}]
    pred = [{"source_span": [0, 4], "modality": "obligation", "actor": "a", "action": "wrong", "condition": None, "exception": None}]
    metrics = evaluate_extraction(pred, gold)
    assert metrics["detection"]["f1"] == 1.0
    assert metrics["field_accuracy"]["action"] == 0.0
    assert evaluate_compliance(["compliant", "non_compliant"], ["compliant", "compliant"])["accuracy"] == 0.5
    assert propagation_report([{"upstream_ambiguous": True, "downstream_changed": True}])["conditional_propagation_rate"] == 1.0


def test_readiness_is_evidence_gated_and_never_legal_certification():
    audit = {"summary": {"error_count": 0, "possible_omission_count": 0}}
    failed = readiness_report(extraction_audit=audit, review_statuses=["unreviewed"], unresolved_interpretations=0, unresolved_conflicts=1)
    assert failed["status"] == "not_ready"
    passed = readiness_report(
        extraction_audit=audit, review_statuses=["approved"], unresolved_interpretations=0, unresolved_conflicts=0,
        benchmark_metrics={"extraction": {"detection": {"f1": 0.95}}, "compliance": {"accuracy": 0.97}},
    )
    assert passed["status"] == "research_ready"
    assert passed["legal_certification"] is False


def test_versioning_reference_resolution_and_monitoring():
    kb = NormKnowledgeBase()
    n = make_norm()
    first = kb.add(n, "alice", "initial review")
    changed = replace(n, action="notify authority")
    second = kb.add(changed, "bob", "scope correction")
    assert second.supersedes == first.revision_id
    resolver = ReferenceResolver({"Article 7": {"source_id": "law", "text": "Text"}})
    assert resolver.resolve(" article   7 ").status == "resolved"
    assert resolver.resolve("Article 99").status == "unresolved"
    monitor = ComplianceMonitor([n])
    assert monitor.ingest("unrelated")["report"]["status"] == "non_compliant"
    assert monitor.ingest("notify")["report"]["status"] == "compliant"


def test_lifecycle_connects_formalization_compliance_evidence_and_readiness():
    text = "The operator must notify."
    item = {
        "id": "n1", "source_id": "s1", "source_span": [0, len(text)], "sentence": text,
        "modality": "obligation", "actor": "The operator", "action": "notify",
        "condition": None, "exception": None, "references": [], "ambiguity_flags": [], "confidence": 0.9,
    }
    audit = {"summary": {"error_count": 0, "possible_omission_count": 0}}
    result = run_lifecycle(
        text=text, source_id="s1", candidates=[item], extraction_audit=audit,
        context=FactContext(facts={}, observed_actions=("notify",)), review_statuses=["approved"],
    )
    assert result["compliance"]["status"] == "compliant"
    assert result["readiness"]["status"] == "research_ready"
    assert len(result["evidence"]["edges"]) >= 3
