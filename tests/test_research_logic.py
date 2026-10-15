from dataclasses import replace
from itertools import product

import pytest

from src.conflicts import detect_conflicts, detect_temporal_conflicts
from src.defeasible import activation, reason
from src.model_checking import verify, verify_request
from src.normative_model import FactContext, Modality, NormInterpretation, Predicate, SourceAuthority
from src.preferences import Preference, analyze_preferences
from src.priority import resolve_pair
from src.research import run_research
from src.temporal import Formula, atom, combine, evaluate_formula
from src.temporal_deontic import TemporalNorm, verify_temporal_norm


def norm(id="o", modality=Modality.OBLIGATION, **kwargs):
    return NormInterpretation(id=id, source_norm_id=id, modality=modality, subject="operator", action="notify", **kwargs)


def test_finite_boundary_and_until_witness():
    p, q = atom("p"), atom("q")
    assert evaluate_formula(combine("next", p), [{"p": True}]) is False
    assert evaluate_formula(combine("weak_next", p), [{"p": False}]) is True
    until = combine("until", p, q)
    assert evaluate_formula(until, [{"p": True, "q": False}, {"p": False, "q": True}]) is True
    assert evaluate_formula(until, [{"p": True, "q": False}]) is False
    assert evaluate_formula(until, [{"p": False, "q": False}, {"p": True, "q": True}]) is False
    assert evaluate_formula(combine("always", p), [{"p": True}, {}]) is None
    assert evaluate_formula(combine("eventually", p), [{}, {"p": True}]) is True
    assert evaluate_formula(combine("or", p, combine("not", p)), [{}]) is None


@pytest.mark.parametrize("bad", [[], [{"p": 1}], [{"p": "false"}], [None]])
def test_trace_types_are_not_coerced(bad):
    with pytest.raises(ValueError):
        evaluate_formula(atom("p"), bad)


@pytest.mark.parametrize("bad", [{"op": "eval", "name": "x"}, {"op": "atom"},
                                  {"op": "true", "args": [{"op": "true"}]},
                                  {"op": "true", "ignored": True}])
def test_formula_schema_rejects_unsupported_semantics(bad):
    with pytest.raises(ValueError):
        Formula.from_dict(bad)


def test_formula_roundtrip_and_limits():
    p = atom("p")
    assert Formula.from_dict(combine("always", p).to_dict()) == combine("always", p)
    raw = {"op": "true"}
    for _ in range(50):
        raw = {"op": "not", "args": [raw]}
    with pytest.raises(ValueError, match="limit"):
        Formula.from_dict(raw)
    with pytest.raises(ValueError):
        verify(p, horizon=True)
    with pytest.raises(ValueError):
        verify_request({"property": p.to_dict(), "horizon": 1, "ignore_me": True})


def test_missing_temporal_metadata_fails_closed_and_actor_is_scoped():
    rule = norm(authority=SourceAuthority(effective_from="2026-01-01"))
    assert activation(rule, FactContext({})).status == "undetermined"
    assert activation(norm(), FactContext({}, actor="someone else")).status == "inactive"
    with pytest.raises(ValueError):
        Predicate("p").evaluate({"p": "false"})
    with pytest.raises(ValueError):
        SourceAuthority(effective_from="2026-10-10", effective_to="2026-01-01")


def test_temporal_achievement_deadlines_and_observation_gaps():
    rule = TemporalNorm(norm(), 1, 2)
    a = rule.action_atom
    assert rule.evaluate([{a: True}, {a: False}], closed=False)["status"] == "pending"
    assert rule.evaluate([{a: True}, {a: False}])["status"] == "unknown"
    assert rule.evaluate([{a: True}, {a: False}, {a: False}])["status"] == "violated"
    assert rule.evaluate([{}, {}, {a: True}])["status"] == "satisfied"
    assert rule.evaluate([{}, {}, {a: False}])["status"] == "unknown"
    assert rule.evaluate([{}, {a: True}], closed=False)["status"] == "satisfied"
    with pytest.raises(ValueError, match="entire"):
        verify_temporal_norm(rule, horizon=2)


def test_temporal_prohibition_maintenance_and_permission():
    prohibition = TemporalNorm(norm(modality=Modality.PROHIBITION), 0, 1)
    a = prohibition.action_atom
    assert prohibition.evaluate([{a: True}], closed=False)["status"] == "violated"
    assert prohibition.evaluate([{a: False}, {a: False}])["status"] == "satisfied"
    assert prohibition.evaluate([{a: False}], closed=False)["status"] == "pending"
    maintenance = TemporalNorm(norm(), 0, 1, "maintenance")
    assert maintenance.evaluate([{a: False}], closed=False)["status"] == "violated"
    permission = TemporalNorm(norm(modality=Modality.PERMISSION), 0, 1)
    assert permission.evaluate([{}])["status"] == "permitted"
    assert verify_temporal_norm(permission, horizon=2).status == "verified_bounded"


def test_temporal_guard_is_sampled_at_activation_once():
    rule = TemporalNorm(norm(conditions=(Predicate("breach"),), exceptions=(Predicate("waiver"),)), 0, 1)
    trace = [{"fact:breach": True, "fact:waiver": False, rule.action_atom: False},
             {"fact:waiver": True, rule.action_atom: False}]
    assert rule.evaluate(trace)["status"] == "violated"
    trace[0]["fact:waiver"] = True
    assert rule.evaluate(trace)["status"] == "inactive"
    assert rule.evaluate([{}])["status"] == "unknown"
    with pytest.raises(ValueError):
        TemporalNorm(norm(deadline="within 24 hours"), 0, 1)
    with pytest.raises(ValueError):
        TemporalNorm(norm(authority=SourceAuthority(effective_from="2026-01-01")), 0, 1)


@pytest.mark.parametrize("modality,mode", [(Modality.OBLIGATION, "achievement"),
                                          (Modality.OBLIGATION, "maintenance"),
                                          (Modality.PROHIBITION, "achievement")])
def test_deontic_compilation_matches_trace_monitor_exhaustively(modality, mode):
    rule = TemporalNorm(norm(modality=modality, conditions=(Predicate("trigger"),)), 1, 2, mode)
    for values in product((False, True), repeat=4):
        trace = [{rule.action_atom: values[0]},
                 {rule.action_atom: values[1], "fact:trigger": values[3]}, {rule.action_atom: values[2]}]
        result = rule.evaluate(trace)["status"]
        assert evaluate_formula(rule.constraint(), trace) == (result in {"satisfied", "inactive"})


def test_verification_reports_vacuity_limits_and_real_witnesses():
    p = atom("p")
    assert verify(p, horizon=1, assumptions=[Formula("false")]).status == "inconsistent_assumptions"
    assert verify(p, horizon=128, max_traces=4).status == "unknown"
    result = verify(combine("always", p), horizon=3)
    assert result.status == "counterexample"
    assert evaluate_formula(combine("always", p), result.counterexample) is False
    assert verify(p, horizon=1, assumptions=[p]).status == "verified_bounded"


def test_exhaustive_timeout_never_becomes_proof(monkeypatch):
    import src.model_checking as module
    times = iter([0, 10])
    monkeypatch.setattr(module, "monotonic", lambda: next(times))
    assert verify(atom("p"), horizon=1, timeout_ms=1).status == "unknown"


def test_independent_smt_and_exhaustive_backends_agree():
    pytest.importorskip("z3")
    p, q = atom("p"), atom("q")
    formulas = [Formula("true"), Formula("false"), p, q, combine("not", p)]
    for op in ("next", "weak_next", "always", "eventually"):
        formulas.extend([combine(op, p), combine(op, combine("not", p))])
    for op in ("and", "or", "until"):
        formulas.extend([combine(op, p, q), combine(op, p, combine("not", p))])
    formulas.extend([combine("not", combine("until", p, q)),
                     combine("or", combine("next", p), combine("weak_next", combine("not", p))),
                     combine("or", combine("not", combine("eventually", p)), combine("eventually", p))])
    for formula, horizon in product(formulas, (1, 2, 3)):
        for assumptions in ([], [p], [Formula("false")]):
            exhaustive = verify(formula, horizon=horizon, assumptions=assumptions)
            smt = verify(formula, horizon=horizon, assumptions=assumptions, backend="z3")
            assert smt.status == exhaustive.status, (formula, horizon, assumptions, smt)
            if smt.counterexample:
                assert evaluate_formula(formula, smt.counterexample) is False
                assert all(evaluate_formula(a, smt.counterexample) for a in assumptions)


def test_scoped_conflicts_and_specificity_inclusion():
    obligation = norm("o", object="public-record")
    prohibition = norm("p", Modality.PROHIBITION, object="private-record")
    assert detect_conflicts([obligation, prohibition]) == []
    prohibition = replace(prohibition, object=obligation.object)
    conflict = detect_conflicts([obligation, prohibition])[0]
    assert conflict.left_id == "o" and conflict.right_id == "p"
    assert conflict.object == "public-record"
    broader = norm("o", conditions=(Predicate("a"),))
    unrelated = norm("p", Modality.PROHIBITION, conditions=(Predicate("b"), Predicate("c")))
    assert resolve_pair(broader, unrelated).status == "unresolved"
    specific = replace(unrelated, conditions=(Predicate("a"), Predicate("b")))
    assert resolve_pair(broader, specific).winner_id == "p"
    disjoint = replace(specific, conditions=(Predicate("a", True),))
    assert detect_conflicts([broader, disjoint]) == []
    assert detect_conflicts([broader, replace(disjoint, conditions=(), exceptions=(Predicate("a"),))]) == []


def test_posterior_requires_known_equal_rank():
    a = norm("a", authority=SourceAuthority(enacted_on="2025-01-01"))
    b = norm("b", Modality.PROHIBITION, authority=SourceAuthority(enacted_on="2026-01-01"))
    assert resolve_pair(a, b).status == "unresolved"


def test_grounded_priority_reinstates_argument_defended_by_winner():
    a = norm("a", authority=SourceAuthority(authority_rank=3))
    b = norm("b", Modality.PROHIBITION, authority=SourceAuthority(authority_rank=2))
    c = norm("c", authority=SourceAuthority(authority_rank=1))
    report = reason([a, b, c], FactContext({}))
    assert report["argumentation"]["labels"] == {"a": "in", "b": "out", "c": "in"}


def test_preferences_incomparability_transitivity_and_cycles():
    attacks = {("a", "b"), ("b", "a")}
    assert analyze_preferences(["a", "b"], attacks)["extensions"] == [[]]
    prefs = [Preference("a", "c", "reviewed ranking"), Preference("c", "b", "reviewed ranking")]
    report = analyze_preferences(["a", "b", "c"], attacks, prefs)
    assert report["extensions"] == [["a", "c"]]
    assert report["suppressed_attacks"] == [["b", "a"]]
    with pytest.raises(ValueError):
        analyze_preferences(["a", "b", "c"], attacks, prefs + [Preference("b", "a", "cycle")])
    with pytest.raises(ValueError):
        analyze_preferences(["a"], set(), [Preference("a", "missing", "unknown")])
    stable = analyze_preferences(["a"], {("a", "a")}, semantics="stable")
    assert stable["extensions"] == stable["skeptical"] == []
    assert not stable["has_extensions"]
    with pytest.raises(ValueError):
        analyze_preferences([str(i) for i in range(17)], set(), semantics="preferred")


def test_temporal_conflicts_include_overlap_and_different_trigger_steps():
    a = TemporalNorm(norm("o"), 0, 2)
    b = TemporalNorm(norm("p", Modality.PROHIBITION), 1, 2)
    conflict = detect_temporal_conflicts([a, b])[0]
    assert conflict.severity == "potential"
    assert conflict.temporal_overlap == (1, 2)
    assert detect_temporal_conflicts([replace(a, start=1), b])[0].severity == "incompatible"
    assert detect_temporal_conflicts([replace(a, end=0), b]) == []
    a = replace(a, norm=replace(a.norm, conditions=(Predicate("p"),)))
    b = replace(b, norm=replace(b.norm, conditions=(Predicate("p", True),)))
    assert detect_temporal_conflicts([a, b])  # p@0 and not p@1 can both hold
    assert detect_temporal_conflicts([replace(a, start=1), b]) == []


def test_integrated_research_preferences_and_unknowns():
    def item(id, modality):
        return {"norm": {"id": id, "modality": modality, "subject": "operator", "action": "notify"},
                "start": 0, "end": 0}
    request = {"schema": "vle-research/v1", "norms": [item("o", "obligation"), item("p", "prohibition")],
               "trace": [{TemporalNorm(norm(), 0, 0).action_atom: True}]}
    assert run_research(request)["status"] == "indeterminate"
    request["preferences"] = [{"preferred": "o", "disfavored": "p", "rationale": "reviewed exception"}]
    assert run_research(request)["status"] == "compliant"
    request["trace"] = [{}]
    assert run_research(request)["status"] == "indeterminate"


def test_missing_solver_is_explicitly_unavailable(monkeypatch):
    import builtins

    original = builtins.__import__

    def without_z3(name, *args, **kwargs):
        if name == "z3":
            raise ImportError("optional solver unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_z3)
    assert verify(atom("p"), horizon=1, backend="z3").status == "unavailable"


def test_smt_budget_and_encoding_timeout_fail_closed(monkeypatch):
    pytest.importorskip("z3")
    import src.model_checking as module

    formula = atom("p")
    for _ in range(8):
        formula = combine("and", atom("p"), formula)
    assert verify(formula, horizon=128, backend="z3").reason == "encoding_budget"
    times = iter([0, 10])
    monkeypatch.setattr(module, "monotonic", lambda: next(times))
    result = verify(atom("p"), horizon=1, backend="z3", timeout_ms=1)
    assert result.status == "unknown" and result.reason == "encoding_timeout"


def test_unknown_norm_blocks_readiness_even_alongside_proven_violation():
    from src.lifecycle import run_lifecycle

    text = "The operator must notify. If risk is high, the operator must log."
    candidates = [
        {"id": "n1", "source_id": "s", "source_span": [0, 25], "sentence": text[:25],
         "modality": "obligation", "actor": "operator", "action": "notify", "condition": None,
         "exception": None, "references": [], "ambiguity_flags": [], "confidence": 1.0},
        {"id": "n2", "source_id": "s", "source_span": [26, len(text)], "sentence": text[26:],
         "modality": "obligation", "actor": "operator", "action": "log", "condition": "If risk is high",
         "exception": None, "references": [], "ambiguity_flags": [], "confidence": 1.0},
    ]
    report = run_lifecycle(text=text, source_id="s", candidates=candidates,
                           extraction_audit={"summary": {"error_count": 0, "possible_omission_count": 0}},
                           context=FactContext({}), review_statuses=["approved", "approved"])
    assert report["compliance"]["status"] == "non_compliant"
    assert report["readiness"]["status"] == "not_ready"
    assert "determinate_reasoning" in report["readiness"]["failed_gates"]
