import json
from dataclasses import replace
from pathlib import Path

import jsonschema
import pytest

from src.ddl import interchange
from src.ddl.export import export_document, export_norms
from src.ddl.import_ import import_document, import_exported
from src.ddl.metrics import metric_vector
from src.ddl.model import DDLRule, ModalLiteral, document
from src.ddl.refine import propose_refinement
from src.ddl.segment import segment
from src.normative_model import InterpretationSet, Modality, NormInterpretation, Predicate, SourceAuthority

FIXTURES = json.loads(Path("tests/fixtures/phase4/ddl-v1.json").read_text())


def validate(result):
    jsonschema.validate(
        json.loads(json.dumps(result)), json.loads(Path("schemas/ddl-interchange.schema.json").read_text())
    )
    assert not result["decision_authorized"] and not result["legal_certification"]


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda c: c["id"])
def test_24_synthetic_unreviewed_roundtrips(case):
    assert case["synthetic"] and case["review_status"] == "unreviewed"
    exported = export_document(case["document"])
    assert import_document(exported["text"])["document"] == exported["document"]
    assert export_document(import_document(exported["text"])["document"]) == exported
    assert exported["lossy_fields"] == []
    validate(interchange({"schema": "vle-ddl/v1", "operation": "export", "document": case["document"]}))


def test_golden_export_and_typed_norm_sidecar_roundtrip():
    assert (
        export_document(FIXTURES["cases"][3]["document"])["text"]
        == Path("tests/fixtures/phase4/ddl-golden.txt").read_text()
    )
    norm = NormInterpretation(
        "n",
        "s",
        Modality.OBLIGATION,
        "operator",
        "notify",
        conditions=(Predicate("p", True),),
        deadline="tomorrow",
        authority=SourceAuthority(authority_rank=3),
    )
    family = InterpretationSet("s", (norm,), None, None)
    exported = export_norms(family)
    assert len(exported["lossy_fields"]) == 2
    imported = import_exported(exported)
    assert imported["typed_norms"] == [norm.to_dict()]
    assert imported["interpretation_set"]["selected_id"] is None
    modified = dict(exported, text=exported["text"].replace("notify", "delete"))
    with pytest.raises(ValueError):
        import_exported(modified)
    recommended = export_norms([replace(norm, modality=Modality.RECOMMENDATION)])
    assert any("recommendation" in f for f in recommended["lossy_fields"])
    assert import_exported(export_document(FIXTURES["cases"][0]["document"]))["lossy_fields"]


def test_segmentation_preserves_exact_unicode_spans_and_no_scope_claim():
    text = "If café is open, operators must notify unless waived; they may appeal."
    snippets = segment(text, "synthetic")
    assert {"condition", "modality", "consequent", "exception"} <= {s["kind"] for s in snippets}
    assert "".join(s["quote"] for s in snippets) == text
    assert all(s["quote"] == text[s["start"] : s["end"]] and not s["scope_verified"] for s in snippets)
    assert segment("A definition without deontic cues.", "s")[0]["kind"] == "context"
    with pytest.raises(ValueError):
        segment("", "s")


def test_refinement_is_a_diff_never_an_overwrite_or_auto_accept():
    before = document([DDLRule("r", "defeasible", ("p",), ModalLiteral(Modality.OBLIGATION, "notify"))])
    after = json.loads(json.dumps(before))
    after["rules"][0]["exceptions"] = ["waiver"]
    p = propose_refinement(
        before,
        after,
        "Operators must notify unless waived.",
        known_predicates=["p", "waiver"],
        references=["article 1"],
        reference_index={},
    )
    assert p["eligible_for_review"] and not p["accepted"] and p["diff"]
    assert p["checks_before"]["missing_exceptions"] > p["checks_after"]["missing_exceptions"]
    assert p["checks_after"]["dangling_references"] == 1
    assert before["rules"][0]["exceptions"] == []
    bad = json.loads(json.dumps(after))
    bad["rules"][0]["body"].append("invented")
    assert not propose_refinement(before, bad, "Must notify.", known_predicates=["p", "waiver"])["eligible_for_review"]
    assert not propose_refinement(before, {}, "Must notify.")["eligible_for_review"]


def test_simple_fragment_cannot_hide_completeness_loss_or_modal_negation():
    gold = document(
        [
            DDLRule(
                "r",
                "defeasible",
                ("p",),
                ModalLiteral(Modality.PERMISSION, "notify", True),
                ("waiver",),
                (ModalLiteral(Modality.OBLIGATION, "repair"),),
            ),
            DDLRule("implicit", "strict", (), ModalLiteral(Modality.OBLIGATION, "record")),
        ],
        [("implicit", "r")],
    )
    simple = document([DDLRule("r", "defeasible", (), ModalLiteral(Modality.PERMISSION, "notify"))])
    vector = metric_vector(simple, gold, source_text="Operators must notify.")
    assert vector["modality_accuracy"] == 0.5 and vector["normative_completeness"] < 0.5
    assert vector["exception_coverage"] == 0 and vector["negation_placement"] < 1
    assert vector["textual_fidelity"]["possible_omissions"] and vector["aggregate_score"] is None
    assert not metric_vector({}, gold)["syntactic_validity"]


@pytest.mark.parametrize(
    "text",
    [
        "DDL/2\n",
        "DDL/1\nignored",
        'DDL/1\nr: [] xx O("x") UNLESS [] COMPENSATE []\n',
        'DDL/1\nr: [] => Q("x") UNLESS [] COMPENSATE []\n',
        'DDL/1\nr: [] => O("x")\n',
    ],
)
def test_unknown_syntax_is_rejected_not_ignored(text):
    with pytest.raises(ValueError):
        import_document(text)


def test_invalid_modal_literals_and_superiority():
    with pytest.raises(ValueError):
        ModalLiteral(Modality.RECOMMENDATION, "x")
    with pytest.raises(ValueError):
        DDLRule("bad id", "strict", (), ModalLiteral(Modality.OBLIGATION, "x"))
    raw = FIXTURES["cases"][1]["document"]
    with pytest.raises(ValueError):
        export_document({**raw, "superiority": [["unknown", "support"]]})
    with pytest.raises(ValueError):
        export_document({**raw, "superiority": [["r1", "support"], ["support", "r1"]]})


def test_all_interchange_operations_validate_their_schema():
    doc = FIXTURES["cases"][0]["document"]
    exported = export_document(doc)
    cases = [
        dict(operation="import", text=exported["text"]),
        dict(operation="import", exported=exported),
        dict(operation="segment", text="If p, an operator must notify unless waived.", source_id="synthetic"),
        dict(operation="refine", before=doc, after=doc, text="Operators must notify."),
        dict(operation="metrics", document=doc, expected=doc),
        dict(operation="export", norms=[NormInterpretation("n", "s", Modality.OBLIGATION, "s", "a").to_dict()]),
    ]
    for c in cases:
        validate(interchange({"schema": "vle-ddl/v1", **c}))
    with pytest.raises(ValueError):
        interchange({"schema": "vle-ddl/v1", "operation": "invent"})
