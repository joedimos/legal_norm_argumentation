from __future__ import annotations

from typing import Dict, Mapping, Sequence


def _safe_div(num: int, den: int) -> float:
    return num / den if den else 0.0


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def evaluate_extraction(predicted: Sequence[Mapping], gold: Sequence[Mapping]) -> Dict:
    pred_by_span = {tuple(item.get("source_span", ())): item for item in predicted}
    gold_by_span = {tuple(item.get("source_span", ())): item for item in gold}
    pred_spans, gold_spans = set(pred_by_span), set(gold_by_span)
    matched = pred_spans & gold_spans
    metrics = _prf(len(matched), len(pred_spans - gold_spans), len(gold_spans - pred_spans))
    fields = ("modality", "actor", "action", "condition", "exception")
    field_accuracy = {}
    for field in fields:
        correct = sum(pred_by_span[span].get(field) == gold_by_span[span].get(field) for span in matched)
        field_accuracy[field] = round(_safe_div(correct, len(matched)), 4)
    return {"detection": metrics, "matched_candidates": len(matched), "field_accuracy": field_accuracy}


def evaluate_compliance(predicted: Sequence[str], gold: Sequence[str]) -> Dict:
    if len(predicted) != len(gold):
        raise ValueError("predicted and gold compliance labels must have equal length")
    labels = sorted(set(predicted) | set(gold))
    per_label = {}
    for label in labels:
        tp = sum(p == label and g == label for p, g in zip(predicted, gold))
        fp = sum(p == label and g != label for p, g in zip(predicted, gold))
        fn = sum(p != label and g == label for p, g in zip(predicted, gold))
        per_label[label] = _prf(tp, fp, fn)
    accuracy = _safe_div(sum(p == g for p, g in zip(predicted, gold)), len(gold))
    macro_f1 = _safe_div(sum(item["f1"] for item in per_label.values()), len(per_label))
    return {"accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4), "per_label": per_label}


def propagation_report(records: Sequence[Mapping]) -> Dict:
    total = len(records)
    ambiguous = [r for r in records if r.get("upstream_ambiguous")]
    downstream_changed = [r for r in ambiguous if r.get("downstream_changed")]
    return {
        "records": total,
        "upstream_ambiguous": len(ambiguous),
        "downstream_changed": len(downstream_changed),
        "conditional_propagation_rate": round(_safe_div(len(downstream_changed), len(ambiguous)), 4),
    }
