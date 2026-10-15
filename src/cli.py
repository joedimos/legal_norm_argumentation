from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .compliance import evaluate
from .formalization import formalize_candidates
from .lifecycle import run_lifecycle
from .normative_model import FactContext
from .norm_extraction import extract_all_norms, extract_norms
from .verification import audit_extraction


def _load_json(value: str) -> Any:
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def _read_text(args) -> str:
    if getattr(args, "text_file", None):
        return Path(args.text_file).read_text(encoding="utf-8")
    return args.text


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--text-file")
    parser.add_argument("--source-id", default="cli")


def main() -> None:
    parser = argparse.ArgumentParser(description="VLE-LEGATO normative AI lifecycle toolkit")
    subs = parser.add_subparsers(dest="command", required=True)

    for name in ("extract", "audit", "formalize"):
        _add_source_args(subs.add_parser(name))

    comply = subs.add_parser("comply")
    _add_source_args(comply)
    comply.add_argument("--facts", default="{}", help="JSON object or path")
    comply.add_argument("--actions", default="[]", help="JSON array or path")
    comply.add_argument("--at-date")

    lifecycle = subs.add_parser("lifecycle")
    _add_source_args(lifecycle)
    lifecycle.add_argument("--facts", default="{}")
    lifecycle.add_argument("--actions", default="[]")
    lifecycle.add_argument("--at-date")
    lifecycle.add_argument("--assume-reviewed", action="store_true", help="Demo/testing convenience only")

    args = parser.parse_args()
    text = _read_text(args)
    norms = extract_norms(text, source_id=args.source_id)

    if args.command == "extract":
        payload = extract_all_norms(text, source_id=args.source_id)
    elif args.command == "audit":
        payload = {"norms": norms, "audit": audit_extraction(text, norms)}
    elif args.command == "formalize":
        sets = formalize_candidates(norms)
        payload = {"formalizations": [{"source_norm_id": item.source_norm_id, "selected_id": item.selected_id, "unresolved": item.unresolved, "interpretations": [x.to_dict() for x in item.interpretations]} for item in sets]}
    else:
        facts = _load_json(args.facts)
        actions = tuple(_load_json(args.actions))
        context = FactContext(facts=facts, observed_actions=actions, at_date=args.at_date)
        if args.command == "comply":
            sets = formalize_candidates(norms)
            selected = [item.selected() for item in sets if item.selected() is not None]
            payload = {
                "unresolved_norm_ids": [item.source_norm_id for item in sets if item.unresolved],
                "compliance": evaluate(selected, context) if selected else {"status": "indeterminate", "findings": []},
            }
        else:
            audit = audit_extraction(text, norms)
            statuses = ["approved"] * len(norms) if args.assume_reviewed else ["unreviewed"] * len(norms)
            payload = run_lifecycle(text=text, source_id=args.source_id, candidates=norms, extraction_audit=audit, context=context, review_statuses=statuses)

    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
