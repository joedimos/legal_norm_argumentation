from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .compliance import evaluate
from .formalization import formalize_candidates
from .lifecycle import run_lifecycle
from .norm_extraction import extract_all_norms, extract_norms
from .normative_model import FactContext
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

    for name in ("verify", "research", "experiment"):
        command = subs.add_parser(name)
        command.add_argument("--input", required=True, help="Path to a versioned JSON research request")
        command.add_argument("--bundle", help="Write an audit ZIP containing request, result and runtime")
        command.add_argument("--revision", help="Full Git SHA of the executing build")
    benchmark = subs.add_parser("benchmark")
    benchmark.add_argument("--corpus", default="benchmarks/research-v1.json")
    benchmark.add_argument("--backend", choices=("exhaustive", "z3"), default="exhaustive")
    experiments = subs.add_parser("experiment-benchmark")
    experiments.add_argument("--input", default="benchmarks/experiments-v1.json")
    experiments.add_argument("--backend", choices=("exhaustive", "z3"), default="exhaustive")
    bundle = subs.add_parser("verify-bundle")
    bundle.add_argument("--input", required=True)
    bundle.add_argument("--expected-manifest-digest")

    args = parser.parse_args()
    if args.command in {"verify", "research", "experiment", "benchmark", "experiment-benchmark", "verify-bundle"}:
        raise SystemExit(_research_command(args))
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


def _research_command(args) -> int:
    from .audit_bundle import MAX_BYTES, create_bundle, runtime_evidence, verify_bundle
    from .experiments import run_experiment, run_experiment_suite
    from .model_checking import verify_request
    from .research import run_research
    from .research_benchmark import run_benchmarks

    try:
        if args.command == "benchmark":
            payload = run_benchmarks(args.corpus, backend=args.backend)
            code = 0 if payload["all_passed"] else 1
        else:
            with Path(args.input).open("rb") as handle:
                raw = handle.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ValueError("input exceeds 16 MiB limit")
            if args.command == "verify-bundle":
                payload = verify_bundle(raw, expected_manifest_digest=args.expected_manifest_digest)
                code = 0 if payload["valid"] else 1
            else:
                def unique_keys(pairs):
                    result = {}
                    for key, value in pairs:
                        if key in result:
                            raise ValueError("duplicate JSON request key")
                        result[key] = value
                    return result

                request = json.loads(raw, object_pairs_hook=unique_keys)
                if args.command == "experiment-benchmark":
                    payload = run_experiment_suite(request, backend=args.backend)
                    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
                    return 0 if payload["all_passed"] else 1
                runner = {"verify": verify_request, "research": run_research, "experiment": run_experiment}[args.command]
                payload = runner(request)
                code = 0 if payload["status"] in {"verified_bounded", "compliant", "definitely_true", "equivalent_bounded"} else 1
                if args.bundle:
                    archive, root = create_bundle({"request.json": request, "result.json": payload,
                                                   "runtime.json": runtime_evidence(revision=args.revision)},
                                                  metadata={"kind": args.command, "legal_certification": False})
                    # Refuse to silently replace an existing evidence bundle.
                    with Path(args.bundle).open("xb") as handle:
                        handle.write(archive)
                    payload = {**payload, "bundle_manifest_digest": root}
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        payload, code = {"error": str(exc)}, 2
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    return code


if __name__ == "__main__":
    main()
