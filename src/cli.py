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
from .resources import resource_path
from .verification import audit_extraction


def _load_json(value: str) -> Any:
    from .audit_bundle import MAX_BYTES
    from .json_boundary import strict_json

    path = Path(value)
    if path.exists():
        with path.open("rb") as handle:
            raw = handle.read(MAX_BYTES + 1)
    else:
        raw = value.encode("utf-8")
    if len(raw) > MAX_BYTES:
        raise ValueError("input exceeds 16 MiB limit")
    return strict_json(raw)


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
    comply.add_argument("--reasoning-backend", choices=("core", "pyreason"), default="core")
    comply.add_argument("--reasoning-timeout-seconds", type=int, default=60)

    lifecycle = subs.add_parser("lifecycle")
    _add_source_args(lifecycle)
    lifecycle.add_argument("--facts", default="{}")
    lifecycle.add_argument("--actions", default="[]")
    lifecycle.add_argument("--at-date")
    lifecycle.add_argument("--assume-reviewed", action="store_true", help="Demo/testing convenience only")
    lifecycle.add_argument("--reasoning-backend", choices=("core", "pyreason"), default="core")
    lifecycle.add_argument("--reasoning-timeout-seconds", type=int, default=60)

    for name in (
        "verify",
        "research",
        "experiment",
        "plan-evidence",
        "family",
        "statutory-revise",
        "live-generate",
        "annotation-prepare",
        "annotation-assess",
        "model-doctor",
        "annotation-verify",
        "norm-evaluate",
        "pyreason-infer",
    ):
        command = subs.add_parser(name)
        command.add_argument("--input", required=True, help="Path to a versioned JSON research request")
        command.add_argument("--bundle", help="Write an audit ZIP containing request, result and runtime")
        command.add_argument("--revision", help="Full Git SHA of the executing build")
    subs.choices["norm-evaluate"].add_argument("--backend", choices=("core", "pyreason"))
    subs.choices["norm-evaluate"].add_argument("--timeout-seconds", type=int)
    subs.choices["pyreason-infer"].add_argument("--timeout-seconds", type=int, default=60)
    subs.choices["pyreason-infer"].add_argument("--worker-mode", choices=("isolated", "pooled"), default="isolated")
    signed = subs.choices["annotation-verify"]
    signed.add_argument("--registry", required=True, help="Public reviewer registry checked outside this assessment")
    signed.add_argument("--expected-registry-sha256", required=True)
    signed.add_argument("--expected-packet-sha256", required=True)
    template = subs.add_parser("annotation-template")
    template.add_argument("--input", required=True)
    template.add_argument("--reviewer", required=True)
    template.add_argument("--qualification", required=True)
    signing = subs.add_parser("annotation-sign")
    signing.add_argument("--input", required=True)
    signing.add_argument("--private-key-file", required=True)
    signing.add_argument("--packet-sha256", required=True)
    signing.add_argument("--role", choices=("annotator", "adjudicator"), required=True)
    keygen = subs.add_parser("annotation-keygen")
    keygen.add_argument("--private-key-file", required=True)
    keygen.add_argument("--public-key-file", required=True)
    benchmark = subs.add_parser("benchmark")
    benchmark.add_argument("--corpus", default=str(resource_path("benchmarks", "research-v1.json")))
    benchmark.add_argument("--backend", choices=("exhaustive", "z3"), default="exhaustive")
    experiments = subs.add_parser("experiment-benchmark")
    experiments.add_argument("--input", default=str(resource_path("benchmarks", "experiments-v1.json")))
    experiments.add_argument("--backend", choices=("exhaustive", "z3"), default="exhaustive")
    families = subs.add_parser("family-benchmark")
    families.add_argument("--input", default=str(resource_path("benchmarks", "families-v1.json")))
    review = subs.add_parser("family-review")
    review.add_argument("--input", required=True)
    review.add_argument("--scope", choices=("family", "grounding", "behavior"), required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--notes", required=True)
    review.add_argument("--decision", choices=("approved", "rejected", "needs_revision"), default="approved")
    bundle = subs.add_parser("verify-bundle")
    bundle.add_argument("--input", required=True)
    bundle.add_argument("--expected-manifest-digest")
    bundle.add_argument("--expected-certificate-query-sha256", action="append")

    from .phase4_cli import COMMANDS, configure, execute

    configure(subs)
    from .phase5_cli import COMMANDS as PHASE5_COMMANDS
    from .phase5_cli import configure as configure_phase5
    from .phase5_cli import execute as execute_phase5

    configure_phase5(subs)
    args = parser.parse_args()
    if args.command in PHASE5_COMMANDS:
        raise SystemExit(execute_phase5(args))
    if args.command in COMMANDS:
        raise SystemExit(execute(args))
    if args.command in {
        "verify",
        "research",
        "experiment",
        "plan-evidence",
        "family",
        "statutory-revise",
        "family-review",
        "family-benchmark",
        "benchmark",
        "experiment-benchmark",
        "verify-bundle",
        "live-generate",
        "annotation-prepare",
        "annotation-assess",
        "model-doctor",
        "annotation-verify",
        "annotation-template",
        "annotation-sign",
        "annotation-keygen",
        "norm-evaluate",
        "pyreason-infer",
    }:
        raise SystemExit(_research_command(args))
    text = _read_text(args)
    norms = extract_norms(text, source_id=args.source_id)

    if args.command == "extract":
        payload = extract_all_norms(text, source_id=args.source_id)
    elif args.command == "audit":
        payload = {"norms": norms, "audit": audit_extraction(text, norms)}
    elif args.command == "formalize":
        sets = formalize_candidates(norms)
        payload = {
            "formalizations": [
                {
                    "source_norm_id": item.source_norm_id,
                    "selected_id": item.selected_id,
                    "unresolved": item.unresolved,
                    "interpretations": [x.to_dict() for x in item.interpretations],
                }
                for item in sets
            ]
        }
    else:
        facts = _load_json(args.facts)
        actions = tuple(_load_json(args.actions))
        context = FactContext(facts=facts, observed_actions=actions, at_date=args.at_date)
        if args.command == "comply":
            sets = formalize_candidates(norms)
            selected = [item.selected() for item in sets if item.selected() is not None]
            payload = {
                "unresolved_norm_ids": [item.source_norm_id for item in sets if item.unresolved],
                "compliance": evaluate(selected, context, backend=args.reasoning_backend,
                                       pyreason_timeout_seconds=args.reasoning_timeout_seconds)
                              if selected else {"status": "indeterminate", "findings": []},
                **({"decision_authorized": False, "review_status": "unreviewed"}
                   if args.reasoning_backend == "pyreason" else {}),
            }
        else:
            audit = audit_extraction(text, norms)
            statuses = ["approved"] * len(norms) if args.assume_reviewed else ["unreviewed"] * len(norms)
            payload = run_lifecycle(
                text=text,
                source_id=args.source_id,
                candidates=norms,
                extraction_audit=audit,
                context=context,
                review_statuses=statuses,
                reasoning_backend=args.reasoning_backend,
                pyreason_timeout_seconds=args.reasoning_timeout_seconds,
            )

    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _research_command(args) -> int:
    from .audit_bundle import MAX_BYTES, create_bundle, runtime_evidence, verify_bundle
    from .evidence_planning import plan_evidence
    from .experiments import run_experiment, run_experiment_suite
    from .family_contract import record_family_review
    from .interpretation_family import analyze_family, run_family_benchmark
    from .json_boundary import strict_json
    from .legal_annotation import annotation_template, assess_annotations, prepare_annotation
    from .live_generation import generate_candidate
    from .model_checking import verify_request
    from .model_runtime import model_doctor
    from .normative_evaluation import evaluate_request
    from .pyreason_connector import worker_transport
    from .pyreason_graph import infer_graph
    from .research import run_research
    from .research_benchmark import run_benchmarks
    from .response_alignment import run_revision_sequence
    from .signed_annotation import assess_signed_annotations, create_reviewer_key, sign_judgment

    try:
        if args.command == "annotation-keygen":
            print(json.dumps(create_reviewer_key(args.private_key_file, args.public_key_file)))
            return 0
        if args.command == "benchmark":
            payload = run_benchmarks(args.corpus, backend=args.backend)
            code = 0 if payload["all_passed"] else 1
        else:
            with Path(args.input).open("rb") as handle:
                raw = handle.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ValueError("input exceeds 16 MiB limit")
            if args.command == "verify-bundle":
                payload = verify_bundle(
                    raw,
                    expected_manifest_digest=args.expected_manifest_digest,
                    expected_certificate_query_sha256s=args.expected_certificate_query_sha256,
                )
                code = 0 if payload["valid"] else 1
            else:

                request = strict_json(raw)
                registry = None
                if args.command == "annotation-template":
                    payload = annotation_template(request, reviewer_id=args.reviewer, qualification=args.qualification)
                    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
                    return 0
                if args.command == "annotation-sign":
                    payload = sign_judgment(
                        request,
                        packet_sha256=args.packet_sha256,
                        role=args.role,
                        private_key_file=args.private_key_file,
                    )
                    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
                    return 0
                if args.command == "annotation-verify":
                    with Path(args.registry).open("rb") as handle:
                        registry_raw = handle.read(MAX_BYTES + 1)
                    if len(registry_raw) > MAX_BYTES:
                        raise ValueError("registry exceeds 16 MiB")
                    registry = strict_json(registry_raw)
                if args.command == "experiment-benchmark":
                    payload = run_experiment_suite(request, backend=args.backend)
                    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
                    return 0 if payload["all_passed"] else 1
                if args.command in {"family-review", "family-benchmark"}:
                    payload = (
                        run_family_benchmark(request)
                        if args.command == "family-benchmark"
                        else record_family_review(
                            request, scope=args.scope, reviewer=args.reviewer, notes=args.notes, decision=args.decision
                        )
                    )
                    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
                    return 0 if args.command == "family-review" or payload["all_passed"] else 1
                def graph_inference(r):
                    with worker_transport(args.worker_mode):
                        return infer_graph(r, timeout_seconds=args.timeout_seconds)

                runner = {
                    "pyreason-infer": graph_inference,
                    "norm-evaluate": lambda r: evaluate_request(r, backend=args.backend, timeout_seconds=args.timeout_seconds),
                    "model-doctor": model_doctor,
                    "annotation-verify": lambda r: assess_signed_annotations(
                        r,
                        registry=registry,
                        expected_registry_sha256=args.expected_registry_sha256,
                        expected_packet_sha256=args.expected_packet_sha256,
                    ),
                    "live-generate": generate_candidate,
                    "annotation-prepare": prepare_annotation,
                    "annotation-assess": assess_annotations,
                    "statutory-revise": run_revision_sequence,
                    "family": analyze_family,
                    "verify": verify_request,
                    "research": run_research,
                    "experiment": run_experiment,
                    "plan-evidence": plan_evidence,
                }[args.command]
                payload = runner(request)
                code = (
                    0
                    if payload.get("status", payload.get("annotation_status"))
                    in {
                        "model_available",
                        "computed",
                        "awaiting_independent_annotations",
                        "annotations_collected",
                        "verified_bounded",
                        "compliant",
                        "definitely_true",
                        "equivalent_bounded",
                        "planned",
                        "already_decided",
                        "robust_satisfied",
                        "model_satisfied",
                    }
                    else 1
                )
                if args.bundle:
                    artifacts = {
                        "request.json": request,
                        "result.json": payload,
                        "runtime.json": runtime_evidence(revision=args.revision),
                    }
                    if registry is not None:
                        artifacts["reviewer-registry.json"] = registry
                    archive, root = create_bundle(
                        artifacts, metadata={"kind": args.command, "legal_certification": False}
                    )
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
