"""Additive research CLI; never records human approvals."""

from __future__ import annotations

import json
from pathlib import Path

from .audit_bundle import MAX_BYTES, create_bundle
from .model_runtime import strict_json
from .phase4 import envelope

COMMANDS = ("certify", "check-certificate", "obstruction", "metamorphic", "plan-review")


def configure(subs):
    for name in COMMANDS:
        p = subs.add_parser(name)
        p.add_argument("--input", required=True)
        p.add_argument("--bundle")
        if name == "metamorphic":
            p.add_argument("--provider", choices=("extractor", "fake", "ollama", "vllm"), default="fake")
        if name == "obstruction":
            p.add_argument("--backend", choices=("exhaustive", "z3"))
        if name == "check-certificate":
            p.add_argument("--expected-query-sha256")


def execute(args):
    try:
        with Path(args.input).open("rb") as f:
            raw = f.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("input exceeds16MiB")
        request = strict_json(raw)
        if args.command == "certify":
            from .certificates import certify

            result = certify(request)
        elif args.command == "plan-review":
            from .review_planning import plan_review

            result = plan_review(request)
        elif args.command == "metamorphic":
            from .metamorphic import metamorphic

            result = metamorphic(request, provider=args.provider)
        elif args.command == "obstruction":
            from .obstruction import obstruction

            result = obstruction(request, backend=args.backend)
        else:
            from .checker import check_certificate

            result = envelope(
                "vle-check-result/v1",
                request,
                **check_certificate(request, expected_query_sha256=args.expected_query_sha256),
            )
        if args.bundle:
            artifacts = {"request.json": request, "result.json": result}
            if "certificate" in result:
                artifacts["certificate.json"] = result["certificate"]
            archive, root = create_bundle(artifacts)
            with Path(args.bundle).open("xb") as f:
                f.write(archive)
            import sys

            print(json.dumps({"bundle_manifest_digest": root}), file=sys.stderr)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 1 if result.get("valid") is False or result.get("status") == "unknown" else 0
    except (ValueError, TypeError, KeyError, OSError, RecursionError) as exc:
        print(json.dumps(envelope("vle-phase5-error/v1", {"command": args.command}, status="unknown", reason=str(exc))))
        return 2
