"""Additive Phase 4 CLI dispatch; never writes human review state."""
from __future__ import annotations

import json
from pathlib import Path

from .audit_bundle import MAX_BYTES, create_bundle
from .model_runtime import strict_json

COMMANDS = ('aspic',)


def configure(subs):
    for name in COMMANDS:
        p = subs.add_parser(name)
        p.add_argument('--input', required=True)
        p.add_argument('--bundle')
        if name == 'aspic':
            p.add_argument('--lifting', choices=('last', 'weakest'), default=None)


def execute(args):
    from .aspic import analyze_aspic

    try:
        with Path(args.input).open('rb') as f:
            raw = f.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError('input exceeds 16 MiB')
        request = strict_json(raw)
        result = analyze_aspic(request, lifting=args.lifting)
        if args.bundle:
            archive, root = create_bundle({'request.json': request, 'result.json': result})
            with Path(args.bundle).open('xb') as f:
                f.write(archive)
            # Digest belongs in stderr so stdout remains the versioned report.
            import sys
            print(json.dumps({'bundle_manifest_digest': root}), file=sys.stderr)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if result['status'] == 'unknown' else 0
    except (ValueError, TypeError, KeyError, OSError, RecursionError) as exc:
        print(json.dumps({'error': str(exc)}))
        return 2
