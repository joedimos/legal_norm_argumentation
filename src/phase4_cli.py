"""Additive Phase 4 CLI dispatch; never writes human review state."""
from __future__ import annotations

import json
from pathlib import Path

from .audit_bundle import MAX_BYTES, create_bundle
from .model_runtime import strict_json

COMMANDS = ('aspic', 'redundant-formalize', 'redundancy-report', 'assumptions', 'ddl')


def configure(subs):
    for name in COMMANDS:
        p = subs.add_parser(name)
        if name == 'redundant-formalize':
            p.add_argument('--source-id', required=True)
            p.add_argument('--text-file', required=True)
            p.add_argument('--candidates', type=int, default=3)
            p.add_argument('--provider', choices=('fake', 'ollama', 'vllm'), default='fake')
            p.add_argument('--model')
        else:
            p.add_argument('--input', required=True)
        p.add_argument('--bundle')
        if name == 'assumptions':
            p.add_argument('--backend', choices=('exhaustive', 'z3'))
            p.add_argument('--max-mcs', type=int)
        if name == 'aspic':
            p.add_argument('--lifting', choices=('last', 'weakest'), default=None)


def execute(args):
    from .aspic import analyze_aspic

    try:
        path = args.text_file if args.command == 'redundant-formalize' else args.input
        with Path(path).open('rb') as f:
            raw = f.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError('input exceeds 16 MiB')
        if args.command == 'redundant-formalize':
            from .redundancy.generate import RuntimeProvider, baseline_fake, generate_formalizations
            text = raw.decode()
            config = {'provider': args.provider}
            if args.model:
                config['model'] = args.model
            provider = baseline_fake(text, args.source_id) if args.provider == 'fake' else RuntimeProvider(config)
            request = {'source_id': args.source_id, 'text': text}
            result = generate_formalizations(text, args.source_id, provider, candidates=args.candidates)
        else:
            request = strict_json(raw)
            if args.command == 'aspic':
                result = analyze_aspic(request, lifting=args.lifting)
            elif args.command == 'ddl':
                from .ddl import interchange
                result = interchange(request)
            elif args.command == 'assumptions':
                from .assumptions import assumption_report
                result = assumption_report(request, backend=args.backend, max_mcs=args.max_mcs)
            else:
                from .redundancy.crosscheck import consensus_report
                if request.get('schema') == 'vle-redundancy-generation/v1':
                    if request.get('failures') or request.get('status') != 'unreviewed':
                        raise ValueError('incomplete generation cannot be silently reduced to surviving candidates')
                    request = {'schema': 'vle-redundancy/v1', 'candidates': request['candidates'], 'horizon': 1}
                result = consensus_report(request)
        if args.bundle:
            from .audit_bundle import redundancy_artifacts
            artifacts = (redundancy_artifacts(request, result) if args.command == 'redundancy-report' else
                         {'request.json': request, 'result.json': result})
            archive, root = create_bundle(artifacts)
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
