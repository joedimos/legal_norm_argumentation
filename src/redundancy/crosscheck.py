"""Shared-horizon semantic inclusion; consensus never selects an interpretation."""
from __future__ import annotations

from itertools import combinations

from ..experiments import run_experiment
from ..formalization import behaviorally_equivalent
from ..model_checking import verify
from ..norm_codec import decode_norm
from ..phase4 import envelope, fields, integer
from ..temporal import combine
from ..temporal_deontic import TemporalNorm
from .align import align


def consensus_report(request):
    fields(request, 'vle-redundancy/v1', {'candidates', 'alignment', 'horizon', 'k', 'backend',
           'max_traces', 'timeout_ms', 'source'}, {'candidates', 'horizon'})
    candidates = request['candidates']
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 8:
        raise ValueError('requires 2-8 candidate objects')
    norms = [decode_norm(c['norm']) for c in candidates]
    if len({n.id for n in norms}) != len(norms) or len({n.source_norm_id for n in norms}) != 1:
        raise ValueError('unique candidates must formalize the same source norm')
    horizon = integer(request['horizon'], 1, 33, 'horizon')
    k = integer(request.get('k', len(norms)), 2, len(norms), 'k')
    max_traces = integer(request.get('max_traces', 65536), 1, 1000000, 'max_traces')
    timeout = integer(request.get('timeout_ms', 5000), 1, 60000, 'timeout_ms')
    backend = request.get('backend', 'exhaustive')
    if backend not in {'exhaustive', 'z3'}:
        raise ValueError('invalid backend')
    norms, alignment = align(norms, explicit=request.get('alignment'))
    pairs, edges = [], set()
    for left, right in combinations(norms, 2):
        classification, evidence, witnesses = 'unknown', {}, []
        reason = 'unresolved_alignment' if alignment['unresolved'] else None
        if not reason and any(n.deadline or n.references or n.assumptions
                              or n.authority.effective_from or n.authority.effective_to for n in (left, right)):
            reason = 'unmodeled_deadline_reference_assumption_or_calendar_scope'
        if not reason:
            try:
                lt, rt = TemporalNorm(left, 0, horizon - 1), TemporalNorm(right, 0, horizon - 1)
                evidence = run_experiment({'schema': 'vle-experiment/v1', 'kind': 'amendment',
                    'before': lt.constraint().to_dict(), 'after': rt.constraint().to_dict(),
                    'horizon': horizon, 'backend': backend, 'max_traces': max_traces, 'timeout_ms': timeout})
                classification = {'equivalent_bounded': 'equivalent', 'strict_strengthening': 'strictly_weaker',
                    'strict_weakening': 'strictly_stronger', 'incomparable': 'incomparable'}.get(evidence['status'], 'unknown')
                witnesses = [{'kind': key, 'trace': evidence[key], 'horizon': horizon}
                             for key in ('removed_behavior', 'added_behavior') if evidence.get(key) is not None]
                if left.modality != right.modality:
                    # Hard satisfaction alone maps permission to True. Preserve a
                    # separate normative-output distinction with activation evidence.
                    guard = combine('or', lt.guard(), rt.guard())
                    w = verify(combine('not', guard), horizon=horizon, backend=backend,
                               max_traces=max_traces, timeout_ms=timeout)
                    classification = 'incomparable' if w.counterexample is not None else 'unknown'
                    witnesses.append({'kind': 'modal_distinction', 'trace': w.counterexample,
                                      'left_modality': left.modality.value, 'right_modality': right.modality.value,
                                      'horizon': horizon})
                if classification == 'unknown':
                    reason = 'incomplete_or_unsupported_comparison'
            except ValueError:
                reason = 'unsupported_temporal_fragment'
        if classification == 'equivalent':
            edges.add((left.id, right.id))
        contexts = [w['trace'][0] for w in witnesses if w.get('trace')]
        pairs.append({'left': left.id, 'right': right.id, 'classification': classification,
                      'reason': reason, 'evidence': evidence, 'witnesses': witnesses,
                      'signature_match_on_witnesses': behaviorally_equivalent(left, right, contexts) if contexts else None})
    # Build only cliques. Unknown pair checks never become equivalent via a path.
    classes = []
    for n in sorted(norms, key=lambda n: n.id):
        group = next((c for c in classes if all(tuple(sorted((n.id, x))) in {
            tuple(sorted(e)) for e in edges} for x in c)), None)
        if group is None:
            classes.append([n.id])
        else:
            group.append(n.id)
    classes.sort(key=lambda c: (-len(c), c))
    largest = [c for c in classes if len(c) == len(classes[0])]
    complete = not alignment['unresolved'] and all(p['classification'] != 'unknown' for p in pairs)
    supported = complete and len(classes[0]) >= k
    return envelope('vle-redundancy-report/v1', request,
        status='consensus_supported' if supported else 'review_required' if complete else 'unknown',
        consensus_supported=supported, selected_id=None, required_agreement=k, candidates=candidates,
        alignment=alignment, pairs=pairs, equivalence_classes=classes, largest_classes=largest,
        bounds={'horizon': horizon, 'max_candidates': 8, 'backend': backend,
                'max_traces_per_check': max_traces, 'timeout_ms_per_check': timeout,
                'max_solver_calls': len(pairs) * 3},
        review_prompts=[{'kind': 'redundancy_divergence', 'pair': [p['left'], p['right']],
                         'witnesses': p['witnesses'], 'reason': p['reason']} for p in pairs
                        if p['classification'] != 'equivalent'])
