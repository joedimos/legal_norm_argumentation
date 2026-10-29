"""Local DDL interchange, segmentation, refinement proposals and metric vectors."""


def interchange(request):
    from ..norm_codec import decode_norm
    from ..phase4 import envelope, fields
    from .export import export_document, export_norms
    from .import_ import import_document, import_exported
    from .metrics import metric_vector
    from .refine import propose_refinement
    from .segment import segment

    fields(request, 'vle-ddl/v1', {'operation', 'document', 'norms', 'text', 'source_id', 'exported',
           'before', 'after', 'known_predicates', 'references', 'reference_index', 'expected', 'extracted_candidates'}, {'operation'})
    op = request['operation']
    if op == 'export':
        result = export_norms([decode_norm(n) for n in request['norms']]) if 'norms' in request else export_document(request['document'])
    elif op == 'import':
        result = import_exported(request['exported']) if 'exported' in request else import_document(request['text'])
    elif op == 'segment':
        result = {'snippets': segment(request['text'], request['source_id']), 'lossy_fields': []}
    elif op == 'refine':
        result = {'proposal': propose_refinement(request['before'], request['after'], request['text'],
            known_predicates=request.get('known_predicates'), references=request.get('references', []),
            reference_index=request.get('reference_index')), 'lossy_fields': []}
    elif op == 'metrics':
        result = {'metrics': metric_vector(request['document'], request['expected'], source_text=request.get('text', ''),
                                           extracted_candidates=request.get('extracted_candidates', [])), 'lossy_fields': []}
    else:
        raise ValueError('invalid DDL operation')
    return envelope('vle-ddl-interchange/v1', request, status='unreviewed', operation=op, **result)
