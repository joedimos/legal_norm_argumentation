"""Small same-origin operations interface over the governed workflow API."""
from fastapi.responses import HTMLResponse

PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inference and legal review</title><style>
body{font:16px system-ui;background:#0a1420;color:#e4eef5;max-width:1100px;margin:auto;padding:24px}a{color:#77e0c4}
section{border:1px solid #34485a;padding:20px;border-radius:12px;margin:20px 0}input,select,textarea,button{font:inherit;margin:6px 0;padding:9px;border-radius:6px;background:#12283a;color:inherit;border:1px solid #486077}
textarea{box-sizing:border-box;width:100%;min-height:150px}button{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#07101a;padding:14px}label{display:block}small{color:#a9bfce}
</style><a href="/">← Source operations</a><h1>Inference and independent legal review</h1>
<p>Submit a normative family, obtain a separate review, then queue inference. Generated drafts require another review. Reviewed operation is enabled by default. Strict release policy additionally requires external release and signed case-review evidence.</p>
<label>API key <input id="key" type="password" autocomplete="off"></label><button id="connect">Connect</button><span id="identity"></span>
<section><h2>1. Normative family</h2><p>Paste a family request without approval receipts or draft answers.</p><textarea id="family" aria-label="Family JSON"></textarea><button id="submitFamily">Store family</button></section>
<section><h2>2. Reviewer decision</h2><label>Workflow ID <input id="reviewId"></label><label>Review scope <select id="scope"><option>family</option><option>grounding</option><option>behavior</option></select></label>
<label>Decision <select id="decision"><option>approved</option><option>needs_revision</option><option>rejected</option></select></label><label>Reason <textarea id="notes"></textarea></label><button id="review">Record review</button><small>A separate reviewer identity is required. Review the complete stored family or draft first.</small></section>
<section><h2>3. Model draft</h2><label>Family workflow ID <input id="familyId"></label><label>Request <textarea id="prompt"></textarea></label><label>Previous draft workflow IDs, in order <input id="history" placeholder="Comma-separated, optional"></label><label>Failed/interrupted job to retry (optional) <input id="retryId"></label><label>Retry investigation reason <input id="retryReason"></label><button id="generate">Queue draft</button><p>The server selects the model; a separate worker processes the durable job. Refresh to inspect the result.</p></section>
<section><h2>4. Signed case review</h2><label>Source ID <input id="source"></label><label>Evaluation context <textarea id="context">{"facts":{},"observed_actions":[],"actor":"","capacity":"","jurisdiction":"FR","at_date":null,"high_stakes":true}</textarea></label><button id="target">Prepare exact review target</button>
<p>Give the complete target to the independent reviewers. Both submissions must sign its target hash. Upload the completed signed assessment below; signing keys stay on the reviewers’ machines.</p>
<label>Signed case upload <textarea id="signed" placeholder='{"target_sha256":"…","assessment":{…}}'></textarea></label><button id="uploadSigned">Verify and store assessment</button>
<label>Verified assessment workflow ID <input id="assessmentId"></label><button id="evaluate">Evaluate with release gates</button></section>
<section><h2>Persistent work</h2><button id="refresh">Refresh</button><label>Workflow ID <input id="inspectId"></label><button id="inspect">Inspect</button><button id="export">Download audit bundle</button><div id="jobs"></div><pre id="output" role="status">Connect to load work.</pre></section>
<script src="/workflow-console.js" defer></script></html>'''

SCRIPT = '''const $=id=>document.getElementById(id);let token=sessionStorage.getItem('vle_api_key')||'';$('key').value=token;
function show(x){$('output').textContent=JSON.stringify(x,null,2);}function body(id){return JSON.parse($(id).value);}
async function call(path,data){const headers={Authorization:'Bearer '+token};const opts={headers};if(data!==undefined){opts.method='POST';headers['Content-Type']='application/json';headers['Idempotency-Key']=crypto.randomUUID();opts.body=JSON.stringify(data);}const r=await fetch(path,opts);const value=await r.json();show(value);if(!r.ok)throw new Error(value.detail||'Request blocked');return value;}
function bind(id,fn){$(id).onclick=async()=>{try{await fn();}catch(e){$('output').textContent=e.message;}};}
bind('connect',async()=>{token=$('key').value.trim();sessionStorage.setItem('vle_api_key',token);const r=await call('/v1/session');$('identity').textContent=r.subject+' · '+r.roles.join(', ');});
bind('submitFamily',async()=>{const r=await call('/v1/workflows/families',{family:body('family')});$('familyId').value=r.id;$('reviewId').value=r.id;});
bind('review',()=>call('/v1/workflows/'+encodeURIComponent($('reviewId').value)+'/reviews',{scope:$('scope').value,decision:$('decision').value,notes:$('notes').value}));
bind('generate',()=>call('/v1/workflows/generations',{family_workflow_id:$('familyId').value,prompt:$('prompt').value,history_ids:$('history').value.split(',').map(x=>x.trim()).filter(Boolean),max_cycles:5,...($('retryId').value?{retry_of:$('retryId').value,retry_reason:$('retryReason').value}:{})}));
bind('refresh',async()=>{const r=await call('/v1/workflows');$('jobs').textContent=r.workflows.map(x=>x.kind+' · '+x.status+' · '+x.id).join('\\n');});
bind('inspect',async()=>{const r=await call('/v1/workflows/'+encodeURIComponent($('inspectId').value));$('reviewId').value=r.id;});
bind('target',()=>call('/v1/sources/'+encodeURIComponent($('source').value)+'/release-target',body('context')));
bind('uploadSigned',async()=>{const r=await call('/v1/sources/'+encodeURIComponent($('source').value)+'/legal-assessments',body('signed'));$('assessmentId').value=r.id;});
bind('evaluate',()=>call('/v1/sources/'+encodeURIComponent($('source').value)+'/evaluate',{...body('context'),high_stakes:true,legal_assessment_id:$('assessmentId').value||null}));
bind('export',async()=>{const r=await fetch('/v1/workflows/'+encodeURIComponent($('inspectId').value)+'/audit-bundle',{headers:{Authorization:'Bearer '+token}});if(!r.ok)throw new Error('Audit access denied');const url=URL.createObjectURL(await r.blob());const a=document.createElement('a');a.href=url;a.download='workflow-evidence.zip';a.click();URL.revokeObjectURL(url);});'''


def register_ui(app):
    @app.get("/workflows", include_in_schema=False)
    def workflows_page():
        return HTMLResponse(PAGE)

    @app.get("/workflow-console.js", include_in_schema=False)
    def workflow_script():
        from fastapi.responses import Response
        return Response(SCRIPT, media_type="application/javascript", headers={"Cache-Control": "no-store"})
