from __future__ import annotations

HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>VLE–LEGATO | Norm Intelligence</title>
  <style>
    :root{--bg:#081018;--panel:#0d1823;--panel2:#111f2c;--line:#243443;--text:#eef5f8;--muted:#8fa4b3;--accent:#64dfc4;--accent2:#83a8ff;--warn:#ffd166;--bad:#ff7b7b;--shadow:0 24px 70px rgba(0,0,0,.28);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    *{box-sizing:border-box} body{margin:0;background:radial-gradient(circle at 18% -10%,rgba(100,223,196,.12),transparent 32%),radial-gradient(circle at 82% 4%,rgba(131,168,255,.12),transparent 28%),var(--bg);color:var(--text);min-height:100vh} button,input,select{font:inherit} button{cursor:pointer}
    .shell{display:grid;grid-template-columns:248px minmax(0,1fr);min-height:100vh}.sidebar{border-right:1px solid var(--line);padding:26px 18px;position:sticky;top:0;height:100vh;background:rgba(8,16,24,.82);backdrop-filter:blur(16px)}
    .brand{display:flex;gap:12px;align-items:center;margin:0 8px 34px}.mark{width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,var(--accent),var(--accent2));display:grid;place-items:center;color:#071018;font-weight:900;box-shadow:0 8px 30px rgba(100,223,196,.2)}.brand strong{display:block;letter-spacing:.04em}.brand small{color:var(--muted)}
    .nav{display:grid;gap:7px}.nav button{border:0;background:transparent;color:var(--muted);padding:11px 12px;border-radius:10px;text-align:left}.nav button:hover,.nav button.active{background:var(--panel2);color:var(--text)}.nav button.active{box-shadow:inset 2px 0 0 var(--accent)}
    .sidefoot{position:absolute;left:18px;right:18px;bottom:24px;padding-top:16px;border-top:1px solid var(--line)}.status{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:13px}.dot{width:8px;height:8px;border-radius:50%;background:var(--warn)}.dot.ok{background:var(--accent)}.dot.bad{background:var(--bad)}
    main{padding:28px 34px 60px;min-width:0}.topbar{display:flex;justify-content:space-between;gap:20px;align-items:center;margin-bottom:30px}.topbar h1{font-size:24px;margin:0}.topbar p{color:var(--muted);margin:5px 0 0;font-size:14px}.auth{display:flex;gap:8px;align-items:center}.auth input{width:290px;background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:10px;padding:10px 12px;outline:none}.auth input:focus{border-color:var(--accent)}
    .btn{border:1px solid var(--line);background:var(--panel2);color:var(--text);padding:10px 14px;border-radius:10px;font-weight:650}.btn:hover{border-color:#3f5668}.btn.primary{background:linear-gradient(135deg,var(--accent),#53c9b2);color:#06110e;border-color:transparent}.btn.ghost{background:transparent}.btn:disabled{opacity:.45;cursor:not-allowed}
    .hero{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(260px,.7fr);gap:18px;margin-bottom:18px}.card{background:linear-gradient(180deg,rgba(17,31,44,.92),rgba(13,24,35,.92));border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}.hero-main{padding:26px;min-height:215px;display:flex;flex-direction:column;justify-content:space-between}.eyebrow{color:var(--accent);text-transform:uppercase;font-size:11px;letter-spacing:.16em;font-weight:800}.hero-main h2{font-size:35px;line-height:1.08;margin:9px 0 10px;max-width:740px}.hero-main p{color:var(--muted);max-width:760px;line-height:1.55;margin:0}.hero-actions{display:flex;gap:10px;margin-top:22px;flex-wrap:wrap}
    .metrics{padding:20px;display:grid;grid-template-columns:1fr 1fr;gap:12px}.metric{padding:14px;border:1px solid var(--line);border-radius:12px;background:rgba(8,16,24,.4)}.metric b{font-size:24px;display:block}.metric span{font-size:12px;color:var(--muted)}
    .grid{display:grid;grid-template-columns:repeat(12,1fr);gap:18px}.span8{grid-column:span 8}.span4{grid-column:span 4}.span12{grid-column:span 12}.section{padding:20px}.section-head{display:flex;align-items:center;justify-content:space-between;gap:14px;margin-bottom:16px}.section-head h3{margin:0;font-size:16px}.section-head p{margin:3px 0 0;color:var(--muted);font-size:13px}
    .corpora{display:grid;grid-template-columns:1fr 1fr;gap:12px}.corpus{padding:17px;border:1px solid var(--line);border-radius:13px;background:rgba(8,16,24,.35);position:relative}.corpus.selected{border-color:var(--accent);box-shadow:0 0 0 1px rgba(100,223,196,.18)}.corpus h4{margin:0 0 6px}.corpus p{color:var(--muted);font-size:13px;line-height:1.5;margin:0}.corpus .tag{display:inline-block;margin-top:12px;color:var(--accent);font-size:11px;text-transform:uppercase;letter-spacing:.12em}
    .controls{display:grid;gap:12px}.field label{display:block;color:var(--muted);font-size:12px;margin-bottom:6px}.field input,.field select{width:100%;background:#09131c;color:var(--text);border:1px solid var(--line);padding:10px 11px;border-radius:9px}.hint{font-size:12px;color:var(--muted);line-height:1.45}.result{font-size:13px;padding:12px;border:1px solid var(--line);border-radius:10px;background:#09131c;min-height:44px;color:var(--muted);white-space:pre-wrap}
    .table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px}table{width:100%;border-collapse:collapse;min-width:720px}th,td{padding:12px 13px;text-align:left;border-bottom:1px solid var(--line);font-size:13px}th{color:var(--muted);font-weight:600;background:#0a151f;position:sticky;top:0}tr:last-child td{border-bottom:0}tbody tr{cursor:pointer}tbody tr:hover{background:rgba(100,223,196,.04)}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.pill{display:inline-flex;align-items:center;padding:4px 7px;border-radius:999px;background:#132636;color:#a9c3d3;font-size:11px}.pill.green{background:rgba(100,223,196,.12);color:var(--accent)}
    .detail{min-height:280px}.empty{color:var(--muted);padding:28px 6px;text-align:center}.detail h4{margin:0 0 8px}.detail .meta{color:var(--muted);font-size:12px;margin-bottom:14px}.candidate{border:1px solid var(--line);border-radius:11px;padding:12px;margin:9px 0;background:rgba(8,16,24,.35)}.candidate strong{font-size:12px;color:var(--accent)}.candidate p{font-size:13px;line-height:1.45;margin:6px 0;color:#d9e7ed}.candidate small{color:var(--muted)}
    .hidden{display:none!important}.toast{position:fixed;right:24px;bottom:24px;background:#122331;border:1px solid var(--line);padding:12px 15px;border-radius:11px;box-shadow:var(--shadow);font-size:13px;max-width:360px;z-index:20}.toast.bad{border-color:rgba(255,123,123,.55)}
    @media(max-width:980px){.shell{grid-template-columns:1fr}.sidebar{height:auto;position:relative;border-right:0;border-bottom:1px solid var(--line);padding:16px}.brand{margin:0 0 14px}.nav{grid-template-columns:repeat(4,1fr)}.nav button{text-align:center;padding:9px 5px;font-size:12px}.sidefoot{display:none}main{padding:22px}.topbar{align-items:flex-start;flex-direction:column}.auth{width:100%}.auth input{flex:1;width:auto}.hero{grid-template-columns:1fr}.span8,.span4{grid-column:span 12}}
    @media(max-width:640px){main{padding:17px}.hero-main h2{font-size:28px}.corpora{grid-template-columns:1fr}.metrics{grid-template-columns:1fr 1fr}.nav button:nth-child(4){display:none}.auth{flex-wrap:wrap}.auth input{min-width:100%}}
  </style>
</head>
<body>
<div class="shell">
  <aside class="sidebar">
    <div class="brand"><div class="mark">VL</div><div><strong>VLE–LEGATO</strong><small>Norm intelligence</small></div></div>
    <nav class="nav" aria-label="Primary">
      <button class="active" data-jump="overview">Overview</button>
      <button data-jump="ingestion">Ingestion</button>
      <button data-jump="sources">Sources</button>
      <button data-jump="evidence">Evidence</button>
    </nav>
    <div class="sidefoot"><div class="status"><span id="healthDot" class="dot"></span><span id="healthText">Checking service…</span></div></div>
  </aside>
  <main>
    <header class="topbar">
      <div><h1>Legal Norm Operations</h1><p>Acquire, formalise, review and audit French normative sources.</p></div>
      <div class="auth"><input id="apiKey" type="password" autocomplete="off" placeholder="API key (session only)" aria-label="API key"><button id="saveKey" class="btn">Connect</button><button id="clearKey" class="btn ghost" title="Clear API key">Clear</button></div>
    </header>

    <section id="overview" class="hero">
      <div class="card hero-main">
        <div><div class="eyebrow">Research-grade legal reasoning</div><h2>From authoritative text to verifiable normative arguments.</h2><p>The console connects public Légifrance sources to the existing extraction, formalisation, human-review, temporal/deontic reasoning and audit-bundle pipeline.</p></div>
        <div class="hero-actions"><button id="jumpIngest" class="btn primary">Ingest Légifrance</button><button id="refreshSources" class="btn">Refresh sources</button></div>
      </div>
      <div class="card metrics" aria-label="Source metrics">
        <div class="metric"><b id="sourceCount">—</b><span>Visible source versions</span></div>
        <div class="metric"><b id="candidateCount">—</b><span>Extracted candidates</span></div>
        <div class="metric"><b id="reviewedCount">—</b><span>Reviewed candidates</span></div>
        <div class="metric"><b id="corpusCount">2</b><span>French legal corpora</span></div>
      </div>
    </section>

    <div class="grid">
      <section id="ingestion" class="card section span8">
        <div class="section-head"><div><h3>Public Légifrance ingestion</h3><p>Fixed, bounded collection from official French code pages.</p></div><span class="pill green">Official source</span></div>
        <div id="corpora" class="corpora"><div class="empty">Loading corpora…</div></div>
      </section>
      <section class="card section span4">
        <div class="section-head"><div><h3>Run collection</h3><p>Controlled batch parameters</p></div></div>
        <div class="controls">
          <div class="field"><label for="limit">Articles this run</label><input id="limit" type="number" min="1" max="50" value="20"></div>
          <div class="field"><label for="offset">Article offset</label><input id="offset" type="number" min="0" value="0"></div>
          <button id="scrape" class="btn primary">Collect & ingest</button>
          <div class="hint">The server follows only approved <span class="mono">legifrance.gouv.fr/codes/</span> URLs and records provenance inside each source extraction audit.</div>
          <div id="scrapeResult" class="result">No collection run yet.</div>
        </div>
      </section>

      <section id="sources" class="card section span8">
        <div class="section-head"><div><h3>Source registry</h3><p>Newest source versions first</p></div><select id="sourceFilter" class="btn" aria-label="Source filter"><option value="legifrance:">Légifrance</option><option value="">All sources</option></select></div>
        <div class="table-wrap"><table><thead><tr><th>Source</th><th>Created</th><th>Candidates</th><th>Review state</th></tr></thead><tbody id="sourceRows"><tr><td colspan="4" class="empty">Connect an API key to load sources.</td></tr></tbody></table></div>
      </section>
      <section id="evidence" class="card section span4 detail">
        <div class="section-head"><div><h3>Norm detail</h3><p>Candidate and provenance view</p></div></div>
        <div id="detail"><div class="empty">Select a source to inspect extracted norms.</div></div>
      </section>
    </div>
  </main>
</div>
<div id="toast" class="toast hidden" role="status" aria-live="polite"></div>
<script>
(() => {
  const $ = (id) => document.getElementById(id);
  const state = { corpus: 'ip', sources: [] };
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const token = () => sessionStorage.getItem('vleApiKey') || '';
  const headers = (json=false) => ({...(token()?{Authorization:`Bearer ${token()}`} : {}), ...(json?{'Content-Type':'application/json'}:{})});
  const toast = (message,bad=false) => { const el=$('toast'); el.textContent=message; el.className=`toast${bad?' bad':''}`; setTimeout(()=>el.classList.add('hidden'),3200); };
  async function api(path, options={}) { const r=await fetch(path,options); const ct=r.headers.get('content-type')||''; const body=ct.includes('json')?await r.json():await r.text(); if(!r.ok) throw new Error(body?.detail||body||`HTTP ${r.status}`); return body; }
  async function health(){try{const d=await api('/health/live');$('healthDot').className='dot ok';$('healthText').textContent=`Online · ${d.version}`;}catch{$('healthDot').className='dot bad';$('healthText').textContent='Service unavailable';}}
  async function loadCorpora(){try{const d=await api('/v1/legifrance/corpora'); $('corpora').innerHTML=d.corpora.map(c=>`<button class="corpus ${c.key===state.corpus?'selected':''}" data-corpus="${esc(c.key)}" style="text-align:left;color:inherit"><h4>${esc(c.label)}</h4><p>${esc(c.description)}</p><span class="tag">${esc(c.legal_scope)}</span></button>`).join(''); document.querySelectorAll('[data-corpus]').forEach(b=>b.onclick=()=>{state.corpus=b.dataset.corpus;loadCorpora();});$('corpusCount').textContent=d.corpora.length;}catch(e){$('corpora').innerHTML=`<div class="empty">${esc(e.message)}</div>`;}}
  function renderSources(){const rows=$('sourceRows'); if(!state.sources.length){rows.innerHTML='<tr><td colspan="4" class="empty">No matching sources found.</td></tr>';return;} rows.innerHTML=state.sources.map(s=>{const reviewed=s.review_statuses.filter(x=>x!=='unreviewed').length; return `<tr data-source="${esc(s.id)}"><td><div class="mono">${esc(s.source_id)}</div><small>${esc(s.content_hash.slice(0,12))}…</small></td><td>${esc(new Date(s.created_at).toLocaleString())}</td><td>${s.candidate_count}</td><td><span class="pill ${reviewed===s.candidate_count&&s.candidate_count?'green':''}">${reviewed}/${s.candidate_count} reviewed</span></td></tr>`;}).join(''); document.querySelectorAll('[data-source]').forEach(r=>r.onclick=()=>loadDetail(r.dataset.source));}
  async function loadSources(){if(!token()){$('sourceRows').innerHTML='<tr><td colspan="4" class="empty">Connect an API key to load sources.</td></tr>';return;} try{const p=new URLSearchParams({limit:'100'});const prefix=$('sourceFilter').value;if(prefix)p.set('prefix',prefix);const d=await api(`/v1/sources?${p}`,{headers:headers()});state.sources=d.sources;renderSources();$('sourceCount').textContent=d.sources.length;$('candidateCount').textContent=d.sources.reduce((n,s)=>n+s.candidate_count,0);$('reviewedCount').textContent=d.sources.reduce((n,s)=>n+s.review_statuses.filter(x=>x!=='unreviewed').length,0);}catch(e){toast(e.message,true); if(/token|401|403/i.test(e.message)) $('sourceRows').innerHTML='<tr><td colspan="4" class="empty">Authentication failed or role is insufficient.</td></tr>';}}
  async function loadDetail(id){try{const s=await api(`/v1/sources/${encodeURIComponent(id)}`,{headers:headers()});const prov=s.extraction_audit?.provenance||{};$('detail').innerHTML=`<h4>${esc(s.source_id)}</h4><div class="meta">${esc(prov.article||'Source')} · ${esc(prov.provider||s.created_by)}</div>${s.candidates.length?s.candidates.slice(0,12).map((c,i)=>`<div class="candidate"><strong>${esc(c.modality||c.type)} · ${esc(s.review_statuses[i]||'unreviewed')}</strong><p>${esc(c.text)}</p><small>${esc(c.actor||'Actor unresolved')} → ${esc(c.action||'Action unresolved')}</small></div>`).join(''):'<div class="empty">No deontic candidates were extracted from this provision.</div>'}`;}catch(e){toast(e.message,true);}}
  async function scrape(){if(!token()){toast('Connect an operator API key first.',true);return;}const btn=$('scrape');btn.disabled=true;$('scrapeResult').textContent='Collecting from Légifrance…';try{const d=await api('/v1/legifrance/scrape',{method:'POST',headers:headers(true),body:JSON.stringify({corpus:state.corpus,limit:Number($('limit').value),offset:Number($('offset').value)})});$('scrapeResult').textContent=`Discovered ${d.discovered}; selected ${d.selected}; created ${d.created}; updated ${d.updated}; unchanged ${d.unchanged}; errors ${d.errors.length}.`;toast('Légifrance collection completed.');await loadSources();}catch(e){$('scrapeResult').textContent=`Collection failed: ${e.message}`;toast(e.message,true);}finally{btn.disabled=false;}}
  $('saveKey').onclick=()=>{const value=$('apiKey').value.trim();if(!value){toast('Enter an API key.',true);return;}sessionStorage.setItem('vleApiKey',value);$('apiKey').value='';toast('API key connected for this tab.');loadSources();};
  $('clearKey').onclick=()=>{sessionStorage.removeItem('vleApiKey');state.sources=[];renderSources();$('sourceCount').textContent='—';$('candidateCount').textContent='—';$('reviewedCount').textContent='—';toast('Session API key cleared.');};
  $('scrape').onclick=scrape;$('refreshSources').onclick=loadSources;$('sourceFilter').onchange=loadSources;$('jumpIngest').onclick=()=>$('ingestion').scrollIntoView({behavior:'smooth'});document.querySelectorAll('[data-jump]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-jump]').forEach(x=>x.classList.remove('active'));b.classList.add('active');$(b.dataset.jump).scrollIntoView({behavior:'smooth'});});
  health();loadCorpora();loadSources();
})();
</script>
</body>
</html>'''
