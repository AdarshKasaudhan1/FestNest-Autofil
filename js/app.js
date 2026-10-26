/* FestNest AutoFill UI. Talks to the existing Flask API only:
   GET /api/samples, GET /api/preview/<file>, POST /api/autofill (multipart: file | sample). */
(() => {
const CFG = { maxMB: 25, timeoutMs: 240000 };
const FIELDS = [
  ['event_name','Event name','text','full'], ['event_type','Event type','select'], ['time','Time','text'],
  ['date','Date','text','full'], ['venue','Venue','text','full'], ['organizer','Organizer','text','full'],
  ['prize_pool','Prize pool','text'], ['registration_deadline','Registration deadline / link','text'],
  ['contact','Contact','area','full'], ['description','Description','area','full']
];
const TYPES = ['Hackathon','Cultural Fest','Workshop','Competition','Tech Talk','Sports','Mega Fest'];
const $ = id => document.getElementById(id);
let ctrl = null, last = null, stageTimer = null, original = {}, objUrl = null, allSamples = [], cur = null;

async function readJson(r) {
  const t = await r.text(); let b = null; try { b = JSON.parse(t); } catch {}
  if (!r.ok || (b && b.status === 'error')) throw new Error((b && (b.message || b.error)) || `Server responded ${r.status}`);
  if (!b) throw new Error('The server returned an unexpected response.');
  return b;
}
const show = s => ['idle','busy','error','result'].forEach(n => { $('s-' + n).hidden = n !== s; });
const clean = n => n.replace(/\.[^.]+$/, '').replace(/[_]+/g, ' ').replace(/\s+/g, ' ').trim();

function startStages() {
  const steps = [[0,'Reading poster…','Preparing the file.'],[2000,'Reading text…','Detecting text and layout.'],
    [6000,'Extracting event details…','Filling in the form fields.'],[15000,'Still working…','New or multi-page posters can take longer.']];
  const t0 = Date.now();
  const tick = () => { const e = Date.now() - t0, s = steps.filter(x => e >= x[0]).pop();
    $('busy-title').textContent = s[1]; $('busy-sub').textContent = s[2]; $('elapsed').textContent = (e / 1000).toFixed(0) + 's'; };
  tick(); stageTimer = setInterval(tick, 500);
}
const stopStages = () => { clearInterval(stageTimer); stageTimer = null; };

function buildForm() {
  $('form').innerHTML = FIELDS.map(([k, label, type, w]) => {
    const id = 'f-' + k;
    const ctl = type === 'select'
      ? `<select id="${id}" name="${k}"><option value="">Not found</option>${TYPES.map(t => `<option>${t}</option>`).join('')}</select>`
      : type === 'area' ? `<textarea id="${id}" name="${k}" rows="${k === 'description' ? 5 : 2}" placeholder="Not found on poster"></textarea>`
      : `<input id="${id}" name="${k}" type="text" placeholder="Not found on poster">`;
    return `<div class="f${w ? ' full' : ''}" data-k="${k}"><label for="${id}">${label}<i>Edited</i></label>${ctl}</div>`;
  }).join('');
  $('form').addEventListener('input', e => { const w = e.target.closest('.f'); if (w) w.classList.toggle('edited', e.target.value !== (original[w.dataset.k] || '')); });
  $('form').addEventListener('submit', e => e.preventDefault());
}
function fillForm(f) {
  original = {}; FIELDS.forEach(([k]) => { original[k] = f[k] == null ? '' : String(f[k]); });
  FIELDS.forEach(([k]) => {
    const el = $('f-' + k), v = original[k];
    [...(el.options || [])].filter(o => o.dataset.extra).forEach(o => o.remove());
    if (k === 'event_type' && v && !TYPES.includes(v)) { const o = new Option(v + ' (unrecognised)', v); o.dataset.extra = 1; el.add(o); }
    el.value = v; el.closest('.f').classList.remove('edited');
  });
}
const readForm = () => Object.fromEntries(FIELDS.map(([k]) => [k, $('f-' + k).value.trim()]));
const filled = () => Object.values(readForm()).filter(Boolean).length;

function tab(n) {
  ['form','poster','raw'].forEach(t => { $('t-' + t).setAttribute('aria-selected', String(t === n)); $('p-' + t).hidden = t !== n; });
}
function validateFile(f) {
  if (!/\.(pdf|png|jpe?g|webp)$/i.test(f.name)) return 'Please choose a PDF, PNG, JPG or WebP file.';
  if (f.size === 0) return 'That file is empty.';
  if (f.size > CFG.maxMB * 1048576) return `That file is ${(f.size / 1048576).toFixed(1)} MB. The limit is ${CFG.maxMB} MB.`;
  return '';
}
function fail(msg, title) { stopStages(); $('err-title').textContent = title || 'Couldn’t read that poster'; $('err-msg').textContent = msg; show('error'); }

function setPoster(job, d) {
  const img = $('poster'), none = $('poster-none');
  if (objUrl) { URL.revokeObjectURL(objUrl); objUrl = null; }
  img.hidden = true; none.hidden = true;
  img.onload = () => { img.hidden = false; none.hidden = true; }; img.onerror = () => { img.hidden = true; none.hidden = false; };
  if (job.file) {
    if (/^image\//.test(job.file.type)) {
      img.src = objUrl = URL.createObjectURL(job.file);
    } else if (d && (d.preview_url || d.filename)) {
      img.src = d.preview_url || ('/api/preview/' + encodeURIComponent(d.filename));
    } else {
      none.hidden = false;
    }
  } else {
    img.src = (d && d.preview_url) ? d.preview_url : ('/api/preview/' + encodeURIComponent(job.sample));
  }
}

async function run(job) {
  if (ctrl) ctrl.abort();
  const mine = ctrl = new AbortController(); last = job;
  document.querySelectorAll('.sample').forEach(b => b.setAttribute('aria-current', String(job.sample === b.dataset.n)));
  show('busy'); startStages();
  const timer = setTimeout(() => { mine.timedOut = true; mine.abort(); }, CFG.timeoutMs);
  try {
    const fd = new FormData();
    if (job.file) fd.append('file', job.file, job.file.name); else fd.append('sample', job.sample);
    const d = await readJson(await fetch('/api/autofill', { method: 'POST', body: fd, signal: mine.signal }));
    if (mine !== ctrl) return;
    stopStages(); fillForm(d.structured_data || {});
    $('ocr').textContent = d.ocr_text || 'No raw text was returned.';
    const n = filled();
    $('res-title').textContent = n ? 'Review and edit' : 'Nothing could be extracted. Fill in manually.';
    $('res-meta').textContent = `${n} of 10 fields filled` + (d.execution_time_seconds != null ? ` · ${d.execution_time_seconds}s` : '');
    $('res-src').textContent = d.filename || (job.file ? job.file.name : job.sample);
    setPoster(job, d); tab('form'); show('result');
  } catch (e) {
    if (mine !== ctrl) return;
    if (e.name === 'AbortError') { if (mine.timedOut) fail('This is taking too long. Try again, or use a clearer or smaller file.', 'Timed out'); return; }
    const net = e instanceof TypeError;
    fail(net ? 'Can’t reach the server. Check that the app is running, then try again.' : e.message, net ? 'Connection problem' : undefined);
  } finally { clearTimeout(timer); }
}
function pick(f) { const m = validateFile(f); if (m) return fail(m, 'Can’t use that file'); run({ file: f }); }

let tt; function toast(t, err) { const el = $('toast'); el.textContent = t; el.className = 'toast on' + (err ? ' err' : ''); clearTimeout(tt); tt = setTimeout(() => el.className = 'toast', 2400); }
async function copy(text, msg) { try { await navigator.clipboard.writeText(text); toast(msg); } catch { toast('Copy failed. Select the text and copy manually.', 1); } }
const json = () => JSON.stringify(readForm(), null, 2);

function renderSamples(q = '') {
  const box = $('samples'), list = allSamples.filter(s => (s.filename + ' ' + clean(s.filename)).toLowerCase().includes(q.toLowerCase()));
  $('count').textContent = allSamples.length ? (q ? `${list.length} of ${allSamples.length}` : `${allSamples.length} posters`) : '';
  if (!list.length) { box.innerHTML = `<p class="muted">${allSamples.length ? 'No samples match.' : 'No samples available. Upload a poster instead.'}</p>`; return; }
  box.innerHTML = '';
  list.forEach(s => { const b = document.createElement('button'); b.type = 'button'; b.className = 'sample'; b.dataset.n = s.filename;
    if (job_is(s.filename)) b.setAttribute('aria-current', 'true');
    const sp = document.createElement('span'); sp.textContent = clean(s.filename); sp.title = s.filename;
    const em = document.createElement('em'); em.textContent = (s.type || '').toUpperCase(); b.append(sp, em); b.onclick = () => run({ sample: s.filename }); box.append(b); });
}
const job_is = n => last && last.sample === n;
async function loadSamples() {
  try {
    const d = await readJson(await fetch('/api/samples')); allSamples = d.samples || [];
    $('q').hidden = allSamples.length < 8; renderSamples();
  } catch { $('samples').innerHTML = '<p class="muted">Samples couldn’t be loaded. You can still upload a poster. <button class="link" type="button" id="rs">Retry</button></p>'; $('rs').onclick = loadSamples; }
}

function init() {
  buildForm(); const drop = $('drop'), fi = $('file');
  fi.addEventListener('change', () => { if (fi.files[0]) pick(fi.files[0]); fi.value = ''; });
  drop.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fi.click(); } });
  ['dragenter','dragover'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave','drop'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', e => { if (e.dataTransfer.files[0]) pick(e.dataTransfer.files[0]); });
  ['dragover','drop'].forEach(t => window.addEventListener(t, e => e.preventDefault()));
  $('q').addEventListener('input', e => renderSamples(e.target.value));
  ['form','poster','raw'].forEach(t => $('t-' + t).onclick = () => tab(t));
  $('cancel').onclick = () => { if (ctrl) ctrl.abort(); stopStages(); show('idle'); };
  $('retry').onclick = () => last ? run(last) : show('idle');
  $('err-reset').onclick = $('reset').onclick = () => { show('idle'); last = null; document.querySelectorAll('.sample').forEach(b => b.removeAttribute('aria-current')); };
  $('revert').onclick = () => { fillForm(original); toast('Edits reverted'); };
  $('copy').onclick = () => copy(json(), 'Copied JSON');
  $('copy-ocr').onclick = () => copy($('ocr').textContent, 'Copied text');
  $('download').onclick = () => { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([json()], { type: 'application/json' }));
    a.download = ((readForm().event_name || 'event').replace(/[^\w]+/g, '-').replace(/^-|-$/g, '').toLowerCase() || 'event') + '.json'; a.click(); URL.revokeObjectURL(a.href); toast('Downloaded'); };
  $('validate').onclick = () => { const f = readForm(), miss = ['event_name','event_type','date','venue'].filter(k => !f[k]);
    miss.length ? toast('Still needed: ' + miss.map(k => k.replace('_', ' ')).join(', '), 1) : toast('Required details are filled in'); };
  loadSamples();
}
init();
})();
