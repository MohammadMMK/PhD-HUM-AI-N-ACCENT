"""
Build online listening-comprehension experiments from SAVED audiovisualize_interactive()
pages — no Kokoro, no torch, no model. This module only needs numpy + soundfile,
so it runs anywhere, in seconds, and is meant to be the thing you re-run every
time you add or change a condition.

WORKFLOW
  1. Generate + save (needs the kokoro conda env, does the actual synthesis):
       functions.py: text_to_ipa_chunks / manipulate / synth_aligned
                     / audiovisualize_interactive(..., out_html="outputs/<version>.html")
     That saved HTML *is* your metadata store: it embeds the full audio (as
     base64) and, for every token Kokoro produced, its duration, word index,
     word text, and whether a manipulation rule altered it. Nothing else needs
     saving — the file is self-contained and safe to archive as-is.

  2. Build the experiment (this module, any environment):
       experiment.py: load_visualization() reads that HTML back into audio +
                      per-token metadata; build_listening_test_from_html() turns
                      it straight into a participant page. See build_experiment.py
                      for the manifest-driven version (a list of conditions you
                      extend whenever you add a new version).

Both functions.py's build_listening_test() (used on a *live* synth_aligned()
result, without saving/reloading) and build_listening_test_from_html() here
(used on a *saved* file) call the same private _write_test_page(), so the two
paths always produce an identical participant page.
"""
import base64, io, json, os, re
from pathlib import Path
from html import escape as _html_escape
import numpy as np
import soundfile as sf


TEST_UI = {
    "en": {
        "title": "Listening test",
        "intro": ["You will hear a short story, one sentence at a time.",
                  "After each sentence, its text appears. Tap every word you did not understand, then go on to the next sentence.",
                  "Please use headphones in a quiet place."],
        "plays_once": "You can listen to each sentence once.",
        "plays_n": "You can listen to each sentence up to {n} times before the text appears.",
        "plays_any": "You can listen to each sentence as often as you like before the text appears.",
        "pid_label": "Your name",
        "pid_missing": "Enter your name to start.",
        "start": "Start",
        "progress": "Sentence {i} of {n}",
        "listen": "Listen",
        "playing": "Playing",
        "again": "Replay",
        "again_n": "Replay\n({n} left)",
        "listened": "Heard",
        "reveal": "Show the text",
        "mark": "Tap the words you did not understand. Tap again to undo.",
        "next": "Next sentence",
        "finish": "Finish",
        "tap_to_play": "Tap Listen to play the sentence.",
        "audio_error": "This sentence could not be played. Reload the page to try again.",
        "end_title": "Thank you",
        "sending": "Saving your answers…",
        "saved": "Your answers are saved.",
        "save_failed": "Your answers could not be sent. Download the file below and send it to the researcher.",
        "no_url": "Download the file below and send it to the researcher.",
        "download": "Download answers (CSV)",
        "done": "Return to the study",
    },
    "it": {
        "title": "Test di ascolto",
        "intro": ["Ascolterai un breve racconto, una frase alla volta.",
                  "Dopo ogni frase comparirà il testo. Tocca tutte le parole che non hai capito, poi passa alla frase successiva.",
                  "Usa le cuffie in un luogo tranquillo."],
        "plays_once": "Puoi ascoltare ogni frase una sola volta.",
        "plays_n": "Puoi ascoltare ogni frase fino a {n} volte prima che compaia il testo.",
        "plays_any": "Puoi ascoltare ogni frase tutte le volte che vuoi prima che compaia il testo.",
        "pid_label": "Nome e cognome",
        "pid_missing": "Inserisci il tuo nome per iniziare.",
        "start": "Inizia",
        "progress": "Frase {i} di {n}",
        "listen": "Ascolta",
        "playing": "In ascolto",
        "again": "Riascolta",
        "again_n": "Riascolta\n(ancora {n})",
        "listened": "Ascoltata",
        "reveal": "Mostra il testo",
        "mark": "Tocca le parole che non hai capito. Tocca di nuovo per annullare.",
        "next": "Frase successiva",
        "finish": "Termina",
        "tap_to_play": "Tocca Ascolta per riprodurre la frase.",
        "audio_error": "Non è stato possibile riprodurre questa frase. Ricarica la pagina e riprova.",
        "end_title": "Grazie",
        "sending": "Salvataggio delle risposte…",
        "saved": "Le tue risposte sono state salvate.",
        "save_failed": "Non è stato possibile inviare le risposte. Scarica il file qui sotto e invialo al ricercatore.",
        "no_url": "Scarica il file qui sotto e invialo al ricercatore.",
        "download": "Scarica le risposte (CSV)",
        "done": "Torna allo studio",
    },
}



def _audio_data_uri(a, sr, fmt="mp3"):
    """Encode one chunk as a data: URI. MP3 is ~10-20x smaller than WAV and
    plays in every browser; falls back to 16-bit WAV if MP3 isn't available."""
    a = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)
    if fmt == "mp3":
        try:
            buf = io.BytesIO(); sf.write(buf, a, sr, format="MP3")
            return "data:audio/mpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception as ex:
            print(f"⚠ MP3 encoding unavailable ({ex}); using WAV")
    buf = io.BytesIO(); sf.write(buf, a, sr, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()



def _write_test_page(payload, out_html, out_dir, test_id, version, submit_url,
                     max_plays, lang, ui, completion_url):
    texts = dict(TEST_UI[lang]); texts.update(ui or {})
    cfg = {"test": test_id, "version": version, "submit_url": submit_url,
           "max_plays": max_plays, "completion_url": completion_url, "ui": texts}
    page = (_TEST_TEMPLATE
            .replace("__LANG__", lang)
            .replace("__TITLE_TEXT__", _html_escape(texts["title"]))
            .replace("__CFG__", json.dumps(cfg, ensure_ascii=False))
            .replace("__CHUNKS__", json.dumps(payload, ensure_ascii=False)))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, out_html)
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Listening test: {path}  ({len(payload)} sentences, {len(page)/1e6:.1f} MB)")
    return path


# ---- reading a saved audiovisualize_interactive() page back into data --------
def load_visualization(html_path):
    """Parse a page saved by functions.audiovisualize_interactive() back into
    its audio and per-token metadata, without re-running any synthesis.
    Returns {"audio": np.ndarray, "sr": int, "segs": [...], "title": str,
    "rules_text": str, "stats": [...]}. Each entry in `segs` is one Kokoro
    token: {"l": display label, "t": raw token, "k": kind (phone / diacritic /
    stress / punct / space / bos / eos), "s"/"e": start/end seconds, "ms",
    "fr": frames, "w": word index (None for space/bos/eos), "wl": word text,
    "alt": True if a manipulation rule altered/introduced this token}.

    Only works on files made with the CURRENT functions.py (the one with
    per-token — not merged — segments). An older saved file will raise a
    clear error rather than silently returning wrong data."""
    html = Path(html_path).read_text(encoding="utf-8")

    m = re.search(r'data:audio/(\w+);base64,([A-Za-z0-9+/=]+)"', html)
    if not m:
        raise ValueError(f"no embedded audio found in {html_path} — "
                         "is this a file made by audiovisualize_interactive()?")
    audio, sr = sf.read(io.BytesIO(base64.b64decode(m.group(2))), dtype="float32")

    dec = json.JSONDecoder()
    def after(marker, start=0):
        return html.index(marker, start) + len(marker)
    def read_json(i):
        return dec.raw_decode(html, i)
    try:
        i = after("const segs="); segs, i = read_json(i)
        i = after(", env=", i); _, i = read_json(i)
        i = after(", DUR=", i); _, i = read_json(i)
        i = after("const TITLE=", i); title, i = read_json(i)
        i = after(", RULES=", i); rules_text, i = read_json(i)
        i = after(", STATS=", i); stats, i = read_json(i)
    except ValueError as ex:
        raise ValueError(f"couldn't read metadata out of {html_path} ({ex}). "
                         "This usually means the file was made with an older "
                         "functions.py — regenerate it with the current one.") from ex
    if not segs or "k" not in segs[0] or "w" not in segs[0]:
        raise ValueError(f"{html_path} has no per-token kind/word metadata — "
                         "regenerate it with the current functions.py before loading it here.")
    return {"audio": audio, "sr": sr, "segs": segs, "title": title,
            "rules_text": rules_text, "stats": stats}


def _chunks_from_segs(segs, audio, sr):
    """Group parsed tokens (see load_visualization) back into per-sentence
    {"words": [...], "altered": [0/1 ...], "audio": ndarray} using their own
    BOS/EOS boundaries and word indices — the same shape functions._test_chunks
    produces from a live synth_aligned() call."""
    chunks, cur = [], None
    for g in segs:
        k = g.get("k")
        if k == "bos":
            cur = {"s": g["s"], "words": {}, "order": []}
            continue
        if cur is None:
            continue
        if k == "eos":
            cur["e"] = g["e"]
            a = audio[int(round(cur["s"] * sr)):int(round(cur["e"] * sr))]
            chunks.append({
                "words":   [cur["words"][w]["label"] for w in cur["order"]],
                "altered": [1 if cur["words"][w]["alt"] else 0 for w in cur["order"]],
                "audio":   a})
            cur = None
            continue
        w = g.get("w")
        if w is None:                                  # word gap
            continue
        if w not in cur["words"]:
            cur["words"][w] = {"label": g.get("wl") or "", "alt": False}
            cur["order"].append(w)
        if g.get("alt"):
            cur["words"][w]["alt"] = True
    return chunks


def build_listening_test_from_html(html_path, out_html=None, out_dir="listening_test",
                                   test_id=None, version=None, submit_url=None, max_plays=1,
                                   lang="en", ui=None, completion_url=None, audio_format="mp3"):
    """Build a participant page straight from a SAVED audiovisualize_interactive()
    file — no Kokoro/torch import, no re-synthesis. This is the function to call
    from build_experiment.py every time you add or change a condition.

    html_path  : path to the saved outputs/<version>.html
    version    : condition name stored with every answer (default: that page's title)
    test_id    : study name stored with every answer (default: same as version —
                 set it explicitly so every condition of one study shares it)
    Everything else is the same as functions.build_listening_test()."""
    data = load_visualization(html_path)
    version = version or data["title"] or Path(html_path).stem
    test_id = test_id or version
    out_html = out_html or f"{version}.html"
    chunks = _chunks_from_segs(data["segs"], data["audio"], data["sr"])
    if not chunks:
        raise ValueError(f"no complete sentence (BOS...EOS pair) found in {html_path}")
    payload = [{"audio": _audio_data_uri(c["audio"], data["sr"], audio_format),
                "words": c["words"], "altered": c["altered"]} for c in chunks]
    return _write_test_page(payload, out_html, out_dir, test_id, version, submit_url,
                            max_plays, lang, ui, completion_url)



_TEST_TEMPLATE = r"""<!DOCTYPE html>
<html lang="__LANG__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>__TITLE_TEXT__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#E9EEF1; --panel:#FFFFFF; --ink:#18262F; --muted:#586874; --line:#CBD5DB;
  --accent:#16586E; --accent-ink:#FFFFFF; --miss:#A01D35; --miss-bg:#F7E1E5; --focus:#E9A93A;
  color-scheme:light;
}
*{box-sizing:border-box}
html,body{margin:0;min-height:100%}
body{background:var(--bg);color:var(--ink);
  font-family:"Atkinson Hyperlegible",system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
  font-size:18px;line-height:1.55;
  padding:env(safe-area-inset-top,0px) 0 env(safe-area-inset-bottom,0px)}
main{max-width:620px;margin:0 auto;padding:28px 20px 40px;min-height:100vh;display:flex;flex-direction:column}
.screen{display:none;flex:1;flex-direction:column}
.screen.on{display:flex}
h1{font-size:32px;line-height:1.15;margin:12px 0 20px;font-weight:700;letter-spacing:-.01em}
p{margin:0 0 12px;max-width:60ch}
.muted{color:var(--muted)}
label{display:block;font-weight:700;margin:22px 0 6px}
input[type=text]{width:100%;max-width:340px;font:inherit;padding:12px 14px;border:2px solid var(--line);
  border-radius:10px;background:var(--panel);color:var(--ink)}
.err{color:var(--miss);min-height:1.5em;margin:6px 0 0;font-size:16px}
button,a.btn{font:inherit}
:focus-visible{outline:3px solid var(--focus);outline-offset:3px}
.btn{display:inline-flex;align-items:center;justify-content:center;font-weight:700;border:0;border-radius:12px;
  padding:13px 24px;min-height:52px;background:var(--accent);color:var(--accent-ink);cursor:pointer;text-decoration:none}
.btn[disabled]{opacity:.3;cursor:not-allowed}
.btn.line{background:transparent;color:var(--accent);box-shadow:inset 0 0 0 2px var(--accent)}
.actions{margin-top:auto;padding-top:24px;display:flex;gap:12px;flex-wrap:wrap;justify-content:flex-end}
.intro-actions{justify-content:flex-start}
/* progress */
.track{height:5px;background:var(--line);border-radius:3px;overflow:hidden}
.track i{display:block;height:100%;width:0;background:var(--accent);transition:width .35s ease}
.count{font-size:15px;color:var(--muted);margin:8px 0 0}
/* the listen button: the ring fills while the sentence plays */
.player{display:flex;flex-direction:column;align-items:center;gap:14px;margin:36px 0 28px}
.ring{position:relative;width:156px;height:156px}
.ring > svg{position:absolute;inset:0;transform:rotate(-90deg)}
.ring circle{fill:none;stroke-width:7}
.ring .bg{stroke:var(--line)}
.ring .fg{stroke:var(--accent);stroke-linecap:round}
.play{position:absolute;inset:13px;border-radius:50%;border:0;background:var(--panel);color:var(--accent);
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;
  font-weight:700;font-size:17px;cursor:pointer}
.play svg{width:30px;height:30px;fill:currentColor}
.play span{font-size:15px;line-height:1.25;padding:0 10px;text-align:center;white-space:pre-line}
.play[disabled]{cursor:default;color:var(--muted)}
.play[disabled] .icon{display:none}
.play.busy .icon{display:none}
.status{min-height:1.5em;color:var(--muted);text-align:center;font-size:16px;margin:0}
.reveal{margin-top:4px}
/* transcript */
.text{display:none}
.text.on{display:block}
.hint{color:var(--muted);font-size:16px;margin-bottom:12px}
.words{display:flex;flex-wrap:wrap;gap:6px 4px;font-size:25px;line-height:1.25}
.w{border:2px solid transparent;background:var(--panel);color:var(--ink);border-radius:8px;
  padding:6px 9px;cursor:pointer;font:inherit}
.w:hover{border-color:var(--line)}
.w[aria-pressed="true"]{background:var(--miss-bg);border-color:var(--miss);color:var(--miss);
  text-decoration:line-through;text-decoration-thickness:2px}
.end-box{margin-top:8px}
.end-box .btn{margin:14px 12px 0 0}
[hidden]{display:none!important}
@media (max-width:420px){ h1{font-size:27px} .words{font-size:22px} }
@media (prefers-reduced-motion:reduce){ *{transition:none!important} }
</style>
</head>
<body>
<main>
  <section class="screen on" id="intro">
    <h1 id="introTitle"></h1>
    <div id="introText"></div>
    <label for="pid" id="pidLabel"></label>
    <input type="text" id="pid" autocomplete="name" spellcheck="false">
    <p class="err" id="pidErr" role="alert"></p>
    <div class="actions intro-actions"><button class="btn" id="start"></button></div>
  </section>

  <section class="screen" id="trial">
    <div class="track"><i id="bar"></i></div>
    <p class="count" id="count"></p>
    <div class="player">
      <div class="ring">
        <svg viewBox="0 0 156 156" aria-hidden="true">
          <circle class="bg" cx="78" cy="78" r="70"/>
          <circle class="fg" id="fg" cx="78" cy="78" r="70"/>
        </svg>
        <button class="play" id="play">
          <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l11-6.5z"/></svg>
          <span id="playLabel"></span>
        </button>
      </div>
      <p class="status" id="status" aria-live="polite"></p>
      <button class="btn line reveal" id="reveal" hidden></button>
    </div>
    <div class="text" id="text">
      <p class="hint" id="hint"></p>
      <div class="words" id="words"></div>
    </div>
    <div class="actions"><button class="btn" id="next" disabled></button></div>
  </section>

  <section class="screen" id="end">
    <h1 id="endTitle"></h1>
    <p id="endMsg" aria-live="polite"></p>
    <div class="end-box">
      <a class="btn line" id="download" href="#"></a>
      <a class="btn" id="done" hidden></a>
    </div>
  </section>
</main>
<script>
(function(){
const CFG = __CFG__;
const CHUNKS = __CHUNKS__;
const T = CFG.ui;
const $ = id => document.getElementById(id);
const fmt = (s, v) => s.replace(/\{(\w+)\}/g, (m, k) => v[k]);
const params = new URLSearchParams(location.search);
const session = Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
const COLS = ['test','version','participant','session','chunk','n_chunks','word_pos','word','word_clean',
              'altered','not_understood','plays','listen_ms','response_ms','time'];
let pid = '', ci = 0, st = null;
const rows = [], failed = [], pending = [];

// ---------- intro ----------
$('introTitle').textContent = T.title;
const plays = CFG.max_plays ? (CFG.max_plays === 1 ? T.plays_once : fmt(T.plays_n, {n: CFG.max_plays})) : T.plays_any;
[...T.intro, plays].forEach(t => { const p = document.createElement('p'); p.textContent = t; $('introText').appendChild(p); });
$('pidLabel').textContent = T.pid_label;
$('pid').value = params.get('pid') || params.get('PROLIFIC_PID') || params.get('participant') || '';
$('start').textContent = T.start;
$('start').onclick = () => {
  pid = $('pid').value.trim();
  if (!pid) { $('pidErr').textContent = T.pid_missing; $('pid').focus(); return; }
  window.addEventListener('beforeunload', guard);
  show('trial'); startChunk(0);
};
$('pid').addEventListener('keydown', e => { if (e.key === 'Enter') $('start').click(); });
function guard(e){ e.preventDefault(); e.returnValue = ''; }
function show(id){ document.querySelectorAll('.screen').forEach(s => s.classList.toggle('on', s.id === id)); window.scrollTo(0, 0); }

// ---------- player ----------
const audio = new Audio(); audio.preload = 'auto';
const fg = $('fg'), CIRC = 2 * Math.PI * 70;
fg.style.strokeDasharray = CIRC;
const ring = f => { fg.style.strokeDashoffset = CIRC * (1 - Math.max(0, Math.min(1, f))); };
const playsLeft = () => CFG.max_plays ? CFG.max_plays - st.plays : Infinity;

function startChunk(i){
  ci = i;
  st = {plays: 0, revealed: false, firstPlay: null, revealAt: null, marked: new Set()};
  $('count').textContent = fmt(T.progress, {i: i + 1, n: CHUNKS.length});
  $('bar').style.width = (100 * i / CHUNKS.length) + '%';
  audio.src = CHUNKS[i].audio; ring(0);
  $('play').disabled = false; $('play').classList.remove('busy');
  $('playLabel').textContent = T.listen;
  $('status').textContent = '';
  $('reveal').hidden = true; $('reveal').textContent = T.reveal;
  $('text').classList.remove('on'); $('words').innerHTML = ''; $('hint').textContent = T.mark;
  $('next').disabled = true;
  $('next').textContent = i === CHUNKS.length - 1 ? T.finish : T.next;
  $('play').focus();
}
function play(){
  if (st.revealed || playsLeft() <= 0) return;
  st.plays++; if (st.firstPlay === null) st.firstPlay = performance.now();
  $('play').disabled = true; $('play').classList.add('busy');
  $('playLabel').textContent = T.playing; $('reveal').hidden = true; $('status').textContent = '';
  ring(0); audio.currentTime = 0;
  audio.play().catch(() => {
    st.plays--; $('play').disabled = false; $('play').classList.remove('busy');
    $('playLabel').textContent = T.listen; $('status').textContent = T.tap_to_play;
  });
}
$('play').onclick = play;
$('reveal').onclick = () => reveal();
audio.ontimeupdate = () => { if (audio.duration) ring(audio.currentTime / audio.duration); };
audio.onended = () => {
  ring(1); $('play').classList.remove('busy');
  const left = playsLeft();
  if (left <= 0) { reveal(); return; }
  $('play').disabled = false;
  $('playLabel').textContent = isFinite(left) ? fmt(T.again_n, {n: left}) : T.again;
  $('reveal').hidden = false; $('reveal').focus();
};
audio.onerror = () => { if (audio.src) $('status').textContent = T.audio_error; };

// ---------- transcript ----------
function reveal(){
  if (st.revealed) return;
  st.revealed = true; st.revealAt = performance.now();
  $('play').disabled = true; $('playLabel').textContent = T.listened; $('reveal').hidden = true;
  const box = $('words'); box.innerHTML = '';
  CHUNKS[ci].words.forEach((w, k) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'w'; b.textContent = w; b.setAttribute('aria-pressed', 'false');
    b.onclick = () => {
      const on = !st.marked.has(k);
      on ? st.marked.add(k) : st.marked.delete(k);
      b.setAttribute('aria-pressed', String(on));
    };
    box.appendChild(b);
  });
  $('text').classList.add('on'); $('next').disabled = false;
}
const clean = w => w.toLowerCase().replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');

$('next').onclick = () => {
  if (!st.revealed) return;
  const ch = CHUNKS[ci], now = new Date().toISOString();
  const r = ch.words.map((w, k) => ({
    test: CFG.test, version: CFG.version, participant: pid, session,
    chunk: ci + 1, n_chunks: CHUNKS.length, word_pos: k + 1, word: w, word_clean: clean(w),
    altered: ch.altered[k] ? 1 : 0, not_understood: st.marked.has(k) ? 1 : 0,
    plays: st.plays, listen_ms: Math.round(st.revealAt - st.firstPlay),
    response_ms: Math.round(performance.now() - st.revealAt), time: now}));
  rows.push(...r);
  pending.push(send(r));
  audio.pause();
  if (ci + 1 < CHUNKS.length) startChunk(ci + 1); else finish();
};

// ---------- saving ----------
function send(r){
  if (!CFG.submit_url) return Promise.resolve(true);
  return fetch(CFG.submit_url, {method: 'POST', mode: 'no-cors',
      headers: {'Content-Type': 'text/plain;charset=utf-8'}, body: JSON.stringify({rows: r})})
    .then(() => true).catch(() => { failed.push(...r); return false; });
}
function toCSV(list){
  const esc = v => { const s = String(v ?? ''); return /[",\n;]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
  return [COLS.join(','), ...list.map(r => COLS.map(c => esc(r[c])).join(','))].join('\n');
}
async function finish(){
  window.removeEventListener('beforeunload', guard);
  $('bar').style.width = '100%';
  show('end');
  $('endTitle').textContent = T.end_title;
  const a = $('download');
  a.textContent = T.download;
  a.href = URL.createObjectURL(new Blob(['\ufeff' + toCSV(rows)], {type: 'text/csv;charset=utf-8'}));
  a.download = [CFG.test, CFG.version, pid].join('_').replace(/[^\w.-]+/g, '-') + '.csv';
  if (CFG.completion_url) { $('done').href = CFG.completion_url; $('done').textContent = T.done; $('done').hidden = false; }
  if (!CFG.submit_url) { $('endMsg').textContent = T.no_url; return; }
  $('endMsg').textContent = T.sending;
  await Promise.all(pending);
  if (failed.length) await send(failed.splice(0));
  $('endMsg').textContent = failed.length ? T.save_failed : T.saved;
}
})();
</script>
</body>
</html>
"""


# =============================================================================
# MULTI-STORY EXPERIMENT — one page, participant picks story + version,
# free navigation between sentences, replay any time, autosave.
# =============================================================================

EXP_UI = {
    "en": {
        "title": "Listening test",
        "intro": ["Choose a version, then listen to it one sentence at a time.",
                  "After you have heard a sentence, its text appears. Tap every word you did not understand.",
                  "You can listen again, go back to any sentence and change your answers until you press Finish.",
                  "Please use headphones in a quiet place."],
        "name_label": "Your name",
        "version_label": "Version",
        "name_missing": "Enter your name to start.",
        "choice_missing": "Choose a version to start.",
        "start": "Start",
        "resumed": "Welcome back. Your earlier answers were restored.",
        "sentence_n": "Sentence {i} of {n}",
        "nav_label": "Go to sentence",
        "listen": "Listen",
        "playing": "Playing",
        "again": "Listen again",
        "wait_text": "The text appears after you have heard the sentence.",
        "mark": "Tap the words you did not understand. Tap again to undo.",
        "prev": "Previous",
        "next": "Next",
        "finish": "Finish",
        "confirm_unheard": "{n} sentence(s) not heard yet. Finish anyway?",
        "tap_to_play": "Tap Listen to play the sentence.",
        "audio_error": "This sentence could not be played. Reload the page to try again.",
        "end_title": "Thank you",
        "sending": "Saving your answers…",
        "saved": "Your answers are saved.",
        "save_failed": "Your answers could not be sent. Download the file below and send it to the researcher.",
        "no_url": "Download the file below and send it to the researcher.",
        "download": "Download answers (CSV)",
        "review": "Review your answers",
        "another_version": "Choose another version",
        "done": "Return to the study",
    },
    "it": {
        "title": "Test di ascolto",
        "intro": ["Scegli una versione, poi ascoltala una frase alla volta.",
                  "Dopo aver ascoltato una frase, compare il testo. Tocca tutte le parole che non hai capito.",
                  "Puoi riascoltare, tornare a qualsiasi frase e cambiare le risposte finché non premi Termina.",
                  "Usa le cuffie in un luogo tranquillo."],
        "name_label": "Nome e cognome",
        "version_label": "Versione",
        "name_missing": "Inserisci il tuo nome per iniziare.",
        "choice_missing": "Scegli una versione per iniziare.",
        "start": "Inizia",
        "resumed": "Bentornato. Le risposte precedenti sono state ripristinate.",
        "sentence_n": "Frase {i} di {n}",
        "nav_label": "Vai alla frase",
        "listen": "Ascolta",
        "playing": "In ascolto",
        "again": "Riascolta",
        "wait_text": "Il testo compare dopo aver ascoltato la frase.",
        "mark": "Tocca le parole che non hai capito. Tocca di nuovo per annullare.",
        "prev": "Precedente",
        "next": "Successiva",
        "finish": "Termina",
        "confirm_unheard": "{n} frase/i non ancora ascoltata/e. Terminare comunque?",
        "tap_to_play": "Tocca Ascolta per riprodurre la frase.",
        "audio_error": "Non è stato possibile riprodurre questa frase. Ricarica la pagina e riprova.",
        "end_title": "Grazie",
        "sending": "Salvataggio delle risposte…",
        "saved": "Le tue risposte sono state salvate.",
        "save_failed": "Non è stato possibile inviare le risposte. Scarica il file qui sotto e invialo al ricercatore.",
        "no_url": "Scarica il file qui sotto e invialo al ricercatore.",
        "download": "Scarica le risposte (CSV)",
        "review": "Rivedi le tue risposte",
        "another_version": "Scegli un'altra versione",
        "done": "Torna allo studio",
    },
}


def _safe_name(s):
    return re.sub(r"[^\w.-]+", "-", str(s)).strip("-") or "x"


def build_experiment(versions, out_html="index.html", out_dir="experiment", test_id="story",
                     version_labels=None, submit_url=None, text_after_listen=True,
                     lang="it", ui=None, completion_url=None, audio_format="mp3",
                     embed_audio=True):
    """ONE participant page for ONE story, offering several VERSIONS of it, built
    from SAVED audiovisualize_interactive() pages (no Kokoro needed).

    versions : {version_id: "outputs/<name>.html", ...} — all versions must be
              the same story/text (just synthesized differently). version_id is
              what gets stored with every answer, e.g.:
                {"native": "outputs/story1__clean.html",
                 "consistent": "outputs/story1__consistent_dur_forced_native.html",
                 "inconsistent": "outputs/story1__less_consistent_dur_forced_native.html"}
    version_labels : {version_id: text shown to participants}, default = the id.
    text_after_listen : True = a sentence's text appears only after it has been
              heard once; False = text visible immediately.
    embed_audio : True = one self-contained HTML (audio inside); False = audio
              as separate MP3 files under out_dir/audio/ (upload the whole folder).
    A link like index.html?version=native&pid=... pre-fills the choice.

    SAVED DATA — deliberately minimal: only the words a participant marked as
    not understood, nothing about timing or how many times they replayed a
    sentence. One row per marked word: test, version, participant, session,
    chunk, word_pos, word, word_clean, altered (1 if your rules changed this
    word). `rev`/`final` are plain sequence counters (not timestamps) kept
    only so load_responses() can tell which save of a sentence is the latest
    when a participant revisits it — pass the loaded data straight to
    load_responses() and they disappear from the result."""
    version_labels = version_labels or {}
    texts = dict(EXP_UI[lang]); texts.update(ui or {})
    out_versions, ref_words, total = [], None, 0
    for vid, html_path in versions.items():
        data = load_visualization(html_path)
        chunks = _chunks_from_segs(data["segs"], data["audio"], data["sr"])
        if not chunks:
            raise ValueError(f"no complete sentence found in {html_path}")
        words = [c["words"] for c in chunks]
        if ref_words is None:
            ref_words = words
        elif words != ref_words:
            print(f"⚠ version '{vid}' has different text/sentences than the first version — "
                  "check all versions were generated from the same text")
        out_chunks = []
        for i, c in enumerate(chunks, 1):
            if embed_audio:
                src = _audio_data_uri(c["audio"], data["sr"], audio_format)
            else:
                rel = f"audio/{_safe_name(vid)}/{i:03d}.mp3"
                os.makedirs(os.path.dirname(os.path.join(out_dir, rel)), exist_ok=True)
                sf.write(os.path.join(out_dir, rel),
                         np.clip(c["audio"], -1, 1).astype(np.float32), data["sr"], format="MP3")
                src = rel
            out_chunks.append({"audio": src, "words": c["words"], "altered": c["altered"]})
        out_versions.append({"id": vid, "label": version_labels.get(vid, vid), "chunks": out_chunks})
        total += len(chunks)
        print(f"  {vid}: {len(chunks)} sentences  ({html_path})")

    cfg = {"test": test_id, "submit_url": submit_url, "text_after_listen": bool(text_after_listen),
           "completion_url": completion_url, "ui": texts}
    page = (_EXP_TEMPLATE
            .replace("__LANG__", lang)
            .replace("__TITLE_TEXT__", _html_escape(texts["title"]))
            .replace("__CFG__", json.dumps(cfg, ensure_ascii=False))
            .replace("__VERSIONS__", json.dumps(out_versions, ensure_ascii=False)))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, out_html)
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Experiment: {path}  ({len(out_versions)} versions, {total} sentence recordings, {len(page)/1e6:.1f} MB)")
    return path


def load_responses(src):
    """Read the exported sheet (CSV path or DataFrame) and return only the
    words each participant actually marked as not understood, keeping the
    LATEST answer for every (session, chunk) — free navigation means a
    sentence's marks can be saved several times as the participant revisits
    it; `rev` (a plain counter, not a time) says which save is newest.
    Drops the internal bookkeeping columns (rev, final) from the result."""
    import pandas as pd
    df = src.copy() if isinstance(src, pd.DataFrame) else pd.read_csv(src)
    if "rev" not in df: df["rev"] = 1
    df["word_pos"] = pd.to_numeric(df["word_pos"], errors="coerce")   # "" (no word marked) -> NaN
    latest = df["rev"] == df.groupby(["session", "chunk"])["rev"].transform("max")
    df = df[latest & df["word_pos"].notna()].copy()
    df["word_pos"] = df["word_pos"].astype(int)
    key = ["session", "chunk", "word_pos"]
    return (df.sort_values(key).drop_duplicates(key, keep="last")
              .drop(columns=[c for c in ("rev", "final") if c in df], errors="ignore")
              .reset_index(drop=True))


_EXP_TEMPLATE = r"""<!DOCTYPE html>
<html lang="__LANG__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>__TITLE_TEXT__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#E9EEF1; --panel:#FFFFFF; --ink:#18262F; --muted:#586874; --line:#CBD5DB;
  --accent:#16586E; --accent-soft:#D6E6EC; --accent-ink:#FFFFFF; --miss:#A01D35; --miss-bg:#F7E1E5;
  --focus:#E9A93A; color-scheme:light;
}
*{box-sizing:border-box}
html,body{margin:0;min-height:100%}
body{background:var(--bg);color:var(--ink);
  font-family:"Atkinson Hyperlegible",system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
  font-size:18px;line-height:1.55;
  padding:env(safe-area-inset-top,0px) 0 env(safe-area-inset-bottom,0px)}
main{max-width:660px;margin:0 auto;padding:24px 20px 32px;min-height:100vh;display:flex;flex-direction:column}
.screen{display:none;flex:1;flex-direction:column}
.screen.on{display:flex}
h1{font-size:32px;line-height:1.15;margin:12px 0 18px;font-weight:700;letter-spacing:-.01em}
p{margin:0 0 10px;max-width:60ch}
button,a.btn{font:inherit}
:focus-visible{outline:3px solid var(--focus);outline-offset:3px}
.field{margin-top:22px}
.field > .lab{display:block;font-weight:700;margin-bottom:8px}
input[type=text]{width:100%;max-width:360px;font:inherit;padding:12px 14px;border:2px solid var(--line);
  border-radius:10px;background:var(--panel);color:var(--ink)}
.choices{display:flex;flex-wrap:wrap;gap:8px}
.choice{position:relative}
.choice input{position:absolute;opacity:0;inset:0;margin:0;cursor:pointer}
.choice span{display:block;padding:10px 16px;border:2px solid var(--line);border-radius:10px;background:var(--panel);cursor:pointer}
.choice input:checked + span{border-color:var(--accent);background:var(--accent-soft);color:var(--accent);font-weight:700}
.choice input:focus-visible + span{outline:3px solid var(--focus);outline-offset:2px}
.err{color:var(--miss);min-height:1.5em;margin:12px 0 0;font-size:16px}
.btn{display:inline-flex;align-items:center;justify-content:center;font-weight:700;border:0;border-radius:12px;
  padding:12px 22px;min-height:50px;background:var(--accent);color:var(--accent-ink);cursor:pointer;text-decoration:none}
.btn[disabled]{opacity:.3;cursor:not-allowed}
.btn.line{background:transparent;color:var(--accent);box-shadow:inset 0 0 0 2px var(--accent)}
.intro-actions{margin-top:auto;padding-top:24px}
/* trial header */
.head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
.head .verName{font-weight:700}
.head .count{font-size:15px;color:var(--muted)}
.notice{font-size:15px;color:var(--accent);margin:6px 0 0}
/* sentence navigator */
.nav{display:flex;flex-wrap:wrap;gap:6px;margin:14px 0 4px}
.nav button{min-width:40px;height:40px;padding:0 6px;border-radius:9px;border:2px solid var(--line);
  background:var(--panel);color:var(--muted);cursor:pointer;position:relative;font-size:15px;font-weight:700}
.nav button.heard{border-color:var(--accent);color:var(--accent)}
.nav button.cur{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
.nav button .dot{position:absolute;top:-5px;right:-5px;min-width:18px;height:18px;border-radius:9px;
  background:var(--miss);color:#fff;font-size:11px;line-height:18px;padding:0 4px;font-weight:700}
/* player */
.player{display:flex;flex-direction:column;align-items:center;gap:10px;margin:26px 0 20px}
.ring{position:relative;width:148px;height:148px}
.ring > svg{position:absolute;inset:0;transform:rotate(-90deg)}
.ring circle{fill:none;stroke-width:7}
.ring .bg{stroke:var(--line)}
.ring .fg{stroke:var(--accent);stroke-linecap:round}
.play{position:absolute;inset:13px;border-radius:50%;border:0;background:var(--panel);color:var(--accent);
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;font-weight:700;cursor:pointer}
.play svg{width:28px;height:28px;fill:currentColor}
.play span{font-size:15px;line-height:1.2;padding:0 10px;text-align:center}
.play[disabled]{cursor:default;color:var(--muted)}
.play[disabled] .icon{display:none}
.status{min-height:1.4em;color:var(--muted);text-align:center;font-size:15px;margin:0}
/* transcript */
.hint{color:var(--muted);font-size:16px;margin-bottom:12px}
.words{display:flex;flex-wrap:wrap;gap:6px 4px;font-size:25px;line-height:1.25}
.w{border:2px solid transparent;background:var(--panel);color:var(--ink);border-radius:8px;padding:6px 9px;cursor:pointer;font:inherit}
.w:hover{border-color:var(--line)}
.w[aria-pressed="true"]{background:var(--miss-bg);border-color:var(--miss);color:var(--miss);
  text-decoration:line-through;text-decoration-thickness:2px}
.bar{margin-top:auto;padding-top:26px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.bar .spacer{flex:1}
.nb{gap:6px}
.nb svg{width:20px;height:20px;fill:none;stroke:currentColor;stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round}
.end-box .btn{margin:14px 12px 0 0}
[hidden]{display:none!important}
@media (max-width:440px){ h1{font-size:27px} .words{font-size:22px} .btn{padding:12px 16px} .nb span{display:none} .nb{width:50px;padding:0} }
@media (prefers-reduced-motion:reduce){ *{transition:none!important} }
</style>
</head>
<body>
<main>
  <section class="screen on" id="intro">
    <h1 id="introTitle"></h1>
    <div id="introText"></div>
    <div class="field"><label class="lab" for="pid" id="pidLabel"></label>
      <input type="text" id="pid" autocomplete="name" spellcheck="false"></div>
    <div class="field"><span class="lab" id="versionLabel"></span>
      <div class="choices" id="versionChoices" role="radiogroup"></div></div>
    <p class="err" id="introErr" role="alert"></p>
    <div class="intro-actions"><button class="btn" id="start"></button></div>
  </section>

  <section class="screen" id="trial">
    <div class="head"><span class="verName" id="verName"></span><span class="count" id="count"></span></div>
    <p class="notice" id="notice" hidden></p>
    <nav class="nav" id="nav"></nav>
    <div class="player">
      <div class="ring">
        <svg viewBox="0 0 148 148" aria-hidden="true">
          <circle class="bg" cx="74" cy="74" r="66"/><circle class="fg" id="fg" cx="74" cy="74" r="66"/>
        </svg>
        <button class="play" id="play">
          <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l11-6.5z"/></svg>
          <span id="playLabel"></span>
        </button>
      </div>
      <p class="status" id="status" aria-live="polite"></p>
    </div>
    <div id="text">
      <p class="hint" id="hint"></p>
      <div class="words" id="words"></div>
    </div>
    <div class="bar">
      <button class="btn line nb" id="prev"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 5l-7 7 7 7"/></svg><span id="prevT"></span></button>
      <button class="btn line nb" id="next"><span id="nextT"></span><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 5l7 7-7 7"/></svg></button>
      <span class="spacer"></span>
      <button class="btn" id="finish"></button>
    </div>
  </section>

  <section class="screen" id="end">
    <h1 id="endTitle"></h1>
    <p id="endMsg" aria-live="polite"></p>
    <div class="end-box">
      <a class="btn line" id="download" href="#"></a>
      <button class="btn line" id="review"></button>
      <button class="btn line" id="another"></button>
      <a class="btn" id="done" hidden></a>
    </div>
  </section>
</main>
<script>
(function(){
const CFG = __CFG__;
const VERSIONS = __VERSIONS__;
const T = CFG.ui;
const $ = id => document.getElementById(id);
const fmt = (s, v) => s.replace(/\{(\w+)\}/g, (m, k) => v[k]);
const params = new URLSearchParams(location.search);
const newSession = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
const COLS = ['test','version','participant','session','chunk','word_pos','word','word_clean','altered','rev','final'];
let pid = '', ver = null, chunks = [], st = [], cur = 0, session = '', storeKey = '';
let pending = [], failed = [];

// ---------- local backup (survives a reload; private to this browser) ----------
function saveLocal(){
  try { localStorage.setItem(storeKey, JSON.stringify({session, cur, st: st.map(s => ({...s, marked: [...s.marked]}))})); } catch(e){}
}
function loadLocal(){
  try { const v = JSON.parse(localStorage.getItem(storeKey) || 'null'); return v && v.st && v.st.length === chunks.length ? v : null; }
  catch(e){ return null; }
}

// ---------- intro ----------
$('introTitle').textContent = T.title;
T.intro.forEach(t => { const p = document.createElement('p'); p.textContent = t; $('introText').appendChild(p); });
$('pidLabel').textContent = T.name_label;
$('versionLabel').textContent = T.version_label;
$('start').textContent = T.start;
$('pid').value = params.get('pid') || params.get('name') || '';

function radio(group, value, label, checked, onchange){
  const l = document.createElement('label'); l.className = 'choice';
  const i = document.createElement('input'); i.type = 'radio'; i.name = group; i.value = value; i.checked = !!checked;
  if (onchange) i.onchange = onchange;
  const s = document.createElement('span'); s.textContent = label;
  l.appendChild(i); l.appendChild(s); return l;
}
function pickedVersion(){ const r = document.querySelector('input[name=version]:checked');
  return r ? VERSIONS.find(v => v.id === r.value) : null; }
function drawVersions(pref){
  const box = $('versionChoices'); box.innerHTML = '';
  VERSIONS.forEach(v => box.appendChild(radio('version', v.id, v.label, v.id === pref)));
}
drawVersions(params.get('version') || (VERSIONS.length === 1 ? VERSIONS[0].id : null));

function enterVersion(){
  ver = pickedVersion();
  if (!pid) { $('introErr').textContent = T.name_missing; $('pid').focus(); return false; }
  if (!ver) { $('introErr').textContent = T.choice_missing; return false; }
  $('introErr').textContent = '';
  chunks = ver.chunks;
  storeKey = ['exp', CFG.test, ver.id, pid.toLowerCase()].join('|');
  const saved = loadLocal();
  if (saved) {
    session = saved.session; cur = saved.cur || 0;
    st = saved.st.map(s => ({...s, marked: new Set(s.marked)}));
    $('notice').textContent = T.resumed; $('notice').hidden = false;
  } else {
    session = newSession(); cur = 0;
    st = chunks.map(() => ({heard: false, marked: new Set(), dirty: false, rev: 0}));
    $('notice').hidden = true;
  }
  $('verName').textContent = ver.label;
  window.addEventListener('beforeunload', guard);
  show('trial'); buildNav(); go(cur, true);
  return true;
}
$('start').onclick = () => { pid = $('pid').value.trim(); enterVersion(); };
$('pid').addEventListener('keydown', e => { if (e.key === 'Enter') $('start').click(); });
function guard(e){ e.preventDefault(); e.returnValue = ''; }
function show(id){ document.querySelectorAll('.screen').forEach(s => s.classList.toggle('on', s.id === id)); window.scrollTo(0, 0); }

// ---------- navigation ----------
function buildNav(){
  const nav = $('nav'); nav.innerHTML = ''; nav.setAttribute('aria-label', T.nav_label);
  chunks.forEach((c, i) => { const b = document.createElement('button'); b.type = 'button';
    b.onclick = () => go(i); nav.appendChild(b); });
  updateNav();
}
function updateNav(){
  [...$('nav').children].forEach((b, i) => {
    const s = st[i], m = s.marked.size;
    b.className = (s.heard ? 'heard' : '') + (i === cur ? ' cur' : '');
    b.innerHTML = ''; b.appendChild(document.createTextNode(i + 1));
    if (m) { const d = document.createElement('span'); d.className = 'dot'; d.textContent = m; b.appendChild(d); }
    b.setAttribute('aria-current', i === cur ? 'step' : 'false');
    b.setAttribute('aria-label', fmt(T.sentence_n, {i: i + 1, n: chunks.length}));
  });
}
function leave(){
  const s = st[cur]; audio.pause();
  if (s.dirty) { s.dirty = false; pending.push(sendChunk(cur, 0)); }
  saveLocal();
}
function go(i, first){
  if (!first) leave();
  cur = i;
  audio.src = chunks[i].audio; ring(0);
  $('count').textContent = fmt(T.sentence_n, {i: i + 1, n: chunks.length});
  $('status').textContent = '';
  $('prev').disabled = i === 0; $('next').disabled = i === chunks.length - 1;
  setPlayIdle(); drawWords(); updateNav(); saveLocal();
}
$('prevT').textContent = T.prev; $('nextT').textContent = T.next; $('finish').textContent = T.finish;
$('prev').setAttribute('aria-label', T.prev); $('next').setAttribute('aria-label', T.next);
$('prev').onclick = () => { if (cur > 0) go(cur - 1); };
$('next').onclick = () => { if (cur < chunks.length - 1) go(cur + 1); };

// ---------- player ----------
const audio = new Audio(); audio.preload = 'auto';
const fg = $('fg'), CIRC = 2 * Math.PI * 66;
fg.style.strokeDasharray = CIRC;
const ring = f => { fg.style.strokeDashoffset = CIRC * (1 - Math.max(0, Math.min(1, f))); };
function setPlayIdle(){
  $('play').disabled = false;
  $('playLabel').textContent = st[cur].heard ? T.again : T.listen;
}
$('play').onclick = () => {
  $('play').disabled = true; $('playLabel').textContent = T.playing; $('status').textContent = '';
  ring(0); audio.currentTime = 0;
  audio.play().catch(() => { setPlayIdle(); $('status').textContent = T.tap_to_play; });
};
audio.ontimeupdate = () => { if (audio.duration) ring(audio.currentTime / audio.duration); };
audio.onended = () => {
  ring(1); const s = st[cur];
  const first = !s.heard; s.heard = true; s.dirty = true;
  setPlayIdle(); updateNav(); saveLocal();
  if (first) drawWords();
};
audio.onerror = () => { if (audio.src) { $('status').textContent = T.audio_error; setPlayIdle(); } };

// ---------- transcript ----------
function drawWords(){
  const s = st[cur], box = $('words'); box.innerHTML = '';
  if (CFG.text_after_listen && !s.heard) { $('hint').textContent = T.wait_text; return; }
  $('hint').textContent = T.mark;
  chunks[cur].words.forEach((w, k) => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'w'; b.textContent = w;
    b.setAttribute('aria-pressed', String(s.marked.has(k)));
    b.onclick = () => {
      const on = !s.marked.has(k); on ? s.marked.add(k) : s.marked.delete(k);
      b.setAttribute('aria-pressed', String(on)); s.dirty = true; updateNav(); saveLocal();
    };
    box.appendChild(b);
  });
}

// ---------- saving: ONLY the marked (not-understood) words go to the sheet ----------
const clean = w => w.toLowerCase().replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
function chunkRows(i, final){
  const s = st[i], c = chunks[i];
  s.rev += 1;
  const marked = [...s.marked].sort((a, b) => a - b);
  const base = {test: CFG.test, version: ver.id, participant: pid, session, chunk: i + 1, rev: s.rev, final};
  if (!marked.length) return [{...base, word_pos: '', word: '', word_clean: '', altered: ''}];
  return marked.map(k => ({...base, word_pos: k + 1, word: c.words[k], word_clean: clean(c.words[k]), altered: c.altered[k] ? 1 : 0}));
}
function post(rows){
  if (!CFG.submit_url || !rows.length) return Promise.resolve(true);
  return fetch(CFG.submit_url, {method: 'POST', mode: 'no-cors',
      headers: {'Content-Type': 'text/plain;charset=utf-8'}, body: JSON.stringify({rows})})
    .then(() => true).catch(() => { failed.push(...rows); return false; });
}
const sendChunk = (i, final) => post(chunkRows(i, final));
function toCSV(list){
  const esc = v => { const s = String(v ?? ''); return /[",\n;]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
  return [COLS.join(','), ...list.map(r => COLS.map(c => esc(r[c])).join(','))].join('\n');
}
async function finishNow(){
  const unheard = st.filter(s => !s.heard).length;
  if (unheard && !confirm(fmt(T.confirm_unheard, {n: unheard}))) return;
  leave();
  const all = [];
  chunks.forEach((c, i) => all.push(...chunkRows(i, 1)));
  window.removeEventListener('beforeunload', guard);
  show('end');
  $('endTitle').textContent = T.end_title;
  const flagged = all.filter(r => r.word_pos !== '');
  const a = $('download'); a.textContent = T.download;
  a.href = URL.createObjectURL(new Blob(['\ufeff' + toCSV(flagged)], {type: 'text/csv;charset=utf-8'}));
  a.download = [CFG.test, ver.id, pid].join('_').replace(/[^\w.-]+/g, '-') + '.csv';
  $('review').textContent = T.review;
  $('another').textContent = T.another_version;
  if (CFG.completion_url) { $('done').href = CFG.completion_url; $('done').textContent = T.done; $('done').hidden = false; }
  if (!CFG.submit_url) { $('endMsg').textContent = T.no_url; return; }
  $('endMsg').textContent = T.sending;
  await Promise.all(pending);
  failed.splice(0);
  const ok = await post(all);
  $('endMsg').textContent = ok ? T.saved : T.save_failed;
}
$('finish').onclick = finishNow;
$('review').onclick = () => { show('trial'); window.addEventListener('beforeunload', guard); buildNav(); go(cur, true); };
$('another').onclick = () => { show('intro'); $('introErr').textContent = ''; drawVersions(); };
})();
</script>
</body>
</html>
"""


def story_words(versions):
    """Every word of the story, from the SAVED analysis pages — the denominator
    the response CSV does not contain (it only holds the words people flagged).

    versions : the same {version_id: "outputs/<name>.html"} dict you passed to
               build_experiment().
    Returns a DataFrame: version, chunk, word_pos, word, word_clean, altered."""
    import pandas as pd
    rows = []
    for vid, html_path in versions.items():
        data = load_visualization(html_path)
        for ci, c in enumerate(_chunks_from_segs(data["segs"], data["audio"], data["sr"]), 1):
            for k, (w, alt) in enumerate(zip(c["words"], c["altered"]), 1):
                rows.append({"version": vid, "chunk": ci, "word_pos": k, "word": w,
                             "word_clean": re.sub(r"^[^\w]+|[^\w]+$", "", w.lower(), flags=re.UNICODE),
                             "altered": int(alt)})
    return pd.DataFrame(rows)


def responses_by_word(csv_path, versions):
    """Join the response CSV onto the full word list, so every word of every
    version a participant did gets not_understood = 0 or 1.
    This is the table to use for rates and for plotting.

    csv_path : the responses CSV exported from the sheet
    versions : {version_id: "outputs/<name>.html"} (same as build_experiment)
    Returns one row per participant-session × word."""
    import pandas as pd
    marked = load_responses(csv_path)
    words = story_words(versions)
    who = marked[["participant", "session", "version"]].drop_duplicates()
    grid = who.merge(words, on="version", how="left")
    hit = marked[["session", "version", "chunk", "word_pos"]].drop_duplicates()
    hit["not_understood"] = 1
    out = grid.merge(hit, on=["session", "version", "chunk", "word_pos"], how="left")
    out["not_understood"] = out["not_understood"].fillna(0).astype(int)
    return out.sort_values(["participant", "session", "chunk", "word_pos"]).reset_index(drop=True)
