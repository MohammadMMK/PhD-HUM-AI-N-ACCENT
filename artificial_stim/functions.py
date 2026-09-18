"""
functions.py — phoneme/audio alignment and the interactive HTML player.

The pipeline is split in three:

    text_manipulation.py   text -> IPA, substitution rules, statistics
    speech_generation.py   the Kokoro model, synthesis, prosody transfer
    functions.py           this file: line up phonemes with the audio and draw it

Both other modules are re-exported here, so `from functions import ...` keeps
working for everything.
"""

import json
import os

# text_manipulation sets the PHONEMIZER_ESPEAK_* paths the espeak backend needs,
# so it has to be imported before anything pulls in phonemizer.
from text_manipulation import (
    # text -> phonemes
    one_sentence_chunks, text_to_ipa_chunks, get_g2p, LANG_CODE,
    # vocab / validation
    VOCAB, INV, check_ipa,
    # phoneme inventory (the aligner below groups characters with these)
    VOWELS, CONSONANTS, BASE, LEADING, TRAILING, SKIP, BOUNDARY, phoneme_units,
    # rules, manipulation, statistics
    Rule, as_rules, describe_rules, manipulate, substitution_stats, print_stats,
)
from speech_generation import (
    SR, FRAME_SAMPLES, get_model, load_voice,
    Prosody, extract_prosody, Speech, Chunk,
    synthesize, synthesize_forced, synthesize_pair,
)

# =============================================================================
# ALIGNMENT + INTERACTIVE VISUALIZATION
#   align_phonemes(...)        -> (segments, seg_word_idx, seg_altered)
#   align_speech(...)          -> (audio, segments, seg_words, seg_altered)
#   synth_aligned(...)         -> synthesise, then align, in one call
#   audiovisualize_interactive -> optional kwargs: seg_words=None, text=None,
#                                 stats=None, rules=None
# =============================================================================
import numpy as np
import base64, io, json, soundfile as sf
from IPython.display import HTML


def align_phonemes(phonemes, pred_dur, n_samples=None, sr=SR, lead_trim_frames=0, altered_mask=None):
    """altered_mask (optional): bool list the same length as `phonemes`, e.g.
    from manipulate(..., return_mask=True). If given, each emitted segment
    gets an `altered` flag (True if any character contributing to it — base
    or a merged trailing diacritic — was flagged)."""
    kept, kept_altered = [], []
    for idx, c in enumerate(phonemes):
        if c in VOCAB:
            kept.append(c)
            kept_altered.append(bool(altered_mask[idx]) if altered_mask is not None else False)
    dur = [int(x) for x in pred_dur]
    assert len(dur) == len(kept) + 2, \
        f"pred_dur ({len(dur)}) != kept phonemes+2 ({len(kept)+2}) — filtered string mismatch"
    spf = (n_samples / sum(dur)) if n_samples else FRAME_SAMPLES
    f2s = lambda f: max(0.0, f - lead_trim_frames) * spf / sr
    start_f, acc = [], 0
    for d in dur:
        start_f.append(acc); acc += d
    segs, pending, pending_altered = [], None, False
    word_idx = 0
    seg_word_idx, seg_altered = [], []
    for k, c in enumerate(kept, start=1):
        s, e = start_f[k], start_f[k] + dur[k]
        c_alt = kept_altered[k - 1]
        if c in BASE:
            segs.append([c, pending if pending is not None else s, e])
            seg_word_idx.append(word_idx)
            seg_altered.append(c_alt or pending_altered)
            pending, pending_altered = None, False
        elif c in TRAILING and segs:
            segs[-1][0] += c; segs[-1][2] = e
            if c_alt:
                seg_altered[-1] = True
        elif c in LEADING:
            pending = s if pending is None else pending
            pending_altered = pending_altered or c_alt
        else:
            pending, pending_altered = None, False
            if c == ' ':
                word_idx += 1
    return [(lab, f2s(s), f2s(e)) for lab, s, e in segs], seg_word_idx, seg_altered


def align_speech(speech, masks=None, sr=SR):
    """Lay a Speech out on one timeline: (audio, segments, seg_words, seg_altered).

    `masks` (optional) is a list parallel to the chunks, each entry the bool mask
    for that chunk from manipulate(..., return_mask=True), so the player can
    highlight the manipulated phonemes.
    """
    segments, seg_words, seg_altered = [], [], []
    t0, word_offset = 0.0, 0
    for ci, chunk in enumerate(speech.chunks):
        chunk_mask = masks[ci] if masks is not None else None
        segs, widx, altd = align_phonemes(chunk.phonemes, chunk.durations,
                                          n_samples=len(chunk.audio), sr=sr,
                                          altered_mask=chunk_mask)
        segments += [(lab, s + t0, e + t0) for lab, s, e in segs]
        seg_words += [w + word_offset for w in widx]
        seg_altered += altd
        word_offset += (widx[-1] + 1) if widx else 0
        t0 += len(chunk.audio) / sr
    return speech.audio, segments, seg_words, seg_altered


def synth_aligned(ipa_chunks, voice, speed=1.0, sr=SR, masks=None, donor=None, **kw):
    """Synthesise and align in one call: (audio, segments, seg_words, seg_altered).

    `donor` (optional) forces another generation's prosody onto this one — pass
    the ORIGINAL ipa chunks, or an extract_prosody() result — so a manipulated
    stimulus keeps its baseline's timing and pitch. Extra keyword arguments go
    to synthesize_forced (duration=, f0=, energy=, seed=, ...).
    """
    speech = synthesize_forced(ipa_chunks, voice, donor=donor, speed=speed, **kw)
    return align_speech(speech, masks=masks, sr=sr)


def audiovisualize_interactive(audio, segments, sr=24000, out_html=None, per_row=None,
                                seg_words=None, text=None, seg_altered=None,
                                title=None, rules=None, stats=None):
    """title (optional): short heading shown at the top of the player.
    stats (optional): from substitution_stats(ipa_chunks, manip_chunks, rules).
    Renders a legend with, for every phoneme the rules touched, how often each
    outcome actually occurred vs the probability the rule asked for.
    rules (optional): the same `rules` list you passed to manipulate() — it's
    turned into a one-line human-readable summary and shown under the title.
    Falls back gracefully (rule kinds only) if describe_rules() isn't in
    scope yet, so this never raises just because the engine module wasn't
    imported first."""
    audio = np.asarray(audio, dtype=np.float32)
    buf = io.BytesIO(); sf.write(buf, audio, sr, format='WAV')
    b64 = base64.b64encode(buf.getvalue()).decode()
    N = 2000
    step = max(1, len(audio)//N)
    env = np.abs(audio[:step*(len(audio)//step)].reshape(-1, step)).max(axis=1)
    env = (env/(env.max() or 1)).round(3).tolist()
    segs = [{"l": lab, "s": round(s, 4), "e": round(e, 4)} for lab, s, e in segments]
    dur = len(audio)/sr
    if seg_words is not None and text is not None:
        n_words = (max(seg_words) + 1) if seg_words else 0
        word_labels = text.split()
        if len(word_labels) != n_words:
            print(f"⚠ word count mismatch: text has {len(word_labels)} words, "
                  f"phoneme string implies {n_words} — labels may be misaligned")
        for i, seg in enumerate(segs):
            wi = seg_words[i]
            seg["w"] = wi
            seg["wl"] = word_labels[wi] if wi < len(word_labels) else ""
    if seg_altered is not None:
        if len(seg_altered) != len(segs):
            print(f"⚠ seg_altered length ({len(seg_altered)}) != segments ({len(segs)}) — skipping alter-highlighting")
        else:
            for i, seg in enumerate(segs):
                seg["alt"] = bool(seg_altered[i])

    rules_text = ""
    if rules:
        try:
            rules_text = describe_rules(rules)
        except NameError:
            rules_text = "; ".join(r.get("kind", "?") for r in rules)

    html = _TEMPLATE.replace("__B64__", b64).replace("__SEGS__", json.dumps(segs)) \
                    .replace("__ENV__", json.dumps(env)).replace("__DUR__", str(dur)) \
                    .replace("__TITLE__", json.dumps(title or "")) \
                    .replace("__RULES__", json.dumps(rules_text)) \
                    .replace("__STATS__", json.dumps(stats or []))
    if out_html:
        out_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(out_dir, exist_ok=True)

        full_path = os.path.join(out_dir, out_html)
        print(f"Writing interactive HTML to: {full_path}")
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(html)

    return html


_TEMPLATE = r"""
<div id="pv" style="font-family:system-ui,sans-serif;max-width:1080px">
<h2 id="pageTitle" style="margin:0 0 4px;font-size:20px;font-weight:700;color:#111;"></h2>
<div id="rulesBox" style="font-size:12.5px;color:#555;margin-bottom:12px;padding:7px 11px;background:#f3f4f6;border-radius:6px;display:none;max-height:60px;overflow-y:auto;"></div>
<div id="statsBox" style="display:none;margin:0 0 12px;padding:9px 12px;border:1px solid #e5e7eb;border-radius:8px;background:#fffdf7;font-size:13px;color:#333;">
    <div style="font-weight:700;margin-bottom:6px;color:#111;">Substitutions <span id="statsTotal" style="font-weight:400;color:#666;"></span></div>
    <div id="statsBody"></div>
</div>
<audio id="au" src="data:audio/wav;base64,__B64__"></audio>
<div style="display:flex;gap:8px;align-items:center;margin-bottom:2px">
    <button id="pp" style="padding:6px 14px;border:0;border-radius:6px;background:#2563eb;color:#fff;cursor:pointer">▶ play</button>
    <input id="sp" type="range" min="0.5" max="1.5" step="0.05" value="1" style="width:120px">
    <span id="spl" style="font-size:12px;color:#555">1.00×</span>
    <span id="tt" style="font-size:12px;color:#555;margin-left:auto">0.00 / __DUR__s</span>
</div>
<div id="legend" style="font-size:11px;color:#666;margin:2px 0 8px;display:none;">
    <span style="border-bottom:3px solid #f59e0b;padding:0 3px;color:#b45309;font-weight:700;">phoneme</span>
    &nbsp;= altered by manipulation rule
</div>
<canvas id="wf" width="1040" height="80" style="width:100%;height:80px;background:#0b1020;border-radius:6px;cursor:pointer"></canvas>
<div id="ph" style="display:flex;flex-wrap:wrap;align-items:flex-start;gap:0;margin-top:16px;"></div>
</div>
<script>
(function(){
const segs=__SEGS__, env=__ENV__, DUR=__DUR__;
const TITLE=__TITLE__, RULES=__RULES__, STATS=__STATS__;
const au=document.getElementById('au'), pp=document.getElementById('pp');
const ph=document.getElementById('ph'), wf=document.getElementById('wf'), ctx=wf.getContext('2d');
const tt=document.getElementById('tt'), sp=document.getElementById('sp'), spl=document.getElementById('spl');
const legend=document.getElementById('legend');
let view=[0,DUR];

if (TITLE) document.getElementById('pageTitle').textContent = TITLE;
if (RULES) {
    const rb = document.getElementById('rulesBox');
    rb.textContent = 'Rules applied: ' + RULES;
    rb.style.display = 'block';
}

if (segs.some(g=>g.alt)) legend.style.display='block';

// ---- substitution-count legend ----
if (STATS && STATS.length) {
    const box=document.getElementById('statsBox'), body=document.getElementById('statsBody');
    const el=(tag,css,txt)=>{ const e=document.createElement(tag); if(css) e.style.cssText=css; if(txt!==undefined) e.textContent=txt; return e; };
    let grand=0;
    STATS.forEach((r,ri)=>{
        grand+=r.changed;
        const blk=el('div', 'margin:0 0 8px;' + (ri ? 'padding-top:7px;border-top:1px dashed #e5e7eb;' : ''));
        const head=el('div','font-size:12px;color:#555;margin-bottom:3px;');
        if (STATS.length > 1) head.appendChild(el('b','color:#111;', 'Rule '+(ri+1)+': '));
        head.appendChild(document.createTextNode(r.label+'  '));
        head.appendChild(el('span','color:#b45309;font-weight:700;', '('+r.changed+' change'+(r.changed===1?'':'s')+')'));
        blk.appendChild(head);
        if (!r.groups.length) blk.appendChild(el('div','color:#999;font-size:12px;margin-left:12px;','nothing changed'));
        r.groups.forEach(g=>{
            const row = el('div','display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:3px 0 3px 12px;');
            row.appendChild(el('span','font-size:12px;color:#666;min-width:64px;', g.old+' ×'+g.total));
            g.rows.forEach(o=>{
                const pct = g.total ? Math.round(100*o.n/g.total) : 0;
                const chip=el('span','display:inline-flex;align-items:center;gap:6px;padding:3px 8px;border-radius:6px;'
                    + (o.changed ? 'background:#fef3c7;border:1px solid #fcd34d;' : 'background:#f3f4f6;border:1px solid #e5e7eb;'));
                chip.appendChild(el('span','font-size:16px;', g.old+' → '+o.new));
                chip.appendChild(el('b', o.changed ? 'color:#b45309;' : 'color:#555;', String(o.n)));
                if (o.p !== null) {
                    chip.appendChild(el('span','font-size:11px;color:#666;', pct+'% (target '+Math.round(100*o.p)+'%)'));
                    const bar=el('span','display:inline-block;width:46px;height:6px;background:#e5e7eb;border-radius:3px;position:relative;overflow:hidden;');
                    bar.appendChild(el('span','position:absolute;left:0;top:0;bottom:0;width:'+pct+'%;background:'+(o.changed?'#f59e0b':'#9ca3af')+';'));
                    bar.appendChild(el('span','position:absolute;top:0;bottom:0;width:2px;background:#111;left:calc('+Math.round(100*o.p)+'% - 1px);'));
                    chip.appendChild(bar);
                }
                if (!o.changed) chip.title='kept as itself (not highlighted)';
                row.appendChild(chip);
            });
            if (!row.parentNode) blk.appendChild(row);
        });
        body.appendChild(blk);
    });
    document.getElementById('statsTotal').textContent='— '+grand+' change'+(grand===1?'':'s')+' in total';
    box.style.display='block';
}
let groups=[];
if (segs.length && segs[0].w !== undefined) {
    segs.forEach((g,i)=>{
        const last=groups[groups.length-1];
        if (last && last.w===g.w) { last.segIdx.push(i); last.e=g.e; }
        else { groups.push({w:g.w, wl:g.wl, s:g.s, e:g.e, segIdx:[i]}); }
    });
} else {
    groups = segs.map((g,i)=>({w:i, wl:null, s:g.s, e:g.e, segIdx:[i]}));
}
groups.forEach((grp, gi)=>{
    const wrap=document.createElement('div');
    wrap.dataset.gi=gi;
    wrap.style.cssText='display:inline-block;vertical-align:top;margin:2px 10px 12px 0;padding:6px 10px;border-radius:9px;background:#f8f9fb;border:1.5px solid #e5e7eb;transition:.08s;';
    if (grp.wl !== null) {
        const wl=document.createElement('div');
        wl.textContent = grp.wl || '·';
        wl.style.cssText='font-size:23px;font-weight:700;color:#111;margin-bottom:4px;letter-spacing:.2px;';
        wrap.appendChild(wl);
    }
    const prow=document.createElement('div');
    prow.style.cssText='font-size:21px;letter-spacing:1.5px;white-space:nowrap;';
    grp.segIdx.forEach(i=>{
        const s=document.createElement('span');
        s.textContent=segs[i].l; s.dataset.i=i;
        const altered = !!segs[i].alt;
        s.dataset.alt = altered ? '1' : '0';
        s.style.cssText='padding:2px 6px;margin:0 1px;border-radius:6px;cursor:pointer;transition:.05s;'
            + (altered ? 'border-bottom:3px solid #f59e0b;color:#b45309;font-weight:700;' : '');
        if (altered) s.title = 'manipulated phoneme';
        s.onclick=()=>{ au.currentTime=segs[i].s; au.play(); };
        prow.appendChild(s);
    });
    wrap.appendChild(prow);
    ph.appendChild(wrap);
});
const chips=[...ph.querySelectorAll('[data-i]')];
const wordWraps=[...ph.querySelectorAll('[data-gi]')];
function drawWave(){ const W=wf.width,H=wf.height; ctx.clearRect(0,0,W,H);
    const [a,b]=view, n=env.length;
    segs.forEach(g=>{ if(g.e<a||g.s>b)return; const x=(g.s-a)/(b-a)*W, w=(g.e-g.s)/(b-a)*W;
        ctx.fillStyle=(g.s<=au.currentTime&&au.currentTime<g.e)?'#2563eb55':'#ffffff10'; ctx.fillRect(x,0,Math.max(w,1),H);});
    ctx.strokeStyle='#ffffff30';
    groups.forEach(g=>{ if(g.s<a||g.s>b)return; const x=(g.s-a)/(b-a)*W;
        ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); });
    ctx.strokeStyle='#7dd3fc'; ctx.beginPath();
    for(let x=0;x<W;x++){ const t=a+(b-a)*x/W, idx=Math.floor(t/DUR*n); const v=env[Math.max(0,Math.min(n-1,idx))]||0;
        ctx.moveTo(x,H/2-v*H/2); ctx.lineTo(x,H/2+v*H/2);} ctx.stroke();
    const cx=(au.currentTime-a)/(b-a)*W; ctx.strokeStyle='#f43f5e'; ctx.lineWidth=2;
    ctx.beginPath(); ctx.moveTo(cx,0); ctx.lineTo(cx,H); ctx.stroke(); ctx.lineWidth=1;
}
function tick(){ const t=au.currentTime; let act=-1;
    for(let i=0;i<segs.length;i++){ if(segs[i].s<=t && t<segs[i].e){act=i;break;} }
    chips.forEach((c,i)=>{ const on=(+c.dataset.i===act); const altered=c.dataset.alt==='1';
        c.style.background=on?'#2563eb':'transparent';
        c.style.color=on?'#fff':(altered?'#b45309':'#111'); });
    wordWraps.forEach(w=>{ w.style.borderColor='#e5e7eb'; w.style.background='#f8f9fb'; });
    if (act>=0){
        const wrap=chips[act].closest('[data-gi]');
        if (wrap){ wrap.style.borderColor='#2563eb'; wrap.style.background='#eef2ff'; }
    }
    tt.textContent=t.toFixed(2)+' / '+DUR.toFixed(2)+'s'; drawWave();
    if(!au.paused) requestAnimationFrame(tick);
}
pp.onclick=()=>{ au.paused?au.play():au.pause(); };
au.onplay=()=>{pp.textContent='⏸ pause'; tick();};
au.onpause=()=>{pp.textContent='▶ play';};
au.onended=()=>{pp.textContent='▶ play';};
sp.oninput=()=>{ au.playbackRate=+sp.value; spl.textContent=(+sp.value).toFixed(2)+'×'; };
wf.onclick=e=>{ const r=wf.getBoundingClientRect(); const f=(e.clientX-r.left)/r.width;
    au.currentTime=view[0]+(view[1]-view[0])*f; if(au.paused) drawWave(); };
wf.onwheel=e=>{ e.preventDefault(); const r=wf.getBoundingClientRect(); const f=(e.clientX-r.left)/r.width;
    const c=view[0]+(view[1]-view[0])*f, z=e.deltaY<0?0.8:1.25; let w=(view[1]-view[0])*z;
    w=Math.max(0.3,Math.min(DUR,w)); let a=c-f*w, b=a+w; a=Math.max(0,a); b=Math.min(DUR,a+w);
    view=[a,b]; drawWave(); };
drawWave();
})();
</script>
"""