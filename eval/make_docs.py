#!/usr/bin/env python3
"""Erzeugt die statischen Vergleichsseiten unter docs/ (GitHub-Pages-tauglich).

results/ wird automatisch gescannt und nach (TTS-Modell, Stimme) gruppiert;
je Kombination wird der NEUESTE vollständige Lauf publiziert. Der Seitenname
ist stabil aus Modell+Stimme abgeleitet — ein neuer Lauf derselben
Kombination überschreibt also die bestehende Seite. Seiten/Audio zu
Kombinationen, die es nicht mehr gibt, werden entfernt.

Es entsteht:
  docs/index.html          – Modellvergleich (Metriken, Kategorien, Links)
  docs/<modell>-<stimme>.html
  docs/audio/<modell>-<stimme>/*.mp3

Je Fall EIN Clip (Wiederholung r0 — repräsentativ, nicht der beste Repeat),
MP3 statt WAV, damit das Repo klein bleibt (~3 KB/s statt ~44 KB/s). Die
WER-Angabe pro Fall ist der Mittelwert über alle Repeats.

Unvollständige Läufe (Smoke, --limit, --category) werden übersprungen
(weniger als MIN_CASES Fälle).

Abhängigkeit: soundfile (pip install --break-system-packages soundfile — das
System-Python ist PEP-668-verwaltet; braucht libsndfile >= 1.2 für MP3).

Aufruf:  python eval/make_docs.py            # aus dem Repo-Root
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
DOCS = REPO / "docs"
MIN_CASES = 40  # Testset hat 43 Fälle; alles darunter ist ein Teil-Lauf

# Lizenz-Kurzhinweis je Modell (Substring-Match auf tts_model) — Hinweise,
# keine Rechtsberatung; verbindlich sind die Lizenztexte der Anbieter.
LICENSES = [
    ("chatterbox", "MIT (Audio enthält Perth-Wasserzeichen)"),
    ("Qwen", "Apache-2.0"),
    ("VoxCPM", "Apache-2.0"),
    ("magpie", "NVIDIA Open Model License"),
    ("Voxtral", "CC BY-NC 4.0 (nicht-kommerziell)"),
]

DISCLAIMER = (
    "Alle Clips sind KI-generiert (synthetische Sprache). Die Lizenzangaben "
    "sind Hinweise auf die jeweiligen Modell-Lizenzen, keine Rechtsberatung — "
    "verbindlich sind allein die Lizenztexte der Modellanbieter."
)

# SouthByte Web-CI (southbyte-brand skill, references/web-ci.md + colors.md):
# Dark-Theme, Matrix-Grid, Wortmarke SOUTH.BYTE, Mono-Überschriften in Grün.
CSS = """
 :root{--bg:#060C0A;--bg-raised:#0A1410;--bg-card:#0E1A14;--border:#162A1E;--border-hi:#1A5C38;
   --green:#00E676;--green-dim:#00994A;--amber:#F59E0B;--text:#D4EDE0;--text-muted:#5E8A72;--text-dim:#2E5040;
   --ko:#FF5A5A;--mono:'Courier New',Consolas,'Cascadia Code','SF Mono',Menlo,monospace;
   --sans:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);line-height:1.7}
 .grid-bg{position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.5;
   background-image:linear-gradient(rgba(0,230,118,.15) 1px,transparent 1px),
     linear-gradient(90deg,rgba(0,230,118,.15) 1px,transparent 1px);background-size:80px 80px}
 .wrap{position:relative;z-index:1;max-width:80rem;margin:0 auto;padding:2.5rem 1.25rem}
 .masthead .wordmark{font-family:var(--mono);font-weight:700;font-size:1.4rem;letter-spacing:1.4px;
   color:var(--text);text-decoration:none}
 .masthead .wordmark .dot{color:var(--green)}
 .masthead .tagline{font-family:var(--mono);font-size:.68rem;letter-spacing:.25em;text-transform:uppercase;
   color:var(--text-muted);margin-top:.3rem}
 h1{font-family:var(--mono);color:var(--text);font-size:1.7rem;margin:1.4rem 0 .4rem}
 h2{font-family:var(--mono);text-transform:uppercase;letter-spacing:.15em;color:var(--green);
    border-top:1px solid var(--border-hi);padding-top:.6rem;margin-top:2.4rem}
 a{color:var(--green)} a:hover{color:var(--green-dim)}
 .tablewrap{overflow-x:auto}
 table{border-collapse:collapse;margin:1rem 0}
 th,td{border:1px solid var(--border);padding:.35rem .6rem;text-align:right}
 th{font-family:var(--mono);font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;
   color:var(--text-muted);background:var(--bg-raised)}
 th:first-child,td:first-child{text-align:left}
 td.best{font-weight:700;color:var(--green);background:var(--bg-raised)}
 td.lic{text-align:left;font-size:.82em;white-space:nowrap}
 .case{border:1px solid var(--border);border-radius:8px;padding:.8rem 1rem;margin:.8rem 0;background:var(--bg-card)}
 .case.bad{border-left:6px solid var(--ko)} .case.mid{border-left:6px solid var(--amber)}
 .case.good{border-left:6px solid var(--green)}
 .text{font-weight:600;color:var(--text)}
 .transcript{color:var(--text-muted);font-style:italic}
 .prompt{margin:.6rem 0;padding:.6rem .8rem;background:var(--bg-raised);border-left:3px solid var(--green);border-radius:3px}
 .prompt b{display:block;font-size:.85rem;text-transform:uppercase;letter-spacing:.03em;
   color:var(--text-muted);margin-bottom:.3rem;font-family:var(--mono)}
 .prompt code{font-size:.9rem;white-space:pre-wrap;word-break:break-word;color:var(--green)}
 .meta{color:var(--text-muted);font-size:.85rem}
 footer{margin-top:3rem;color:var(--text-muted);font-size:.85rem;border-top:1px solid var(--border);padding-top:1rem}
 footer .wm{font-family:var(--mono);font-weight:700;letter-spacing:1px;color:var(--text)}
 footer .wm .dot{color:var(--green)}
 audio{height:2rem;vertical-align:middle;margin:.2rem .4rem .2rem 0}
 @keyframes scanline{0%{transform:translateY(-100vh)}100%{transform:translateY(100vh)}}
 .scanline{position:fixed;left:0;top:0;width:100%;height:80px;background:linear-gradient(to bottom,transparent,rgba(0,230,118,.03) 40%,rgba(0,230,118,.07) 50%,rgba(0,230,118,.03) 60%,transparent);pointer-events:none;z-index:0;animation:scanline 8s linear infinite;will-change:transform}
 @media(prefers-reduced-motion:reduce){.scanline{display:none}}
 table.sortable th{cursor:pointer;user-select:none}
 table.sortable th::after{content:' ';opacity:.35;font-size:.75em}
 table.sortable th[aria-sort=ascending]::after{content:' \\25B2';opacity:.9}
 table.sortable th[aria-sort=descending]::after{content:' \\25BC';opacity:.9}
"""

FAVICON = '<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzMiAzMiIgcm9sZT0iaW1nIiBhcmlhLWxhYmVsPSJTb3V0aEJ5dGUiPgogIDx0aXRsZT5Tb3V0aEJ5dGU8L3RpdGxlPgogIDxyZWN0IHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgZmlsbD0iIzA2MEMwQSIvPgogIDx0ZXh0IHg9IjIiIHk9IjIzIgogICAgICAgIGZvbnQtZmFtaWx5PSInQ291cmllciBOZXcnLCBDb25zb2xhcywgJ1NGIE1vbm8nLCBtb25vc3BhY2UiCiAgICAgICAgZm9udC1zaXplPSIxNiIKICAgICAgICBmb250LXdlaWdodD0iNzAwIgogICAgICAgIGxldHRlci1zcGFjaW5nPSIwLjUiPgogICAgPHRzcGFuIGZpbGw9IiNENEVERTAiPlM8L3RzcGFuPjx0c3BhbiBmaWxsPSIjMDBFNjc2Ij4uPC90c3Bhbj48dHNwYW4gZmlsbD0iI0Q0RURFMCI+QjwvdHNwYW4+CiAgPC90ZXh0PgogIDxyZWN0IHg9IjIiIHk9IjI2IiB3aWR0aD0iMjgiIGhlaWdodD0iMS41IiBmaWxsPSIjMDBFNjc2IiBvcGFjaXR5PSIwLjQiLz4KPC9zdmc+Cg==">'

SORT_SCRIPT = """
<script>
(function(){
  function val(td){var s=td.getAttribute('data-sort');if(s===null){var el=td.querySelector('[data-sort]');if(el)s=el.getAttribute('data-sort');}return (s!==null?s:(td.textContent||'')).trim();}
  function num(t){var m=t.replace(/\\u00a0/g,'').replace(/\\s+/g,'').replace(',','.').match(/-?\\d+(?:\\.\\d+)?/);return m?parseFloat(m[0]):null;}
  function isEmpty(t){return t===''||t==='\\u2014'||t==='-';}
  function sortTable(table,idx,asc){
    var tb=table.tBodies[0]; if(!tb) return;
    var rows=Array.prototype.slice.call(tb.rows);
    var allNum=rows.every(function(r){var c=r.cells[idx];if(!c)return true;var v=val(c);return isEmpty(v)||num(v)!==null;});
    rows.sort(function(a,b){
      var av=a.cells[idx]?val(a.cells[idx]):'',bv=b.cells[idx]?val(b.cells[idx]):'';
      var e1=isEmpty(av),e2=isEmpty(bv);
      if(e1&&e2)return 0; if(e1)return 1; if(e2)return -1;
      var r=allNum?((num(av)||0)-(num(bv)||0)):av.localeCompare(bv,'de',{numeric:true});
      return asc?r:-r;
    });
    rows.forEach(function(r){tb.appendChild(r);});
  }
  document.querySelectorAll('table.sortable').forEach(function(table){
    var head=table.tHead; if(!head||!head.rows.length) return;
    Array.prototype.forEach.call(head.rows[0].cells,function(th,idx){
      th.setAttribute('title','Klick: sortieren');
      th.addEventListener('click',function(){
        var asc=th.getAttribute('aria-sort')!=='ascending';
        Array.prototype.forEach.call(head.rows[0].cells,function(o){o.removeAttribute('aria-sort');});
        th.setAttribute('aria-sort',asc?'ascending':'descending');
        sortTable(table,idx,asc);
      });
    });
  });
})();
</script>
"""

_MASTHEAD = ('<header class="masthead"><a class="wordmark" href="index.html">'
             'SOUTH<span class="dot">.</span>BYTE</a>'
             '<div class="tagline">AI Governance &amp; IT-Beratung</div></header>')


def page(title: str, body: str) -> str:
    foot = (f'<footer><span class="wm">SOUTH<span class="dot">.</span>BYTE</span> · '
            f'{html.escape(DISCLAIMER)} · '
            f'<a href="https://southbyte.de">southbyte.de</a></footer>')
    return (f'<!doctype html><html lang="de"><head><meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{html.escape(title)}</title>\n{FAVICON}\n<style>{CSS}</style></head><body>\n"
            f'<div class="grid-bg"></div><div class="scanline"></div><div class="wrap">\n{_MASTHEAD}\n'
            f"{body}\n{foot}\n</div>{SORT_SCRIPT}</body></html>")


def model_display(tts_model: str) -> str:
    """'/hf_models/Qwen--Qwen3-TTS-…' → 'Qwen3-TTS-…'; Pfad/Vendor-Präfix weg."""
    base = tts_model.rstrip("/").rsplit("/", 1)[-1]
    if "--" in base:
        base = base.split("--", 1)[1]
    return base


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def license_note(tts_model: str) -> str:
    for needle, note in LICENSES:
        if needle.lower() in tts_model.lower():
            return note
    return "unbekannt — Modellkarte prüfen"


def run_date(res_dir: Path) -> tuple[str, float]:
    """Sortierschlüssel 'neuester Lauf': Datums-Präfix, Gleichstand per mtime."""
    m = re.match(r"\d{4}-\d{2}-\d{2}", res_dir.name)
    return (m.group(0) if m else "", res_dir.stat().st_mtime)


def discover_runs() -> list[dict]:
    """Neuester vollständiger Lauf je (tts_model, voice), sortiert nach WER."""
    by_combo: dict[tuple, dict] = {}
    for d in sorted(RESULTS.iterdir()):
        sfile = d / "summary.json"
        if not sfile.exists():
            continue
        s = json.loads(sfile.read_text(encoding="utf-8"))
        if s.get("n_ok", 0) < MIN_CASES:
            print(f"übersprungen (Teil-Lauf, n_ok={s.get('n_ok')}): {d.name}")
            continue
        combo = (s["tts_model"], s["voice"])
        prev = by_combo.get(combo)
        if prev and run_date(prev["res_dir"]) >= run_date(d):
            continue
        disp = model_display(s["tts_model"])
        rescore = None
        if (d / "rescore_judge2.json").exists():
            rescore = json.loads((d / "rescore_judge2.json").read_text(encoding="utf-8"))
        by_combo[combo] = {
            "res_dir": d,
            "summary": s,
            "slug": f"{slugify(disp)}-{slugify(str(s['voice']))}",
            "title": f"{disp} · {s['voice']}",
            "license": license_note(s["tts_model"]),
            "rescore": rescore,
            "rows": [json.loads(l) for l in
                     (d / "results_raw.jsonl").read_text(encoding="utf-8").splitlines()],
        }
    return sorted(by_combo.values(),
                  key=lambda r: r["summary"].get("wer_capped_mean")
                  or r["summary"].get("wer_mean") or 9)


def encode_clips(run: dict) -> int:
    """<id>_r0.wav → docs/audio/<slug>/<id>.mp3 (mono). Liefert Byte-Summe."""
    out_dir = DOCS / "audio" / run["slug"]
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for r in run["rows"]:
        src = run["res_dir"] / "audio" / f"{r['id']}_r0.wav"
        if not src.exists():
            src = run["res_dir"] / "audio" / f"{r['id']}.wav"
        if not src.exists():
            print(f"  WARNUNG: kein Audio für {r['id']}", file=sys.stderr)
            continue
        data, rate = sf.read(src)
        if data.ndim > 1:
            data = data.mean(axis=1)
        dst = out_dir / f"{r['id']}.mp3"
        sf.write(dst, data, rate)
        total += dst.stat().st_size
    return total


def fmt(v, digits: int = 3) -> str:
    return f"{v:.{digits}f}" if isinstance(v, (int, float)) else "–"


def model_page(run: dict) -> None:
    s = run["summary"]
    parts = ['<p><a href="index.html">← Übersicht</a></p>',
             f"<h1>{html.escape(run['title'])}</h1>",
             f'<p class="meta">Modell: {html.escape(str(s["tts_model"]))} · '
             f'Stimme: {html.escape(str(s["voice"]))} · '
             f'STT-Judge: {html.escape(str(s.get("stt_model", "?")))} · '
             f'WER {fmt(s.get("wer_capped_mean", s.get("wer_mean")))} (Cap 1.0) · '
             f'CER {fmt(s.get("cer_capped_mean", s.get("cer_mean")))} · '
             f'RTF {fmt(s.get("rtf_mean"), 2)} · '
             + (f'Tempo {fmt(s.get("sec_per_char_median"), 4)} s/Zeichen · '
                if s.get("sec_per_char_median") else '')
             + f'Lizenz: {html.escape(run["license"])} · '
             f'Lauf: {html.escape(run["res_dir"].name)}</p>',
             '<p class="meta">Je Fall ein Clip (erste von '
             f'{s.get("n_repeats", 1)} Wiederholung(en)); WER ist der Mittelwert über '
             'alle Wiederholungen. Rot ≥ 0.3, Orange ≥ 0.1, Grün &lt; 0.1.</p>']

    # Bei prompt-gesteuerten Stimmen (Qwen VoiceDesign, VoxCPM2) IST der
    # instruct-Text die Stimme — ohne ihn ist die Seite nicht nachvollziehbar.
    if s.get("voice_instruct"):
        parts.append(
            f'<div class="prompt"><b>Stimm-Prompt (instruct)</b>'
            f'<code>{html.escape(s["voice_instruct"])}</code></div>')
    if s.get("stt_prompt"):
        parts.append(
            f'<div class="prompt"><b>Judge-Prompt</b>'
            f'<code>{html.escape(s["stt_prompt"])}</code></div>')

    rescore_by_id = {}
    if run["rescore"]:
        rescore_by_id = {c["id"]: c for c in run["rescore"]["cases"]}
        parts.append(
            f'<p class="meta">Kreuzvalidiert mit Zweit-Judge '
            f'{html.escape(run["rescore"]["judge2"])} (r0, beste WER über refs+Text): '
            f'WER {run["rescore"]["wer_judge2_mean"]:.3f} '
            f'(Judge 1 im selben Protokoll: {run["rescore"]["wer_judge1_mean"]:.3f}).</p>')

    by_cat: dict[str, list] = {}
    for r in run["rows"]:
        by_cat.setdefault(r["category"], []).append(r)

    for cat, cases in sorted(by_cat.items()):
        parts.append(f"<h2>{html.escape(cat)}</h2>")
        for r in sorted(cases, key=lambda x: -(x.get("wer") or 0)):
            w = r.get("wer")
            cls = "bad" if (w or 0) >= 0.3 else ("mid" if (w or 0) >= 0.1 else "good")
            tx = (r.get("repeats") or [r])[0].get("transcript", "")
            wer_str = f"{w:.2f}" if w is not None else "FEHLER"
            j2 = rescore_by_id.get(r["id"])
            j2_html = (f'<div class="transcript">→ Judge 2: {html.escape(j2["judge2_transcript"])}'
                       f' <span class="meta">(WER {j2["wer_judge2"]:.2f})</span></div>' if j2 else "")
            parts.append(f"""<div class="case {cls}">
 <div class="text">{html.escape(r["text"])}</div>
 <div><audio controls preload="none" src="audio/{run["slug"]}/{r["id"]}.mp3"></audio></div>
 <div class="transcript">→ {html.escape(tx)}</div>
 {j2_html}
 <div class="meta">{r["id"]} · WER {wer_str}</div>
</div>""")

    (DOCS / f"{run['slug']}.html").write_text(page(run["title"], "\n".join(parts)),
                                              encoding="utf-8")


def index_page(runs: list[dict]) -> None:
    cats = sorted({c for run in runs
                   for c in (run["summary"].get("wer_by_category") or {})})

    # ── Sortierbares Leaderboard: Stimme-als-Zeilen, Name → HF-Card, Bestwert je Spalte grün ──
    def _hf(tm):
        raw = str(tm).strip()
        for seg in raw.split("/"):                              # 'owner--model'-Segment im Pfad finden
            if "--" in seg:
                return "https://huggingface.co/" + seg.replace("--", "/", 1)
        if re.match(r"^[\w.-]+/[\w.-]+$", raw):                 # direkte owner/model-Form (z.B. Voxtral)
            return "https://huggingface.co/" + raw
        return ""
    _lb_cols = [("WER (Cap 1.0)", lambda r: r["summary"].get("wer_capped_mean", r["summary"].get("wer_mean"))),
                ("WER ungekappt", lambda r: r["summary"].get("wer_mean")),
                ("WER Whisper", lambda r: (r["rescore"] or {}).get("wer_judge1_mean")),
                ("WER Voxtral", lambda r: (r["rescore"] or {}).get("wer_judge2_mean")),
                ("CER", lambda r: r["summary"].get("cer_capped_mean", r["summary"].get("cer_mean"))),
                ("RTF", lambda r: r["summary"].get("rtf_mean"))]
    _cbest = [min([g(r) for r in runs if g(r) is not None], default=None) for _, g in _lb_cols]
    def _vcell(r):
        hf = _hf(r["summary"].get("tts_model", "")); nm = html.escape(r["title"])
        lk = f'<a href="{hf}" target="_blank" rel="noopener">{nm}</a>' if hf else nm
        return (f'<td data-sort="{html.escape(r["title"])}">{lk} '
                f'<a href="{r["slug"]}.html">anhören ↗</a></td>')
    _lbh = ("<tr><th>Stimme</th>"
            + "".join(f"<th>{html.escape(l)}</th>" for l, _ in _lb_cols)
            + "<th>Lizenz</th></tr>")
    _lbb = ""
    for r in runs:
        cells = ""
        for (lbl, g), b in zip(_lb_cols, _cbest):
            v = g(r); cls = "best" if v is not None and b is not None and v == b else ""
            cells += f'<td class="{cls}" data-sort="{"" if v is None else v}">{fmt(v)}</td>'
        lic = html.escape(str(r.get("license") or "—"))
        cells += f'<td class="lic" data-sort="{lic}">{lic}</td>'
        _lbb += f"<tr>{_vcell(r)}{cells}</tr>"
    leaderboard = (f'<div class="tablewrap"><table class="sortable"><thead>{_lbh}</thead>'
                   f'<tbody>{_lbb}</tbody></table></div>')

    # ── „WER je Kategorie": gleiche Orientierung wie das Leaderboard (Stimme-als-Zeilen,
    #    Kategorie-Spalten sortierbar, Bestwert je Spalte grün) → Wiedererkennungswert. ──
    def _catval(r, c):
        return ((r["summary"].get("wer_capped_by_category")
                 or r["summary"].get("wer_by_category") or {}).get(c))
    _catbest = [min([v for r in runs if (v := _catval(r, c)) is not None], default=None)
                for c in cats]
    _cath = "<tr><th>Stimme</th>" + "".join(f"<th>{html.escape(c)}</th>" for c in cats) + "</tr>"
    _catb = ""
    for r in runs:
        cells = ""
        for c, b in zip(cats, _catbest):
            v = _catval(r, c)
            cls = "best" if v is not None and b is not None and v == b else ""
            cells += f'<td class="{cls}" data-sort="{"" if v is None else v}">{fmt(v)}</td>'
        _catb += f"<tr>{_vcell(r)}{cells}</tr>"
    cat_table = (f'<div class="tablewrap"><table class="sortable"><thead>{_cath}</thead>'
                 f'<tbody>{_catb}</tbody></table></div>')

    n = runs[0]["summary"]
    judge_note = ""
    if any(r["rescore"] for r in runs):
        judge2 = next(r["rescore"]["judge2"] for r in runs if r["rescore"])
        judge_note = (
            f'<p class="note">Haupt-Judge ist <b>Whisper large-v3</b> mit Casing-Prompt (verbalisiert '
            f'Zahlen, statt „17.45 Uhr" zu notieren). <b>WER Whisper</b> und <b>WER Voxtral</b> zeigen '
            f'dieselben Clips durch zwei ASR-Modelle — die Spanne ist ein Unsicherheitsband, kein '
            f'Doppelmaß ({html.escape(judge2)} normalisiert aggressiver zu Ziffern). Endlos-Decoder-Läufe '
            f'werden erkannt, einmal wiederholt und sonst bei WER&nbsp;1.0 gekappt.</p>')
    body = f"""<h1>Deutscher TTS-Vergleich auf dem DGX Spark</h1>
<p>{n["n_total"]} Testfälle (<a href="https://github.com/MvdB/southbyte-tts">Testset &amp; Eval-Code</a>),
Judge {html.escape(str(n.get("stt_model", "?")))}. Bester Wert je Spalte grün, niedriger = besser;
Spalten sortierbar. Die WER enthält auch STT-Fehler (obere Schranke) — Kategorien-Deltas sind
aussagekräftiger als Absolutwerte. Leitmetrik ist die bei 1.0 gekappte WER.</p>
{leaderboard}
{judge_note}
<h2>WER je Kategorie</h2>
{cat_table}"""
    (DOCS / "index.html").write_text(page("Deutscher TTS-Vergleich (DGX Spark)", body),
                                     encoding="utf-8")


def prune(runs: list[dict]) -> None:
    """Seiten/Audio zu nicht mehr vorhandenen Kombinationen entfernen."""
    keep = {r["slug"] for r in runs}
    for p in DOCS.glob("*.html"):
        if p.stem != "index" and p.stem not in keep:
            print(f"entfernt (veraltet): {p.relative_to(REPO)}")
            p.unlink()
    for d in (DOCS / "audio").glob("*/"):
        if d.name not in keep:
            print(f"entfernt (veraltet): {d.relative_to(REPO)}/")
            for f in d.iterdir():
                f.unlink()
            d.rmdir()


def main() -> int:
    DOCS.mkdir(exist_ok=True)
    runs = discover_runs()
    if not runs:
        print("keine vollständigen Läufe in results/ gefunden", file=sys.stderr)
        return 1
    total = 0
    for run in runs:
        size = encode_clips(run)
        total += size
        model_page(run)
        print(f"{run['slug']}: {len(run['rows'])} Clips, {size / 1e6:.1f} MB "
              f"(aus {run['res_dir'].name})")
    index_page(runs)
    prune(runs)
    print(f"gesamt: {total / 1e6:.1f} MB Audio → {DOCS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
