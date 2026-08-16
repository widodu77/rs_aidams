"""Build a self-contained demo page from the held-out generations.

There is no live inference here, and that is deliberate rather than a shortcut.
The local card is 4 GB, too small for unquantised Qwen3-1.7B, and the quantised
alternative is exactly what produced the two findings that later reversed sign
(see `notes/2026-08-09.md`). A demo that generated fresh text on the wrong
precision would show something the paper does not claim.

So the page replays the *actual* evaluated completions: the same 248 held-out
items, the same outputs, scored by the same BFCL checker that produced every
table in the paper. Every verdict shown was computed here, not asserted.

    PYTHONPATH=src uv run python demo/build_demo.py
"""

from __future__ import annotations

import ast
import html
import json
import os

from analysis.pareto import CATEGORIES
from scoring.bfcl_scorer import load_category, parse_model_output, score_sample
from train.dataset import load_manifest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# The three policies the demo contrasts. `never` and `always` are the fixed-policy
# anchors; `gate` is the trained one. The twelve prompt-decision runs are left out
# on purpose: the point of the page is the contrast, and thirteen columns of
# indistinguishable output would bury it.
POLICIES = [
    ("never", "never reason", "results/raw/vllm/qwen3-1.7b_never.jsonl"),
    ("always", "always reason", "results/raw/vllm/qwen3-1.7b_always.jsonl"),
    ("gate", "trained (λ=2.0)", "results/raw/vllm_eval/trained_gate_lam2_0.jsonl"),
]


def read_jsonl(path: str) -> dict:
    records = {}
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                records[(record["category"], record["id"])] = record
    return records


def structured(raw):
    """BFCL hands these fields back already parsed on some categories and as a
    repr'd string on others, so accept either rather than assuming."""
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw
    return raw


def user_question(raw) -> str:
    turns = structured(raw)
    if isinstance(turns, str):
        return turns
    while isinstance(turns, list) and turns and isinstance(turns[0], list):
        turns = turns[0]
    if isinstance(turns, dict):
        turns = [turns]
    parts = [t.get("content", "") for t in turns if isinstance(t, dict)]
    return " ".join(p for p in parts if p).strip()


def function_names(raw) -> list[str]:
    functions = structured(raw)
    if isinstance(functions, dict):
        functions = [functions]
    if not isinstance(functions, list):
        return []
    return [f.get("name", "?") for f in functions if isinstance(f, dict)]


def build_items() -> list[dict]:
    _, eval_ids = load_manifest(os.path.join(ROOT, "results/split_manifest.json"))
    ground_truth = {(c, s["id"]): s for c in CATEGORIES for s in load_category(c)}
    runs = {tag: read_jsonl(path) for tag, _, path in POLICIES}

    items = []
    for key in sorted(eval_ids):
        if any(key not in runs[tag] for tag, _, _ in POLICIES):
            continue
        category, item_id = key
        sample = ground_truth[key]

        entry = {
            "id": item_id,
            "category": category,
            "question": user_question(sample["question"]),
            "functions": function_names(sample["function"]),
            "outputs": {},
        }
        for tag, _, _ in POLICIES:
            record = runs[tag][key]
            parsed = parse_model_output(record["output_text"])
            result = score_sample(
                sample["function"], sample["ground_truth"], parsed, category
            )
            entry["outputs"][tag] = {
                "text": record["output_text"],
                "tokens": record["completion_tokens"],
                "correct": bool(result["correct"]),
                "thought": bool(parsed.did_think),
            }
        items.append(entry)
    return items


# A meta charset has to land inside the first 1024 bytes to take effect, so it
# goes above everything. Without it `python -m http.server` serves no charset,
# the browser falls back to latin-1 and every non-ASCII character mojibakes.
PAGE = """<meta charset="utf-8">
<title>Adaptive Thinking for Tool Use</title>
<style>
  /* Light is the base palette; the two blocks below redefine only the tokens, so
     the un-stamped "system" state and both explicit stamps all resolve as a set.
     The neutrals carry a slight cool bias toward the navy accent rather than
     being pure grey, which keeps the page reading as an instrument readout. */
  :root {
    --bg: #fbfbfc; --panel: #ffffff; --sunk: #f3f5f7;
    --ink: #14181d; --muted: #5c6773; --faint: #6b7681;
    --line: #e2e6ea; --rule: #cbd2d9;
    --accent: #1f4e79; --good: #157347; --bad: #b02a20;
    --track: #e8ebee;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #101318; --panel: #171b21; --sunk: #1c2128;
      --ink: #e4e8ec; --muted: #8f9ba8; --faint: #7d8894;
      --line: #262c33; --rule: #333b44;
      --accent: #7fb3e0; --good: #4ec98a; --bad: #ff8878;
      --track: #232930;
    }
  }
  :root[data-theme="dark"] {
    --bg: #101318; --panel: #171b21; --sunk: #1c2128;
    --ink: #e4e8ec; --muted: #8f9ba8; --faint: #7d8894;
    --line: #262c33; --rule: #333b44;
    --accent: #7fb3e0; --good: #4ec98a; --bad: #ff8878;
    --track: #232930;
  }

  /* Two voices: a grotesque for prose, and monospace for every measured value,
     label and model output. The mono is the instrument register -- if a number
     was counted or a verdict computed, it is set in mono. */
  :root {
    --sans: ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, sans-serif;
    --mono: ui-monospace, "SF Mono", "Cascadia Mono", "Consolas", monospace;
  }

  * { box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--ink); margin: 0;
    font: 15px/1.6 var(--sans);
    padding: 28px 20px 64px;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1140px; margin: 0 auto; display: flex;
          flex-direction: column; gap: 18px; }
  header { display: flex; flex-direction: column; gap: 6px; }
  h1 { font-size: 21px; font-weight: 600; margin: 0; letter-spacing: -.015em;
       text-wrap: balance; }
  .sub { color: var(--muted); font-size: 14px; margin: 0; max-width: 68ch; }
  .sub a { color: var(--accent); }

  .eyebrow { font-family: var(--mono); font-size: 10.5px; letter-spacing: .1em;
             text-transform: uppercase; color: var(--faint); }

  /* Measurements. Squared, hairline-ruled, digits aligned -- a ledger row, not
     a set of rounded cards. */
  .totals { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
            background: var(--line); border: 1px solid var(--line); }
  @media (max-width: 720px) { .totals { grid-template-columns: repeat(2, 1fr); } }
  .tile { background: var(--panel); padding: 11px 14px 12px;
          display: flex; flex-direction: column; gap: 5px; }
  .tile.lead { background: var(--sunk); }
  .tile .k { font-family: var(--mono); color: var(--faint); font-size: 10.5px;
             text-transform: uppercase; letter-spacing: .08em; }
  .tile .v { font-family: var(--mono); font-size: 18px; font-weight: 600;
             font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
  .tile .v small { font-size: 12px; font-weight: 400; color: var(--muted);
                   letter-spacing: 0; }

  .bar { display: flex; gap: 7px; flex-wrap: wrap; align-items: center; }
  button, select {
    font-family: var(--sans); font-size: 13.5px; padding: 6px 11px; border-radius: 3px;
    border: 1px solid var(--rule); background: var(--panel); color: var(--ink);
    cursor: pointer;
  }
  button:hover, select:hover { border-color: var(--accent); }
  button:focus-visible, select:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 1px;
  }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  .counter { font-family: var(--mono); color: var(--muted); font-size: 12.5px;
             margin-left: auto; font-variant-numeric: tabular-nums; }

  /* The item under test. */
  .q { background: var(--panel); border: 1px solid var(--line);
       border-left: 2px solid var(--rule);
       padding: 13px 16px; display: flex; flex-direction: column; gap: 7px; }
  .q .text { font-size: 15.5px; text-wrap: pretty; }
  .q .tools { color: var(--muted); font-size: 12.5px;
              overflow-x: auto; white-space: nowrap; }
  .q .tools code { font-family: var(--mono); background: var(--sunk);
                   border: 1px solid var(--line);
                   padding: 1px 6px; margin-right: 5px; font-size: 11.5px; }

  .cols { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
          align-items: stretch; }
  @media (max-width: 880px) { .cols { grid-template-columns: 1fr; } }
  .card { background: var(--panel); border: 1px solid var(--line);
          border-top: 2px solid var(--rule);
          display: flex; flex-direction: column; }
  .card.ok  { border-top-color: var(--good); }
  .card.no  { border-top-color: var(--bad); }
  .card h2 { font-size: 13px; margin: 0; padding: 10px 14px 8px; font-weight: 600;
             display: flex; align-items: baseline; gap: 8px; }
  .verdict { font-family: var(--mono); font-size: 10.5px; font-weight: 600;
             letter-spacing: .07em; text-transform: uppercase; }
  .card.ok .verdict { color: var(--good); }
  .card.no .verdict { color: var(--bad); }
  .cost { padding: 0 14px 10px; border-bottom: 1px solid var(--line);
          display: flex; flex-direction: column; gap: 5px; }
  .cost .n { font-family: var(--mono); font-size: 12.5px; color: var(--muted);
             font-variant-numeric: tabular-nums; }
  .cost .track { height: 3px; background: var(--track); }
  .cost .fill { height: 100%; background: var(--accent); }
  pre {
    margin: 0; padding: 12px 14px; font-size: 12.5px; line-height: 1.55;
    font-family: var(--mono);
    white-space: pre-wrap; word-break: break-word; overflow-x: auto;
    flex: 1; min-height: 148px;
  }
  .think { color: var(--faint); }
  .empty { color: var(--faint); font-style: italic; }
  footer { color: var(--muted); font-size: 12.5px;
           border-top: 1px solid var(--line); padding-top: 14px; margin-top: 8px; }
  footer a { color: var(--accent); }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">Qwen3-1.7B &middot; BFCL v4 single-turn &middot; 248 held-out items</div>
    <h1>Adaptive Thinking for Tool Use</h1>
    <p class="sub">
      The same function call, answered three ways. Every completion below was produced
      during evaluation and scored by BFCL's own AST checker, the same one used as the
      training reward. Nothing is re-generated in the browser.
    </p>
  </header>

  <div class="totals" id="totals"></div>

  <div class="bar">
    <button id="prev">&larr; prev</button>
    <button id="next">next &rarr;</button>
    <button id="rand" class="primary">Show a random call</button>
    <select id="filter" aria-label="filter items">
      <option value="all">all items</option>
      <option value="disagree">where trained and always-reason disagree</option>
      <option value="gatewin">where trained beats never-reason</option>
    </select>
    <select id="cat" aria-label="filter by category"></select>
    <span class="counter" id="counter"></span>
  </div>

  <div class="q">
    <div class="eyebrow" id="qlabel"></div>
    <div class="text" id="question"></div>
    <div class="tools" id="tools"></div>
  </div>

  <div class="cols" id="cols"></div>

  <footer id="foot"></footer>
</div>

<script>
const ITEMS = __ITEMS__;
const POLICIES = __POLICIES__;

let view = ITEMS.slice();
let at = 0;

const el = id => document.getElementById(id);
const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function renderTotals() {
  const n = ITEMS.length;
  const tiles = POLICIES.map(([tag, name]) => {
    const acc = ITEMS.filter(i => i.outputs[tag].correct).length / n;
    const tok = ITEMS.reduce((s, i) => s + i.outputs[tag].tokens, 0) / n;
    return `<div class="tile"><div class="k">${name}</div>
      <div class="v">${(acc * 100).toFixed(1)}%
        <small>${tok.toFixed(0)} tok / call</small></div></div>`;
  });
  const saved = ITEMS.reduce(
    (s, i) => s + i.outputs.always.tokens - i.outputs.gate.tokens, 0);
  tiles.push(`<div class="tile lead"><div class="k">saved vs always</div>
    <div class="v">${saved.toLocaleString()} <small>tokens</small></div></div>`);
  el('totals').innerHTML = tiles.join('');
}

// The empty think block is the decision: the model emitted it rather than being
// given it, which is the whole point of the trained run. Show it, greyed, instead
// of stripping it, so the viewer can see the choice being made.
function body(text) {
  const m = text.match(/^([\\s\\S]*?<\\/think>)([\\s\\S]*)$/);
  if (!m) return esc(text.trim()) || '<span class="empty">(no output)</span>';
  const inner = m[1].replace(/<\\/?think>/g, '').trim();
  const head = inner
    ? `<span class="think">&lt;think&gt;\\n${esc(inner)}\\n&lt;/think&gt;</span>`
    : `<span class="think empty">&lt;think&gt;&lt;/think&gt;  (decided not to reason)</span>`;
  return head + '\\n\\n' + esc(m[2].trim());
}

function render() {
  if (!view.length) {
    el('question').textContent = 'No items match this filter.';
    el('tools').innerHTML = ''; el('cols').innerHTML = ''; el('counter').textContent = '';
    el('qlabel').textContent = ''; return;
  }
  at = (at % view.length + view.length) % view.length;
  const item = view[at];

  el('qlabel').textContent = item.id;
  el('question').textContent = item.question;
  el('tools').innerHTML = item.functions.length
    ? 'available: ' + item.functions.map(f => `<code>${esc(f)}</code>`).join('')
    : '';
  el('counter').textContent = `${at + 1} of ${view.length}`;

  // The cost bars are scaled against the most expensive policy on *this* item, so
  // the comparison a viewer makes is the one the item actually supports.
  const worst = Math.max(...POLICIES.map(([t]) => item.outputs[t].tokens), 1);

  el('cols').innerHTML = POLICIES.map(([tag, name]) => {
    const o = item.outputs[tag];
    const state = o.correct ? 'ok' : 'no';
    return `<div class="card ${state}">
      <h2>${name}<span class="verdict">${o.correct ? 'correct' : 'wrong'}</span></h2>
      <div class="cost">
        <div class="n">${o.tokens} tokens</div>
        <div class="track"><div class="fill" style="width:${o.tokens / worst * 100}%"></div></div>
      </div>
      <pre>${body(o.text)}</pre>
    </div>`;
  }).join('');
}

function applyFilter() {
  const f = el('filter').value, c = el('cat').value;
  view = ITEMS.filter(i => {
    if (c !== 'all' && i.category !== c) return false;
    if (f === 'disagree') return i.outputs.gate.correct !== i.outputs.always.correct;
    if (f === 'gatewin') return i.outputs.gate.correct && !i.outputs.never.correct;
    return true;
  });
  at = 0; render();
}

el('prev').onclick = () => { at--; render(); };
el('next').onclick = () => { at++; render(); };
el('rand').onclick = () => { at = Math.floor(Math.random() * view.length); render(); };
el('filter').onchange = applyFilter;
el('cat').onchange = applyFilter;
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') { at--; render(); }
  if (e.key === 'ArrowRight') { at++; render(); }
});

const cats = ['all', ...new Set(ITEMS.map(i => i.category))];
el('cat').innerHTML = cats.map(c =>
  `<option value="${c}">${c === 'all' ? 'all categories' : c}</option>`).join('');

el('foot').innerHTML =
  `The trained policy is GRPO with a length-penalised reward and the thinking decision `
  + `moved into the completion. It reasons on 0% of items, not the 17.4% a per-item oracle `
  + `would: it did not learn <em>when</em> to reason, it learned not to need it. `
  + `Method and caveats in the paper; code at `
  + `<a href="https://github.com/widodu77/rs_aidams">github.com/widodu77/rs_aidams</a>.`;

renderTotals();
render();
</script>
"""


def main() -> None:
    items = build_items()
    page = (
        PAGE.replace("__ITEMS__", json.dumps(items))
        .replace("__POLICIES__", json.dumps([[t, n] for t, n, _ in POLICIES]))
    )
    out = os.path.join(HERE, "index.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)

    n = len(items)
    print(f"{n} items")
    for tag, name, _ in POLICIES:
        acc = sum(i["outputs"][tag]["correct"] for i in items) / n
        tok = sum(i["outputs"][tag]["tokens"] for i in items) / n
        print(f"  {name:16s} {acc:6.1%} {tok:7.1f} tok")
    print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
