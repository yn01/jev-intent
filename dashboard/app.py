"""results.csv を視覚的に確認するStreamlitダッシュボード。

data/queries.csv (正解ラベル・実績値) と results.csv (Jevの判定結果) を
query_id で結合して表示する。classify.py / questions.py には一切変更を
加えず、questions.py はimportして質問文の表示にのみ使う。

起動:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yaml

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FOOTER_TEXT = "Synthetic data. Not actual search data of any company."

QUERIES_PATH = ROOT / "data" / "queries.csv"
VOCAB_PATH = ROOT / "data" / "vocab.yaml"

# Streamlitの静的配信は「実行中のスクリプト(=dashboard/app.py)からの相対パス」の
# ./static/ 配下しか対象にならないため、dashboard/static/live に置く。
# URLは /app/static/live/... で配信される(.streamlit/config.toml の
# enableStaticServing=trueが必要。設定変更後はStreamlitの再起動が必要)。
LIVE_DIR = DASHBOARD_DIR / "static" / "live"
LIVE_STREAM_PATH = LIVE_DIR / "results_live.csv"
LIVE_OUT_PATH = LIVE_DIR / "results_last.csv"
LIVE_STATUS_PATH = LIVE_DIR / "status.json"
LIVE_STATIC_URL_PREFIX = "app/static/live"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

# 公称値に基づく推定単価(USD / 100万 input tokens)。実際の契約条件とは異なる場合がある。
PRICE_PER_M_INPUT_TOKENS = 0.042

INTENT_LABELS = ["informational", "comparison", "transactional", "navigational"]
PROB_COLS = {label: f"intent_prob_{label}" for label in INTENT_LABELS}

CONFIDENCE_BIN_EDGES = [0.0, 0.5, 0.7, 0.9, 0.99, 1.0 + 1e-9]
CONFIDENCE_BIN_LABELS = ["0-0.5", "0.5-0.7", "0.7-0.9", "0.9-0.99", "0.99-1.0"]

# 合成/実測の各Noulと、対応する正解列(queries.csv)・値の由来の対応表
NOUL_SPECS = [
    # key, 表示名, results.csv内の列, 正解列(Noneなら vocab.yaml 由来の参考ground truth)
    {"key": "mentions_competitor", "label": "mentions_competitor (合成)", "pred_col": "mentions_competitor_pred", "truth_col": "mentions_competitor_true"},
    {"key": "own_product_only", "label": "own_product_only (合成)", "pred_col": "own_product_only_pred", "truth_col": "own_product_only_true"},
    {"key": "out_of_category", "label": "out_of_category", "pred_col": "out_of_category_pred", "truth_col": "out_of_category_true"},
    {"key": "mentions_own_brand", "label": "mentions_own_brand (元)", "pred_col": "mentions_own_brand", "truth_col": "_gt_own_brand"},
    {"key": "mentions_own_product", "label": "mentions_own_product (元)", "pred_col": "mentions_own_product", "truth_col": "_gt_own_product"},
    {"key": "mentions_competitor_brand", "label": "mentions_competitor_brand (元)", "pred_col": "mentions_competitor_brand", "truth_col": "_gt_competitor_brand"},
    {"key": "mentions_competitor_product", "label": "mentions_competitor_product (元)", "pred_col": "mentions_competitor_product", "truth_col": "_gt_competitor_product"},
]


def footer() -> None:
    st.caption(FOOTER_TEXT)


# ---------------------------------------------------------------------------
# データ読み込み
# ---------------------------------------------------------------------------

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_vocab(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except OSError:
        return None


def _entry_aliases(entry) -> list[str]:
    if isinstance(entry, str):
        return [entry]
    names = [entry["name"]]
    names.extend(entry.get("aliases") or [])
    return names


_NO_MATCH = re.compile(r"(?!x)x")


def _terms_to_pattern(terms: list[str]) -> re.Pattern:
    terms = sorted({t for t in terms if t}, key=len, reverse=True)
    if not terms:
        return _NO_MATCH
    return re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)


@st.cache_data
def build_vocab_patterns(_vocab: dict | None) -> dict[str, re.Pattern]:
    """vocab.yamlのブランド名・製品名から、元の4Noul用の参考ground truthを作る。

    queries.csvにはブランド/製品を分けた正解列が無いため、生成に使われた
    語彙リストとの文字列一致で近似する(あくまで参考値)。
    """
    keys = ["own_brand", "own_product", "competitor_brand", "competitor_product"]
    if _vocab is None:
        return {k: _NO_MATCH for k in keys}

    own_brand_terms = _vocab["brands"]["own"]["aliases"]
    competitor_brand_terms: list[str] = []
    for comp in _vocab["brands"]["competitors"].values():
        competitor_brand_terms.extend(comp["aliases"])

    own_product_terms: list[str] = []
    for group in _vocab["products"]["own"].values():
        for item in group:
            own_product_terms.extend(_entry_aliases(item))

    competitor_product_terms: list[str] = []
    for comp in _vocab["products"]["competitors"].values():
        for group in comp.values():
            for item in group:
                competitor_product_terms.extend(_entry_aliases(item))

    return {
        "own_brand": _terms_to_pattern(own_brand_terms),
        "own_product": _terms_to_pattern(own_product_terms),
        "competitor_brand": _terms_to_pattern(competitor_brand_terms),
        "competitor_product": _terms_to_pattern(competitor_product_terms),
    }


def try_load_questions():
    """questions.py をimportする。失敗したら例外を返す(呼び出し側で表示をスキップ)。"""
    try:
        import questions  # type: ignore

        return questions
    except Exception as exc:  # noqa: BLE001 - importの失敗理由を問わず表示だけスキップしたい
        return exc


@st.cache_data
def merge_data(queries_path: str, results_path: str) -> tuple[pd.DataFrame, int]:
    queries = load_csv(queries_path)
    results = load_csv(results_path)

    error_count = 0
    if "error" in results.columns:
        has_error = results["error"].notna() & (results["error"].astype(str).str.strip() != "")
        error_count = int(has_error.sum())
        results = results[~has_error].copy()

    merged = queries.merge(results, on="query_id", suffixes=("_true", "_pred"))

    vocab = load_vocab(str(VOCAB_PATH))
    patterns = build_vocab_patterns(vocab)
    query_text = merged["query_true"] if "query_true" in merged.columns else merged["query"]
    merged["_gt_own_brand"] = query_text.str.contains(patterns["own_brand"]).astype(int)
    merged["_gt_own_product"] = query_text.str.contains(patterns["own_product"]).astype(int)
    merged["_gt_competitor_brand"] = query_text.str.contains(patterns["competitor_brand"]).astype(int)
    merged["_gt_competitor_product"] = query_text.str.contains(patterns["competitor_product"]).astype(int)

    merged["correct"] = merged["intent"] == merged["true_intent"]

    return merged, error_count


def query_col(df: pd.DataFrame) -> str:
    return "query_true" if "query_true" in df.columns else "query"


def competitor_brand_pattern(vocab: dict | None) -> re.Pattern:
    return build_vocab_patterns(vocab)["competitor_brand"]


# ---------------------------------------------------------------------------
# ライブモード用ヘルパー(キャッシュしない: 常に最新ファイルを読む)
# ---------------------------------------------------------------------------

def read_live_status() -> dict | None:
    if not LIVE_STATUS_PATH.exists():
        return None
    try:
        with LIVE_STATUS_PATH.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def read_live_stream() -> pd.DataFrame:
    if not LIVE_STREAM_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(LIVE_STREAM_PATH, engine="python", on_bad_lines="skip")
    except Exception:  # noqa: BLE001 - 書き込み途中の不完全な内容は空扱いにして次のtickに任せる
        return pd.DataFrame()


def is_live_process_running() -> bool:
    proc = st.session_state.get("live_proc")
    return proc is not None and proc.poll() is None


# ---------------------------------------------------------------------------
# app本体
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Jev Intent Dashboard", layout="wide")

st.sidebar.header("データ")
if st.sidebar.button("再読み込み"):
    load_csv.clear()
    load_vocab.clear()
    build_vocab_patterns.clear()
    merge_data.clear()

results_path_str = st.sidebar.text_input("results.csv のパス", value="results.csv", key="results_path_input")
results_path = (ROOT / results_path_str) if not Path(results_path_str).is_absolute() else Path(results_path_str)

if not QUERIES_PATH.exists():
    st.error(f"{QUERIES_PATH} が見つかりません。")
    st.stop()
if not results_path.exists():
    st.error(f"{results_path} が見つかりません。パスを確認してください。")
    st.stop()

df, error_count = merge_data(str(QUERIES_PATH), str(results_path))
qcol = query_col(df)

st.sidebar.header("しきい値")
threshold = st.sidebar.slider("Noul二値化のしきい値", 0.0, 1.0, 0.5, 0.01)

st.sidebar.header("フィルタ")
only_incorrect = st.sidebar.checkbox("不正解のみ")
ambiguous_choice = st.sidebar.radio("ambiguous", ["すべて", "0のみ", "1のみ"], horizontal=True)
only_ooc = st.sidebar.checkbox("out_of_category=1 のみ")
only_competitor_product_only = st.sidebar.checkbox("競合製品名のみ(社名表記なし)")
intent_options = sorted(df["true_intent"].dropna().unique().tolist())
selected_intents = st.sidebar.multiselect("true_intent", intent_options, default=intent_options)
search_text = st.sidebar.text_input("クエリ文字列の部分一致検索")

vocab = load_vocab(str(VOCAB_PATH))
comp_brand_pat = competitor_brand_pattern(vocab)

filtered = df.copy()
if only_incorrect:
    filtered = filtered[~filtered["correct"]]
if ambiguous_choice == "0のみ":
    filtered = filtered[filtered["ambiguous"] == 0]
elif ambiguous_choice == "1のみ":
    filtered = filtered[filtered["ambiguous"] == 1]
if only_ooc:
    filtered = filtered[filtered["out_of_category_true"] == 1]
if only_competitor_product_only:
    filtered = filtered[
        (filtered["mentions_competitor_true"] == 1)
        & ~filtered[qcol].str.contains(comp_brand_pat)
    ]
if selected_intents:
    filtered = filtered[filtered["true_intent"].isin(selected_intents)]
else:
    filtered = filtered.iloc[0:0]
if search_text:
    filtered = filtered[filtered[qcol].str.contains(re.escape(search_text), case=False, na=False)]

if error_count:
    st.sidebar.warning(f"error列が空でない行を{error_count}件、集計から除外しました。")

tab_live, tab_overview, tab_log, tab_intent, tab_noul, tab_latency = st.tabs(
    ["ライブ", "概要", "判定ログ", "インテント分析", "Noul分析", "レイテンシ"]
)

# ---------------------------------------------------------------------------
# ライブ
# ---------------------------------------------------------------------------
with tab_live:
    st.caption("classify.py をここから起動し、完了順の結果をリアルタイムに表示します。")

    if "live_run_id" not in st.session_state:
        st.session_state["live_run_id"] = 0

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        live_sample_n = st.radio("件数", [100, 300, 800], horizontal=True, key="live_sample_n")
    with c2:
        live_workers = st.number_input("workers", min_value=1, max_value=32, value=8, step=1, key="live_workers")
    with c3:
        live_max_rps = st.number_input("max-rps", min_value=1.0, max_value=100.0, value=18.0, step=1.0, key="live_max_rps")
    with c4:
        st.write("")
        running = is_live_process_running()
        if st.button("開始", type="primary", disabled=running, key="live_start_btn"):
            LIVE_DIR.mkdir(parents=True, exist_ok=True)
            python_exe = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
            cmd = [
                python_exe,
                str(ROOT / "classify.py"),
                "--sample", str(live_sample_n),
                "--stream", str(LIVE_STREAM_PATH),
                "--out", str(LIVE_OUT_PATH),
                "--workers", str(int(live_workers)),
                "--max-rps", str(live_max_rps),
            ]
            proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            st.session_state["live_proc"] = proc
            st.session_state["live_run_id"] += 1
            st.rerun()
        if running and st.button("中止", key="live_stop_btn"):
            proc = st.session_state.get("live_proc")
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            st.session_state["live_proc"] = None
            st.rerun()

    # ------------------------------------------------------------------
    # 判定ログ: st.components.v1.html の単一HTML/JSで、fragmentの外に置く。
    # run_id が変わった時だけJS側でログをクリアする(sessionStorageで判定)。
    # ------------------------------------------------------------------
    _run_id = st.session_state["live_run_id"]
    _queries_df_for_truth = load_csv(str(QUERIES_PATH))
    _truth_map = (
        _queries_df_for_truth[["query_id", "true_intent", "ambiguous"]]
        .assign(ambiguous=lambda d: d["ambiguous"].astype(int))
        .set_index("query_id")
        .to_dict(orient="index")
    )
    _truth_json = json.dumps(_truth_map, ensure_ascii=False)
    _stream_url_path = json.dumps(f"{LIVE_STATIC_URL_PREFIX}/results_live.csv")

    _live_log_html = f"""
<style>
  body {{ margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  #counters {{ display: flex; gap: 32px; padding: 8px 4px 12px 4px; }}
  .counter {{ line-height: 1.1; }}
  .counter .value {{ font-size: 30px; font-weight: 700; }}
  .counter .label {{ font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.04em; }}
  #log-container {{ display: flex; flex-direction: column; }}
  .log-line {{
    padding: 2px 6px;
    white-space: pre;
    font-size: 13px;
    border-bottom: 1px solid rgba(128,128,128,0.12);
  }}
  .log-line.correct {{ color: #2ca02c; }}
  .log-line.incorrect {{ color: #d62728; }}
  .log-line.flash {{ animation: flash 0.6s ease-out; }}
  @keyframes flash {{
    from {{ background-color: rgba(255, 221, 0, 0.55); }}
    to {{ background-color: transparent; }}
  }}
  #status-line {{ font-size: 12px; color: #888; padding: 0 4px 6px 4px; }}
</style>
<div id="counters">
  <div class="counter"><div class="value" id="c-done">0</div><div class="label">完了件数</div></div>
  <div class="counter"><div class="value" id="c-elapsed">0.0s</div><div class="label">経過秒</div></div>
  <div class="counter"><div class="value" id="c-rps">0</div><div class="label">直近1秒 件/秒</div></div>
</div>
<div id="status-line">待機中...</div>
<div id="log-container"></div>
<script>
(function() {{
  const RUN_ID = {_run_id};
  const TRUTH = {_truth_json};
  const STREAM_URL_PATH = {_stream_url_path};
  const STORAGE_KEY = "jev_live_log_state_v1";

  const origin = window.parent.location.origin;
  const streamUrl = origin + "/" + STREAM_URL_PATH;

  const logContainer = document.getElementById("log-container");
  const statusLine = document.getElementById("status-line");
  const elDone = document.getElementById("c-done");
  const elElapsed = document.getElementById("c-elapsed");
  const elRps = document.getElementById("c-rps");

  let stored = null;
  try {{ stored = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "null"); }} catch (e) {{ stored = null; }}

  let processedCount = 0;
  let displayedCount = 0;
  let lastDisplayedTdone = null;

  if (stored && stored.runId === RUN_ID) {{
    processedCount = stored.processedCount || 0;
    displayedCount = stored.displayedCount || 0;
    lastDisplayedTdone = stored.lastDisplayedTdone;
    if (stored.logHtml) {{ logContainer.innerHTML = stored.logHtml; }}
  }} else {{
    try {{ sessionStorage.removeItem(STORAGE_KEY); }} catch (e) {{}}
  }}

  let header = null;
  let headerCount = 0;
  let queue = [];
  let displayedTimestamps = [];

  function persist() {{
    try {{
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify({{
        runId: RUN_ID,
        processedCount: processedCount,
        displayedCount: displayedCount,
        lastDisplayedTdone: lastDisplayedTdone,
        logHtml: logContainer.innerHTML,
      }}));
    }} catch (e) {{}}
  }}

  function parseCsvLine(line) {{
    const result = [];
    let cur = "";
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {{
      const c = line[i];
      if (inQuotes) {{
        if (c === '"') {{
          if (line[i + 1] === '"') {{ cur += '"'; i++; }}
          else {{ inQuotes = false; }}
        }} else {{ cur += c; }}
      }} else {{
        if (c === '"') {{ inQuotes = true; }}
        else if (c === ",") {{ result.push(cur); cur = ""; }}
        else {{ cur += c; }}
      }}
    }}
    result.push(cur);
    return result;
  }}

  function escapeHtml(s) {{
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }}

  function addLogLine(row) {{
    const truth = TRUTH[row.query_id] || {{}};
    const trueIntent = truth.true_intent;
    const ambiguous = truth.ambiguous;
    const correct = trueIntent !== undefined && row.intent === trueIntent;
    const conf = parseFloat(row.intent_confidence).toFixed(2);
    const latency = Math.round(parseFloat(row.latency_ms));
    let text = row.query + " \\u2192 " + row.intent.padEnd(14) + " conf " + conf + "  " + latency + "ms";
    if (!correct && trueIntent !== undefined) {{ text += " (\\u6b63\\u89e3: " + trueIntent + ")"; }}
    if (String(ambiguous) === "1") {{ text += " \\u26a0"; }}

    const div = document.createElement("div");
    div.className = "log-line " + (correct ? "correct" : "incorrect") + " flash";
    div.textContent = text;
    logContainer.prepend(div);
    while (logContainer.children.length > 40) {{
      logContainer.removeChild(logContainer.lastChild);
    }}
    setTimeout(function() {{ div.classList.remove("flash"); }}, 600);

    displayedCount++;
    const tDone = parseFloat(row.t_done_s);
    displayedTimestamps.push(tDone);
    while (displayedTimestamps.length && displayedTimestamps[0] < tDone - 1) {{ displayedTimestamps.shift(); }}

    elDone.textContent = displayedCount;
    elElapsed.textContent = tDone.toFixed(1) + "s";
    elRps.textContent = displayedTimestamps.length;

    persist();
  }}

  async function poll() {{
    try {{
      const url = streamUrl + "?t=" + Date.now();
      const resp = await fetch(url, {{ cache: "no-store" }});
      if (resp.ok) {{
        const text = await resp.text();
        let lines = text.split("\\n");
        if (lines.length && lines[lines.length - 1] === "") {{ lines.pop(); }}
        if (lines.length > 0) {{
          if (!header) {{
            header = parseCsvLine(lines[0]);
            headerCount = header.length;
          }}
          const dataLines = lines.slice(1);
          let newCount = 0;
          for (let i = processedCount; i < dataLines.length; i++) {{
            const fields = parseCsvLine(dataLines[i]);
            if (fields.length !== headerCount) {{ break; }}
            const row = {{}};
            for (let j = 0; j < header.length; j++) {{ row[header[j]] = fields[j]; }}
            queue.push(row);
            processedCount++;
            newCount++;
          }}
          console.log("[live-log] fetch: " + newCount + " new row(s), queue=" + queue.length + ", processed=" + processedCount);
          statusLine.textContent = "\\u53d6\\u5f97\\u6e08\\u307f: " + processedCount + " \\u4ef6 / \\u30ad\\u30e5\\u30fc: " + queue.length;
        }}
      }}
    }} catch (e) {{
      console.warn("[live-log] fetch failed", e);
    }}
    setTimeout(poll, 100);
  }}

  function displayNext() {{
    if (queue.length === 0) {{
      setTimeout(displayNext, 100);
      return;
    }}
    const row = queue.shift();
    addLogLine(row);
    const tDone = parseFloat(row.t_done_s);
    let delay = (lastDisplayedTdone === null) ? 50 : Math.max(0, (tDone - lastDisplayedTdone) * 1000);
    lastDisplayedTdone = tDone;
    if (queue.length >= 30) {{ delay = delay / 2; }}
    setTimeout(displayNext, delay);
  }}

  poll();
  displayNext();
}})();
</script>
"""
    components.html(_live_log_html, height=560, scrolling=True)

    _status_for_interval = read_live_status()
    _is_running_now = (_status_for_interval is not None and _status_for_interval.get("state") == "running") or is_live_process_running()
    _fragment_interval = 1.0 if _is_running_now else None

    @st.fragment(run_every=_fragment_interval)
    def live_fragment() -> None:
        status = read_live_status()
        stream_df = read_live_stream()

        if status is None and stream_df.empty:
            st.info("「開始」を押すとここにリアルタイムの進捗が表示されます。")
            return

        queries_df = load_csv(str(QUERIES_PATH))
        if len(stream_df):
            merged_live = stream_df.merge(
                queries_df[
                    ["query_id", "true_intent", "ambiguous", "out_of_category", "mentions_competitor", "own_product_only"]
                ],
                on="query_id",
                how="left",
                suffixes=("", "_true"),
            )
            merged_live["correct"] = merged_live["intent"] == merged_live["true_intent"]
        else:
            merged_live = stream_df

        total = status.get("total", 0) if status else len(stream_df)
        done = status.get("done", len(stream_df)) if status else len(stream_df)
        state = status.get("state", "unknown") if status else "unknown"
        started_at = status.get("started_at") if status else None

        elapsed = None
        if started_at:
            try:
                started_dt = datetime.fromisoformat(started_at)
                elapsed = (datetime.now(timezone.utc) - started_dt).total_seconds()
            except ValueError:
                elapsed = None

        if elapsed and elapsed > 0 and len(merged_live):
            window_start = elapsed - 2
            recent = merged_live[merged_live["t_done_s"] >= window_start]
            throughput = len(recent) / min(2.0, elapsed)
        else:
            throughput = 0.0

        avg_latency = merged_live["latency_ms"].mean() if len(merged_live) else float("nan")
        acc_all = merged_live["correct"].mean() if len(merged_live) else float("nan")
        clean_subset = (
            merged_live[(merged_live["ambiguous"] == 0) & (merged_live["out_of_category_true"] == 0)]
            if len(merged_live)
            else merged_live
        )
        acc_clean = clean_subset["correct"].mean() if len(clean_subset) else float("nan")

        metric_cols = st.columns(6)
        metric_cols[0].metric("完了/総数", f"{done}/{total}")
        metric_cols[1].metric("経過秒", f"{elapsed:.1f}s" if elapsed is not None else "-")
        metric_cols[2].metric("スループット", f"{throughput:.1f} 件/秒")
        metric_cols[3].metric("平均レイテンシ", f"{avg_latency:.0f} ms" if len(merged_live) else "-")
        metric_cols[4].metric("正解率", f"{acc_all:.1%}" if len(merged_live) else "-")
        metric_cols[5].metric("正解率(非曖昧・カテゴリ内)", f"{acc_clean:.1%}" if len(clean_subset) else "-")

        st.progress(done / total if total else 0.0)

        st.markdown("#### 累積正解率")
        if len(merged_live):
            sorted_live = merged_live.sort_values("t_done_s").reset_index(drop=True)
            sorted_live["n"] = range(1, len(sorted_live) + 1)
            sorted_live["cum_acc"] = sorted_live["correct"].expanding().mean()
            sorted_live["group"] = "全体"
            parts = [sorted_live[["n", "cum_acc", "group"]]]
            amb0_live = sorted_live[sorted_live["ambiguous"] == 0].copy()
            if len(amb0_live):
                amb0_live["cum_acc"] = amb0_live["correct"].expanding().mean()
                amb0_live["group"] = "ambiguous=0"
                parts.append(amb0_live[["n", "cum_acc", "group"]])
            cum_live_df = pd.concat(parts, ignore_index=True)
            chart = (
                alt.Chart(cum_live_df)
                .mark_line()
                .encode(
                    x=alt.X("n:Q", title="完了件数"),
                    y=alt.Y("cum_acc:Q", title="累積正解率", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("group:N", title=""),
                )
                .properties(height=280)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("まだ結果がありません。")

        st.markdown("#### スループット / レイテンシ推移")
        col_t, col_l = st.columns(2)
        with col_t:
            if len(merged_live):
                tdf = merged_live.copy()
                tdf["sec_bucket"] = tdf["t_done_s"].astype(float).astype(int)
                thpt = tdf.groupby("sec_bucket").size().reset_index(name="count")
                st.altair_chart(
                    alt.Chart(thpt)
                    .mark_bar()
                    .encode(
                        x=alt.X("sec_bucket:O", title="経過秒"),
                        y=alt.Y("count:Q", title="件/秒"),
                    )
                    .properties(height=250),
                    use_container_width=True,
                )
            else:
                st.info("まだ結果がありません。")
        with col_l:
            if len(merged_live):
                ldf = merged_live.sort_values("t_done_s")
                st.altair_chart(
                    alt.Chart(ldf)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("t_done_s:Q", title="経過秒"),
                        y=alt.Y("latency_ms:Q", title="latency_ms"),
                    )
                    .properties(height=250),
                    use_container_width=True,
                )
            else:
                st.info("まだ結果がありません。")

        if state == "done" and total and len(merged_live):
            total_time = merged_live["t_done_s"].max()
            mean_rps = total / total_time if total_time else 0.0
            total_input_tokens = (
                merged_live["input_tokens"].dropna().sum() if "input_tokens" in merged_live.columns else 0
            )
            cost = total_input_tokens / 1_000_000 * PRICE_PER_M_INPUT_TOKENS
            st.success(f"{total}件を{total_time:.1f}秒で分類、平均{mean_rps:.1f}件/秒、推定コスト ${cost:.4f}")
            st.caption("単価は公称値に基づく推定です。")

            if st.button(f"results.csvパスに {LIVE_OUT_PATH.relative_to(ROOT)} をセット", key="apply_live_results"):
                st.session_state["results_path_input"] = str(LIVE_OUT_PATH.relative_to(ROOT))
                st.rerun()
        elif state == "error":
            st.error("classify.pyの実行中にエラーが発生しました。ターミナルのログを確認してください。")

    live_fragment()

    footer()

# ---------------------------------------------------------------------------
# 概要
# ---------------------------------------------------------------------------
with tab_overview:
    n = len(filtered)
    acc_all = filtered["correct"].mean() if n else float("nan")
    amb0 = filtered[filtered["ambiguous"] == 0]
    amb1 = filtered[filtered["ambiguous"] == 1]
    acc_amb0 = amb0["correct"].mean() if len(amb0) else float("nan")
    acc_amb1 = amb1["correct"].mean() if len(amb1) else float("nan")
    avg_latency = filtered["latency_ms"].mean() if n else float("nan")
    p95_latency = filtered["latency_ms"].quantile(0.95) if n else float("nan")
    models = filtered["model"].dropna().unique().tolist()
    model_label = models[0] if len(models) == 1 else (", ".join(models) if models else "-")

    cols = st.columns(7)
    cols[0].metric("件数", f"{n}")
    cols[1].metric("正解率(全体)", f"{acc_all:.1%}" if n else "-")
    cols[2].metric("正解率(ambiguous=0)", f"{acc_amb0:.1%}" if len(amb0) else "-")
    cols[3].metric("正解率(ambiguous=1)", f"{acc_amb1:.1%}" if len(amb1) else "-")
    cols[4].metric("平均レイテンシ", f"{avg_latency:.0f} ms" if n else "-")
    cols[5].metric("p95レイテンシ", f"{p95_latency:.0f} ms" if n else "-")
    cols[6].metric("モデル", model_label)

    questions_mod = try_load_questions()
    with st.expander("質問文(questions.py)"):
        if isinstance(questions_mod, Exception):
            st.info(f"questions.py のimportに失敗したため質問文の表示をスキップします: {questions_mod}")
        else:
            intent_q = questions_mod.QUESTIONS["intent"]
            st.markdown("**intent**")
            st.write(intent_q.instructions)
            st.write({k: v for k, v in intent_q.criteria.items()})
            for key, q in questions_mod.QUESTIONS.items():
                if key == "intent":
                    continue
                st.markdown(f"**{key}**")
                st.write(q.instructions)

    st.subheader("累積正解率(query_id順)")
    if n:
        sorted_all = filtered.sort_values("query_id").copy()
        sorted_all["cum_acc"] = sorted_all["correct"].expanding().mean()
        sorted_all["group"] = "全体"

        parts = [sorted_all[["query_id", "cum_acc", "group"]]]
        for label, subset in [("ambiguous=0", amb0), ("ambiguous=1", amb1)]:
            if len(subset):
                s = subset.sort_values("query_id").copy()
                s["cum_acc"] = s["correct"].expanding().mean()
                s["group"] = label
                parts.append(s[["query_id", "cum_acc", "group"]])
        cum_df = pd.concat(parts, ignore_index=True)

        chart = (
            alt.Chart(cum_df)
            .mark_line()
            .encode(
                x=alt.X("query_id:N", sort=None, title="query_id"),
                y=alt.Y("cum_acc:Q", title="累積正解率", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("group:N", title=""),
            )
            .properties(height=350)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("フィルタ条件に一致する行がありません。")

    footer()

# ---------------------------------------------------------------------------
# 判定ログ
# ---------------------------------------------------------------------------
with tab_log:
    log_cols = ["query_id", qcol, "true_intent", "intent", "correct", "intent_confidence", "ambiguous", "latency_ms"]
    log_df = filtered[log_cols].rename(columns={qcol: "query"}).reset_index(drop=True)

    def highlight_incorrect(row: pd.Series) -> list[str]:
        color = "background-color: rgba(255, 0, 0, 0.15)" if not row["correct"] else ""
        return [color] * len(row)

    styled = log_df.style.apply(highlight_incorrect, axis=1)
    event = st.dataframe(
        styled,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key="log_table",
    )

    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        sel_idx = selected_rows[0]
        sel_query_id = log_df.iloc[sel_idx]["query_id"]
        detail = filtered[filtered["query_id"] == sel_query_id].iloc[0]

        st.markdown(f"### 詳細: {sel_query_id} — {detail[qcol]}")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**4クラス確率**")
            prob_rows = [
                {"intent": label, "probability": detail[PROB_COLS[label]], "is_true": label == detail["true_intent"]}
                for label in INTENT_LABELS
            ]
            prob_df = pd.DataFrame(prob_rows)
            bar = (
                alt.Chart(prob_df)
                .mark_bar()
                .encode(
                    x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("intent:N", sort=INTENT_LABELS),
                    color=alt.Color(
                        "is_true:N",
                        scale=alt.Scale(domain=[True, False], range=["#2ca02c", "#4c78a8"]),
                        legend=alt.Legend(title="正解クラス"),
                    ),
                )
                .properties(height=200)
            )
            st.altair_chart(bar, use_container_width=True)

        with c2:
            st.markdown("**Noulの値と正解フラグ**")
            noul_rows = []
            for spec in NOUL_SPECS:
                truth_col = spec["truth_col"]
                truth_val = detail[truth_col] if truth_col in detail else None
                noul_rows.append(
                    {
                        "noul": spec["label"],
                        "value": detail[spec["pred_col"]],
                        "正解フラグ": truth_val,
                    }
                )
            st.dataframe(pd.DataFrame(noul_rows), use_container_width=True, hide_index=True)
            st.markdown(f"**product_surface:** {detail.get('product_surface', '')}")
    else:
        st.info("表の行をクリックすると詳細が表示されます。")

    footer()

# ---------------------------------------------------------------------------
# インテント分析
# ---------------------------------------------------------------------------
with tab_intent:
    if len(filtered) == 0:
        st.info("フィルタ条件に一致する行がありません。")
    else:
        st.subheader("混同行列 (true_intent × intent)")
        cm = (
            filtered.groupby(["true_intent", "intent"]).size().reset_index(name="count")
        )
        heat = (
            alt.Chart(cm)
            .mark_rect()
            .encode(
                x=alt.X("intent:N", sort=INTENT_LABELS, title="予測 intent"),
                y=alt.Y("true_intent:N", sort=INTENT_LABELS, title="true_intent"),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="blues")),
            )
            .properties(height=320)
        )
        text = (
            alt.Chart(cm)
            .mark_text()
            .encode(
                x=alt.X("intent:N", sort=INTENT_LABELS),
                y=alt.Y("true_intent:N", sort=INTENT_LABELS),
                text="count:Q",
            )
        )
        st.altair_chart(heat + text, use_container_width=True)

        st.subheader("confidence帯ごとの件数・正解率(較正の確認)")
        binned = filtered.copy()
        binned["conf_bin"] = pd.cut(
            binned["intent_confidence"], bins=CONFIDENCE_BIN_EDGES, labels=CONFIDENCE_BIN_LABELS, include_lowest=True
        )
        calib = binned.groupby("conf_bin", observed=True).agg(count=("correct", "size"), accuracy=("correct", "mean")).reset_index()
        c1, c2 = st.columns(2)
        with c1:
            st.altair_chart(
                alt.Chart(calib).mark_bar().encode(
                    x=alt.X("conf_bin:N", sort=CONFIDENCE_BIN_LABELS, title="confidence帯"),
                    y=alt.Y("count:Q", title="件数"),
                ).properties(height=300, title="件数"),
                use_container_width=True,
            )
        with c2:
            st.altair_chart(
                alt.Chart(calib).mark_bar().encode(
                    x=alt.X("conf_bin:N", sort=CONFIDENCE_BIN_LABELS, title="confidence帯"),
                    y=alt.Y("accuracy:Q", title="正解率", scale=alt.Scale(domain=[0, 1])),
                ).properties(height=300, title="正解率"),
                use_container_width=True,
            )

        st.subheader("ambiguous=1: true_intentが確率上位2位以内の割合")
        amb1_df = filtered[filtered["ambiguous"] == 1].copy()
        if len(amb1_df):
            def rank_of_true(row) -> int:
                probs = {label: row[PROB_COLS[label]] for label in INTENT_LABELS}
                ranked = sorted(probs, key=probs.get, reverse=True)
                return ranked.index(row["true_intent"]) + 1 if row["true_intent"] in ranked else len(ranked)

            amb1_df["true_rank"] = amb1_df.apply(rank_of_true, axis=1)
            top2_ratio = (amb1_df["true_rank"] <= 2).mean()
            st.metric("top-2以内の割合", f"{top2_ratio:.1%}", help=f"対象 {len(amb1_df)} 件")
        else:
            st.info("ambiguous=1の行がありません。")

        st.subheader("CTR / CVR (判定 intent 別 vs true_intent 別)")

        def ctr_cvr(group_col: str) -> pd.DataFrame:
            agg = filtered.groupby(group_col).agg(
                impressions=("impressions", "sum"), clicks=("clicks", "sum"), conversions=("conversions", "sum")
            ).reset_index()
            agg["CTR"] = agg["clicks"] / agg["impressions"]
            agg["CVR"] = agg["conversions"] / agg["clicks"]
            return agg.melt(id_vars=group_col, value_vars=["CTR", "CVR"], var_name="metric", value_name="value")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**判定 intent 別**")
            pred_agg = ctr_cvr("intent").rename(columns={"intent": "group"})
            st.altair_chart(
                alt.Chart(pred_agg).mark_bar().encode(
                    x=alt.X("group:N", sort=INTENT_LABELS, title="intent(予測)"),
                    y=alt.Y("value:Q", title=""),
                    color="metric:N",
                    column=alt.Column("metric:N", title=""),
                ).properties(height=280),
                use_container_width=True,
            )
        with c2:
            st.markdown("**true_intent 別**")
            true_agg = ctr_cvr("true_intent").rename(columns={"true_intent": "group"})
            st.altair_chart(
                alt.Chart(true_agg).mark_bar().encode(
                    x=alt.X("group:N", sort=INTENT_LABELS, title="true_intent"),
                    y=alt.Y("value:Q", title=""),
                    color="metric:N",
                    column=alt.Column("metric:N", title=""),
                ).properties(height=280),
                use_container_width=True,
            )

    footer()

# ---------------------------------------------------------------------------
# Noul分析
# ---------------------------------------------------------------------------
with tab_noul:
    if len(filtered) == 0:
        st.info("フィルタ条件に一致する行がありません。")
    else:
        st.caption(
            "mentions_own_* / mentions_competitor_* (元の4つ) の正例・負例は、"
            "queries.csvに直接の正解列が無いためdata/vocab.yamlの語彙との文字列一致による参考値です。"
        )

        st.subheader("値のヒストグラム(正例・負例を重ねて表示)")
        for spec in NOUL_SPECS:
            pred_col, truth_col = spec["pred_col"], spec["truth_col"]
            if truth_col not in filtered.columns:
                continue
            hist_df = filtered[[pred_col, truth_col]].dropna().copy()
            hist_df["label"] = hist_df[truth_col].map({1: "正例", 0: "負例"})
            chart = (
                alt.Chart(hist_df)
                .mark_bar(opacity=0.6)
                .encode(
                    x=alt.X(f"{pred_col}:Q", bin=alt.Bin(maxbins=20), title=spec["label"]),
                    y=alt.Y("count():Q", stack=None, title="件数"),
                    color=alt.Color("label:N", title=""),
                )
                .properties(height=220, title=spec["label"])
            )
            st.altair_chart(chart, use_container_width=True)

        st.subheader(f"しきい値 {threshold:.2f} での precision / recall / F1")
        pr_rows = []
        for spec in NOUL_SPECS:
            pred_col, truth_col = spec["pred_col"], spec["truth_col"]
            if truth_col not in filtered.columns:
                continue
            sub = filtered[[pred_col, truth_col]].dropna()
            pred_bin = (sub[pred_col] >= threshold).astype(int)
            truth = sub[truth_col].astype(int)
            tp = int(((pred_bin == 1) & (truth == 1)).sum())
            fp = int(((pred_bin == 1) & (truth == 0)).sum())
            fn = int(((pred_bin == 0) & (truth == 1)).sum())
            precision = tp / (tp + fp) if (tp + fp) else float("nan")
            recall = tp / (tp + fn) if (tp + fn) else float("nan")
            f1 = 2 * precision * recall / (precision + recall) if precision and recall and (precision + recall) else float("nan")
            pr_rows.append(
                {"noul": spec["label"], "precision": precision, "recall": recall, "f1": f1, "support(正例)": int(truth.sum())}
            )
        st.dataframe(pd.DataFrame(pr_rows), use_container_width=True, hide_index=True)

        st.subheader("誤判定の行一覧")
        noul_choice = st.selectbox("対象のNoul", [s["label"] for s in NOUL_SPECS], key="noul_error_select")
        spec = next(s for s in NOUL_SPECS if s["label"] == noul_choice)
        pred_col, truth_col = spec["pred_col"], spec["truth_col"]
        if truth_col in filtered.columns:
            sub = filtered.dropna(subset=[pred_col, truth_col]).copy()
            sub["pred_bin"] = (sub[pred_col] >= threshold).astype(int)
            fp_df = sub[(sub["pred_bin"] == 1) & (sub[truth_col] == 0)]
            fn_df = sub[(sub["pred_bin"] == 0) & (sub[truth_col] == 1)]
            cols_to_show = ["query_id", qcol, pred_col, truth_col, "product_surface"]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**偽陽性 ({len(fp_df)}件)**")
                st.dataframe(fp_df[cols_to_show].rename(columns={qcol: "query"}), use_container_width=True, hide_index=True)
            with c2:
                st.markdown(f"**偽陰性 ({len(fn_df)}件)**")
                st.dataframe(fn_df[cols_to_show].rename(columns={qcol: "query"}), use_container_width=True, hide_index=True)

    footer()

# ---------------------------------------------------------------------------
# レイテンシ
# ---------------------------------------------------------------------------
with tab_latency:
    if len(filtered) == 0:
        st.info("フィルタ条件に一致する行がありません。")
    else:
        st.subheader("レイテンシのヒストグラム")
        hist = (
            alt.Chart(filtered)
            .mark_bar()
            .encode(
                x=alt.X("latency_ms:Q", bin=alt.Bin(maxbins=30), title="latency_ms"),
                y=alt.Y("count():Q", title="件数"),
            )
            .properties(height=300)
        )
        st.altair_chart(hist, use_container_width=True)

        st.subheader("query_id順の推移")
        line = (
            alt.Chart(filtered.sort_values("query_id"))
            .mark_line(point=True)
            .encode(
                x=alt.X("query_id:N", sort=None, title="query_id"),
                y=alt.Y("latency_ms:Q", title="latency_ms"),
            )
            .properties(height=300)
        )
        st.altair_chart(line, use_container_width=True)

    footer()
