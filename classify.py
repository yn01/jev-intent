#!/usr/bin/env python3
"""data/queries.csv の検索語句をJevで分類し、results.csvに保存する。

questions.py で定義した質問(intent / mentions_own_brand /
mentions_own_product / mentions_competitor_brand /
mentions_competitor_product / out_of_category)をquery列だけから組み立てた
stateに対して投げ、選択結果・各クラスの確率・confidence・各Noulの値・
合成列(mentions_competitor / own_product_only)・レイテンシ・使用モデル名・
トークン使用量を1行ずつ記録する。

--stream を指定すると、完了順に1行ずつ即時追記(t_done_s列つき)し、
同じディレクトリの .status.json に進捗を書く。ダッシュボードの
「ライブ」タブはこれを1秒間隔でポーリングして進捗を表示する。

使い方:
    python3 classify.py --limit 10
    python3 classify.py --sample 100 --seed 42
    python3 classify.py --sample 100 --stream live/results_live.csv --out live/results_last.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from questions import QUESTIONS, build_state

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = Path("data/queries.csv")
DEFAULT_OUTPUT = Path("results.csv")
DEFAULT_WORKERS = 8
DEFAULT_MAX_RPS = 18.0

INTENT_LABELS = ["informational", "comparison", "transactional", "navigational"]
ROUND_NDIGITS = 4

FIELDNAMES = [
    "query_id",
    "query",
    "intent",
    "intent_prob_informational",
    "intent_prob_comparison",
    "intent_prob_transactional",
    "intent_prob_navigational",
    "intent_confidence",
    "mentions_own_brand",
    "mentions_own_product",
    "mentions_competitor_brand",
    "mentions_competitor_product",
    "out_of_category",
    "mentions_competitor",
    "own_product_only",
    "latency_ms",
    "model",
    "input_tokens",
    "output_tokens",
]
STREAM_FIELDNAMES = FIELDNAMES + ["t_done_s"]


def load_api_key() -> None:
    """TYPESAFE_API_KEYが未設定の場合、.claude/settings.local.jsonから読み込む。

    キーの値は絶対に出力しない。
    """
    if os.environ.get("TYPESAFE_API_KEY"):
        return
    settings_path = SCRIPT_DIR / ".claude" / "settings.local.json"
    if not settings_path.exists():
        return
    try:
        with settings_path.open(encoding="utf-8") as f:
            data = json.load(f)
        key = data.get("env", {}).get("TYPESAFE_API_KEY")
    except (OSError, json.JSONDecodeError):
        return
    if key:
        os.environ["TYPESAFE_API_KEY"] = key


class TokenBucket:
    """シンプルなトークンバケットによる送信レート制限。"""

    def __init__(self, rate: float):
        self.rate = rate
        self.capacity = max(rate, 1.0)
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            time.sleep(wait)


def classify_row(client, query: str, start_time: float) -> dict:
    state = build_state(query)

    req_start = time.perf_counter()
    result = client.system_one(state, QUESTIONS)
    latency_ms = (time.perf_counter() - req_start) * 1000

    intent = result.answers["intent"]
    own_brand = result.answers["mentions_own_brand"].noul
    own_product = result.answers["mentions_own_product"].noul
    competitor_brand = result.answers["mentions_competitor_brand"].noul
    competitor_product = result.answers["mentions_competitor_product"].noul

    row = {
        "intent": intent.choice,
        "intent_confidence": round(intent.confidence, ROUND_NDIGITS),
        "mentions_own_brand": round(own_brand, ROUND_NDIGITS),
        "mentions_own_product": round(own_product, ROUND_NDIGITS),
        "mentions_competitor_brand": round(competitor_brand, ROUND_NDIGITS),
        "mentions_competitor_product": round(competitor_product, ROUND_NDIGITS),
        "out_of_category": round(result.answers["out_of_category"].noul, ROUND_NDIGITS),
        "mentions_competitor": round(max(competitor_brand, competitor_product), ROUND_NDIGITS),
        "own_product_only": round(own_product * (1 - own_brand), ROUND_NDIGITS),
        "latency_ms": round(latency_ms, 1),
        "model": result.model,
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
    }
    for label in INTENT_LABELS:
        row[f"intent_prob_{label}"] = round(intent.probabilities.get(label, 0.0), ROUND_NDIGITS)
    return row


def select_rows(rows: list[dict], limit: int | None, sample: int | None, seed: int | None) -> list[dict]:
    if sample is not None:
        rng = random.Random(seed)
        return rng.sample(rows, min(sample, len(rows)))
    if limit is not None:
        return rows[:limit]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Jevで検索語句を分類してCSVに保存する")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="入力CSV(既定: data/queries.csv)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="query_id順の最終出力CSV(既定: results.csv)")
    parser.add_argument("--stream", type=Path, default=None, help="完了順に逐次追記するCSV。指定時は同ディレクトリに.status.jsonも書く")
    parser.add_argument("--limit", type=int, default=None, help="先頭から処理する行数の上限")
    parser.add_argument("--sample", type=int, default=None, help="ランダムに抽出する行数(--limitより優先)")
    parser.add_argument("--seed", type=int, default=None, help="--sample使用時の乱数シード")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="並列ワーカー数(既定: 8)")
    parser.add_argument("--max-rps", type=float, default=DEFAULT_MAX_RPS, help="送信レートの上限(req/s、既定: 18。0以下で無制限)")
    args = parser.parse_args()

    load_api_key()
    from typesafe_sdk import TypeSafeClient  # 遅延import: APIキー設定後に読み込む

    with args.input.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    rows = select_rows(rows, args.limit, args.sample, args.seed)
    total = len(rows)

    status_path = (args.stream.parent / "status.json") if args.stream else None
    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()
    done_count = 0
    status_lock = threading.Lock()
    last_status_write = 0.0

    def write_status(state: str) -> None:
        if status_path is None:
            return
        payload = {
            "state": state,
            "total": total,
            "done": done_count,
            "started_at": started_at,
            "workers": args.workers,
            "max_rps": args.max_rps,
        }
        status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = status_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
        tmp_path.replace(status_path)

    def maybe_write_status() -> None:
        nonlocal last_status_write
        now = time.time()
        if now - last_status_write >= 1.0:
            write_status("running")
            last_status_write = now

    client = TypeSafeClient()
    bucket = TokenBucket(args.max_rps) if args.max_rps and args.max_rps > 0 else None

    stream_file = None
    stream_writer = None
    if args.stream:
        args.stream.parent.mkdir(parents=True, exist_ok=True)
        stream_file = args.stream.open("w", encoding="utf-8", newline="")
        stream_writer = csv.DictWriter(stream_file, fieldnames=STREAM_FIELDNAMES)
        stream_writer.writeheader()
        stream_file.flush()

    def process(row: dict) -> tuple[str, str, dict]:
        if bucket is not None:
            bucket.acquire()
        return row["query_id"], row["query"], classify_row(client, row["query"], start_time)

    results_by_id: dict[str, tuple[str, dict]] = {}
    write_status("running")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process, row) for row in rows]
            for future in as_completed(futures):
                query_id, query, result_row = future.result()
                t_done_s = round(time.perf_counter() - start_time, 3)
                results_by_id[query_id] = (query, result_row)

                with status_lock:
                    done_count += 1
                    current_done = done_count

                if stream_writer is not None:
                    stream_writer.writerow({"query_id": query_id, "query": query, **result_row, "t_done_s": t_done_s})
                    stream_file.flush()
                maybe_write_status()
                print(f"[{current_done}/{total}] {query_id}: {query!r} -> intent={result_row['intent']}")
    except BaseException:
        write_status("error")
        raise
    finally:
        if stream_file is not None:
            stream_file.close()

    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for query_id in sorted(results_by_id):
            query, result_row = results_by_id[query_id]
            writer.writerow({"query_id": query_id, "query": query, **result_row})

    write_status("done")
    print(f"完了: {total}件を{args.out}に保存しました。")


if __name__ == "__main__":
    main()
