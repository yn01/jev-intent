#!/usr/bin/env python3
"""
data/queries.csv 生成スクリプト

デジタルカメラ・交換レンズ分野の検索語句レポートを模したダミーデータを作る。
自社=ニコン、競合=キヤノン/ソニー。data/vocab.yaml の語彙とテンプレートを
組み合わせてクエリ文を生成し、意図(true_intent)・競合言及(mentions_competitor)・
カテゴリ外(out_of_category)・境界例(ambiguous)のラベルを付与したうえで、
インテント別のCVR/CTR母数から乱数で impressions/clicks/conversions を生成する。

標準ライブラリ + PyYAML のみに依存（numpy/pandas 不要）。

使い方:
    python3 data/generate.py
    python3 data/generate.py --n 800 --seed 42 --out data/queries.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import yaml

INTENTS = ["informational", "comparison", "transactional", "navigational"]

# 正常系クエリにおけるインテント出現比率（合計1.0）
INTENT_WEIGHTS = {
    "informational": 0.35,
    "transactional": 0.30,
    "comparison": 0.20,
    "navigational": 0.15,
}

# インテント別のボリューム・CTR・CVR母数（合成データ用の仮パラメータ）
# impressions は対数正規分布 lognormvariate(mu, sigma) = exp(N(mu, sigma)) から生成。
INTENT_METRIC_PARAMS = {
    "informational": {"imp_lognorm": (6.5, 0.9), "ctr_range": (0.02, 0.06), "cvr_range": (0.002, 0.015)},
    "comparison": {"imp_lognorm": (5.5, 0.8), "ctr_range": (0.03, 0.08), "cvr_range": (0.010, 0.040)},
    "transactional": {"imp_lognorm": (5.0, 0.9), "ctr_range": (0.05, 0.12), "cvr_range": (0.030, 0.100)},
    "navigational": {"imp_lognorm": (4.5, 0.9), "ctr_range": (0.15, 0.35), "cvr_range": (0.050, 0.200)},
}

# 生成する行のうち、境界例(ambiguous)・カテゴリ外(out_of_category)の比率
AMBIGUOUS_RATIO = 0.15
OUT_OF_CATEGORY_RATIO = 0.05

# out_of_category テンプレート -> 疑似的なインテント割り当て（ベストエフォート）
OOC_TEMPLATE_INTENT_MAP = {
    "{brand} {term}": "navigational",
    "{term} おすすめ": "comparison",
    "{term} 比較": "comparison",
    "{term} とは": "informational",
    "{term} 最安値": "transactional",
}

COMPETITOR_KEYS = ["canon", "sony"]
ALL_BRAND_KEYS = ["nikon", "canon", "sony"]

# ブランド名なし（自社型番のみ）テンプレを使う確率（対象: informational/transactional/navigational）
BRANDLESS_PROB = 0.30
BRANDLESS_CAPABLE_INTENTS = ["informational", "transactional", "navigational"]

# 製品の表記を選ぶ際、正式名(name)ではなく略称(aliases)を選ぶ確率
# （aliasesが1件も無い製品は常にnameが使われる）
PRODUCT_ALIAS_PROB = 0.70


def load_vocab(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def brand_alias(brand_key: str, vocab: dict, rng: random.Random) -> str:
    if brand_key == "nikon":
        aliases = vocab["brands"]["own"]["aliases"]
    else:
        aliases = vocab["brands"]["competitors"][brand_key]["aliases"]
    return rng.choice(aliases)


def _normalize_product_entry(item) -> dict:
    """products配下の1件を {"name": str, "aliases": list[str]} に正規化する。
    既存の単純な文字列形式（例: "Z9"）も aliases=[] として扱う。"""
    if isinstance(item, str):
        return {"name": item, "aliases": []}
    return {"name": item["name"], "aliases": list(item.get("aliases") or [])}


def _product_pool(brand_key: str, vocab: dict) -> list[dict]:
    products = vocab["products"]
    if brand_key == "nikon":
        raw = list(products["own"]["cameras"]) + list(products["own"]["lenses"])
    else:
        comp = products["competitors"][brand_key]
        raw = list(comp["cameras"]) + list(comp["lenses"])
    return [_normalize_product_entry(item) for item in raw]


def choose_surface(entry: dict, rng: random.Random) -> str:
    """製品1件の name/aliases から、実際にクエリに使う表記を重み付きで選ぶ
    （aliasesがあれば PRODUCT_ALIAS_PROB の確率でaliasesの方を優先）。"""
    aliases = entry["aliases"]
    if aliases and rng.random() < PRODUCT_ALIAS_PROB:
        return rng.choice(aliases)
    return entry["name"]


def pick_product(brand_key: str, vocab: dict, rng: random.Random) -> tuple[str, bool]:
    """製品の表記文字列と、実在の(ブランド固有の)製品名かどうかを返す。
    products が空で product_categories にフォールバックした場合は False。"""
    pool = _product_pool(brand_key, vocab)
    if pool:
        entry = rng.choice(pool)
        return choose_surface(entry, rng), True
    # 製品名が未記入の場合は一般名詞にフォールバック（ブランド固有ではない）
    cats = vocab["product_categories"]
    fallback_pool = list(cats["cameras"]) + list(cats["lenses"])
    return rng.choice(fallback_pool), False


def weighted_intent(rng: random.Random) -> str:
    return rng.choices(INTENTS, weights=[INTENT_WEIGHTS[i] for i in INTENTS], k=1)[0]


def pick_brand_for_single(rng: random.Random) -> str:
    """comparison以外のテンプレート用: 65%自社 / 35%競合"""
    if rng.random() < 0.65:
        return "nikon"
    return rng.choice(COMPETITOR_KEYS)


def uses_placeholder(name: str, template: str) -> bool:
    return ("{" + name + "}") in template


def build_query(intent: str, vocab: dict, rng: random.Random) -> tuple[str, set, set, list]:
    """クエリ文、ブランド表記スロットに使ったブランドキー集合、
    製品名スロットの出所ブランドキー集合、実際に使った製品表記のリストを返す。"""
    if intent == "comparison":
        b1, b2 = rng.sample(ALL_BRAND_KEYS, 2)
        template = rng.choice(vocab["templates"]["comparison"])
        modifier = rng.choice(vocab["modifiers"]["comparison"])
        product1, is_real1 = pick_product(b1, vocab, rng)
        product2, is_real2 = pick_product(b2, vocab, rng)
        text = template.format(
            brand=brand_alias(b1, vocab, rng),
            brand2=brand_alias(b2, vocab, rng),
            product=product1,
            product2=product2,
            modifier=modifier,
        )
        brand_alias_keys = set()
        if uses_placeholder("brand", template):
            brand_alias_keys.add(b1)
        if uses_placeholder("brand2", template):
            brand_alias_keys.add(b2)
        product_source_keys = set()
        surfaces = []
        if uses_placeholder("product", template) and is_real1:
            product_source_keys.add(b1)
            surfaces.append(product1)
        if uses_placeholder("product2", template) and is_real2:
            product_source_keys.add(b2)
            surfaces.append(product2)
        return normalize_text(text), brand_alias_keys, product_source_keys, surfaces

    use_brandless = (
        intent in BRANDLESS_CAPABLE_INTENTS
        and vocab.get("templates_brandless", {}).get(intent)
        and rng.random() < BRANDLESS_PROB
    )
    if use_brandless:
        b = "nikon"
        template = rng.choice(vocab["templates_brandless"][intent])
        modifier = rng.choice(vocab["modifiers"][intent])
        product, is_real = pick_product(b, vocab, rng)
        text = template.format(product=product, modifier=modifier)
    else:
        b = pick_brand_for_single(rng)
        template = rng.choice(vocab["templates"][intent])
        modifier = rng.choice(vocab["modifiers"][intent])
        product, is_real = pick_product(b, vocab, rng)
        text = template.format(
            brand=brand_alias(b, vocab, rng),
            product=product,
            modifier=modifier,
        )
    brand_alias_keys = {b} if uses_placeholder("brand", template) else set()
    product_used = uses_placeholder("product", template) and is_real
    product_source_keys = {b} if product_used else set()
    surfaces = [product] if product_used else []
    return normalize_text(text), brand_alias_keys, product_source_keys, surfaces


def build_ambiguous_query(vocab: dict, rng: random.Random) -> tuple[str, str, set, set, list]:
    """2つの異なるインテントの修飾語を1クエリに混在させ、支配的なほうをtrue_intentとする"""
    i1 = weighted_intent(rng)
    i2 = rng.choice([i for i in INTENTS if i != i1])
    template = rng.choice(vocab["ambiguous_templates"])
    modifier1 = rng.choice(vocab["modifiers"][i1])
    modifier2 = rng.choice(vocab["modifiers"][i2])
    b1 = pick_brand_for_single(rng)
    b2 = rng.choice([b for b in ALL_BRAND_KEYS if b != b1])
    product, is_real = pick_product(b1, vocab, rng)
    text = template.format(
        brand=brand_alias(b1, vocab, rng),
        brand2=brand_alias(b2, vocab, rng),
        product=product,
        modifier1=modifier1,
        modifier2=modifier2,
    )
    brand_alias_keys = set()
    if uses_placeholder("brand", template):
        brand_alias_keys.add(b1)
    if uses_placeholder("brand2", template):
        brand_alias_keys.add(b2)
    product_used = uses_placeholder("product", template) and is_real
    product_source_keys = {b1} if product_used else set()
    surfaces = [product] if product_used else []
    return normalize_text(text), i1, brand_alias_keys, product_source_keys, surfaces


def build_out_of_category_query(vocab: dict, rng: random.Random) -> tuple[str, str, set, set, list]:
    ooc = vocab["out_of_category"]
    template = rng.choice(ooc["templates"])
    term = rng.choice(ooc["terms"])
    brand_alias_keys = set()
    if uses_placeholder("brand", template):
        b = pick_brand_for_single(rng)
        text = template.format(brand=brand_alias(b, vocab, rng), term=term)
        brand_alias_keys.add(b)
    else:
        text = template.format(term=term)
    true_intent = OOC_TEMPLATE_INTENT_MAP.get(template, weighted_intent(rng))
    return normalize_text(text), true_intent, brand_alias_keys, set(), []


def compute_flags(brand_alias_keys: set, product_source_keys: set) -> tuple[int, int]:
    """スロット追跡結果から mentions_competitor / own_product_only を判定する
    （生成後の文字列マッチは行わない）。"""
    used_keys = brand_alias_keys | product_source_keys
    mentions_competitor = int(any(k in COMPETITOR_KEYS for k in used_keys))
    own_product_only = int(
        not mentions_competitor
        and not brand_alias_keys
        and product_source_keys == {"nikon"}
    )
    return mentions_competitor, own_product_only


def binomial_like(n: int, p: float, rng: random.Random) -> int:
    """二項分布 B(n, p) の簡易近似（正規近似）。合成データ用途のため十分。"""
    if n <= 0:
        return 0
    mean = n * p
    variance = n * p * (1 - p)
    sd = math.sqrt(variance) if variance > 0 else 0.0
    value = rng.gauss(mean, sd) if sd > 0 else mean
    return max(0, min(n, round(value)))


def gen_metrics(intent: str, rng: random.Random) -> tuple[int, int, int]:
    params = INTENT_METRIC_PARAMS[intent]
    mu, sigma = params["imp_lognorm"]
    impressions = round(rng.lognormvariate(mu, sigma))
    impressions = max(1, min(impressions, 50000))

    ctr = rng.uniform(*params["ctr_range"])
    clicks = binomial_like(impressions, ctr, rng)

    cvr = rng.uniform(*params["cvr_range"])
    conversions = binomial_like(clicks, cvr, rng)

    return impressions, clicks, conversions


def generate_rows(n: int, vocab: dict, rng: random.Random) -> list[dict]:
    n_ooc = round(n * OUT_OF_CATEGORY_RATIO)
    n_amb = round(n * AMBIGUOUS_RATIO)
    n_normal = n - n_ooc - n_amb

    kinds = ["out_of_category"] * n_ooc + ["ambiguous"] * n_amb + ["normal"] * n_normal
    rng.shuffle(kinds)

    seen_queries: set[str] = set()
    rows = []
    for i, kind in enumerate(kinds, start=1):
        ambiguous_flag = 1 if kind == "ambiguous" else 0
        ooc_flag = 1 if kind == "out_of_category" else 0

        text = ""
        true_intent = ""
        brand_alias_keys: set = set()
        product_source_keys: set = set()
        surfaces: list = []
        for _attempt in range(5):
            if kind == "out_of_category":
                text, true_intent, brand_alias_keys, product_source_keys, surfaces = build_out_of_category_query(vocab, rng)
            elif kind == "ambiguous":
                text, true_intent, brand_alias_keys, product_source_keys, surfaces = build_ambiguous_query(vocab, rng)
            else:
                true_intent = weighted_intent(rng)
                text, brand_alias_keys, product_source_keys, surfaces = build_query(true_intent, vocab, rng)
            if text not in seen_queries:
                break
        seen_queries.add(text)

        mentions_competitor, own_product_only = compute_flags(brand_alias_keys, product_source_keys)
        impressions, clicks, conversions = gen_metrics(true_intent, rng)

        rows.append(
            {
                "query_id": f"Q{i:04d}",
                "query": text,
                "true_intent": true_intent,
                "mentions_competitor": mentions_competitor,
                "out_of_category": ooc_flag,
                "ambiguous": ambiguous_flag,
                "own_product_only": own_product_only,
                "product_surface": " / ".join(surfaces),
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
            }
        )
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = [
        "query_id",
        "query",
        "true_intent",
        "mentions_competitor",
        "out_of_category",
        "ambiguous",
        "own_product_only",
        "product_surface",
        "impressions",
        "clicks",
        "conversions",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=800, help="生成する行数（既定: 800）")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード（既定: 42）")
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path(__file__).parent / "vocab.yaml",
        help="語彙定義ファイルのパス",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "queries.csv",
        help="出力CSVのパス",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    vocab = load_vocab(args.vocab)
    rows = generate_rows(args.n, vocab, rng)
    write_csv(rows, args.out)
    print(f"generated {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
