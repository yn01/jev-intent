"""Jevに投げる質問定義。

state は data/queries.csv の query 列のみから組み立てる。true_intent や
各種フラグ、product_surface、impressions 等の正解ラベル・実績値は
リークになるため state に絶対に含めないこと。
"""

from __future__ import annotations

from typesafe_sdk import Choice, Noul

BRAND = "ニコン"
COMPETITORS = ["キヤノン", "ソニー"]
CATEGORY = "デジタルカメラ・交換レンズ"


def build_state(query: str) -> dict:
    """queries.csv の1行(query列)からJevに渡すstateを作る。"""
    return {
        "brand": BRAND,
        "competitors": COMPETITORS,
        "category": CATEGORY,
        "query": query,
    }


QUESTIONS = {
    "intent": Choice(
        instructions=(
            "`query`は検索エンジンに入力された検索語句である。`category`分野における"
            "この検索語句のユーザー意図を、以下の4つの選択肢から1つ選べ。"
        ),
        criteria={
            "informational": (
                "製品や技術、使い方、比較の前提となる知識などについて情報を知りたい・"
                "調べたい意図（例:「〇〇とは」「使い方」「設定方法」）。"
            ),
            "comparison": (
                "複数の製品・ブランドを比較検討したい意図（例:「A B 違い」"
                "「A vs B」「おすすめ」「どっちがいい」）。"
            ),
            "transactional": (
                "購入・価格確認・在庫確認など、取引に直結する行動を取りたい意図"
                "（例:「最安値」「価格」「通販」「在庫」「購入」）。"
            ),
            "navigational": (
                "特定のブランド公式サイトやサポートページ、店舗など、"
                "特定の行き先に到達したい意図（例:「〇〇 公式」"
                "「〇〇 カスタマーサポート」「〇〇 ログイン」）。"
            ),
        },
    ),
    "mentions_own_brand": Noul(
        instructions=(
            "`query`に`brand`の社名そのもの（ニコン、Nikon、NIKON等の"
            "表記揺れを含む）が書かれているか。NIKKORなどの製品ブランド名や"
            "型番だけの場合は偽。"
        ),
    ),
    "mentions_own_product": Noul(
        instructions=(
            "`query`に`brand`の具体的な製品名・型番（カメラボディ、レンズ、"
            "略称や表記揺れを含む）が書かれているか。"
        ),
    ),
    "mentions_competitor_brand": Noul(
        instructions=(
            "`query`に`competitors`のいずれかの社名そのもの（表記揺れを含む）"
            "が書かれているか。"
        ),
    ),
    "mentions_competitor_product": Noul(
        instructions=(
            "`query`に`competitors`のいずれかの具体的な製品名・型番"
            "（略称や表記揺れを含む）が書かれているか。社名がなくても、"
            "型番から競合製品と分かれば真。"
        ),
    ),
    "out_of_category": Noul(
        instructions=(
            "`query`は`category`と無関係な話題か。`category`の製品の購入・比較・"
            "使い方・サポートに関する検索であれば偽。社名を含んでいても、株価、"
            "採用、企業情報、`category`以外の事業や製品（プリンター、イヤホン、"
            "半導体製造装置など）に関する検索であれば真。"
        ),
    ),
}
