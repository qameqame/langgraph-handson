"""
マルチエージェント・ハンズオンで使う共通ツール定義。

外部のクラウドAPIキーが不要なツールだけを用意しています
(LLM本体はOllamaでローカル実行する前提のため、ツールも極力ローカル/無料で完結させています)。

- web_search: DuckDuckGoでWeb検索(APIキー不要)
- calculator: 簡単な四則演算
"""
from langchain_core.tools import tool

try:
    from ddgs import DDGS
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "ddgs がインストールされていません。"
        "`pip install -r requirements.txt` を実行してください。"
    ) from e


@tool
def web_search(query: str) -> str:
    """インターネット検索を行い、上位の検索結果(タイトル・概要・URL)を返す。
    最新情報や事実確認が必要なときに使う。

    Args:
        query: 検索したいキーワードや質問文。
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as e:
        return f"検索中にエラーが発生しました: {e}"

    if not results:
        return "検索結果が見つかりませんでした。"

    lines = []
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        lines.append(f"- {title}: {body} ({href})")
    return "\n".join(lines)


@tool
def calculator(expression: str) -> str:
    """四則演算・括弧を含む数式文字列を計算する。例: '12 * (3 + 4)'

    Args:
        expression: 計算したい数式(数字と + - * / ( ) . 半角スペースのみ使用可)。
    """
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "エラー: 数式に使用できない文字が含まれています。"
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307 (文字種を制限した上でのローカル計算)
        return str(result)
    except Exception as e:
        return f"計算エラー: {e}"
