"""
Step 3: マルチエージェント連携(Supervisorパターン)

司令塔(Supervisor)エージェントが会話の状況を見て、
「researcher(調査担当)」または「writer(執筆担当)」に処理を委譲し、
両者が協力して最終回答を作り上げるグラフを構築します。

  START -> supervisor --(次はresearcher)--> researcher --> supervisor
                     \--(次はwriter)-----> writer -----> supervisor
                      \--(FINISH)---------------------------> END

ポイント:
- 各ノードは add_edge で固定の遷移先を書く代わりに、
  Command(goto=..., update=...) を返すことで「動的に次のノード」を
  指定できる(= 誰から誰に処理を渡すかをノード自身が決められる)。
- supervisorは with_structured_output を使い、LLMの出力を
  `{"next": "researcher"}` のような決まった形式に強制することで、
  「次に誰を呼ぶか」を安定してプログラムから扱えるようにしている。
- 遷移先をLLMに決めさせるグラフは放っておくと無限ループする
  (GraphRecursionError)。小さなモデルは「もう終わっていい」の判断が苦手なため、
  (a) 作業済みメンバーを state に記録してプロンプトに明示する
  (b) 全員が作業を終えたらLLMに聞かず終了する
  という2つのガードを入れて、ループを構造的に防いでいる。
- ただし(a)の記録は「1ターン分」の情報なので、Step4のようにcheckpointerで
  Stateを引き継ぐと前のターンの記録が残り、逆に何も作業せず終了してしまう。
  そのためReducerを自作(merge_done)して、新しいユーザー発言を受け取った
  タイミングで記録をリセットしている。

事前準備:
    ollama pull qwen2.5:7b-instruct
"""
from typing import Annotated, Literal, Optional, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from tools import web_search

MODEL_NAME = "qwen2.5:7b-instruct"

llm = ChatOllama(model=MODEL_NAME, temperature=0)


# --- State ---
def merge_done(current: list[str], update: Optional[list[str]]) -> list[str]:
    """`done` フィールド用のReducer。

    リストが来たら追記し、None が来たらリセットする。
    リセットが必要なのは、Step4のようにcheckpointerを付けたときに
    `done` が前のターンの値を持ち越してしまうため(単なる operator.add では
    「空にする」という更新が表現できない)。
    """
    if update is None:
        return []
    return current + update


class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: str  # supervisorが選んだ「次に呼ぶエージェント名」を記録(デバッグ用)
    # 作業を終えたメンバー名。各ノードは ["writer"] を返すだけで追記され、
    # None を返すとリセットされる(merge_done 参照)。
    done: Annotated[list[str], merge_done]


# --- Supervisor: 次にどのエージェントを呼ぶか決定するノード ---
MEMBERS = ["researcher", "writer"]

SUPERVISOR_PROMPT = (
    "あなたはチームの司令塔(Supervisor)です。\n"
    f"チームメンバー: {MEMBERS}\n"
    "- researcher: Web検索で事実情報を集める担当\n"
    "- writer: 集まった情報をもとに、読みやすい日本語の最終回答を執筆する担当\n\n"
    "すでに作業を終えたメンバー: {done}\n"
    "まだ作業していないメンバー: {remaining}\n\n"
    "会話履歴を見て、次に作業させるメンバーを1人選んでください。\n"
    "まだ調査が必要なら researcher、調査が済んでいて執筆が必要なら writer、\n"
    "ユーザーへの最終回答がすでに用意できていれば FINISH を選んでください。\n"
    "すでに作業を終えたメンバーを再度選んではいけません。"
)


class Router(TypedDict):
    """Supervisorの出力スキーマ(構造化出力)"""

    next: Literal["researcher", "writer", "FINISH"]


def supervisor_node(state: State) -> Command[Literal["researcher", "writer", "__end__"]]:
    messages = state["messages"]
    done = state.get("done", [])

    # 最後のメッセージがユーザー発言なら、新しいターンの始まり。
    # checkpointer(Step4)を使う場合、doneには前のターンの
    # ["researcher", "writer"] が残っているため、ここで捨てないと
    # 「全員作業済み」と誤判定して新しい質問に答えずに終了してしまう。
    new_turn = bool(messages) and messages[-1].type == "human"
    if new_turn:
        done = []

    remaining = [m for m in MEMBERS if m not in done]

    # ガード(b): 全員が作業を終えたらLLMに聞かずに終了する。
    # 小さなモデルはFINISHを選べずに同じメンバーを選び続けるため、
    # 終了条件はプログラム側で持っておく。
    if not remaining:
        return Command(goto=END, update={"next": END})

    system = SUPERVISOR_PROMPT.format(
        done=", ".join(done) or "なし",
        remaining=", ".join(remaining),
    )
    router = llm.with_structured_output(Router).invoke(
        [("system", system), *messages]
    )
    goto = router["next"]

    if goto == "FINISH":
        goto = END
    elif goto in done:
        # 作業済みメンバーを選び直してきた場合は、未作業のメンバーへ回す
        goto = remaining[0]

    # ガード(c): メンバー間の依存関係を守らせる。
    # writerはresearcherの調査結果を前提に執筆するので、researcherが先。
    if goto == "writer" and "researcher" not in done:
        goto = "researcher"
    # ガード(d): 最終回答を書くのはwriterなので、writerが未実行のまま
    # 終了させない。ここを許すと、Stateの最後のメッセージが調査メモや
    # ユーザー発言のままになり、呼び出し側が最終回答を取り出せない。
    if goto == END and "writer" not in done:
        goto = "writer"

    update = {"next": goto}
    if new_turn:
        update["done"] = None  # 前のターンの記録をStateからも消す
    return Command(goto=goto, update=update)


# --- Researcher: Web検索ツールを持つReActエージェント ---
researcher_agent = create_react_agent(
    llm,
    tools=[web_search],
    prompt=(
        "あなたは調査担当(researcher)です。web_searchツールを使って"
        "事実情報を調べ、要点を日本語の箇条書きで簡潔にまとめてください。"
        "推測ではなく検索結果に基づいて回答してください。"
    ),
)


def researcher_node(state: State) -> Command[Literal["supervisor"]]:
    # create_react_agentは {"messages": [...]} 形式のstateを期待するため、
    # 必要な部分だけを渡す
    result = researcher_agent.invoke({"messages": state["messages"]})
    last_message = result["messages"][-1]
    return Command(
        goto="supervisor",
        update={
            "messages": [("ai", f"[researcher の調査結果]\n{last_message.content}")],
            "done": ["researcher"],
        },
    )


# --- Writer: 収集済みの情報をもとに最終回答を執筆するノード ---
WRITER_PROMPT = (
    "あなたは執筆担当(writer)です。渡された会話とresearcherの調査結果を"
    "踏まえて、ユーザーの最新の質問に対する分かりやすい日本語の回答を"
    "作成してください。回答本文だけを出力し、前置きや見出しは付けないでください。"
)


def format_transcript(messages: list) -> str:
    """会話履歴を1つのテキストに畳み込む。

    researcher / writer の出力は ai メッセージとしてStateに積まれているため、
    これをそのままLLMに渡すと「自分の発言が書きかけの状態」に見えてしまい、
    モデルが回答ではなく“前の文の続き”を書き始める(ときには空文字を返す)。
    履歴をテキストとして human メッセージ1つに畳み込むことで、
    「これを読んで答える」という素直な形にする。
    """
    lines = []
    for m in messages:
        if not m.content.strip():
            continue
        speaker = "ユーザー" if m.type == "human" else "チーム"
        lines.append(f"{speaker}: {m.content}")
    return "\n\n".join(lines)


def writer_node(state: State) -> Command[Literal["supervisor"]]:
    transcript = format_transcript(state["messages"])
    response = llm.invoke(
        [
            ("system", WRITER_PROMPT),
            ("human", f"{transcript}\n\n---\n上記を踏まえて、回答を書いてください。"),
        ]
    )
    return Command(
        goto="supervisor",
        # 「回答案」だとsupervisorが「まだ未完成」と解釈して writer を
        # 選び直しやすいので、完成品であることが分かるラベルにしておく
        update={
            "messages": [("ai", f"[writer の最終回答]\n{response.content}")],
            "done": ["writer"],
        },
    )


# --- Graph構築 ---
graph_builder = StateGraph(State)
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("researcher", researcher_node)
graph_builder.add_node("writer", writer_node)

graph_builder.add_edge(START, "supervisor")
# researcher / writer から supervisor への遷移、および supervisor から
# 各メンバーやENDへの遷移は、すべて各ノードが返す Command(goto=...) で
# 動的に決まるため、ここで add_edge を追加する必要はない。

graph = graph_builder.compile()


if __name__ == "__main__":
    question = "LangGraphの主な特徴を調べて、初心者向けに3行で要約してください。"
    result = graph.invoke(
        {"messages": [("user", question)], "next": "", "done": []},
        {"recursion_limit": 15},
    )
    print("=== 最終回答 ===")
    print(result["messages"][-1].content)

    print("\n=== 全体のやり取り(デバッグ用) ===")
    for m in result["messages"]:
        print(f"[{m.type}] {m.content[:200]}")
