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

事前準備:
    ollama pull qwen2.5:7b-instruct
"""
import operator
from typing import Annotated, Literal, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from tools import web_search

MODEL_NAME = "qwen2.5:7b-instruct"

llm = ChatOllama(model=MODEL_NAME, temperature=0)


# --- State ---
class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: str  # supervisorが選んだ「次に呼ぶエージェント名」を記録(デバッグ用)
    # 作業を終えたメンバー名。operator.add をreducerにすることで、
    # 各ノードが ["writer"] を返すだけでリストに追記される。
    done: Annotated[list[str], operator.add]


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
    done = state.get("done", [])
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
        [("system", system), *state["messages"]]
    )
    goto = router["next"]

    if goto == "FINISH":
        goto = END
    elif goto in done:
        # 作業済みメンバーを選び直してきた場合は、未作業のメンバーへ回す
        goto = remaining[0]

    return Command(goto=goto, update={"next": goto})


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
    "あなたは執筆担当(writer)です。これまでの会話(特にresearcherの調査結果)を"
    "踏まえて、ユーザーの質問に対する分かりやすい日本語の回答を作成してください。"
)


def writer_node(state: State) -> Command[Literal["supervisor"]]:
    messages = [("system", WRITER_PROMPT), *state["messages"]]
    response = llm.invoke(messages)
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
