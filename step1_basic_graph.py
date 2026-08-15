"""
Step 1: LangGraphの基本 - State / Node / Edge を理解する

最小構成のグラフとして、ユーザーの入力を受け取り、
Ollamaでローカル実行するLLMが応答するだけの「1ノードのグラフ」を作ります。

事前準備:
    ollama pull qwen2.5:7b-instruct
    (ollama serve が起動していること)
"""
from typing import Annotated, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

MODEL_NAME = "qwen2.5:7b-instruct"


# 1. State定義
#    グラフ全体で共有される「状態」の型。
#    Annotated[list, add_messages] にしておくと、ノードが返した
#    メッセージは「上書き」ではなく「リストへの追記」として扱われる。
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 2. LLMの準備(ローカルのOllamaサーバーに接続)
llm = ChatOllama(model=MODEL_NAME, temperature=0)


# 3. Node定義
#    Nodeは「Stateを受け取り、Stateへの更新分(差分)をdictで返す関数」
def chat_node(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# 4. Graph構築
#    START -> chat -> END という一本道のグラフ
graph_builder = StateGraph(State)
graph_builder.add_node("chat", chat_node)
graph_builder.add_edge(START, "chat")
graph_builder.add_edge("chat", END)

graph = graph_builder.compile()


if __name__ == "__main__":
    result = graph.invoke(
        {"messages": [("user", "LangGraphとは何か、一文で教えてください。")]}
    )
    print(result["messages"][-1].content)
