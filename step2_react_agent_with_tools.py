"""
Step 2: ツールを使うエージェント(ReActパターン)

LLMが「ツールを呼ぶべきか」を判断し、必要ならツールを実行して
結果をLLMに戻す、というループを「条件分岐エッジ」で実装します。
これがStep3のマルチエージェントの各ワーカーの土台になります。

事前準備:
    ollama pull qwen2.5:7b-instruct
    (ツール呼び出し=Function Callingに対応したモデルであること)
"""
from typing import Annotated, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import calculator, web_search

MODEL_NAME = "qwen2.5:7b-instruct"


class State(TypedDict):
    messages: Annotated[list, add_messages]


tools = [web_search, calculator]

llm = ChatOllama(model=MODEL_NAME, temperature=0)
llm_with_tools = llm.bind_tools(tools)  # ツールをLLMに「使えるように」登録する


def agent_node(state: State) -> State:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def route_after_agent(state: State) -> str:
    """直前のAIメッセージが tool_calls を含むかどうかで分岐先を決める。"""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


graph_builder = StateGraph(State)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges(
    "agent", route_after_agent, {"tools": "tools", END: END}
)
graph_builder.add_edge("tools", "agent")  # ツール実行後は再びagentに戻り、結果を踏まえて応答する

graph = graph_builder.compile()


if __name__ == "__main__":
    question = "LangGraphを開発している会社はどこか調べて、その会社名の文字数を教えて。"
    result = graph.invoke({"messages": [("user", question)]})
    for m in result["messages"]:
        print(f"[{m.type}] {m.content}")
