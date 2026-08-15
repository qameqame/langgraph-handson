"""
Step 4: 会話の記憶(Checkpointer)とCLIチャットアプリ

MemorySaver をコンパイル時に渡すと、thread_id ごとに会話履歴(State)が
自動的に保存・復元されます。これにより、同じセッション内で何度もやり取り
しながらマルチエージェント・チームと対話できるCLIアプリを作ります。

実行方法:
    python step4_memory_and_cli.py
"""
from langgraph.checkpoint.memory import MemorySaver

from step3_multi_agent_supervisor import graph_builder

# チェックポインタを登録してコンパイルし直す
checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)


def main() -> None:
    thread_id = "demo-session-1"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}

    print("=== マルチエージェント・チーム CLI ===")
    print("(終了するには 'exit' または 'quit' と入力してください)\n")

    while True:
        user_input = input("あなた: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("終了します。")
            break
        if not user_input:
            continue

        result = graph.invoke({"messages": [("user", user_input)]}, config)
        print(f"\nチーム: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
