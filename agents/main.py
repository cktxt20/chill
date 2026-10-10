"""
main.py — Entry point cho hệ thống multi-agent.

Chạy: python main.py
"""

import sys
from langchain_core.messages import HumanMessage

from supervisor import build_graph


def print_separator():
    print("\n" + "=" * 60 + "\n")


def run_interactive():
    """Chế độ chat tương tác với multi-agent system."""
    print("=" * 60)
    print("🤖 Multi-Agent System (LangGraph)")
    print("   Gõ 'quit' hoặc 'exit' để thoát")
    print("   Gõ 'help' để xem hướng dẫn")
    print("=" * 60)

    graph = build_graph()

    while True:
        try:
            user_input = input("\n👤 Bạn: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Tạm biệt!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("👋 Tạm biệt!")
            break

        if user_input.lower() == "help":
            print_help()
            continue

        print_separator()
        print("⚙️  Đang xử lý...\n")

        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"recursion_limit": 10},
            )

            # In kết quả của các agents
            for msg in result["messages"]:
                name = getattr(msg, "name", None)
                if name and name != "supervisor":
                    icon = {"researcher": "🔍", "coder": "💻", "reviewer": "✅", "translator": "🌐", "designer": "🎨"}.get(name, "🤖")
                    print(f"{icon} **{name.upper()}**:")
                    print(msg.content)
                    print()

        except Exception as e:
            print(f"❌ Lỗi: {e}")
            print("Hãy kiểm tra API key trong file .env")

        print_separator()


def run_single(query: str):
    """Chạy một câu hỏi đơn lẻ (không tương tác)."""
    graph = build_graph()
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": 10},
    )

    for msg in result["messages"]:
        name = getattr(msg, "name", None)
        if name and name != "supervisor":
            print(f"\n[{name.upper()}]: {msg.content}")


def print_help():
    print("""
📖 Hướng dẫn sử dụng:
  - Hỏi về thông tin, kiến thức → Researcher sẽ trả lời
  - Yêu cầu viết / debug code   → Coder sẽ xử lý
  - Nhờ đánh giá, review        → Reviewer sẽ kiểm tra

Ví dụ:
  > LangGraph là gì?
  > Viết hàm Python tính fibonacci
  > Review đoạn code này: def add(a,b): return a+b
""")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Truyền query qua command line: python main.py "câu hỏi"
        run_single(" ".join(sys.argv[1:]))
    else:
        run_interactive()
