"""
supervisor.py — Supervisor agent điều phối các agents khác.

Supervisor nhận yêu cầu từ user, phân tích và chuyển đến agent phù hợp.
Sau khi agent hoàn thành, supervisor quyết định tiếp tục hay kết thúc.
"""

from typing import Annotated
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END, MessagesState
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langgraph.prebuilt import create_react_agent

from config import get_llm, AGENT_ROLES, extract_text


# ── System prompts ─────────────────────────────────────────────────────────────
SUPERVISOR_SYSTEM = """Bạn là Supervisor — người điều phối một nhóm AI agents chuyên nghiệp.

Các agents:
- **researcher**: Trả lời câu hỏi, giải thích khái niệm, tìm kiếm thông tin, phân tích.
- **coder**: Viết, debug, giải thích code Python.
- **reviewer**: Đánh giá, review code hoặc văn bản.
- **translator**: Dịch thuật sang ngôn ngữ khác.
- **designer**: Thiết kế giao diện web HTML/CSS/JS.
- **FINISH**: Dùng SAU KHI một agent đã trả lời xong.

QUAN TRỌNG:
1. Nếu trong lịch sử CHƯA có tin nhắn từ agent nào (researcher/coder/...) → BẮT BUỘC chọn một agent, KHÔNG được FINISH.
2. Mọi câu hỏi về kiến thức, giải thích, thông tin → researcher.
3. Yêu cầu code Python → coder.
4. Dịch thuật → translator.
5. Review/đánh giá → reviewer.
6. Giao diện web → designer.
7. Chỉ FINISH khi đã thấy câu trả lời từ agent trong lịch sử.

Chỉ trả lời đúng một từ: researcher, coder, reviewer, translator, designer, hoặc FINISH."""

RESEARCHER_SYSTEM = """Bạn là Researcher — chuyên gia tìm kiếm và phân tích thông tin.
Hãy trả lời chi tiết, chính xác và có cấu trúc rõ ràng bằng tiếng Việt."""

CODER_SYSTEM = """Bạn là Coder — chuyên gia Python.
Bạn ĐƯỢC TRANG BỊ một trình thông dịch Python (Python REPL tool).
Nhiệm vụ của bạn:
1. Viết code theo yêu cầu.
2. BẮT BUỘC dùng tool 'python_repl' để chạy thử đoạn code đó xem có lỗi không.
3. Nếu có lỗi, tự sửa và chạy lại.
4. Cuối cùng, trả về đoạn code đã chạy thành công 100% kèm kết quả (stdout) mà bạn thấy.
Tuyệt đối KHÔNG ĐƯỢC "đoán" kết quả, phải tự chạy thật."""

REVIEWER_SYSTEM = """Bạn là Reviewer — chuyên gia đánh giá và cải thiện.
Hãy đưa ra nhận xét cụ thể, mang tính xây dựng bằng tiếng Việt."""

TRANSLATOR_SYSTEM = """Bạn là Translator — chuyên gia dịch thuật đa ngôn ngữ.
Nhiệm vụ của bạn là dịch văn bản một cách tự nhiên, chính xác, giữ nguyên ngữ cảnh và văn phong.
Chỉ trả về bản dịch và giải thích ngắn gọn nếu có từ lóng/đặc biệt."""

DESIGNER_SYSTEM = """Bạn là Designer — chuyên gia UI/UX và Frontend.
Nhiệm vụ của bạn là thiết kế giao diện web, viết code HTML/CSS/JS hoặc Tailwind đẹp mắt, hiện đại.
Tập trung vào thẩm mỹ, responsive và trải nghiệm người dùng."""


# ── Helpers ────────────────────────────────────────────────────────────────────
def _get_user_query(state: MessagesState) -> str:
    """Lấy câu hỏi gốc của user (message cuối cùng không có name)."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and not getattr(msg, "name", None):
            return msg.content
    return ""


# ── Node functions ─────────────────────────────────────────────────────────────
def supervisor_node(state: MessagesState) -> dict:
    """Supervisor quyết định agent nào sẽ xử lý tiếp theo."""
    llm = get_llm()
    messages = [SystemMessage(content=SUPERVISOR_SYSTEM)] + state["messages"]
    response = llm.invoke(messages)

    decision = extract_text(response.content).strip().lower()
    # Dọn dẹp nếu model trả về nhiều hơn 1 từ
    for role in ["researcher", "coder", "reviewer", "translator", "designer", "finish"]:
        if role in decision:
            decision = role
            break
    else:
        decision = "researcher"  # Default: giao cho researcher

    # Safety net: Nếu supervisor định FINISH nhưng chưa có agent nào trả lời
    # → bắt buộc giao cho researcher
    AGENT_NAMES = {"researcher", "coder", "reviewer", "translator", "designer"}
    agent_has_replied = any(
        getattr(msg, "name", None) in AGENT_NAMES
        for msg in state["messages"]
    )
    if decision == "finish" and not agent_has_replied:
        decision = "researcher"

    return {
        "messages": [
            HumanMessage(content=f"[Supervisor → {decision}]", name="supervisor")
        ]
    }


def researcher_node(state: MessagesState) -> dict:
    """Researcher agent xử lý yêu cầu tìm kiếm / phân tích (Được trang bị Internet)."""
    llm = get_llm(AGENT_ROLES["researcher"]["provider"])
    search_tool = DuckDuckGoSearchRun(
        name="web_search",
        description="Dùng công cụ này để tìm kiếm thông tin cập nhật trên Internet."
    )
    researcher_agent = create_react_agent(llm, tools=[search_tool], prompt=RESEARCHER_SYSTEM)
    # Chỉ truyền câu hỏi gốc của user, không truyền toàn bộ lịch sử graph
    query = _get_user_query(state)
    result = researcher_agent.invoke({"messages": [HumanMessage(content=query)]})
    text = extract_text(result["messages"][-1].content)
    return {"messages": [HumanMessage(content=text, name="researcher")]}


def coder_node(state: MessagesState) -> dict:
    """Coder agent xử lý yêu cầu viết / debug code (Được trang bị Python REPL)."""
    llm = get_llm(AGENT_ROLES["coder"]["provider"])
    repl_tool = PythonREPLTool(
        name="python_repl",
        description="Một trình thông dịch Python. Dùng để thực thi mã Python. Truyền mã nguồn vào để xem kết quả (stdout) hoặc lỗi. Nếu muốn thấy giá trị, phải dùng print()."
    )
    coder_agent = create_react_agent(llm, tools=[repl_tool], prompt=CODER_SYSTEM)
    query = _get_user_query(state)
    result = coder_agent.invoke({"messages": [HumanMessage(content=query)]})
    text = extract_text(result["messages"][-1].content)
    return {"messages": [HumanMessage(content=text, name="coder")]}


def reviewer_node(state: MessagesState) -> dict:
    """Reviewer agent đánh giá và cải thiện output."""
    llm = get_llm(AGENT_ROLES["reviewer"]["provider"])
    query = _get_user_query(state)
    messages = [SystemMessage(content=REVIEWER_SYSTEM), HumanMessage(content=query)]
    response = llm.invoke(messages)
    text = extract_text(response.content)
    return {"messages": [HumanMessage(content=text, name="reviewer")]}


def translator_node(state: MessagesState) -> dict:
    """Translator agent dịch thuật."""
    llm = get_llm(AGENT_ROLES["translator"]["provider"])
    query = _get_user_query(state)
    messages = [SystemMessage(content=TRANSLATOR_SYSTEM), HumanMessage(content=query)]
    response = llm.invoke(messages)
    text = extract_text(response.content)
    return {"messages": [HumanMessage(content=text, name="translator")]}


def designer_node(state: MessagesState) -> dict:
    """Designer agent thiết kế giao diện."""
    llm = get_llm(AGENT_ROLES["designer"]["provider"])
    query = _get_user_query(state)
    messages = [SystemMessage(content=DESIGNER_SYSTEM), HumanMessage(content=query)]
    response = llm.invoke(messages)
    text = extract_text(response.content)
    return {"messages": [HumanMessage(content=text, name="designer")]}


# ── Router ─────────────────────────────────────────────────────────────────────
def route_supervisor(state: MessagesState) -> str:
    """Đọc quyết định của supervisor và route đến agent phù hợp."""
    # Tìm message supervisor mới nhất
    for msg in reversed(state["messages"]):
        name = getattr(msg, "name", None)
        if name == "supervisor":
            content = msg.content.lower()
            if "researcher" in content: return "researcher"
            elif "coder" in content: return "coder"
            elif "reviewer" in content: return "reviewer"
            elif "translator" in content: return "translator"
            elif "designer" in content: return "designer"
            else: return "__end__"
    return "__end__"


# ── Build graph ────────────────────────────────────────────────────────────────
def build_graph(checkpointer=None) -> StateGraph:
    """Tạo và compile LangGraph cho hệ thống multi-agent."""
    builder = StateGraph(MessagesState)

    # Thêm các nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("translator", translator_node)
    builder.add_node("designer", designer_node)

    # Entry point
    builder.set_entry_point("supervisor")

    # Supervisor routing
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "coder": "coder",
            "reviewer": "reviewer",
            "translator": "translator",
            "designer": "designer",
            "__end__": END,
        },
    )

    # Sau khi agent xử lý xong → quay lại supervisor để FINISH
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("coder", "supervisor")
    builder.add_edge("reviewer", "supervisor")
    builder.add_edge("translator", "supervisor")
    builder.add_edge("designer", "supervisor")

    return builder.compile(checkpointer=checkpointer)
