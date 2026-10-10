import os
import sqlite3
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from supervisor import build_graph

# ── Cấu hình trang ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Multi-Agent Chat", page_icon="🤖", layout="centered")
st.title("🤖 Multi-Agent System")
st.caption("Được điều phối bởi LangGraph | Bộ nhớ SQLite 🧠")

# ── Khởi tạo graph & DB (cache 1 lần duy nhất) ─────────────────────────────────
@st.cache_resource
def get_graph():
    conn = sqlite3.connect("memory.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    memory.setup()
    return build_graph(checkpointer=memory), conn

graph, _db_conn = get_graph()
thread_config = {"configurable": {"thread_id": "default_session"}, "recursion_limit": 10}

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏢 Điều hành Team AI")
    st.write("Mặc định **Supervisor** sẽ tự chọn agent phù hợp. Bạn cũng có thể chỉ định trực tiếp:")
    assign_mode = st.radio(
        "📍 Giao việc cho:",
        ["🤖 Tự động (Khuyên dùng)", "🔍 Researcher", "💻 Coder", "✅ Reviewer", "🌐 Translator", "🎨 Designer"]
    )
    st.divider()
    if st.button("🗑️ Xoá lịch sử chat"):
        # Đóng connection trước khi xoá file (tránh lỗi PermissionError trên Windows)
        try:
            _db_conn.close()
        except Exception:
            pass
        st.cache_resource.clear()
        for f in ["memory.db", "memory.db-shm", "memory.db-wal"]:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
        st.rerun()
    st.caption("Dữ liệu lưu trên máy của bạn.")

# ── Đọc & hiển thị lịch sử chat từ DB ──────────────────────────────────────────
ICON_MAP = {
    "researcher": "🔍",
    "coder": "💻",
    "reviewer": "✅",
    "translator": "🌐",
    "designer": "🎨",
}

current_state = graph.get_state(thread_config)
history = (
    current_state.values.get("messages", [])
    if current_state and current_state.values
    else []
)

for msg in history:
    name = getattr(msg, "name", None)
    # Tin nhắn user (HumanMessage không có name)
    if isinstance(msg, HumanMessage) and not name:
        with st.chat_message("user"):
            st.markdown(msg.content)
    # Tin nhắn agent (có name, không phải supervisor)
    elif name and name != "supervisor":
        icon = ICON_MAP.get(name, "🤖")
        with st.chat_message("assistant", avatar=icon):
            st.markdown(f"**[{icon} {name.upper()}]**\n\n{msg.content}")

# ── Ô nhập chat ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Nhập câu hỏi hoặc yêu cầu..."):

    # Thêm tiền tố nếu user ép chỉ định agent
    if assign_mode != "🤖 Tự động (Khuyên dùng)":
        target = assign_mode.split(" ")[1].upper()
        prompt = f"[Giao trực tiếp cho {target}]: {prompt}"

    # Hiện tin nhắn user ngay lập tức
    with st.chat_message("user"):
        st.markdown(prompt)

    # Chạy graph với spinner
    with st.spinner("🤖 Team đang xử lý..."):
        try:
            graph.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=thread_config,
            )
        except Exception as e:
            st.error(f"❌ Có lỗi xảy ra: {e}")

    # Rerun SAU KHI spinner kết thúc (không nằm trong with block)
    st.rerun()
