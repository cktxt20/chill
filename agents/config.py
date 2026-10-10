"""
config.py — Cấu hình model và agents cho hệ thống multi-agent.

Hỗ trợ: Google Gemini, OpenAI, Anthropic Claude, Groq, Ollama (local).
Thay đổi DEFAULT_MODEL trong .env hoặc gọi get_llm(provider="...") để chuyển model.
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# ── Model defaults ─────────────────────────────────────────────────────────────
DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "groq": "qwen/qwen3.8-27b",
    "ollama": "llama3.2",
}


def get_llm(provider: Optional[str] = None, model_name: Optional[str] = None):
    """
    Trả về LLM instance dựa theo provider.

    Args:
        provider: "gemini" | "openai" | "anthropic" | "groq" | "ollama"
                  Mặc định lấy từ biến môi trường DEFAULT_MODEL.
        model_name: Tên model cụ thể (tuỳ chọn). Mặc định dùng DEFAULT_MODELS.

    Returns:
        BaseChatModel instance.

    Raises:
        ValueError: Nếu provider không được hỗ trợ hoặc API key thiếu.
    """
    provider = (provider or os.getenv("DEFAULT_MODEL", "gemini")).lower()
    model = model_name or os.getenv("DEFAULT_MODEL_NAME") or DEFAULT_MODELS.get(provider)

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Thiếu GOOGLE_API_KEY trong .env")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
        )

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Thiếu OPENAI_API_KEY trong .env")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=api_key)

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Thiếu ANTHROPIC_API_KEY trong .env")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=api_key)

    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Thiếu GROQ_API_KEY trong .env")
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, api_key=api_key, max_tokens=800)

    elif provider == "ollama":

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url=base_url)

    else:
        raise ValueError(
            f"Provider '{provider}' không được hỗ trợ. "
            f"Chọn một trong: {list(DEFAULT_MODELS.keys())}"
        )



# ── Helper ─────────────────────────────────────────────────────────────────────
def extract_text(content) -> str:
    """
    Trích xuất text thuần từ response content.
    Xử lý cả dạng string và dạng list [{'type': 'text', 'text': '...'}].
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


# ── Agent roles ────────────────────────────────────────────────────────────────
AGENT_ROLES = {
    "researcher": {
        "description": "Tìm kiếm và tổng hợp thông tin, phân tích dữ liệu.",
        "provider": None,  # None = dùng DEFAULT_MODEL
    },
    "coder": {
        "description": "Viết, debug và tối ưu code Python.",
        "provider": None,
    },
    "reviewer": {
        "description": "Đánh giá chất lượng, tìm lỗi và đề xuất cải thiện.",
        "provider": None,
    },
    "translator": {
        "description": "Dịch thuật tài liệu sang các ngôn ngữ khác nhau.",
        "provider": None,
    },
    "designer": {
        "description": "Thiết kế giao diện, viết code HTML/CSS/Tailwind.",
        "provider": None,
    },
}
