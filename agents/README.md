# 🤖 Multi-Agent System với LangGraph

Hệ thống nhiều AI agents phối hợp với nhau, dễ dàng chuyển đổi giữa các model.

## Cấu trúc

```
agents/
├── .env                  # API keys (tạo từ .env.example)
├── .env.example          # Mẫu cấu hình
├── config.py             # Cấu hình model & agents
├── main.py               # Entry point
├── supervisor.py         # Supervisor agent điều phối
└── agents/
    ├── researcher.py     # Agent nghiên cứu thông tin
    ├── coder.py          # Agent viết code
    └── reviewer.py       # Agent kiểm tra / phản hồi
```

## Cài đặt

```bash
# Kích hoạt venv
D:\chill\.venv\Scripts\Activate.ps1

# Tạo file .env
copy .env.example .env
# Điền API keys vào .env
```

## Chạy

```bash
python main.py
```

## Hỗ trợ Models

| Model | Provider | Biến môi trường |
|---|---|---|
| gemini-1.5-flash | Google | `GOOGLE_API_KEY` |
| gpt-4o | OpenAI | `OPENAI_API_KEY` |
| claude-3-5-sonnet | Anthropic | `ANTHROPIC_API_KEY` |
| llama3 | Ollama (local) | Không cần key |
| llama-3.1-70b | Groq | `GROQ_API_KEY` |
