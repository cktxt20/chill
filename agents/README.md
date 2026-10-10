# 🤖 Multi-Agent System với LangGraph

Hệ thống nhiều AI agents phối hợp với nhau, dễ dàng chuyển đổi giữa các model.

## Cấu trúc

```
agents/
├── .env                  # API keys (tạo từ .env.example)
├── .env.example          # Mẫu cấu hình
├── requirements.txt      # Thư viện cần cài
├── config.py             # Cấu hình model & agents
├── main.py               # Entry point (chat dòng lệnh)
├── app.py                # Giao diện web Streamlit
├── supervisor.py         # Supervisor + 5 agents (researcher, coder, reviewer, translator, designer)
└── start_chat.bat        # Chạy nhanh giao diện web trên Windows
```

## Cài đặt

```bash
# 1. Kích hoạt venv
D:\chill\.venv\Scripts\Activate.ps1

# 2. Cài thư viện (chỉ cần làm 1 lần)
pip install -r agents/requirements.txt

# 3. Tạo file .env
copy agents\.env.example agents\.env
# Điền API keys vào agents\.env
```

## Chạy

```bash
# Chat dòng lệnh
python agents/main.py

# Giao diện web (cách khác: double-click agents/start_chat.bat)
streamlit run agents/app.py
```

## Hỗ trợ Models

| Model | Provider | Biến môi trường |
|---|---|---|
| gemini-1.5-flash | Google | `GOOGLE_API_KEY` |
| gpt-4o | OpenAI | `OPENAI_API_KEY` |
| claude-3-5-sonnet | Anthropic | `ANTHROPIC_API_KEY` |
| llama3 | Ollama (local) | Không cần key |
| llama-3.1-70b | Groq | `GROQ_API_KEY` |
