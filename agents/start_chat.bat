@echo off
echo Dang khoi dong Multi-Agent Chat...
cd /d D:\chill\agents
call D:\chill\.venv\Scripts\activate.bat
set PYTHONIOENCODING=utf-8
streamlit run app.py
