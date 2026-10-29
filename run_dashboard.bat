@echo off
echo =========================================================
echo Launching Multimodal Attendance & Engagement Agent
echo =========================================================
echo Opening Streamlit Web Dashboard at http://localhost:8501 ...
python -m streamlit run app/main.py --server.port 8501 --browser.serverAddress localhost
pause
