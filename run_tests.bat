@echo off
echo =========================================================
echo Running Automated Unit ^& Integration Tests
echo =========================================================
python -m pytest tests/ -v
pause
