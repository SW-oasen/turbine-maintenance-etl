@echo off
SET PROJECT_DIR="D:\Projects\DataScience\Portfolio\Turbine-Maintenance-ETL\Workspace"
SET PYTHON_EXE="C:\Users\aipc\AppData\Local\Programs\Python\Python313\python.exe"
D:
cd %PROJECT_DIR%
REM Change directory and execute Python script
streamlit run scripts/streamlit_dashboard.py >> %PROJECT_DIR%\logs\streamlit_log.txt 2>&1
echo Streamlit dashboard launched. Check logs\streamlit_log.txt for details.