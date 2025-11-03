@echo off
SET PROJECT_DIR="D:\Projects\DataScience\Portfolio\Turbine-Maintenance-ETL\Workspace"
SET PYTHON_EXE="C:\Users\aipc\AppData\Local\Programs\Python\Python313\python.exe"
d:
cd %PROJECT_DIR%
REM Change directory and execute Python script
%PYTHON_EXE% %PROJECT_DIR%\scripts\etl_turbofan.py --config %PROJECT_DIR%\scripts\etl_config.yaml