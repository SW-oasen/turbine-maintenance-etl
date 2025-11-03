@echo off
SET PROJECT_DIR="C:\Projects\DataScience\Portfolio\Turbine-Maintenance-ETL\Workspace"
SET PYTHON_EXE="C:\Users\seewi\AppData\Local\Programs\Python\Python313\python.exe"
cd %PROJECT_DIR%
REM Change directory and execute Python script
%PYTHON_EXE% %PROJECT_DIR%\scripts\etl_turbofan.py --config %PROJECT_DIR%\scripts\etl_config.yaml >> %PROJECT_DIR%\logs\etl_log.txt 2>&1
echo ETL process completed. Check logs\etl_log.txt for details.