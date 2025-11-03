@echo off
SET PROJECT_DIR="C:\Projects\DataScience\Portfolio\Turbine-Maintenance-ETL\Workspace"
SET PYTHON_EXE="C:\Users\seewi\AppData\Local\Programs\Python\Python313\python.exe"
cd %PROJECT_DIR%
REM Change directory and execute Python script
%PYTHON_EXE% %PROJECT_DIR%\scripts\ml_pipeline_gpu.py >> %PROJECT_DIR%\logs\ml_pipeline_gpu_log.txt 2>&1
echo ML process completed. Check logs\etl_log.txt for details.