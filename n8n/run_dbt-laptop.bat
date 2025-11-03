@echo off
SET PROJECT_DIR="C:\Projects\DataScience\Portfolio\Turbine-Maintenance-ETL\Workspace\turbine_etl_dbt"
SET PYTHON_EXE="C:\Users\seewi\AppData\Local\Programs\Python\Python313\python.exe"
cd %PROJECT_DIR%
REM Change directory and execute Python script
dbt run --profiles-dir . >> %PROJECT_DIR%\logs\dbt_log.txt 2>&1
echo DBT process completed. Check logs\dbt_log.txt for details.