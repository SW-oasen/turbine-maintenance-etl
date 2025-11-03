# flows/pipeline.py
from prefect import flow, task
import subprocess, mlflow

@task(retries=2, retry_delay_seconds=60)
def run_etl():
    subprocess.check_call(["./venv/Scripts/python.exe","pipelines/run_etl.py","--config","configs/etl.yaml"])

@task
def run_dbt():
    subprocess.check_call(["dbt","run","--profiles-dir",".","--project-dir","./dbt_project"])

@task
def train_model():
    subprocess.check_call(["./venv/Scripts/python.exe","ml/train.py","--config","configs/train.yaml"])

@task
def score_batch():
    subprocess.check_call(["./venv/Scripts/python.exe","ml/score.py","--model","latest","--output","data/predictions/preds.parquet"])

@task
def refresh_powerbi():
    subprocess.check_call([
        "./venv/Scripts/python.exe","ops/powerbi_refresh.py",
        "--tenant-id","...", "--client-id","...", "--client-secret","...", 
        "--workspace-id","...", "--dataset-id","..."
    ])

@flow(name="turbine_maintenance_daily")
def daily_pipeline():
    run_etl()
    run_dbt()
    train_model()
    score_batch()
    refresh_powerbi()

if __name__ == "__main__":
    daily_pipeline()
