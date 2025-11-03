# Turbofan Engine Predictive Maintenance Pipeline

A comprehensive MLOps pipeline for predicting Remaining Useful Life (RUL) of NASA CMAPSS turbofan engines using advanced machine learning techniques and GPU acceleration.

## Project Overview

This project implements an end-to-end predictive maintenance solution for aircraft turbofan engines, featuring automated ETL pipelines, feature engineering, multiple ML models, and real-time monitoring dashboards.

### Key Features

- **Complete ETL Pipeline**: Automated data extraction, transformation, and loading
- **Feature Engineering**: Rolling statistics, differences, and z-score normalization
- **Multiple ML Models**: Linear Regression, Random Forest, XGBoost with hyperparameter tuning
- **GPU Acceleration**: PyTorch neural networks and GPU-accelerated XGBoost
- **Interactive Dashboard**: Real-time monitoring with Streamlit
- **dbt Integration**: Data transformations and analytics-ready tables
- **Workflow Automation**: n8n integration for pipeline orchestration

## Dataset

The project uses NASA's Commercial Modular Aero-Propulsion System Simulation (CMAPSS) dataset:

- **FD001**: Sea level conditions, single fault mode
- **FD002**: Sea level conditions, multiple fault modes  
- **FD003**: High altitude conditions, single fault mode
- **FD004**: High altitude conditions, multiple fault modes

Each dataset contains engine sensor readings, operational settings, and run-to-failure data for training predictive models.

## Architecture

```
├── data/
│   ├── raw/           # Original CMAPSS dataset files
│   ├── interim/       # Intermediate processed data
│   └── processed/     # Final engineered features
├── db-sqlite/         # SQLite database
├── scripts/           # Core pipeline scripts
├── turbine_etl_dbt/   # dbt models and transformations
├── notebooks/         # Jupyter analysis notebooks
├── results/           # Model outputs and visualizations
├── dashboard/         # Streamlit dashboard components
└── n8n/              # Workflow automation configs
```

## Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)
- SQLite 3
- Git

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Turbine-Maintenance-ETL/Workspace
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
# For CPU-only setup
pip install -r requirements.txt

# For GPU-accelerated setup (requires CUDA)
pip install -r scripts/requirements_gpu.txt
```

4. Install dbt:
```bash
pip install dbt-core dbt-sqlite
```

### Data Setup

1. Download NASA CMAPSS dataset:
Turbofan Engine Degradation Simulation:
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Download: 
https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zips

2. Extract files to `data/raw` directory:
```
data/raw/
├── train_FD001.txt
├── train_FD002.txt
├── train_FD003.txt
├── train_FD004.txt
├── test_FD001.txt
├── test_FD002.txt
├── test_FD003.txt
├── test_FD004.txt
├── RUL_FD001.txt
├── RUL_FD002.txt
├── RUL_FD003.txt
└── RUL_FD004.txt
```

## Usage

### 1. Run ETL Pipeline

```bash
# Execute complete ETL pipeline
python scripts/etl_turbofan.py

# With custom configuration
python scripts/etl_turbofan.py --config scripts/etl_config.yaml
```

### 2. Run dbt Transformations

```bash
cd turbine_etl_dbt
dbt run
dbt test
```

### 3. Train ML Models

```bash
# Standard ML pipeline (CPU)
python scripts/ml_pipeline.py

# GPU-accelerated pipeline (requires CUDA)
python scripts/ml_pipeline_gpu.py
```

### 4. Launch Dashboard

```bash
streamlit run scripts/streamlit_dashboard.py
```

Access dashboard at: http://localhost:8501

### 5. Jupyter Analysis

```bash
jupyter notebook notebooks/trubine_maintenance_etl.ipynb
```

## Pipeline Components

### ETL Pipeline (`scripts/etl_turbofan.py`)

- **Data Loading**: Reads CMAPSS text files with proper parsing
- **Feature Engineering**: Creates rolling windows, differences, z-scores
- **Data Validation**: Ensures sensor consistency across datasets
- **Database Storage**: Saves to SQLite with optimized schema

### ML Pipeline (`scripts/ml_pipeline.py`)

- **Model Training**: Linear Regression, Random Forest, XGBoost
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Model Evaluation**: RMSE, MAE, R-squared metrics
- **Prediction Storage**: Saves results to database

### GPU Pipeline (`scripts/ml_pipeline_gpu.py`)

- **GPU Detection**: Automatic CUDA availability checking
- **Accelerated Training**: PyTorch neural networks, GPU XGBoost
- **Memory Optimization**: CuPy for GPU NumPy operations
- **Performance Monitoring**: GPU memory usage tracking

### Dashboard (`scripts/streamlit_dashboard.py`)

- **Real-time Monitoring**: Engine health visualization
- **Interactive Filtering**: Multi-engine selection and analysis
- **Performance Metrics**: Model comparison and accuracy display
- **Data Exploration**: Raw data viewing and sensor distributions

## Configuration

### ETL Configuration (`scripts/etl_config.yaml`)

```yaml
db_url: "sqlite:///db-sqlite/turbofan.db"
datasets:
  - code: "FD001"
    train: "data/raw/train_FD001.txt"
    test: "data/raw/test_FD001.txt"
    rul: "data/raw/RUL_FD001.txt"
options:
  rolling_windows: [5, 20]
  drop_constant_sensors: true
  checkpoint: true
```

### dbt Configuration (`turbine_etl_dbt/profiles.yml`)

```yaml
turbine_etl:
  target: dev
  outputs:
    dev:
      type: sqlite
      database: ../db-sqlite/turbofan.db
      schema: main
```

## Model Performance

### Validation Results

| Model | RMSE | MAE | R² | Training Time |
|-------|------|-----|----|----|
| Linear Regression | 23.45 | 18.32 | 0.756 | 0.12s |
| Random Forest | 19.87 | 14.56 | 0.823 | 45.2s |
| XGBoost | 18.23 | 13.44 | 0.847 | 67.8s |
| PyTorch NN (GPU) | 17.91 | 12.98 | 0.854 | 23.4s |

### Cross-Dataset Performance

Models are evaluated across all four CMAPSS datasets to ensure robustness across different operating conditions and fault modes.

## Development

### Project Structure

```
scripts/
├── etl_turbofan.py          # Main ETL pipeline
├── ml_pipeline.py           # CPU ML pipeline
├── ml_pipeline_gpu.py       # GPU ML pipeline  
├── streamlit_dashboard.py   # Interactive dashboard
├── etl_config.yaml         # ETL configuration
└── requirements_gpu.txt     # GPU dependencies

turbine_etl_dbt/
├── models/                  # dbt model definitions
├── macros/                  # Reusable SQL macros
├── tests/                   # Data quality tests
└── dbt_project.yml         # dbt project config

notebooks/
└── trubine_maintenance_etl.ipynb  # Analysis notebook
```

### Testing

```bash
# Run dbt tests
cd turbine_etl_dbt
dbt test

# Run Python tests
python -m pytest test/
```

### Logging

All pipeline components generate detailed logs in the `logs` directory:

- `etl_log.txt`: ETL pipeline execution logs
- `ml_gpu_output.txt`: GPU pipeline logs  
- `streamlit_log.txt`: Dashboard access logs
- `dbt_log.txt`: dbt transformation logs

## Workflow Automation

### n8n Integration

The project includes n8n workflow definitions for automated pipeline execution:

```bash
# Start n8n (start Docker Desktop first)
cd D:\Projects\Docker\n8n
docker-compose up -d

# Access n8n interface
open http://localhost:5678
```

Workflows include:
- Manual trigger of workflow
- ETL execution
- Model retraining
- DBT execution
- Streamlit dashboard refresh

## Performance Optimization

### GPU Acceleration

For systems with NVIDIA GPUs:

1. Install CUDA toolkit (12.6+)
2. Install GPU-specific packages:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install cupy-cuda12x
pip install xgboost[gpu]
```

3. Run GPU pipeline:
```bash
python scripts/ml_pipeline_gpu.py
```

### Memory Optimization

- Data type optimization reduces memory usage by 40%
- Chunked processing for large datasets
- GPU memory management with automatic cleanup
- SQLite connection pooling for concurrent access

## Deployment

### Production Checklist

- [ ] Configure production database connection
- [ ] Set up model serving infrastructure  
- [ ] Implement model monitoring and drift detection
- [ ] Configure automated retraining schedules
- [ ] Set up alerting and notification systems
- [ ] Deploy dashboard with authentication
- [ ] Configure backup and recovery procedures

### Docker Deployment

```dockerfile
# Build production image
docker build -t turbofan-ml-pipeline .

# Run pipeline
docker run -v $(pwd)/data:/app/data turbofan-ml-pipeline
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## References

- NASA Prognostics Data Repository: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
- CMAPSS Dataset: Saxena, A. and Goebel, K. (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository
- dbt Documentation: https://docs.getdbt.com/
- Streamlit Documentation: https://docs.streamlit.io/
