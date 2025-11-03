"""
GPU-Accelerated Turbofan Predictive Maintenance ML Pipeline
==========================================================

This module implements a complete machine learning pipeline for turbofan engine
Remaining Useful Life (RUL) prediction using GPU-accelerated algorithms for 
improved performance on large datasets.

GPU Acceleration Features:
- PyTorch for neural networks and tensor operations
- XGBoost with GPU support (CUDA)
- CuPy for GPU-accelerated NumPy operations
- Automatic GPU detection with CPU fallback

Hardware Requirements:
- NVIDIA GPU with CUDA support (RTX 4070 Ti SUPER detected)
- CUDA 12.6+ and compatible drivers

Author: Data Science Portfolio  
Date: October 2025
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import time
warnings.filterwarnings('ignore')

# Standard ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# GPU-Accelerated Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sqlalchemy import create_engine, text
import yaml


# Try to import GPU libraries with fallbacks
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy available for GPU NumPy operations")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available, using standard NumPy")

try:
    from numba import cuda
    NUMBA_CUDA_AVAILABLE = cuda.is_available()
    print(f"Numba CUDA available: {NUMBA_CUDA_AVAILABLE}")
except ImportError:
    NUMBA_CUDA_AVAILABLE = False
    print("Numba CUDA not available")

# Database and Configuration
from sqlalchemy import create_engine
import yaml

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class GPUAcceleratedTurbofanMLPipeline:
    """
    GPU-Accelerated Machine Learning Pipeline for Turbofan RUL Prediction
    
    This class handles:
    1. GPU device detection and configuration
    2. Data loading and GPU memory optimization
    3. GPU-accelerated model training (XGBoost, PyTorch)
    4. Performance monitoring and memory usage
    5. Comprehensive evaluation with GPU acceleration
    """
    
    def __init__(self, config_path="scripts/etl_config.yaml"):
        """Initialize the GPU-accelerated ML pipeline."""
        self.config_path = config_path
        self.load_config()
        self.setup_gpu_environment()
        
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_names = None
        
        # Create results directory
        self.results_dir = Path("results/ml_models")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print("GPU-Accelerated Turbofan ML Pipeline Initialized")
        print(f"Database: {self.db_url}")
        print(f"Results directory: {self.results_dir}")
        
    def setup_gpu_environment(self):
        """Setup GPU environment and check availability."""
        print("\n" + "="*60)
        print("GPU ENVIRONMENT SETUP")
        print("="*60)
        
        # PyTorch CUDA setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"PyTorch CUDA: Available")
            print(f"GPU Device: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
        else:
            print("PyTorch CUDA: Not Available")
            print("Using CPU for PyTorch operations")
        
        # XGBoost GPU check
        try:
            # Test XGBoost GPU capability
            test_data = xgb.DMatrix([[1, 2], [3, 4]], label=[1, 0])
            test_params = {'objective': 'binary:logistic', 'tree_method': 'hist', 'device': 'cuda'}
            xgb.train(test_params, test_data, num_boost_round=1)
            self.xgb_gpu_available = True
            print(f"XGBoost GPU: Available")
            
        except Exception as e:
            self.xgb_gpu_available = False
            print(f"XGBoost GPU: Not Available ({e})")
            print("Will use CPU for XGBoost")
        
        # CuPy check
        if CUPY_AVAILABLE:
            try:
                # Test CuPy operations
                x = cp.array([1, 2, 3])
                y = cp.sum(x)
                print(f"CuPy GPU: Available (test result: {float(y)})")
                self.cupy_available = True
            except Exception as e:
                print(f"CuPy GPU: Error ({e})")
                self.cupy_available = False
        else:
            self.cupy_available = False
            
        print(f"\nActive Device: {self.device}")
        print("="*60)
    
    def load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            print("Using default configuration")
            self.config = {}
        
        self.db_url = self.config.get('db_url', 'sqlite:///db-sqlite/turbofan.db')
        self.datasets = self.config.get('datasets', [])
    
    def gpu_memory_info(self):
        """Display current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        
    # =========================================================================
    # SECTION 1: GPU-OPTIMIZED DATA LOADING
    # =========================================================================
    
    def load_features_from_db(self):
        """Load engineered features with GPU memory optimization."""
        print("\n" + "="*60)
        print("SECTION 1: GPU-OPTIMIZED DATA LOADING")
        print("="*60)
        
        try:
            engine = create_engine(self.db_url)
            
            # Check for dbt table first
            query_features = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='fct_cycles_features'
            """
            
            with engine.connect() as conn:
                table_exists = pd.read_sql(text(query_features), conn)

            if len(table_exists) > 0:
                print("Loading from dbt table: fct_cycles_features")
                query = "SELECT * FROM fct_cycles_features"
            else:
                print("Using fallback table: cycles_features")
                query = "SELECT * FROM cycles_features"
            
            # Load data in chunks if large dataset
            print("Loading data with memory optimization...")
            df_features = pd.read_sql(text(query), engine)
            engine.dispose()
            
            # Optimize memory usage
            print("Optimizing memory usage...")
            start_memory = df_features.memory_usage(deep=True).sum() / 1024**2
            
            # Convert to optimal dtypes
            for col in df_features.columns:
                if df_features[col].dtype == 'float64':
                    df_features[col] = pd.to_numeric(df_features[col], downcast='float')
                elif df_features[col].dtype == 'int64':
                    df_features[col] = pd.to_numeric(df_features[col], downcast='integer')
            
            end_memory = df_features.memory_usage(deep=True).sum() / 1024**2
            memory_reduction = (start_memory - end_memory) / start_memory * 100
            
            print(f"Memory optimization: {memory_reduction:.1f}% reduction")
            print(f"Dataset size: {len(df_features):,} records")
            print(f"Memory usage: {end_memory:.1f} MB")
            
            # Store feature names
            metadata_cols = ['unit_nr', 'time_cycles', 'dataset', 'rul']
            self.feature_names = [col for col in df_features.columns 
                                if col not in metadata_cols]
            
            print(f"Features available: {len(self.feature_names)}")
            
            # Display GPU memory if available
            self.gpu_memory_info()
            
            return df_features
            
        except Exception as e:
            print(f"Error loading features: {e}")
            raise
    
    def load_raw_test_data(self):
        """Load test data with GPU-optimized processing."""
        print("\nLoading Test Data with GPU Optimization")
        print("-" * 50)
        
        test_data = {}
        
        # Default dataset configuration if config not available
        if not self.datasets:
            default_datasets = [
                {'code': 'FD001', 'test': 'data/raw/test_FD001.txt', 'rul': 'data/raw/RUL_FD001.txt'},
                {'code': 'FD002', 'test': 'data/raw/test_FD002.txt', 'rul': 'data/raw/RUL_FD002.txt'},
                {'code': 'FD003', 'test': 'data/raw/test_FD003.txt', 'rul': 'data/raw/RUL_FD003.txt'},
                {'code': 'FD004', 'test': 'data/raw/test_FD004.txt', 'rul': 'data/raw/RUL_FD004.txt'},
            ]
            self.datasets = default_datasets
        
        for dataset_config in self.datasets:
            dataset_code = dataset_config['code']
            test_path = dataset_config.get('test')
            rul_path = dataset_config.get('rul')
            
            if not test_path or not rul_path or not os.path.exists(test_path) or not os.path.exists(rul_path):
                print(f"Skipping {dataset_code}: Missing files")
                continue
                
            print(f"Loading {dataset_code}...")
            
            # Read test data
            df_test = self.read_cmapss_txt(test_path)
            df_test['dataset'] = dataset_code
            
            # Load RUL ground truth
            rul_true = pd.read_csv(rul_path, header=None, names=['rul_true'])
            rul_true['unit_nr'] = range(1, len(rul_true) + 1)
            
            # Get last cycle for each unit
            df_test_last = df_test.groupby('unit_nr').last().reset_index()
            df_test_last = df_test_last.merge(rul_true, on='unit_nr')
            
            test_data[dataset_code] = {
                'full_test': df_test,
                'last_cycles': df_test_last,
                'rul_true': rul_true
            }
            
            print(f"   {len(df_test)} cycles, {len(rul_true)} units")
        
        return test_data
    
    def read_cmapss_txt(self, path: str) -> pd.DataFrame:
        """Read CMAPSS dataset text file."""
        SENSOR_COUNT = 21
        COLS = ["unit_nr","time_cycles","setting1","setting2","setting3"] + [f"sensor{i}" for i in range(1,SENSOR_COUNT+1)]
        
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        df = df.iloc[:, :len(COLS)]
        df.columns = COLS
        df["unit_nr"] = df["unit_nr"].astype(int)
        df["time_cycles"] = df["time_cycles"].astype(int)
        for c in ["setting1","setting2","setting3"] + [f"sensor{i}" for i in range(1,SENSOR_COUNT+1)]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        
        return df
    
    # =========================================================================
    # SECTION 2: GPU-ACCELERATED MODEL TRAINING
    # =========================================================================
    
    def prepare_training_data_gpu(self, df_features, test_size=0.2, random_state=42):
        """Prepare training data with GPU acceleration where possible."""
        print("\n" + "="*60)
        print(" SECTION 2: GPU-ACCELERATED DATA PREPARATION")
        print("="*60)
        
        # Prepare features and target
        X = df_features[self.feature_names].copy()
        y = df_features['rul'].copy()
        
        print(f" Feature matrix: {X.shape}")
        print(f" Target vector: {y.shape}")
        
        # GPU-accelerated missing value handling if CuPy available
        if self.cupy_available:
            print(" Using GPU for missing value computation...")
            
            # Convert to CuPy arrays for GPU processing
            X_gpu = cp.asarray(X.values)
            
            # Calculate median on GPU
            median_values = cp.nanmedian(X_gpu, axis=0)
            
            # Fill missing values
            for i in range(X_gpu.shape[1]):
                mask = cp.isnan(X_gpu[:, i])
                X_gpu[mask, i] = median_values[i]
            
            # Convert back to pandas
            X = pd.DataFrame(cp.asnumpy(X_gpu), columns=self.feature_names, index=X.index)
            
            print("   GPU-accelerated missing value imputation completed")
        else:
            print(" Using CPU for missing value handling...")
            X = X.fillna(X.median())
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f" Training set: {X_train.shape[0]:,} samples")
        print(f" Validation set: {X_val.shape[0]:,} samples")
        
        # Display GPU memory usage
        self.gpu_memory_info()
        
        return X_train, X_val, y_train, y_val
    
    def train_gpu_accelerated_xgboost(self, X_train, X_val, y_train, y_val):
        """Train XGBoost with GPU acceleration."""
        print("\n Training GPU-Accelerated XGBoost")
        print("-" * 50)
        
        if not self.xgb_gpu_available:
            print("  GPU not available for XGBoost, using CPU version")
            return self.train_cpu_xgboost(X_train, X_val, y_train, y_val)
        
        # GPU-specific parameters
        base_params = {
            'objective': 'reg:squarederror',
            'device': 'cuda',  # Use GPU
            'tree_method': 'hist',  # GPU-compatible method
            'random_state': 42
        }
        
        # Hyperparameter grid optimized for GPU
        param_grid = {
            'n_estimators': [200, 500],
            'max_depth': [6, 10],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        print(f" GPU XGBoost hyperparameter tuning...")
        print(f"  Testing {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree'])} combinations")
        
        # Custom scoring with GPU
        xgb_regressor = xgb.XGBRegressor(**base_params)
        
        start_time = time.time()
        
        # Use GPU-optimized grid search
        grid_search = GridSearchCV(
            xgb_regressor, 
            param_grid,
            cv=3,
            scoring='neg_root_mean_squared_error',
            n_jobs=1,  # Important: use n_jobs=1 for GPU
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        best_model = grid_search.best_estimator_
        self.models['XGBoost_GPU'] = best_model
        
        # Evaluate
        y_pred = best_model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        print(f"GPU XGBoost Results:")
        print(f"     Training time: {training_time:.1f} seconds")
        print(f"    Best params: {grid_search.best_params_}")
        print(f"    RMSE: {rmse:.2f}")
        print(f"    MAE:  {mae:.2f}")
        print(f"    R²:   {r2:.3f}")
        
        self.results['XGBoost_GPU'] = {'RMSE': rmse, 'MAE': mae, 'R²': r2, 'training_time': training_time}
        
        # Display GPU memory after training
        self.gpu_memory_info()
        
        return best_model
    
    def train_cpu_xgboost(self, X_train, X_val, y_train, y_val):
        """Fallback CPU XGBoost training."""
        print(" Training CPU XGBoost (GPU not available)")
        
        base_params = {
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1
        }
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 10],
            'learning_rate': [0.1, 0.2]
        }
        
        xgb_regressor = xgb.XGBRegressor(**base_params)
        
        start_time = time.time()
        grid_search = GridSearchCV(
            xgb_regressor, param_grid,
            cv=3, scoring='neg_root_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        best_model = grid_search.best_estimator_
        self.models['XGBoost_CPU'] = best_model
        
        # Evaluate
        y_pred = best_model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        print(f"     Training time: {training_time:.1f} seconds")
        print(f"    RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")
        
        self.results['XGBoost_CPU'] = {'RMSE': rmse, 'MAE': mae, 'R²': r2, 'training_time': training_time}
        
        return best_model
        
    def train_pytorch_neural_network(self, X_train, X_val, y_train, y_val):
        """Train a PyTorch neural network on GPU."""
        print("\n Training PyTorch Neural Network")
        print("-" * 40)
        
        # Create datasets
        class RULDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.FloatTensor(X.values)
                self.y = torch.FloatTensor(y.values)
                
            def __len__(self):
                return len(self.X)
                
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]
        
        # Create data loaders
        train_dataset = RULDataset(X_train, y_train)
        val_dataset = RULDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        
        # Define neural network architecture
        class RULPredictor(nn.Module):
            def __init__(self, input_size):
                super(RULPredictor, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.3),
                    
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    nn.Linear(64, 1)
                )
                
            def forward(self, x):
                return self.network(x).squeeze()
        
        # Initialize model
        model = RULPredictor(len(self.feature_names)).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        print(f" Training on {self.device}")
        print(f" Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training loop
        epochs = 100
        best_val_loss = float('inf')
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    predictions = model(batch_X)
                    loss = criterion(predictions, batch_y)
                    val_loss += loss.item()
                    
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.results_dir / 'best_pytorch_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 20 == 0 or patience_counter >= 15:
                print(f"   Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if patience_counter >= 15:
                print(f"   Early stopping at epoch {epoch}")
                break
        
        training_time = time.time() - start_time
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(self.results_dir / 'best_pytorch_model.pth'))
        
        # Calculate final metrics
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        
        rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
        mae = mean_absolute_error(val_targets, val_predictions)
        r2 = r2_score(val_targets, val_predictions)
        
        self.models['PyTorch_NN'] = model
        self.results['PyTorch_NN'] = {
            'RMSE': rmse, 'MAE': mae, 'R²': r2, 
            'training_time': training_time, 'epochs': epoch
        }
        
        print(f" PyTorch NN Results:")
        print(f"     Training time: {training_time:.1f} seconds")
        print(f"    Epochs trained: {epoch}")
        print(f"    RMSE: {rmse:.2f}")
        print(f"    MAE:  {mae:.2f}")
        print(f"    R²:   {r2:.3f}")
        
        # Display GPU memory after training
        self.gpu_memory_info()
        
        return model
    
    def train_all_models(self, X_train, X_val, y_train, y_val):
        """Train all available models (CPU and GPU)."""
        print("\n" + "="*60)
        print(" SECTION 2: TRAINING ALL AVAILABLE MODELS")
        print("="*60)

        # Create predictions table first
        self.create_predictions_table()
                
        # 1. Baseline Linear Regression (CPU)
        print("\n Training Linear Regression (Baseline)")
        scaler_lr = StandardScaler()
        X_train_scaled = scaler_lr.fit_transform(X_train)
        X_val_scaled = scaler_lr.transform(X_val)
        
        lr_model = LinearRegression()
        start_time = time.time()
        lr_model.fit(X_train_scaled, y_train)
        lr_time = time.time() - start_time
        
        y_pred_lr = lr_model.predict(X_val_scaled)
        lr_rmse = np.sqrt(mean_squared_error(y_val, y_pred_lr))
        lr_mae = mean_absolute_error(y_val, y_pred_lr)
        lr_r2 = r2_score(y_val, y_pred_lr)
        
        self.models['LinearRegression'] = lr_model
        self.scalers['LinearRegression'] = scaler_lr
        self.results['LinearRegression'] = {
            'RMSE': lr_rmse, 'MAE': lr_mae, 'R²': lr_r2, 'training_time': lr_time
        }
        
        print(f"     Training time: {lr_time:.3f} seconds")
        print(f"    RMSE: {lr_rmse:.2f}, MAE: {lr_mae:.2f}, R²: {lr_r2:.3f}")

        # Log validation predictions
        self.log_validation_predictions(X_val, y_val, 'LinearRegression', lr_model)

        # 2. GPU-Accelerated XGBoost
        xgb_model = self.train_gpu_accelerated_xgboost(X_train, X_val, y_train, y_val)
        if xgb_model:
            model_name = 'XGBoost_GPU' if self.xgb_gpu_available else 'XGBoost_CPU'
            self.log_validation_predictions(X_val, y_val, model_name, xgb_model)

        # 3. PyTorch Neural Network (GPU)
        if torch.cuda.is_available():
            pytorch_model = self.train_pytorch_neural_network(X_train, X_val, y_train, y_val)
            if pytorch_model:
                self.log_validation_predictions(X_val, y_val, 'PyTorch_NN', pytorch_model)

        # 4. Random Forest (CPU as comparison)
        print("\n Training Random Forest (CPU Comparison)")
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        
        start_time = time.time()
        rf_model.fit(X_train, y_train)
        rf_time = time.time() - start_time
        
        y_pred_rf = rf_model.predict(X_val)
        rf_rmse = np.sqrt(mean_squared_error(y_val, y_pred_rf))
        rf_mae = mean_absolute_error(y_val, y_pred_rf)
        rf_r2 = r2_score(y_val, y_pred_rf)
        
        self.models['RandomForest'] = rf_model
        self.results['RandomForest'] = {
            'RMSE': rf_rmse, 'MAE': rf_mae, 'R²': rf_r2, 'training_time': rf_time
        }
        
        print(f"     Training time: {rf_time:.1f} seconds")
        print(f"    RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}, R²: {rf_r2:.3f}")

        # Log validation predictions
        self.log_validation_predictions(X_val, y_val, 'RandomForest', rf_model)

        # Summary
        print("\n TRAINING SUMMARY")
        print("="*50)
        for model_name, metrics in self.results.items():
            print(f"{model_name:15s} - RMSE: {metrics['RMSE']:6.2f} - Time: {metrics['training_time']:6.1f}s")
    
    # ==========================================================================
    # SECTION 3: Create prediction features for test data
    # ==========================================================================

    # Add this method to the GPUAcceleratedTurbofanMLPipeline class

    def create_predictions_table(self):
        """Create ml_predictions table if it doesn't exist."""
        print("\n Creating ML Predictions Table")
        print("-" * 40)
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            unit_nr INTEGER NOT NULL,
            cycle INTEGER NOT NULL,
            dataset TEXT NOT NULL,
            model_name TEXT NOT NULL,
            predicted_rul REAL NOT NULL,
            actual_rul REAL,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_version TEXT,
            confidence_score REAL,
            features_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            engine = create_engine(self.db_url)
            with engine.connect() as conn:
                conn.execute(text(create_table_query))
                conn.commit()
            engine.dispose()
            print(" ml_predictions table ready")
        except Exception as e:
            print(f" Error creating predictions table: {e}")
            raise

    def save_predictions_to_db(self, predictions_data):
        """Save model predictions to the database."""
        print(f"\n Saving {len(predictions_data)} predictions to database...")
        
        try:
            engine = create_engine(self.db_url)
            
            # Convert to DataFrame
            df_predictions = pd.DataFrame(predictions_data)
            
            # Save to database
            df_predictions.to_sql('ml_predictions', engine, if_exists='append', index=False)
            engine.dispose()
            
            print(f" Saved {len(predictions_data)} predictions to ml_predictions table")
            
        except Exception as e:
            print(f" Error saving predictions: {e}")
            raise

    # Query predictions
    def query_recent_predictions(self, limit=10):
        """Query recent predictions from the database."""
        try:
            engine = create_engine(self.db_url)
            
            query = """
            SELECT unit_number, cycle, dataset, model_name, 
                predicted_rul, actual_rul, prediction_date,
                model_version, confidence_score
            FROM ml_predictions 
            ORDER BY prediction_date DESC 
            LIMIT ?
            """
            
            df_predictions = pd.read_sql(text(query), engine, params=(limit,))
            engine.dispose()
            
            return df_predictions
            
        except Exception as e:
            print(f"Error querying predictions: {e}")
            return pd.DataFrame()

    # Get performance summary
    def get_model_performance_summary(self):
        """Get performance summary from stored predictions."""
        try:
            engine = create_engine(self.db_url)
            
            query = """
            SELECT model_name, dataset,
                COUNT(*) as prediction_count,
                AVG(ABS(predicted_rul - actual_rul)) as avg_absolute_error,
                AVG((predicted_rul - actual_rul) * (predicted_rul - actual_rul)) as mse,
                AVG(confidence_score) as avg_confidence
            FROM ml_predictions 
            WHERE actual_rul IS NOT NULL
            GROUP BY model_name, dataset
            ORDER BY avg_absolute_error
            """
            
            df_summary = pd.read_sql(text(query), engine)
            df_summary['rmse'] = np.sqrt(df_summary['mse'])
            engine.dispose()
            
            return df_summary
            
        except Exception as e:
            print(f"Error getting performance summary: {e}")
            return pd.DataFrame()


    # =========================================================================
    # SECTION 4: GPU-ACCELERATED EVALUATION
    # =========================================================================
    
    def apply_feature_engineering_to_test(self, df_test_full):
        """Apply the same feature engineering to test data as was used in training."""
        
        # Get the sensors used in training by extracting from feature names
        sensor_cols = []
        for fname in self.feature_names:
            if 'sensor' in fname:
                # Extract sensor number from feature names like 'mean5_sensor11', 'd_sensor2', etc.
                parts = fname.split('_')
                if len(parts) >= 2 and parts[1].startswith('sensor'):
                    sensor_col = parts[1]
                    if sensor_col not in sensor_cols:
                        sensor_cols.append(sensor_col)
        
        # Sort sensor columns to ensure consistent order
        sensor_cols = sorted(list(set(sensor_cols)))
        print(f"   Applying feature engineering using sensors: {sensor_cols}")
        
        # Check which sensors are available in test data
        available_sensors = [c for c in df_test_full.columns if c.startswith('sensor')]
        missing_sensors = [s for s in sensor_cols if s not in available_sensors]
        
        if missing_sensors:
            print(f"     Missing sensors in test data: {missing_sensors}")
            # Use only available sensors
            sensor_cols = [s for s in sensor_cols if s in available_sensors]
        
        if not sensor_cols:
            raise ValueError("No matching sensors found between training and test data")
        
        # Apply feature engineering
        df_test_sorted = df_test_full.sort_values(["unit_nr", "time_cycles"])
        
        # Rolling means (windows 5 and 20)
        features_list = []
        
        for window in [5, 20]:
            rolled = (df_test_sorted.groupby("unit_nr", group_keys=False)[sensor_cols]
                     .rolling(window, min_periods=1).mean()
                     .reset_index(level=0, drop=True))
            rolled.columns = [f"mean{window}_{c}" for c in sensor_cols]
            features_list.append(rolled)
        
        # Differences
        diffs = (df_test_sorted.groupby("unit_nr", group_keys=False)[sensor_cols]
                .diff().reset_index(level=0, drop=True))
        diffs.columns = [f"d_{c}" for c in sensor_cols]
        features_list.append(diffs)
        
        # Z-scores by unit
        def zscore_by_unit(group):
            return (group[sensor_cols] - group[sensor_cols].mean()) / group[sensor_cols].std(ddof=0)
        
        z_scores = (df_test_sorted.groupby("unit_nr", group_keys=False)
                   .apply(zscore_by_unit).reset_index(level=0, drop=True))
        z_scores.columns = [f"z_{c}" for c in sensor_cols]
        features_list.append(z_scores)
        
        # Combine all features
        df_features_test = pd.concat([
            df_test_sorted[["unit_nr", "time_cycles", "dataset"]].reset_index(drop=True),
            *features_list
        ], axis=1)
        
        return df_features_test
    
    def evaluate_models_on_test_gpu(self, test_data, df_features):
        """Evaluate models on test sets with GPU acceleration."""
        print("\n" + "="*60)
        print(" SECTION 3: GPU-ACCELERATED MODEL EVALUATION")
        print("="*60)
        
        test_results = {}
        
        for dataset_code, test_info in test_data.items():
            print(f"\n Evaluating on {dataset_code}")
            
            df_test_full = test_info['full_test']
            y_true = test_info['rul_true']['rul_true'].values
            
            # Apply feature engineering to test data
            print("    Applying feature engineering to test data...")
            df_test_features = self.apply_feature_engineering_to_test(df_test_full)
            
            # Get last cycle for each unit with engineered features
            df_test_last = df_test_features.groupby('unit_nr').last().reset_index()
            
            # Prepare test features - only use features that exist in both training and test
            available_features = [f for f in self.feature_names if f in df_test_last.columns]
            missing_features = [f for f in self.feature_names if f not in df_test_last.columns]
            
            if missing_features:
                print(f"     Missing features in test data: {len(missing_features)} features")
                print(f"    Using {len(available_features)} available features")
            
            X_test = df_test_last[available_features].fillna(df_test_last[available_features].median())
            
            dataset_results = {}
            
            for model_name, model in self.models.items():
                start_time = time.time()
                
                try:
                    if model_name == 'LinearRegression':
                        scaler = self.scalers[model_name]
                        X_test_scaled = scaler.transform(X_test)
                        y_pred = model.predict(X_test_scaled)
                        
                    elif model_name == 'PyTorch_NN':
                        # GPU inference for PyTorch
                        # Check if model input size matches test features
                        expected_features = len(self.feature_names)
                        actual_features = X_test.shape[1]
                        
                        if expected_features != actual_features:
                            print(f"     PyTorch model expects {expected_features} features, got {actual_features}")
                            print(f"    Skipping PyTorch model for {dataset_code}")
                            continue
                            
                        model.eval()
                        with torch.no_grad():
                            X_test_tensor = torch.FloatTensor(X_test.values).to(self.device)
                            y_pred_tensor = model(X_test_tensor)
                            y_pred = y_pred_tensor.cpu().numpy()
                            
                    else:
                        # XGBoost and RandomForest - these can handle missing features better
                        y_pred = model.predict(X_test)
                
                except Exception as e:
                    print(f"    Error with {model_name}: {e}")
                    continue
                
                inference_time = time.time() - start_time
                
                # Ensure non-negative predictions
                y_pred = np.maximum(y_pred, 0)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                dataset_results[model_name] = {
                    'RMSE': rmse, 'MAE': mae, 'R²': r2,
                    'y_true': y_true, 'y_pred': y_pred,
                    'inference_time': inference_time
                }
                
                device_info = "GPU" if model_name in ['XGBoost_GPU', 'PyTorch_NN'] else "CPU"
                print(f"   {model_name:15s} ({device_info}) - RMSE: {rmse:6.2f}, Time: {inference_time:6.3f}s")
            
            test_results[dataset_code] = dataset_results
        
        self.test_results = test_results
        return test_results
    
    # Add this method to log validation predictions during training

    def log_validation_predictions(self, X_val, y_val, model_name, model):
        """Log validation predictions to database."""
        print(f"    Logging validation predictions for {model_name}...")
        
        try:
            # Make predictions
            if model_name == 'LinearRegression':
                scaler = self.scalers[model_name]
                X_val_scaled = scaler.transform(X_val)
                y_pred = model.predict(X_val_scaled)
            elif model_name == 'PyTorch_NN':
                model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val.values).to(self.device)
                    y_pred_tensor = model(X_val_tensor)
                    y_pred = y_pred_tensor.cpu().numpy()
            else:
                y_pred = model.predict(X_val)
            
            # Prepare validation predictions
            validation_predictions = []
            for i in range(len(y_val)):
                prediction_record = {
                    'unit_nr': 9999,  # Special unit number for validation data
                    'cycle': i + 1,
                    'dataset': 'VALIDATION',
                    'model_name': f"{model_name}_validation",
                    'predicted_rul': float(y_pred[i]),
                    'actual_rul': float(y_val.iloc[i]),
                    'model_version': "v1.0_training",
                    'confidence_score': None,
                    'features_used': ','.join(self.feature_names[:10])
                }
                validation_predictions.append(prediction_record)
            
            # Save validation predictions
            self.save_predictions_to_db(validation_predictions)
            
        except Exception as e:
            print(f"    Warning: Could not log validation predictions for {model_name}: {e}")
        
    # =========================================================================
    # SECTION 5: ENHANCED VISUALIZATION
    # =========================================================================
    
    def plot_performance_comparison(self):
        """Plot performance comparison including GPU vs CPU."""
        print("\n Creating Performance Comparison Plots")
        
        # Training performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Validation metrics comparison
        models = list(self.results.keys())
        rmse_scores = [self.results[m]['RMSE'] for m in models]
        training_times = [self.results[m]['training_time'] for m in models]
        
        # RMSE comparison
        colors = ['red' if 'GPU' in m or 'PyTorch' in m else 'blue' for m in models]
        axes[0, 0].bar(models, rmse_scores, color=colors)
        axes[0, 0].set_title('Validation RMSE Comparison')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Training time comparison
        axes[0, 1].bar(models, training_times, color=colors)
        axes[0, 1].set_title('Training Time Comparison')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Test performance across datasets
        if hasattr(self, 'test_results'):
            test_rmse_data = []
            for dataset_code, dataset_results in self.test_results.items():
                for model_name, results in dataset_results.items():
                    test_rmse_data.append({
                        'Dataset': dataset_code,
                        'Model': model_name,
                        'RMSE': results['RMSE'],
                        'Device': 'GPU' if 'GPU' in model_name or 'PyTorch' in model_name else 'CPU'
                    })
            
            df_test_perf = pd.DataFrame(test_rmse_data)
            
            # Test RMSE by dataset
            sns.barplot(data=df_test_perf, x='Dataset', y='RMSE', hue='Model', ax=axes[1, 0])
            axes[1, 0].set_title('Test RMSE by Dataset')
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # GPU vs CPU performance
            gpu_cpu_comparison = df_test_perf.groupby(['Device', 'Model'])['RMSE'].mean().reset_index()
            sns.barplot(data=gpu_cpu_comparison, x='Model', y='RMSE', hue='Device', ax=axes[1, 1])
            axes[1, 1].set_title('Average Test RMSE: GPU vs CPU')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'gpu_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Performance summary
        print("\n PERFORMANCE SUMMARY")
        print("="*60)
        
        best_gpu_model = min([m for m in models if 'GPU' in m or 'PyTorch' in m], 
                            key=lambda x: self.results[x]['RMSE'], default=None)
        best_cpu_model = min([m for m in models if 'GPU' not in m and 'PyTorch' not in m], 
                            key=lambda x: self.results[x]['RMSE'])
        
        if best_gpu_model:
            gpu_rmse = self.results[best_gpu_model]['RMSE']
            cpu_rmse = self.results[best_cpu_model]['RMSE']
            gpu_time = self.results[best_gpu_model]['training_time']
            cpu_time = self.results[best_cpu_model]['training_time']
            
            print(f" Best GPU Model: {best_gpu_model}")
            print(f"   RMSE: {gpu_rmse:.2f}, Training Time: {gpu_time:.1f}s")
            print(f" Best CPU Model: {best_cpu_model}")
            print(f"   RMSE: {cpu_rmse:.2f}, Training Time: {cpu_time:.1f}s")
            
            rmse_improvement = ((cpu_rmse - gpu_rmse) / cpu_rmse) * 100
            time_ratio = cpu_time / gpu_time
            
            print(f"\n GPU Performance:")
            print(f"   RMSE improvement: {rmse_improvement:+.1f}%")
            print(f"   Speed ratio: {time_ratio:.1f}x")
    
    # =========================================================================
    # SECTION 6: SAVE AND DEPLOY
    # =========================================================================
    
    def save_gpu_models(self):
        """Save GPU models with device information."""
        print("\n Saving GPU-Accelerated Models")
        
        for model_name, model in self.models.items():
            if model_name == 'PyTorch_NN':
                # Save PyTorch model
                model_path = self.results_dir / f"{model_name.lower()}_model.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_architecture': str(model),
                    'device': str(self.device),
                    'feature_names': self.feature_names
                }, model_path)
                print(f" Saved PyTorch model to {model_path}")
            else:
                # Save other models
                model_path = self.results_dir / f"{model_name.lower()}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f" Saved {model_name} to {model_path}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = self.results_dir / f"{scaler_name.lower()}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        # Save comprehensive results
        results_summary = {
            'validation_results': self.results,
            'test_results': getattr(self, 'test_results', {}),
            'feature_names': self.feature_names,
            'gpu_info': {
                'device': str(self.device),
                'gpu_available': torch.cuda.is_available(),
                'xgb_gpu_available': self.xgb_gpu_available,
                'cupy_available': self.cupy_available
            },
            'timestamp': pd.Timestamp.now()
        }
        
        results_path = self.results_dir / "gpu_model_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results_summary, f)
        
        print(f" Saved comprehensive results to {results_path}")
    
    # =========================================================================
    # MAIN PIPELINE EXECUTION
    # =========================================================================
    
    def run_gpu_pipeline(self):
        """Execute the complete GPU-accelerated ML pipeline."""
        print("\n" + "=" * 30)
        print("GPU-ACCELERATED TURBOFAN ML PIPELINE")
        print("=" * 30)
        
        pipeline_start_time = time.time()
        
        try:
            # Section 1: GPU-Optimized Data Loading
            df_features = self.load_features_from_db()
            test_data = self.load_raw_test_data()
            
            # Section 2: GPU-Accelerated Model Training
            X_train, X_val, y_train, y_val = self.prepare_training_data_gpu(df_features)
            self.train_all_models(X_train, X_val, y_train, y_val)
            
            # Section 3: GPU-Accelerated Model Evaluation
            self.evaluate_models_on_test_gpu(test_data, df_features)
            
            # Section 4: Enhanced Visualization
            self.plot_performance_comparison()
            
            # Section 5: Save GPU Models
            self.save_gpu_models()
            
            total_time = time.time() - pipeline_start_time
            
            print("\n" + "-" * 30)
            print("GPU-ACCELERATED PIPELINE COMPLETED!")
            print("-" * 30)
            print(f"\n  Total pipeline time: {total_time:.1f} seconds")
            print(f" Results saved to: {self.results_dir}")
            print(f" GPU acceleration utilized: {torch.cuda.is_available()}")
            
            # Final GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(" GPU memory cleared")
            
        except Exception as e:
            print(f"\n GPU Pipeline failed: {e}")
            raise


# =========================================================================
# ENTRY POINT FOR GPU PIPELINE
# =========================================================================

if __name__ == "__main__":
    """
    Main execution point for the GPU-accelerated ML pipeline.
    
    Usage:
        python scripts/ml_pipeline_gpu.py
        
    Prerequisites:
        1. NVIDIA GPU with CUDA support
        2. Run ETL pipeline: python scripts/etl_turbofan.py
        3. Run dbt models: cd turbine_etl_dbt && dbt run
        4. Install GPU libraries: pip install -r requirements_gpu.txt
    """
    
    print(" GPU-ACCELERATED TURBOFAN ML PIPELINE - STARTING")
    print("="*70)
    
    # Initialize and run GPU pipeline
    gpu_pipeline = GPUAcceleratedTurbofanMLPipeline()
    gpu_pipeline.run_gpu_pipeline()
    
    print("\n GPU Pipeline execution completed!")
    print(" Check results directory for GPU performance analysis")
    print(" Models ready for GPU-accelerated deployment!")