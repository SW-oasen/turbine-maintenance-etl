"""
Turbofan Predictive Maintenance ML Pipeline
==========================================

This module implements a complete machine learning pipeline for turbofan engine
Remaining Useful Life (RUL) prediction using multiple algorithms and comprehensive
evaluation across different operating conditions and fault modes.

Dataset Overview:
- FD001: Sea level, single fault mode
- FD002: Sea level, multiple fault modes  
- FD003: High altitude, single fault mode
- FD004: High altitude, multiple fault modes

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
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Database and Configuration
from sqlalchemy import create_engine
import yaml

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TurbofanMLPipeline:
    """
    Complete Machine Learning Pipeline for Turbofan RUL Prediction
    
    This class handles:
    1. Data loading from SQLite database and raw files
    2. Model training with multiple algorithms
    3. Comprehensive evaluation and visualization
    4. Model persistence and prediction storage
    """
    
    def __init__(self, config_path="scripts/etl_config.yaml"):
        """Initialize the ML pipeline with configuration."""
        self.config_path = config_path
        self.load_config()
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_names = None
        
        # Create results directory
        self.results_dir = Path("results/ml_models")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(" Turbofan ML Pipeline Initialized")
        print(f" Database: {self.db_url}")
        print(f" Results directory: {self.results_dir}")
    
    def load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.db_url = self.config.get('db_url', 'sqlite:///db-sqlite/turbofan.db')
        self.datasets = self.config.get('datasets', [])
        
    # =========================================================================
    # SECTION 1: DATA LOADING AND PREPARATION
    # =========================================================================
    
    def load_features_from_db(self):
        """
        Load engineered features from the database fct_cycles_features table.
        
        This function loads the features created by the ETL pipeline which include:
        - Rolling window statistics (mean5, mean20)
        - Sensor differences (d_sensor*)
        - Z-score normalized values (z_sensor*)
        - RUL targets
        
        Returns:
            pd.DataFrame: Complete feature dataset with all datasets
        """
        print("\n" + "="*60)
        print(" SECTION 1: LOADING ENGINEERED FEATURES FROM DATABASE")
        print("="*60)
        
        try:
            engine = create_engine(self.db_url)
            
            # Check if fct_cycles_features table exists (from dbt)
            query_features = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='fct_cycles_features'
            """
            
            with engine.connect() as conn:
                table_exists = pd.read_sql(query_features, conn)
            
            if len(table_exists) > 0:
                # Load from dbt-generated table
                print(" Loading features from dbt table: fct_cycles_features")
                query = "SELECT * FROM fct_cycles_features"
                
            else:
                # Fallback to ETL-generated table
                print("  fct_cycles_features not found, using cycles_features")
                query = "SELECT * FROM cycles_features"
            
            df_features = pd.read_sql(query, engine)
            engine.dispose()
            
            print(f" Loaded {len(df_features):,} records from database")
            print(f"  Datasets included: {sorted(df_features['dataset'].unique())}")
            print(f" Feature columns: {len(df_features.columns)}")
            print(f" RUL range: {df_features['rul'].min()} to {df_features['rul'].max()}")
            
            # Store feature names (excluding metadata columns)
            metadata_cols = ['unit_nr', 'time_cycles', 'dataset', 'rul']
            self.feature_names = [col for col in df_features.columns 
                                if col not in metadata_cols]
            
            print(f" Available features: {len(self.feature_names)}")
            print("   - Rolling means (mean5_*, mean20_*)")
            print("   - Sensor differences (d_*)")  
            print("   - Z-score normalized (z_*)")
            
            return df_features
            
        except Exception as e:
            print(f" Error loading features from database: {e}")
            print(" Make sure to run the ETL pipeline first:")
            print("   python scripts/etl_turbofan.py")
            print("   cd turbine_etl_dbt && dbt run")
            raise
    
    def load_raw_test_data(self):
        """
        Load raw test datasets and RUL ground truth for evaluation.
        
        This function loads the test_FD*.txt files and RUL_FD*.txt files
        to create proper test sets with ground truth RUL values.
        
        Returns:
            dict: Dictionary with test data for each dataset
        """
        print("\n" + "="*60)
        print(" SECTION 1B: LOADING RAW TEST DATA FOR EVALUATION")
        print("="*60)
        
        test_data = {}
        
        for dataset_config in self.datasets:
            dataset_code = dataset_config['code']
            test_path = dataset_config.get('test')
            rul_path = dataset_config.get('rul')
            
            if not test_path or not rul_path:
                print(f"  Skipping {dataset_code}: Missing test or RUL file")
                continue
                
            if not os.path.exists(test_path) or not os.path.exists(rul_path):
                print(f"  Skipping {dataset_code}: Files not found")
                continue
            
            print(f" Loading test data for {dataset_code}")
            
            # Load test data (same format as training data)
            from etl_turbofan import read_cmapss_txt  # Import from ETL module
            
            df_test = read_cmapss_txt(test_path)
            df_test['dataset'] = dataset_code
            
            # Load RUL ground truth
            rul_true = pd.read_csv(rul_path, header=None, names=['rul_true'])
            rul_true['unit_nr'] = range(1, len(rul_true) + 1)
            
            # Get last cycle for each unit in test set (for RUL prediction)
            df_test_last = df_test.groupby('unit_nr').last().reset_index()
            df_test_last = df_test_last.merge(rul_true, on='unit_nr')
            
            test_data[dataset_code] = {
                'full_test': df_test,
                'last_cycles': df_test_last,
                'rul_true': rul_true
            }
            
            print(f"    {len(df_test)} test cycles, {len(rul_true)} units")
        
        print(f"\n Test datasets loaded: {list(test_data.keys())}")
        return test_data
    
    # =========================================================================
    # SECTION 2: MODEL TRAINING AND HYPERPARAMETER TUNING
    # =========================================================================
    
    def prepare_training_data(self, df_features, test_size=0.2, random_state=42):
        """
        Prepare training and validation data with proper scaling.
        
        Args:
            df_features: DataFrame with engineered features
            test_size: Proportion of data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_val, y_train, y_val)
        """
        print("\n" + "="*60)
        print(" SECTION 2: PREPARING TRAINING DATA")
        print("="*60)
        
        # Prepare features and target
        X = df_features[self.feature_names].copy()
        y = df_features['rul'].copy()
        
        print(f" Feature matrix shape: {X.shape}")
        print(f" Target vector shape: {y.shape}")
        
        # Handle missing values
        print("\n🧹 Handling missing values...")
        missing_before = X.isnull().sum().sum()
        X = X.fillna(X.median())  # Fill with median values
        print(f"   Filled {missing_before} missing values with median")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        print(f"\n Training set: {X_train.shape[0]:,} samples")
        print(f" Validation set: {X_val.shape[0]:,} samples")
        print(f" RUL distribution in training:")
        print(f"   Mean: {y_train.mean():.1f}, Std: {y_train.std():.1f}")
        print(f"   Range: {y_train.min():.0f} - {y_train.max():.0f}")
        
        return X_train, X_val, y_train, y_val
    
    def train_models(self, X_train, X_val, y_train, y_val):
        """
        Train multiple ML models with hyperparameter tuning.
        
        Models included:
        1. Linear Regression (baseline)
        2. Random Forest Regressor
        3. XGBoost Regressor
        
        Args:
            X_train, X_val, y_train, y_val: Training and validation data
        """
        print("\n" + "="*60)
        print(" SECTION 2: TRAINING ML MODELS")
        print("="*60)
        
        # 1. LINEAR REGRESSION (Baseline Model)
        print("\n Training Linear Regression (Baseline)")
        print("-" * 40)
        
        # Scale features for linear regression
        scaler_lr = StandardScaler()
        X_train_scaled = scaler_lr.fit_transform(X_train)
        X_val_scaled = scaler_lr.transform(X_val)
        
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        
        self.models['LinearRegression'] = lr_model
        self.scalers['LinearRegression'] = scaler_lr
        
        # Evaluate
        y_pred_lr = lr_model.predict(X_val_scaled)
        lr_rmse = np.sqrt(mean_squared_error(y_val, y_pred_lr))
        lr_mae = mean_absolute_error(y_val, y_pred_lr)
        lr_r2 = r2_score(y_val, y_pred_lr)
        
        print(f"    RMSE: {lr_rmse:.2f}")
        print(f"    MAE:  {lr_mae:.2f}")
        print(f"    R²:   {lr_r2:.3f}")
        
        # 2. RANDOM FOREST REGRESSOR
        print("\n Training Random Forest Regressor")
        print("-" * 40)
        
        # Hyperparameter tuning for Random Forest
        rf_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
        
        rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

        # Add progress tracking
        total_rf_combinations = len(rf_param_grid['n_estimators']) * len(rf_param_grid['max_depth']) * len(rf_param_grid['min_samples_split']) * len(rf_param_grid['min_samples_leaf'])
        print(f"    Testing {total_rf_combinations} parameter combinations with 3-fold CV")
        print(f"     Estimated total fits: {total_rf_combinations * 3}")
        print("   Progress will be shown below...")

        rf_grid = GridSearchCV(
            rf_base, rf_param_grid, 
            cv=3, scoring='neg_root_mean_squared_error',
            n_jobs=-1, verbose=2
        )
        
        print("   🔍 Performing hyperparameter tuning...")
        import time
        start_time = time.time()
        rf_grid.fit(X_train, y_train)
        rf_time = time.time() - start_time
        
        rf_model = rf_grid.best_estimator_
        self.models['RandomForest'] = rf_model
        
        # Evaluate
        y_pred_rf = rf_model.predict(X_val)
        rf_rmse = np.sqrt(mean_squared_error(y_val, y_pred_rf))
        rf_mae = mean_absolute_error(y_val, y_pred_rf)
        rf_r2 = r2_score(y_val, y_pred_rf)
        
        print(f"     Training time: {rf_time:.1f} seconds")
        print(f"    Best params: {rf_grid.best_params_}")
        print(f"    RMSE: {rf_rmse:.2f}")
        print(f"    MAE:  {rf_mae:.2f}")
        print(f"    R²:   {rf_r2:.3f}")
        
        # 3. XGBOOST REGRESSOR
        print("\n Training XGBoost Regressor")
        print("-" * 40)
        
        # Hyperparameter tuning for XGBoost
        xgb_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 10],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        xgb_base = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
        # Add progress tracking
        total_xgb_combinations = len(xgb_param_grid['n_estimators']) * len(xgb_param_grid['max_depth']) * len(xgb_param_grid['learning_rate']) * len(xgb_param_grid['subsample']) * len(xgb_param_grid['colsample_bytree'])
        print(f"    Testing {total_xgb_combinations} parameter combinations with 3-fold CV")
        print(f"     Estimated total fits: {total_xgb_combinations * 3}")
        print("    Progress as below...")

        xgb_grid = GridSearchCV(
            xgb_base, xgb_param_grid,
            cv=3, scoring='neg_root_mean_squared_error',
            n_jobs=-1, verbose=2
        )
        
        print("    Performing hyperparameter tuning...")
        start_time = time.time()
        xgb_grid.fit(X_train, y_train)
        xgb_time = time.time() - start_time
        
        xgb_model = xgb_grid.best_estimator_
        self.models['XGBoost'] = xgb_model
        
        # Evaluate
        y_pred_xgb = xgb_model.predict(X_val)
        xgb_rmse = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
        xgb_mae = mean_absolute_error(y_val, y_pred_xgb)
        xgb_r2 = r2_score(y_val, y_pred_xgb)
        
        print(f"     Training time: {xgb_time:.1f} seconds")
        print(f"    Best params: {xgb_grid.best_params_}")
        print(f"    RMSE: {xgb_rmse:.2f}")
        print(f"    MAE:  {xgb_mae:.2f}")
        print(f"    R²:   {xgb_r2:.3f}")
        
        # Store results
        self.results = {
            'LinearRegression': {'RMSE': lr_rmse, 'MAE': lr_mae, 'R²': lr_r2},
            'RandomForest': {'RMSE': rf_rmse, 'MAE': rf_mae, 'R²': rf_r2},
            'XGBoost': {'RMSE': xgb_rmse, 'MAE': xgb_mae, 'R²': xgb_r2}
        }
        
        print("\n MODEL TRAINING COMPLETE")
        print("=" * 40)
    
    # =========================================================================
    # SECTION 3: MODEL EVALUATION ON TEST SETS
    # =========================================================================
    
    def evaluate_models_on_test(self, test_data, df_features):
        """
        Evaluate trained models on actual test sets with ground truth RUL.
        
        Args:
            test_data: Dictionary with test data for each dataset
            df_features: Training features for feature engineering on test data
        """
        print("\n" + "="*60)
        print(" SECTION 3: MODEL EVALUATION ON TEST SETS")
        print("="*60)
        
        test_results = {}
        
        for dataset_code, test_info in test_data.items():
            print(f"\n📊 Evaluating on {dataset_code}")
            print("-" * 30)
            
            # Get test data with features engineered similarly to training
            df_test_last = test_info['last_cycles']
            y_true = test_info['rul_true']['rul_true'].values # get first dataframe, then series, then numpy array
            
            # Feature engineering on test data, fill missing values with median
            X_test = df_test_last[self.feature_names].fillna(df_test_last[self.feature_names].median())  
            
            dataset_results = {}
            
            for model_name, model in self.models.items():
                if model_name == 'LinearRegression':
                    # Apply scaling for linear regression
                    scaler = self.scalers[model_name]
                    X_test_scaled = scaler.transform(X_test)
                    y_pred = model.predict(X_test_scaled)
                else:
                    y_pred = model.predict(X_test)
                
                # Ensure non-negative predictions
                y_pred = np.maximum(y_pred, 0)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                dataset_results[model_name] = {
                    'RMSE': rmse, 'MAE': mae, 'R²': r2,
                    'y_true': y_true, 'y_pred': y_pred
                }
                
                print(f"   {model_name:15s} - RMSE: {rmse:6.2f}, MAE: {mae:6.2f}, R²: {r2:6.3f}")
            
            test_results[dataset_code] = dataset_results
        
        self.test_results = test_results
        return test_results
    
    # =========================================================================
    # SECTION 4: VISUALIZATION AND INTERPRETATION
    # =========================================================================
    
    def plot_feature_importance(self):
        """
        Plot feature importance for tree-based models.
        """
        print("\n" + "="*60)
        print(" SECTION 4: FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Random Forest Feature Importance
        if 'RandomForest' in self.models:
            rf_model = self.models['RandomForest']
            rf_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            axes[0].barh(range(len(rf_importance)), rf_importance['importance'])
            axes[0].set_yticks(range(len(rf_importance)))
            axes[0].set_yticklabels(rf_importance['feature'])
            axes[0].set_title('Random Forest - Top 15 Feature Importance')
            axes[0].set_xlabel('Importance')
            axes[0].invert_yaxis()
        
        # XGBoost Feature Importance
        if 'XGBoost' in self.models:
            xgb_model = self.models['XGBoost']
            xgb_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            axes[1].barh(range(len(xgb_importance)), xgb_importance['importance'])
            axes[1].set_yticks(range(len(xgb_importance)))
            axes[1].set_yticklabels(xgb_importance['feature'])
            axes[1].set_title('XGBoost - Top 15 Feature Importance')
            axes[1].set_xlabel('Importance')
            axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(" Feature importance plots saved to results/ml_models/")
    
    def plot_residual_analysis(self):
        """
        Create residual plots for model diagnostics.
        """
        print("\n Creating Residual Analysis Plots")
        print("-" * 40)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        plot_idx = 0
        for dataset_code, dataset_results in self.test_results.items():
            for model_name, results in dataset_results.items():
                if plot_idx >= 6:  # Limit to 6 plots
                    break
                    
                y_true = results['y_true']
                y_pred = results['y_pred']
                residuals = y_true - y_pred
                
                # Residual vs Predicted plot
                axes[plot_idx].scatter(y_pred, residuals, alpha=0.6)
                axes[plot_idx].axhline(y=0, color='red', linestyle='--')
                axes[plot_idx].set_xlabel('Predicted RUL')
                axes[plot_idx].set_ylabel('Residuals')
                axes[plot_idx].set_title(f'{model_name} - {dataset_code}')
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_cross_dataset_performance(self):
        """
        Visualize model performance across different datasets.
        """
        print("\n Cross-Dataset Performance Analysis")
        print("-" * 40)
        
        # Prepare data for plotting
        performance_data = []
        for dataset_code, dataset_results in self.test_results.items():
            for model_name, results in dataset_results.items():
                performance_data.append({
                    'Dataset': dataset_code,
                    'Model': model_name,
                    'RMSE': results['RMSE'],
                    'MAE': results['MAE'],
                    'R²': results['R²']
                })
        
        df_perf = pd.DataFrame(performance_data)
        
        # Create subplots for each metric
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # RMSE comparison
        sns.barplot(data=df_perf, x='Dataset', y='RMSE', hue='Model', ax=axes[0])
        axes[0].set_title('RMSE Across Datasets')
        axes[0].set_ylabel('RMSE')
        
        # MAE comparison
        sns.barplot(data=df_perf, x='Dataset', y='MAE', hue='Model', ax=axes[1])
        axes[1].set_title('MAE Across Datasets')
        axes[1].set_ylabel('MAE')
        
        # R² comparison
        sns.barplot(data=df_perf, x='Dataset', y='R²', hue='Model', ax=axes[2])
        axes[2].set_title('R² Score Across Datasets')
        axes[2].set_ylabel('R² Score')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'cross_dataset_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary table
        print("\n Performance Summary Table:")
        print("=" * 60)
        pivot_rmse = df_perf.pivot(index='Dataset', columns='Model', values='RMSE')
        print("RMSE by Dataset and Model:")
        print(pivot_rmse.round(2))
        
        print(f"\n🏆 Best overall RMSE: {df_perf.loc[df_perf['RMSE'].idxmin(), 'Model']} "
              f"on {df_perf.loc[df_perf['RMSE'].idxmin(), 'Dataset']} "
              f"({df_perf['RMSE'].min():.2f})")
    
    # =========================================================================
    # SECTION 5: MODEL PERSISTENCE AND PREDICTION STORAGE
    # =========================================================================
    
    def save_models(self):
        """
        Save trained models and scalers to disk.
        """
        print("\n" + "="*60)
        print(" SECTION 5: SAVING MODELS AND RESULTS")
        print("="*60)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = self.results_dir / f"{model_name.lower()}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f" Saved {model_name} model to {model_path}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = self.results_dir / f"{scaler_name.lower()}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f" Saved {scaler_name} scaler to {scaler_path}")
        
        # Save results summary
        results_path = self.results_dir / "model_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump({
                'validation_results': self.results,
                'test_results': self.test_results,
                'feature_names': self.feature_names
            }, f)
        print(f" Saved results summary to {results_path}")
    
    def write_predictions_to_db(self):
        """
        Write model predictions back to the database via dbt-compatible tables.
        
        Creates prediction tables that can be used by dbt for further analysis.
        """
        print("\n Writing Predictions to Database")
        print("-" * 40)
        
        engine = create_engine(self.db_url)
        
        # Prepare predictions for all test datasets
        prediction_records = []
        
        for dataset_code, dataset_results in self.test_results.items():
            for model_name, results in dataset_results.items():
                y_true = results['y_true']
                y_pred = results['y_pred']
                
                for i, (true_val, pred_val) in enumerate(zip(y_true, y_pred)):
                    prediction_records.append({
                        'dataset': dataset_code,
                        'unit_nr': i + 1,  # Unit numbers start from 1
                        'model_name': model_name,
                        'rul_true': float(true_val),
                        'rul_predicted': float(pred_val),
                        'absolute_error': float(abs(true_val - pred_val)),
                        'squared_error': float((true_val - pred_val) ** 2),
                        'prediction_timestamp': pd.Timestamp.now()
                    })
        
        # Create DataFrame and write to database
        df_predictions = pd.DataFrame(prediction_records)
        
        with engine.begin() as conn:
            df_predictions.to_sql('ml_predictions', conn, if_exists='replace', index=False)
        
        engine.dispose()
        
        print(f" Wrote {len(df_predictions)} prediction records to ml_predictions table")
        print(" Run dbt models to analyze predictions:")
        print("   cd turbine_etl_dbt && dbt run")
    
    # =========================================================================
    # MAIN PIPELINE EXECUTION
    # =========================================================================
    
    def run_complete_pipeline(self):
        """
        Execute the complete ML pipeline from data loading to model deployment.
        """
        print("\n" + "=" * 30)
        print("TURBOFAN PREDICTIVE MAINTENANCE ML PIPELINE")
        print("=" * 30)
        
        try:
            # Section 1: Data Loading
            df_features = self.load_features_from_db()
            test_data = self.load_raw_test_data()
            
            # Section 2: Model Training
            X_train, X_val, y_train, y_val = self.prepare_training_data(df_features)
            self.train_models(X_train, X_val, y_train, y_val)
            
            # Section 3: Model Evaluation
            self.evaluate_models_on_test(test_data, df_features)
            
            # Section 4: Visualization and Interpretation
            self.plot_feature_importance()
            self.plot_residual_analysis()
            self.plot_cross_dataset_performance()
            
            # Section 5: Model Persistence
            self.save_models()
            self.write_predictions_to_db()
            
            print("\n" + "-" * 30)
            print("ML PIPELINE COMPLETED SUCCESSFULLY!")
            print("-" * 30)
            print(f"\n Results saved to: {self.results_dir}")
            print(" Visualizations generated and saved")
            print(" Models and scalers saved for deployment")
            print("  Predictions written to database")
            
        except Exception as e:
            print(f"\n Pipeline failed with error: {e}")
            raise


# =========================================================================
# ENTRY POINT FOR SCRIPT EXECUTION
# =========================================================================

if __name__ == "__main__":
    """
    Main execution point for the ML pipeline.
    
    Usage:
        python scripts/ml_pipeline.py
        
    Prerequisites:
        1. Run ETL pipeline: python scripts/etl_turbofan.py
        2. Run dbt models: cd turbine_etl_dbt && dbt run
        3. Ensure raw test data files are available
    """
    
    print(" TURBOFAN ML PIPELINE - STARTING EXECUTION")
    print("=" * 60)
    
    # Initialize and run pipeline
    pipeline = TurbofanMLPipeline()
    pipeline.run_complete_pipeline()
    
    print("\n Pipeline execution completed!")
    print(" Check the results directory for detailed outputs")
    print(" Models are ready for deployment!")