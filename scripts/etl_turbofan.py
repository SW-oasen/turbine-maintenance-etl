import argparse, os, pandas as pd, numpy as np, yaml
from sqlalchemy import create_engine
import requests

SENSOR_COUNT = 21
COLS = ["unit_nr","time_cycles","setting1","setting2","setting3"] + [f"sensor{i}" for i in range(1,SENSOR_COUNT+1)]
N8N_WEBHOOK_URL = "http://localhost:5678/webhook/turbine-progress"

# Read CMAPSS txt file into DataFrame
def read_cmapss_txt(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, :len(COLS)]
    df.columns = COLS
    df["unit_nr"] = df["unit_nr"].astype(int)
    df["time_cycles"] = df["time_cycles"].astype(int)
    for c in ["setting1","setting2","setting3"] + [f"sensor{i}" for i in range(1,SENSOR_COUNT+1)]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns from {path}")
    return df 

# Compute Remaining Useful Life (RUL)
def compute_rul(df: pd.DataFrame) -> pd.Series:
    max_cycles = df.groupby("unit_nr")["time_cycles"].transform("max")
    return (max_cycles - df["time_cycles"]).astype(int)

# Z-score normalization by unit
def zscore_by_unit(df: pd.DataFrame, cols):
    def _z(g): return (g[cols] - g[cols].mean()) / g[cols].std(ddof=0)
    z = df.groupby("unit_nr", group_keys=False).apply(_z)
    z.columns = [f"z_{c}" for c in cols]
    return z

# Compute rolling mean features and differences
def rolling_features(df: pd.DataFrame, cols, windows=(5,20)):
    out = {}
    df_sorted = df.sort_values(["unit_nr","time_cycles"])
    for w in windows:
        rolled = (df_sorted.groupby("unit_nr", group_keys=False)[cols].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True))
        rolled.columns = [f"mean{w}_{c}" for c in cols]
        out[w] = rolled
    diffs = df_sorted.groupby("unit_nr", group_keys=False)[cols].diff().rename(columns={c: f"d_{c}" for c in cols})
    return out, diffs

#  Main ETL function
def run(cfg):
    dataset = cfg.get("dataset","FD001")
    db_url = cfg.get("db_url","sqlite:///turbofan.db")
    paths = cfg.get("files",{})
    opts = cfg.get("options",{})
    checkpoint = bool(opts.get("checkpoint", True))
    
    # Get available sensors after loading data
    train_path = paths.get("train","data/raw/train_FD001.txt")
    if not os.path.exists(train_path): raise FileNotFoundError(f"Missing file: {train_path}")   
    df = read_cmapss_txt(train_path)
    df["dataset"] = dataset
    
    # Use forced sensors if provided (for consistency across datasets)
    if "_force_sensors" in cfg:
        consistent_sensors = cfg["_force_sensors"]
        
        # Keep only the base columns + consistent sensors + dataset
        base_cols = ["unit_nr", "time_cycles", "setting1", "setting2", "setting3"]
        cols_to_keep = base_cols + consistent_sensors + ["dataset"]
        
        # Filter DataFrame to only keep consistent columns
        df = df[cols_to_keep]
        
        sensors = consistent_sensors
        print("Using consistent sensor set:", json_body=sensors)
        
    else:
        # Original logic for single dataset
        if opts.get("drop_constant_sensors", True):
            const_cols = [c for c in df.columns if c.startswith("sensor") and df[c].nunique(dropna=True) <= 1]
            if const_cols: 
                print("Dropping constant sensors:", json_body=const_cols)
                df = df.drop(columns=const_cols)

        # Get available sensor columns after dropping constants
        available_sensors = [c for c in df.columns if c.startswith("sensor")]
        requested_sensors = [f"sensor{i}" for i in opts.get("sensors", [2, 3, 4])]
        
        # Filter to only use sensors that actually exist
        sensors = [s for s in requested_sensors if s in available_sensors]
        if not sensors:
            sensors = available_sensors[:3]  # Use first 3 available sensors as fallback
        
        print(f"Using sensors: {sensors}")

    windows = tuple(opts.get("rolling_windows",[5,20]))

    rul = compute_rul(df)
    roll_dict, diffs = rolling_features(df, sensors, windows=windows)
    z = zscore_by_unit(df, sensors)

    feat = pd.concat([
        df[["unit_nr","time_cycles","dataset"]].reset_index(drop=True),
        rul.rename("rul"),
        *(roll_dict[w] for w in windows),
        diffs.reset_index(drop=True),
        z.reset_index(drop=True)
    ], axis=1)

    # Create db directory if it doesn't exist
    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            print(f"Created directory: {db_dir}")

    # Create engine with specific SQLite settings
    engine = create_engine(db_url, 
                          connect_args={'timeout': 20, 'check_same_thread': False},
                          poolclass=None)  # Disable connection pooling
    
    try:
        # Use replace for first dataset, append for others
        if_exists_raw = "replace" if dataset == "FD001" else "append"
        if_exists_feat = "replace" if dataset == "FD001" else "append"
        
        with engine.begin() as conn:
            # Debug: print column names being inserted
            print(f"DataFrame columns for {dataset}: {list(df.columns)}")
            
            df.to_sql("cycles_raw", conn, if_exists=if_exists_raw, index=False)
            feat.to_sql("cycles_features", conn, if_exists=if_exists_feat, index=False)
            
            units = df.groupby(["dataset","unit_nr"]).agg(cycles_min=("time_cycles","min"),
                                                         cycles_max=("time_cycles","max"),
                                                         cycles_count=("time_cycles","count")).reset_index()
            units.to_sql("units_summary", conn, if_exists="replace", index=False)
        
            print(f"[ETL] Wrote cycles_raw: +{len(df)} rows for {dataset}")
            print(f"[ETL] Wrote cycles_features: +{len(feat)} rows for {dataset}")
    
    finally:
        engine.dispose()

    os.makedirs("data/processed", exist_ok=True)
    feat.to_csv("data/processed/cycles_features.csv", index=False)

    if checkpoint:
        os.makedirs("data/interim", exist_ok=True)
        df.to_csv("data/interim/cycles_raw.csv", index=False)

    print("ETL done.")

# Entry point
if __name__ == "__main__":
    import argparse, yaml
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="scripts/etl_config.yaml")
    p.add_argument("--db", dest="db_url", default=None)
    p.add_argument("--replace", action="store_true",
                   help="replace old data in the database with new data")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.db_url:
        cfg["db_url"] = args.db_url

    datasets = cfg.get("datasets")

    if datasets:
        # If replace flag is used, delete the entire database first
        if args.replace:
            db_path = cfg["db_url"].replace("sqlite:///", "")
            if os.path.exists(db_path):
                os.remove(db_path)
                print(f"Deleted existing database: {db_path}")
        
        # STEP 1: Determine which sensors are variable across ALL datasets
        print("Analyzing sensors across all datasets...")
        all_variable_sensors = None
        
        for ds in datasets:
            train_path = ds["train"]
            if not os.path.exists(train_path):
                print(f"Warning: Missing file {train_path}, skipping...")
                continue

            print(f"Analyzing {ds['code']}...")
            temp_df = read_cmapss_txt(train_path)
            
            if cfg.get("options", {}).get("drop_constant_sensors", True):
                # Find variable sensors in this dataset
                variable_sensors = [c for c in temp_df.columns 
                                  if c.startswith("sensor") and temp_df[c].nunique(dropna=True) > 1]
            else:
                variable_sensors = [c for c in temp_df.columns if c.startswith("sensor")]

            print(f"Variable sensors in {ds['code']}: {len(variable_sensors)} sensors")

            if all_variable_sensors is None:
                all_variable_sensors = set(variable_sensors)
            else:
                # Keep only sensors that are variable in ALL datasets
                all_variable_sensors = all_variable_sensors.intersection(set(variable_sensors))
        
        # Convert to sorted list for consistency
        consistent_sensors = sorted(list(all_variable_sensors))
        print(f"\nSensors variable across ALL datasets: {consistent_sensors}")
        print(f"Using {len(consistent_sensors)} consistent sensors\n")

        # STEP 2: Process datasets with consistent sensor set
        for ds in datasets:
            subcfg = dict(cfg)
            subcfg["dataset"] = ds["code"]
            subcfg["files"] = {"train": ds["train"], "test": ds.get("test"), "rul": ds.get("rul")}
            subcfg["_force_sensors"] = consistent_sensors  # Force same sensors for all datasets

            print(f"\nProcessing dataset: {ds['code']}")
            run(subcfg)
            
    else:
        # fallback: single dataset (old behavior)
        run(cfg)