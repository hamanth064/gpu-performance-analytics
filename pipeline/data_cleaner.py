"""
Data Cleaner — ETL Pipeline
============================
Cleans raw GPU/CPU metrics: handles missing values, normalizes data,
and stores results in both CSV and SQLite.
"""

import numpy as np
import pandas as pd
import sqlite3
import os


def inject_missing_values(df, missing_rate=0.02, seed=42):
    """
    Inject realistic missing values (~2%) to simulate sensor dropouts.

    Parameters
    ----------
    df : pd.DataFrame
        Raw metrics data.
    missing_rate : float
        Fraction of values to set to NaN.

    Returns
    -------
    pd.DataFrame
        DataFrame with injected NaN values.
    """
    np.random.seed(seed)
    df = df.copy()
    numeric_cols = ["gpu_usage", "cpu_usage", "memory_usage",
                    "temperature", "power_consumption"]

    total_cells = len(df) * len(numeric_cols)
    n_missing = int(total_cells * missing_rate)

    for _ in range(n_missing):
        row = np.random.randint(0, len(df))
        col = np.random.choice(numeric_cols)
        df.at[row, col] = np.nan

    return df


def clean_data(df):
    """
    Clean the DataFrame:
    1. Forward-fill short gaps (≤3 consecutive NaNs)
    2. Linear interpolation for remaining gaps
    3. Clip to valid ranges

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potential NaN values.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with no missing values.
    """
    df = df.copy()
    numeric_cols = ["gpu_usage", "cpu_usage", "memory_usage",
                    "temperature", "power_consumption"]

    # Forward-fill short gaps
    df[numeric_cols] = df[numeric_cols].ffill(limit=3)

    # Interpolate remaining gaps
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")

    # Fill any remaining edge NaNs
    df[numeric_cols] = df[numeric_cols].bfill()

    # Clip to valid ranges
    df["gpu_usage"] = df["gpu_usage"].clip(0, 100)
    df["cpu_usage"] = df["cpu_usage"].clip(0, 100)
    df["memory_usage"] = df["memory_usage"].clip(0, 100)
    df["temperature"] = df["temperature"].clip(20, 105)
    df["power_consumption"] = df["power_consumption"].clip(50, 500)

    return df


def normalize_data(df):
    """
    Add Min-Max normalized columns (preserving originals).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional `_normalized` columns.
    """
    df = df.copy()
    numeric_cols = ["gpu_usage", "cpu_usage", "memory_usage",
                    "temperature", "power_consumption"]

    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        range_val = max_val - min_val
        if range_val == 0:
            df[f"{col}_normalized"] = 0.0
        else:
            df[f"{col}_normalized"] = ((df[col] - min_val) / range_val).round(4)

    return df


def save_to_csv(df, output_dir=None):
    """Save cleaned data to data/processed/ as CSV."""
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "processed"
        )
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "cleaned_metrics.csv")
    df.to_csv(path, index=False)
    print(f"✅ Cleaned CSV saved → {path}")
    return path


def save_to_sqlite(df, db_path=None, table_name="system_metrics"):
    """
    Save cleaned data to SQLite database.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and normalized data.
    db_path : str, optional
        Path to SQLite database file.
    table_name : str
        Table name in the database.

    Returns
    -------
    str
        Path to the database file.
    """
    if db_path is None:
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "performance.db"
        )
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    print(f"✅ SQLite saved → {db_path} (table: {table_name})")
    return db_path


def run_etl_pipeline(df):
    """
    Run the full ETL pipeline: inject gaps → clean → normalize → save.

    Parameters
    ----------
    df : pd.DataFrame
        Raw generated metrics.

    Returns
    -------
    tuple
        (cleaned_df, csv_path, db_path)
    """
    print("🔄 Step 1/4: Injecting missing values (~2%)...")
    df_dirty = inject_missing_values(df)
    missing_count = df_dirty.isnull().sum().sum()
    print(f"   Injected {missing_count} NaN values")

    print("🔄 Step 2/4: Cleaning data...")
    df_clean = clean_data(df_dirty)
    remaining = df_clean.isnull().sum().sum()
    print(f"   Remaining NaN: {remaining}")

    print("🔄 Step 3/4: Normalizing data...")
    df_norm = normalize_data(df_clean)

    print("🔄 Step 4/4: Saving to CSV + SQLite...")
    csv_path = save_to_csv(df_norm)
    db_path = save_to_sqlite(df_norm)

    print(f"✅ ETL complete! {len(df_norm):,} rows processed.")
    return df_norm, csv_path, db_path


if __name__ == "__main__":
    from data_generator import generate_system_metrics
    df = generate_system_metrics(days=7)
    run_etl_pipeline(df)
