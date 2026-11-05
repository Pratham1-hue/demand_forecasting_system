import os
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_retail_data(num_stores: int = 3, num_skus: int = 5, days: int = 365) -> pd.DataFrame:
    """Generates synthetic daily multi-store sales data with seasonality and missing values."""
    np.random.seed(42)
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    records = []
    for store in range(1, num_stores + 1):
        for sku in range(1, num_skus + 1):
            base_sales = np.random.randint(50, 200)
            for d in dates:
                weekend_boost = 1.3 if d.weekday() >= 5 else 1.0
                month_boost = 1.2 if d.month == 12 else 1.0
                is_promo = 1 if np.random.rand() < 0.1 else 0
                promo_boost = 1.4 if is_promo == 1 else 1.0
                noise = np.random.normal(1.0, 0.15)
                
                sales = int(base_sales * weekend_boost * month_boost * promo_boost * noise)
                sales = np.nan if np.random.rand() < 0.03 else max(0, sales)
                
                records.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "store_id": f"STORE_{store:02d}",
                    "sku_id": f"SKU_{sku:03d}",
                    "sales": sales,
                    "is_promo": is_promo,
                    "on_holiday": 1 if (d.month == 12 and d.day >= 24) else 0
                })
                
    return pd.DataFrame(records)

def setup_duckdb_storage(df: pd.DataFrame, db_path: str = "data/retail_demand.duckdb") -> duckdb.DuckDBPyConnection:
    """Stores raw data into DuckDB and executes SQL window-function imputation."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = duckdb.connect(db_path)
    conn.register("raw_sales", df)
    
    conn.execute("""
        CREATE OR REPLACE TABLE cleaned_sales AS
        WITH filled_data AS (
            SELECT 
                date,
                store_id,
                sku_id,
                is_promo,
                on_holiday,
                COALESCE(
                    sales, 
                    AVG(sales) OVER(PARTITION BY store_id, sku_id ORDER BY date ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING)
                ) AS imputed_sales
            FROM raw_sales
        )
        SELECT 
            CAST(date AS DATE) as date,
            store_id,
            sku_id,
            is_promo,
            on_holiday,
            ROUND(imputed_sales) as sales
        FROM filled_data;
    """)
    print("✓ DuckDB storage initialized and missing values imputed via SQL window functions.")
    return conn