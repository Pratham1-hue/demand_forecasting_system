import duckdb
import pandas as pd

def build_tft_feature_dataset(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Extracts multi-series time features formatted for TFT and DeepAR models."""
    df = conn.execute("""
        SELECT 
            date,
            store_id,
            sku_id,
            sales,
            is_promo,
            on_holiday,
            DAYOFWEEK(date) as day_of_week,
            MONTH(date) as month,
            DATEDIFF('day', MIN(date) OVER(), date) as time_idx
        FROM cleaned_sales
        ORDER BY store_id, sku_id, date
    """).df()
    
    df["group_id"] = df["store_id"] + "_" + df["sku_id"]
    return df
