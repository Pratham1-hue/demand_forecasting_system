import pandas as pd

def prepare_tft_inputs(tft_df: pd.DataFrame, max_encoder_length: int = 60, max_prediction_length: int = 30):
    """Formats time-series DataFrame with features required for TFT PyTorch forecasting."""
    tft_df["sales"] = tft_df["sales"].astype(float)
    tft_df["time_idx"] = tft_df["time_idx"].astype(int)
    tft_df["store_id"] = tft_df["store_id"].astype(str)
    tft_df["sku_id"] = tft_df["sku_id"].astype(str)
    
    metadata = {
        "static_categoricals": ["store_id", "sku_id"],
        "time_varying_known_reals": ["time_idx", "day_of_week", "month", "is_promo", "on_holiday"],
        "time_varying_unknown_reals": ["sales"],
        "target": "sales"
    }
    
    return tft_df, metadata