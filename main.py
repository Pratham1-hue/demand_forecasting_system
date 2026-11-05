from src.data_loader import generate_synthetic_retail_data, setup_duckdb_storage
from src.preprocessing import build_tft_feature_dataset
from src.models.prophet_baseline import train_prophet_model
from src.models.tft_pipeline import prepare_tft_inputs
from src.viz import plot_forecast_results

def main():
    print("--- 1. Generating Synthetic Store Data ---")
    raw_df = generate_synthetic_retail_data(num_stores=2, num_skus=3, days=365)
    
    print("--- 2. Setting up DuckDB Imputation Pipeline ---")
    db_conn = setup_duckdb_storage(raw_df)
    cleaned_df = db_conn.execute("SELECT * FROM cleaned_sales").df()
    
    print("--- 3. Running Prophet Benchmark ---")
    target_store = "STORE_01"
    target_sku = "SKU_001"
    
    forecast, test_results, metrics = train_prophet_model(
        cleaned_df, target_store, target_sku, forecast_horizon=30
    )
    
    print(f"\nModel Evaluation Metrics ({target_store} | {target_sku}):")
    print(f" -> WAPE : {metrics['WAPE'] * 100:.2f}%")
    print(f" -> SMAPE: {metrics['SMAPE']:.2f}%\n")
    
    print("--- 4. Preparing TFT Multi-Series Features ---")
    tft_df = build_tft_feature_dataset(db_conn)
    formatted_tft_df, meta = prepare_tft_inputs(tft_df)
    print("TFT Dataset Ready. Sample rows:")
    print(formatted_tft_df.head(3))
    
    print("--- 5. Plotting Interactive Results ---")
    plot_df = cleaned_df[
        (cleaned_df["store_id"] == target_store) & 
        (cleaned_df["sku_id"] == target_sku)
    ].rename(columns={"date": "ds", "sales": "y"})
    
    plot_forecast_results(plot_df, forecast, target_store, target_sku)

if __name__ == "__main__":
    main()