import pandas as pd
from prophet import Prophet
from src.evaluation import calculate_wape, calculate_smape

def train_prophet_model(df: pd.DataFrame, store_id: str, sku_id: str, forecast_horizon: int = 30):
    """Trains Prophet on a specific Store-SKU series using exogenous regressors."""
    sub_df = df[(df["store_id"] == store_id) & (df["sku_id"] == sku_id)].copy()
    
    prophet_df = sub_df[["date", "sales", "is_promo", "on_holiday"]].rename(
        columns={"date": "ds", "sales": "y"}
    )
    
    train_df = prophet_df.iloc[:-forecast_horizon]
    test_df = prophet_df.iloc[-forecast_horizon:]
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.add_regressor("is_promo")
    model.add_regressor("on_holiday")
    model.fit(train_df)
    
    future = prophet_df[["ds", "is_promo", "on_holiday"]].copy()
    forecast = model.predict(future)
    
    test_results = test_df.merge(forecast[["ds", "yhat"]], on="ds")
    
    wape = calculate_wape(test_results["y"], test_results["yhat"])
    smape = calculate_smape(test_results["y"], test_results["yhat"])
    
    return forecast, test_results, {"WAPE": wape, "SMAPE": smape}