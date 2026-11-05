import pandas as pd
import plotly.graph_objects as go

def plot_forecast_results(df: pd.DataFrame, forecast: pd.DataFrame, store_id: str, sku_id: str):
    """Generates interactive Plotly dashboard comparing actual vs predicted demand."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['ds'], y=df['y'],
        mode='lines', name='Actual Sales',
        line=dict(color='#2b5c8f', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines', name='Prophet Forecast',
        line=dict(color='#e74c3c', width=2, dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(231, 76, 60, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))

    fig.update_layout(
        title=f"Demand Forecast for {store_id} - {sku_id}",
        xaxis_title="Date",
        yaxis_title="Sales Volume (Units)",
        template="plotly_white",
        hovermode="x unified"
    )
    
    fig.show()