import numpy as np

def calculate_wape(actual, predicted) -> float:
    """Weighted Absolute Percentage Error."""
    actual, predicted = np.array(actual), np.array(predicted)
    return float(np.sum(np.abs(actual - predicted)) / np.sum(actual))

def calculate_smape(actual, predicted) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    actual, predicted = np.array(actual), np.array(predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    return float(np.mean(np.abs(actual - predicted) / denominator) * 100)