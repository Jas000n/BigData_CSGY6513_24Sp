import numpy as np
import scipy
from sklearn.metrics import mean_squared_error, mean_absolute_error

def RMSE(y_true, y_estimate):
    return np.sqrt(mean_squared_error(y_true, y_estimate))

def MAE(y_true, y_estimate):
    return mean_absolute_error(y_true, y_estimate)

def custom_metric(y_true, y_estimate):
    # Implement your custom metric here
    pass

metrics_dict = {
    "RMSE": RMSE,
    "MAE": MAE,
    "custom_metric": custom_metric
}

def calculate_metrics(matrix1, matrix2, estimates, metrics=["RMSE", "MAE"]):
    true_inner_product = matrix1.dot(matrix2)
    metric_results = {}
    for metric_name, metric_func in metrics_dict.items():
        if metric_name == "RMSE":
            metric_results[metric_name] = RMSE(true_inner_product, estimates)
        elif metric_name == "MAE":
            metric_results[metric_name] = MAE(true_inner_product, estimates)
        # 添加其他度量的计算
    return metric_results
