import numpy as np
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

def calculate_metrics(X_train, X_test, estimates, metrics=["RMSE", "MAE"]):
    true_inner_products = [X_train[i].dot(X_test[i].T).toarray()[0][0] for i in range(X_test.shape[0])]
    metric_results = {}
    for metric_name in metrics:
        metric_func = metrics_dict[metric_name]
        metric_results[metric_name] = metric_func(true_inner_products, estimates)
    return metric_results