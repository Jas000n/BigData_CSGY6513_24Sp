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
    if matrix1.ndim == 2 and matrix2.ndim == 2:
        true_inner_product = matrix1.dot(matrix2.T)
        if scipy.sparse.issparse(true_inner_product):
            true_inner_product = true_inner_product.toarray()
        if true_inner_product.ndim == 2:
            true_inner_product = true_inner_product[0, 0]
        true_inner_products = np.full(len(estimates), true_inner_product)
    else:
        true_inner_products = np.array([matrix1[i].dot(matrix2[i].T).toarray()[0][0] for i in range(len(estimates))])

    metric_results = {}
    for metric_name in metrics:
        metric_func = metrics_dict[metric_name]
        metric_results[metric_name] = metric_func(true_inner_products, estimates)
    return metric_results