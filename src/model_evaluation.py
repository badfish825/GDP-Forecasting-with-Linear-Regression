from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def model_evaluation(model_results):
    """

    Evaluates the performance of an ML model using multiple metrics.

    Parameters:
        model_results (dict): {
            "y_test" : pandas df for actual values of test set
            "y_pred" : pandas df for predicted values of test set
        }

    Returns:
        dict: {
            "MAE": float, Mean Absolute Error
            "RMSE": float, Root Mean Squared Error
            "R2": float, Coefficient of Determination
            "MAE_PCT": float, MAE as a percentage of average target value
            "RMSE_PCT": float, RMSE as a percentage of average target value
        }

    """

    y_test = model_results["y_test"]
    y_pred = model_results["y_pred"]

    # Performance evaluation
    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_test, y_pred)
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # R² (Coefficient of Determination)
    r2 = r2_score(y_test, y_pred)

    # MAE % and RMSE %
    mae_pct = mae / y_test.mean() * 100
    rmse_pct = rmse / y_test.mean() * 100

    # Print results
    return {
        "MAE" : mae,
        "RMSE" : rmse,
        "R2" : r2,
        "MAE_PCT" : mae_pct,
        "RMSE_PCT" : rmse_pct,
    }

