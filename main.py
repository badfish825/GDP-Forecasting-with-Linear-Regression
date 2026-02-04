from src.linear_model import linear_model
from src.ridge_model import ridge_model
from src.model_evaluation import model_evaluation

# Create Linear Regression
linear_regression = linear_model()

# Dislay Linear Regression performance metrics
linear_results = model_evaluation(linear_regression)
print("Linear Regression Model Reults: ")
print(f"MAE: {linear_results['MAE']:.2f} ({linear_results['MAE_PCT']:.2f}%)")
print(f"RMSE: {linear_results['RMSE']:.2f} ({linear_results['RMSE_PCT']:.2f}%)")
print(f"R²: {linear_results['R2']:.2f}")


print()

# Create Ridge Regression
ridge_regression = ridge_model()

# Dislay Ridge Regression performance metrics
print("Ridge Regression Model Reults: ")
ridge_results = model_evaluation(ridge_regression)
print(f"MAE: {ridge_results['MAE']:.2f} ({ridge_results['MAE_PCT']:.2f}%)")
print(f"RMSE: {ridge_results['RMSE']:.2f} ({ridge_results['RMSE_PCT']:.2f}%)")
print(f"R²: {ridge_results['R2']:.2f}")
