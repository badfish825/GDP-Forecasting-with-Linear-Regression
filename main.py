import json
from src.linear_model import linear_model
from src.ridge_model import ridge_model
from src.model_evaluation import model_evaluation
from src.visualization import parity_plot, residuals_over_time, residual_distribution

# Create Linear and Ridge models
linear_regression = linear_model()
ridge_regression = ridge_model()

# Evaluate models
linear_results = model_evaluation(linear_regression)
ridge_results = model_evaluation(ridge_regression)

# Save metrics to JSON
metrics = {
    "Linear": {k: round(v, 2) for k, v in linear_results.items()},
    "Ridge": {k: round(v, 2) for k, v in ridge_results.items()}
}

with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Visualize linear and ridge model performance
parity_plot(linear_regression, ridge_regression, "Linear Regression", "Ridge Regression")
residuals_over_time(linear_regression, ridge_regression, "Linear Regression", "Ridge Regression")
residual_distribution(linear_regression, ridge_regression, "Linear Regression", "Ridge Regression")
