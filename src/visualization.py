import matplotlib.pyplot as plt

"""

Visualize the performance of 2 ML models.
Creates and displays graphs to evaluate and compare model performance:
1. Parity Plot: Scatter plot of actual vs. predicted values.
2. Residuals Over Time Plot: Residuals vs. time (quarters).
3. Residual Distribution: Histogram to evaluate error symmetry and variance.

Parameters:
    model1_results (dict): {
        "model1_test" : pandas df for actual values of model 1's test set 
        "model1_pred" : pandas df for predicted values of model 1's test set
    }

    model2_results (dict): {
        "model2_test" : pandas df for actual values of model 2's test set 
        "model2_pred" : pandas df for predicted values of model 2's test set
    }

    model1_name (str): Name of model 1

    model2_name (str): Name of model 2

"""

def parity_plot(model1_results, model2_results, model1_name="Model 1", model2_name="Model 2"):
    model1_test = model1_results["y_test"]
    model1_pred = model1_results["y_pred"]
    model2_test = model2_results["y_test"]
    model2_pred = model2_results["y_pred"]
    
    plt.figure(figsize = (8, 8))

    # Model 1 Line
    plt.scatter(model1_test, model1_pred, color = 'blue', alpha = 0.6, edgecolor = 'k', label = model1_name)
    # Model 2 Line
    plt.scatter(model2_test, model2_pred, color = 'red', alpha = 0.6, edgecolor = 'k', label = model2_name)
    # Perfect Prediction Line
    plt.plot([model1_test.min(), model1_test.max()], [model1_test.min(), model1_test.max()], 'r--', lw=2, label='Perfect Prediction')

    plt.xlabel("Actual GDP ($ billions)")
    plt.ylabel("Predicted GDP ($ billions)")
    plt.title("Parity Plot: Actual vs Predicted GDP")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/parity_plot.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def residuals_over_time(model1_results, model2_results, model1_name="Model 1", model2_name="Model 2"):
    model1_test = model1_results["y_test"]
    model1_pred = model1_results["y_pred"]
    model2_test = model2_results["y_test"]
    model2_pred = model2_results["y_pred"]
    
    # Calculate Residuals
    residuals1 = model1_test - model1_pred
    residuals2 = model2_test - model2_pred

    plt.figure(figsize = (8, 8))
    plt.plot(model1_test.index, residuals1, marker='o', linestyle='-', label=f"{model1_name} Residuals", color='blue')
    plt.plot(model2_test.index, residuals2, marker='x', linestyle='--', label=f"{model2_name} Residuals", color='red')
    
    plt.axhline(0, color='black', linestyle='--', lw=1) # Black line on x-axis
    plt.xlabel("Date (Quarter Start)")
    plt.ylabel("Residual ($ billions)")
    plt.title("Residuals Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/residuals_over_time.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def residual_distribution(model1_results, model2_results, model1_name="Model 1", model2_name="Model 2"):
    model1_test = model1_results["y_test"]
    model1_pred = model1_results["y_pred"]
    model2_test = model2_results["y_test"]
    model2_pred = model2_results["y_pred"]
    
    # Calculate Residuals
    residuals1 = model1_test - model1_pred
    residuals2 = model2_test - model2_pred

    plt.figure(figsize=(10, 6))
    plt.hist(residuals1, bins=15, alpha=0.6, label=f"{model1_name} Residuals", color='blue', edgecolor='black')
    plt.hist(residuals2, bins=15, alpha=0.6, label=f"{model2_name} Residuals", color='red', edgecolor='black')

    plt.axvline(0, color='black', linestyle='--', lw=1) # Black line on x-axis
    plt.xlabel("Residual ($ billions)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/residual_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
