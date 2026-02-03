from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

from src.macro_data_preparation import macro_data_preparation

def main():
    df = macro_data_preparation()

    # x = features for the model
    # Dropping the columns not used as features
    x = df.drop(columns = [
        "GDP", "GDP_growth", "CPI_pct_change", "UNRATE_change"
        ])

    # y = output of the model
    y = df["GDP"]

    # Training vs. testing data split
    x_train, x_test, y_train, y_test = train_test_split (
        x, y, test_size = 0.2, shuffle = False
    )

    # Model training
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_prediction = model.predict(x_test)

    # Performance evaluation
    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_test, y_prediction)
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_test, y_prediction))
    # R² (Coefficient of Determination)
    r2 = r2_score(y_test, y_prediction)

    # MAE % and RMSE %
    mae_pct = mae / y_test.mean() * 100
    rmse_pct = rmse / y_test.mean() * 100

    # Print results
    print(f"MAE: {mae:.2f} ({mae_pct:.2f}%)")
    print(f"RMSE: {rmse:.2f} ({rmse_pct:.2f}%)")
    print(f"R²: {r2:.2f}")
    

if __name__ == "__main__":
    main()
