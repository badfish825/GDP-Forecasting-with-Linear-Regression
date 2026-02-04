from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .macro_data_preparation import macro_data_preparation


def ridge_model():
    """

    Trains a ridge regression model to predict GDP from macroeocnomic features.

    Output:
    dict: {
        "y_test" : pandas df for the actual GDP values of the test set
        "y_pred" : pandas df for the predicted GDP values of the test set
    }

    """
    
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

    # Feature Scaling
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Model training (hyperparameter alpha can be increased for more regularizaiton strength)
    model = Ridge(alpha = 1.0)
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)

    return {
        "y_test" : y_test,
        "y_pred" : y_pred
    }
