
# GDP Prediction Pipeline
### Forecasting U.S. GDP using a linear regression and ridge regression ML pipeline.

## Overview
This project builds an end-to-end regression model using U.S. economic indicators from the Federal Reserve Economic Data (FRED) to predict quarterly GDP. The goal is to demonstrate **data ingestion, feature engineering, and model evaluations** in an interpretable and reproducible way. 

## Data Source
- FRED (GDP, CPI, Unemployment Rate, Federal Funds Rate)

## Data Pipeline
* Loads raw CSVs from 'data/raw/'
* Averages monthly data to quarterly
* Merges GDP, CPI, unemployment rate, and federal funds rate
* Creates additional features:
    * GDP growth
    * CPI % Change
    * Unemployment Rate Change
    * Lagged features for GDP, GDP growth, CPI, Unemployment Rate, and Federal Funds Rate
* Saves final DataFrame with processed featues to 'data/processed/features.csv'

## Linear Regression Model
The linear regression model is engineering to predict quarterly GDP using macroeconomic data as features.
* Target variable: GDP
* Features: lagged macroeconomic indicators
* Training/testing split: 80%/20% (no shuffling and no data leakage)
* Evaluation Metrics:
    * MAE - Mean Absolute Error
    * RMSE - Root Mean Squared Error
    * R<sup>2</sup> - Coefficient of Determination

### Model Performance:
- **MAE (Mean Absolute Error):** \$366.26 billion
    - (1.44% of average GDP)
- **RMSE (Root Mean Squared Error):** \$763.99 billion  
    - (3.01% of average GDP)
- **R<sup>2</sup> (Coefficient of Determination):** 0.95  
    - (95% of variance explained)
These results indicate that the linear regression model has relatively low average error and captures the majority of GDP variation. 

## Ridge Regression Model
The Ridge regression model is a regularized version of linear regression which adds an L2 penalty to reduce overfitting and increase resistance to noise.
* Target variable: GDP
* Features: lagged macroeconomic indicators
* Training/testing split: 80%/20% (no shuffling and no data leakage)
* Evaluation Metrics:
    * MAE - Mean Absolute Error
    * RMSE - Root Mean Squared Error
    * R<sup>2</sup> - Coefficient of Determination

### Model Performance:
- **MAE (Mean Absolute Error):** \$638.54 billion
    - (2.51% of average GDP)
- **RMSE (Root Mean Squared Error):** \$768.05 billion  
    - (3.02% of average GDP)
- **R<sup>2</sup> (Coefficient of Determination):** 0.95  
    - (95% of variance explained)
Similar to the previous model, the ridge regression model has relatively low average error and captures the majority of GDP variation. 

## Model Comparison
| Model | MAE | RMSE | R<sup>2</sup> |
|:------|:---:|-----:|---:|
| Linear Regression | $366.26B | $763.99B | 0.95 |
| Ridge Regression | $638.54B | $768.05B | 0.95 |
The Ridge regression model produces similar results, with the exception of the significantly higher MAE. This increase in error is due to the L2 regularization shrinking the coefficients too aggressively. This model has become over-regularized, resulting in underfitting.

## How to use
1. Install libraries:
```bash
pip install -r requirements.txt
```

2. Run main.py to execute the full pipeline
```bash
python main.py
```

### Future Improvements
* Tune *alpha* for Ridge regression using cross-validation.
* Add additional models (Lasso, logistic regression, tree-based models).
* Add visualizations for model predictions vs. actual GDP.
* Add additional macroeconomic indicators and features.
