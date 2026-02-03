
# GDP Prediction Pipeline
Forecasting U.S. GDP using a linear regression ML pipeline.

## Overview
This project builds a regression model using macroeconomic indicators from the Federal Reserve Economic Data (FRED) to predict quarterly GDP.

## Data Source
- FRED (GDP, CPI, Unemployment Rate, Federal Funds Rate)

## Status
Data ingestion and feature engineering pipeline complete.

### Commit 2:
* Pipeline loads raw CSVs from 'data/raw/'
* Averages monthly data to quarterly
* Creates additional features:
    * GDP growth
    * CPI % Change
    * Unemployment Rate Change
    * Lagged features for GDP, GDP growth, CPI, Unemployment Rate, and Federal Funds Rate
* Saves final DataFrame with processed featues to 'data/processed/features.csv'

## How to use
1. Install libraries:
```bash
pip install -r requirements.txt
```

2. Generate processed features:
```python 
from src.macro_data_preparation import macro_data_preparation

df = macro_data_preparation()
```
