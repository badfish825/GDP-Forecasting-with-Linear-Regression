import pandas as pd

def macro_data_preparation():

        # Load Quarterly GDP (and convert to datetime)
        gdp = pd.read_csv("data/raw/GDP.csv")
        gdp['observation_date'] = pd.to_datetime(gdp['observation_date'])
        gdp = gdp.set_index('observation_date')

        # Load Quarterly CPI (average monthly to quarterly)
        cpi = pd.read_csv("data/raw/CPIAUCSL.csv")
        cpi['observation_date'] = pd.to_datetime(cpi['observation_date'])
        cpi = cpi.set_index('observation_date')
        cpi = cpi.rename(columns={'CPIAUCSL': 'CPI'})
        cpi_quarterly = cpi.resample('QS').mean()

        # Load Unemployment Rate (average monthly to quarterly)
        unrate = pd.read_csv("data/raw/UNRATE.csv")
        unrate['observation_date'] = pd.to_datetime(unrate['observation_date'])
        unrate = unrate.set_index('observation_date')
        unrate_quarterly = unrate.resample('QS').mean()

        # Load Federal Funds Rate (average monthly to quarterly)
        fedfunds = pd.read_csv("data/raw/FEDFUNDS.csv")
        fedfunds['observation_date'] = pd.to_datetime(fedfunds['observation_date'])
        fedfunds = fedfunds.set_index('observation_date')
        fedfunds_quarterly = fedfunds.resample('QS').mean()

        # Merge all data together
        df = gdp.merge(cpi_quarterly, left_index=True, right_index=True) \
                .merge(unrate_quarterly, left_index=True, right_index=True) \
                .merge(fedfunds_quarterly, left_index=True, right_index=True)

        # Isolate data from 1990 to Q3 2025
        df = df.loc['1990-01-01':'2025-09-30']

        # Creating the additional features
        # Percent Change = (new - old) / old  OR  (current - shift 1) / shift 1
        df['GDP_growth'] = (df['GDP'] - df['GDP'].shift(1)) / df['GDP'].shift(1)

        df['CPI_pct_change'] = (df['CPI'] - df['CPI'].shift(1)) / df['CPI'].shift(1)

        df['UNRATE_change'] = df['UNRATE'] - df['UNRATE'].shift(1)

        # Lags for GDP, GDP growth, CPI % Change, and Unemployment Change
        lags = [1, 2, 4]
        for lag in lags:
                df[f'GDP_lag{lag}'] = df['GDP'].shift(lag)
        
                df[f'GDP_growth_lag{lag}'] = df['GDP_growth'].shift(lag)
        
                df[f'CPI_pct_change_lag{lag}'] = df['CPI_pct_change'].shift(lag)
        
                df[f'UNRATE_change_lag{lag}'] = df['UNRATE_change'].shift(lag)

        # Lags for Federal Funds Rate
        df['FEDFUNDS_lag1'] = df['FEDFUNDS'].shift(1)
        df['FEDFUNDS_lag2'] = df['FEDFUNDS'].shift(2)

        # Removing the missing values at the start of the lag columns
        df = df.dropna()

        # Save processed DataFrame to CSV
        df.to_csv("data/processed/features.csv")

        return df


# Final df:
# Columns = ["GDP", "CPI", "UNRATE", "FEDFUNDS",
#           "GDP_growth", "CPI_pct_change", "UNRATE_change",
#           "GDP_lag1", "GDP_lag2", "GDP_lag4", 
#           "CPI_pct_lag1", "CPI_pct_lag2", "CPI_pct_lag4", 
#           "UNRATE_change_lag1", "UNRATE_change_lag2", "UNRATE_change_lag4", 
#           "FEDFUNDS_lag1", "FEDFUNDS_lag2"]
# Index = observation_date
