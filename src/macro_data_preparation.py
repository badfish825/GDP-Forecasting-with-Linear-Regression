import pandas as pd

def macro_data_preparation():

        # Load Quarterly GDP (and convert to datetime)
        gdp = pd.read_csv("data/GDP.csv")
        gdp['observation_date'] = pd.to_datetime(gdp['observation_date'])
        gdp = gdp.set_index('observation_date')

        # Load Quarterly CPI (average monthly to quarterly)
        cpi = pd.read_csv("data/CPIAUCSL.csv")
        cpi['observation_date'] = pd.to_datetime(cpi['observation_date'])
        cpi = cpi.set_index('observation_date')
        cpi = cpi.rename(columns={'CPIAUCSL': 'CPI'})
        cpi_quarterly = cpi.resample('QS').mean()

        # Load Unemployment Rate (average monthly to quarterly)
        unrate = pd.read_csv("data/UNRATE.csv")
        unrate['observation_date'] = pd.to_datetime(unrate['observation_date'])
        unrate = unrate.set_index('observation_date')
        unrate_quarterly = unrate.resample('QS').mean()

        # Load Federal Funds Rate (average monthly to quarterly)
        fedfunds = pd.read_csv("data/FEDFUNDS.csv")
        fedfunds['observation_date'] = pd.to_datetime(fedfunds['observation_date'])
        fedfunds = fedfunds.set_index('observation_date')
        fedfunds_quarterly = fedfunds.resample('QS').mean()

        # Merge all data together
        df = gdp.merge(cpi_quarterly, left_index=True, right_index=True) \
                .merge(unrate_quarterly, left_index=True, right_index=True) \
                .merge(fedfunds_quarterly, left_index=True, right_index=True)

        return df
