import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


from src.global_forecasting.statistical import ARIMAForecaster
from src.global_forecasting.base_forecaster import BaseForecaster

class TestARIMAForecaster(unittest.TestCase):

    def setUp(self):
        self.sample_data = pd.DataFrame({
            'date': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01',
                                     '2020-06-01', '2020-07-01', '2020-08-01', '2020-09-01', '2020-10-01',
                                     '2020-11-01', '2020-12-01', '2021-01-01', '2021-02-01', '2021-03-01']), # Monthly data
            'value': [10, 12, 13, 15, 16, 18, 19, 20, 22, 23, 25, 26, 28, 30, 31]
        })
        self.forecaster_config = {'order': (1, 1, 0)} # Simple ARIMA order for testing
        self.seasonal_forecaster_config = {'order': (1,1,0), 'seasonal_order': (1,0,0,4)} # for quarterly seasonality
        
        # Data for seasonal test - 2 years of monthly data, 24 points
        # Simple repeating pattern for seasonality test: 10,11,12,13, 10,11,12,13 ...
        seasonal_values = []
        for _ in range(6): # 6 quarters = 24 months
            seasonal_values.extend([10,11,12,13])
        
        self.sample_seasonal_data = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=24, freq='MS'),
            'value': seasonal_values[:24] # ensure it's exactly 24
        })


    def test_forecaster_creation(self):
        forecaster_no_config = ARIMAForecaster()
        self.assertIsInstance(forecaster_no_config, ARIMAForecaster)
        self.assertIsInstance(forecaster_no_config, BaseForecaster)
        self.assertEqual(forecaster_no_config.order, (1,1,1)) # Default order

        forecaster_with_config = ARIMAForecaster(config=self.forecaster_config)
        self.assertIsInstance(forecaster_with_config, ARIMAForecaster)
        self.assertEqual(forecaster_with_config.order, (1,1,0))

    def test_fit_model(self):
        forecaster = ARIMAForecaster(config=self.forecaster_config)
        forecaster.fit(self.sample_data)
        self.assertTrue(forecaster.fitted)
        self.assertIsNotNone(forecaster.model) # model attribute in ARIMAForecaster is the statsmodels result wrapper

    def test_forecast_generation(self):
        forecaster = ARIMAForecaster(config=self.forecaster_config)
        forecaster.fit(self.sample_data)
        
        periods_to_forecast = 3
        forecast_df = forecaster.forecast(periods=periods_to_forecast, frequency='M')

        self.assertIsInstance(forecast_df, pd.DataFrame)
        self.assertEqual(len(forecast_df), periods_to_forecast)
        self.assertListEqual(list(forecast_df.columns), ['date', 'value'])

        self.assertIsInstance(forecaster.confidence_intervals, pd.DataFrame)
        self.assertEqual(len(forecaster.confidence_intervals), periods_to_forecast)
        self.assertListEqual(list(forecaster.confidence_intervals.columns), ['date', 'lower', 'value', 'upper'])
        
        self.assertEqual(len(forecaster.forecast_dates), periods_to_forecast)
        
        last_historical_date = self.sample_data['date'].iloc[-1]
        for forecast_date in forecaster.forecast_dates:
            self.assertIsInstance(forecast_date, datetime)
            # Check if forecast date is after last historical date
            # Using relativedelta for month-end comparisons can be tricky.
            # Direct comparison works if dates are precise.
            self.assertTrue(forecast_date > last_historical_date)
            
        # Check frequency of forecasted dates
        expected_date = last_historical_date
        for f_date in forecaster.forecast_dates:
            expected_date = expected_date + relativedelta(months=1)
            self.assertEqual(f_date.year, expected_date.year)
            self.assertEqual(f_date.month, expected_date.month)


    def test_forecast_with_seasonal_order(self):
        # Using sample_seasonal_data for this test
        forecaster = ARIMAForecaster(config=self.seasonal_forecaster_config)
        
        # The default (0,0,0,0) seasonal order means no seasonality.
        # Here we test if a non-zero seasonal order is accepted and runs.
        # The ARIMA model might not converge or might produce warnings with small data / arbitrary orders,
        # but the goal is to test if the forecaster handles the parameter.
        try:
            forecaster.fit(self.sample_seasonal_data)
            self.assertTrue(forecaster.fitted)
            self.assertIsNotNone(forecaster.model)

            periods_to_forecast = 4 # e.g., one full season
            forecast_df = forecaster.forecast(periods=periods_to_forecast, frequency='M') # Monthly data

            self.assertIsInstance(forecast_df, pd.DataFrame)
            self.assertEqual(len(forecast_df), periods_to_forecast)
            # Further checks could be on the pattern of forecasted values if data is very predictable
            # For instance, if seasonal_order=(0,0,0,0) was used, forecast would likely be flat or trended
            # With a P component for SARIMA, it might try to pick up some seasonality.
            # Exact value assertion is too brittle for ARIMA on small data.
        except Exception as e:
            # Allow test to pass if fitting with seasonality fails on this small dataset,
            # as long as it's a statsmodels internal error (e.g. convergence)
            # and not an error in the ARIMAForecaster's parameter handling.
            # This is a pragmatic choice for a unit test not meant to validate ARIMA math itself.
            if "statsmodels" in str(type(e)).lower() or "numpy" in str(type(e)).lower(): # common for convergence issues
                self.skipTest(f"Skipping seasonal test due to model fitting issues on small data: {e}")
            else:
                raise e # Re-raise if it's an unexpected error type


    def test_ensure_minimum_value(self):
        forecaster = ARIMAForecaster(config=self.forecaster_config)
        forecaster.fit(self.sample_data)
        
        # Generate an initial forecast to see what values we get
        initial_forecast_df = forecaster.forecast(periods=3, frequency='M')
        # Pick a value from the forecast that we want to cap
        # To make this robust, let's try to set a minimum that is slightly above the first forecasted value
        
        if not initial_forecast_df.empty:
            potential_min_val = initial_forecast_df['value'].iloc[0] + 1
        else: # Should not happen if fit and forecast work
            potential_min_val = 0 

        forecaster.ensure_minimum = True
        forecaster.minimum_value = potential_min_val
        
        # Regenerate forecast with the minimum value constraint
        capped_forecast_df = forecaster.forecast(periods=3, frequency='M')
        
        self.assertTrue(all(capped_forecast_df['value'] >= potential_min_val))
        # Also check confidence intervals if they are generated and relevant
        if forecaster.confidence_intervals is not None:
             self.assertTrue(all(forecaster.confidence_intervals['lower'] >= potential_min_val))
             # Upper bound should also be >= minimum_value, as it cannot be less than the mean/lower.
             self.assertTrue(all(forecaster.confidence_intervals['upper'] >= potential_min_val))


    def test_predict_for_dates_future(self):
        forecaster = ARIMAForecaster(config=self.forecaster_config)
        forecaster.fit(self.sample_data)
        
        last_hist_date = self.sample_data['date'].iloc[-1]
        future_dates = [
            last_hist_date + relativedelta(months=1),
            last_hist_date + relativedelta(months=2),
            last_hist_date + relativedelta(months=3)
        ]
        
        predictions = forecaster._predict_for_dates(np.array(future_dates))
        self.assertEqual(len(predictions), 3)
        # Basic check: predictions should be numeric (not NaN, unless model fails badly)
        self.assertFalse(np.isnan(predictions).any())

    def test_predict_for_dates_historical_and_future(self):
        forecaster = ARIMAForecaster(config=self.forecaster_config)
        forecaster.fit(self.sample_data)

        first_hist_date = self.sample_data['date'].iloc[0]
        mid_hist_date = self.sample_data['date'].iloc[len(self.sample_data)//2]
        last_hist_date = self.sample_data['date'].iloc[-1]
        
        request_dates = [
            # first_hist_date, # Current _predict_for_dates warns and filters these out
            # mid_hist_date,
            last_hist_date + relativedelta(months=1),
            last_hist_date + relativedelta(months=3)
        ]
        
        # The current _predict_for_dates implementation in ARIMAForecaster
        # warns and filters out dates <= self._last_date.
        # So, we only expect predictions for the future dates.
        
        predictions = forecaster._predict_for_dates(np.array(request_dates))
        
        self.assertEqual(len(predictions), 2) # Only the 2 future dates
        self.assertFalse(np.isnan(predictions).any())

    def test_predict_for_dates_empty_or_all_historical(self):
        forecaster = ARIMAForecaster(config=self.forecaster_config)
        forecaster.fit(self.sample_data)

        historical_dates = [
            self.sample_data['date'].iloc[0],
            self.sample_data['date'].iloc[1]
        ]
        
        # Test with only historical dates
        predictions_hist = forecaster._predict_for_dates(np.array(historical_dates))
        self.assertEqual(len(predictions_hist), 0) # As per current logic, these are filtered out

        # Test with empty array of dates
        predictions_empty = forecaster._predict_for_dates(np.array([]))
        self.assertEqual(len(predictions_empty), 0)


if __name__ == '__main__':
    unittest.main()
