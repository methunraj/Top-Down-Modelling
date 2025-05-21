import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta # For date calculations

from src.global_forecasting.prophet_forecaster import ProphetForecaster
from src.global_forecasting.base_forecaster import BaseForecaster
from prophet import Prophet # Import the Prophet class for type checking

class TestProphetForecaster(unittest.TestCase):

    def setUp(self):
        self.sample_data = pd.DataFrame({
            'date': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01',
                                     '2020-06-01', '2020-07-01', '2020-08-01', '2020-09-01', '2020-10-01',
                                     '2020-11-01', '2020-12-01', '2021-01-01', '2021-02-01', '2021-03-01']), # Monthly data
            'value': [10, 12, 13, 15, 16, 18, 19, 20, 22, 23, 25, 26, 28, 30, 31]
        })
        # Default config for Prophet, keep tests simple
        self.forecaster_config = {
            'yearly_seasonality': False,
            'weekly_seasonality': False,
            'daily_seasonality': False,
            'growth': 'linear' # Default but explicit for clarity
        }
        self.holidays_df = pd.DataFrame({
            'holiday': 'test_event',
            'ds': pd.to_datetime(['2021-01-01', '2020-04-01']), # One historical, one for potential forecast period
            'lower_window': 0,
            'upper_window': 0 # Simple point event
        })


    def test_forecaster_creation(self):
        forecaster_no_config = ProphetForecaster()
        self.assertIsInstance(forecaster_no_config, ProphetForecaster)
        self.assertIsInstance(forecaster_no_config, BaseForecaster)
        # Check some default Prophet params if not overridden by BaseForecaster's defaults
        self.assertEqual(forecaster_no_config.growth, 'linear') 

        forecaster_with_config = ProphetForecaster(config=self.forecaster_config)
        self.assertIsInstance(forecaster_with_config, ProphetForecaster)
        self.assertEqual(forecaster_with_config.yearly_seasonality, False)

    def test_fit_model(self):
        forecaster = ProphetForecaster(config=self.forecaster_config)
        forecaster.fit(self.sample_data)
        self.assertTrue(forecaster.fitted)
        self.assertIsNotNone(forecaster.model) # Prophet model instance
        self.assertIsInstance(forecaster.model, Prophet)

    def test_forecast_generation(self):
        forecaster = ProphetForecaster(config=self.forecaster_config)
        forecaster.fit(self.sample_data)
        
        periods_to_forecast = 3
        # Prophet uses 'MS' (Month Start) for monthly frequency
        forecast_df = forecaster.forecast(periods=periods_to_forecast, frequency='M')

        self.assertIsInstance(forecast_df, pd.DataFrame)
        self.assertEqual(len(forecast_df), periods_to_forecast)
        self.assertListEqual(list(forecast_df.columns), ['date', 'value'])

        self.assertIsInstance(forecaster.confidence_intervals, pd.DataFrame)
        self.assertEqual(len(forecaster.confidence_intervals), periods_to_forecast)
        self.assertListEqual(list(forecaster.confidence_intervals.columns), ['date', 'lower', 'value', 'upper'])
        
        self.assertEqual(len(forecaster.forecast_dates), periods_to_forecast)
        
        last_historical_date = self.sample_data['date'].max()
        for forecast_date in forecaster.forecast_dates:
            self.assertIsInstance(forecast_date, datetime) # Prophet returns them as python datetimes
            self.assertTrue(forecast_date > last_historical_date)

        # Check frequency of forecasted dates (should be monthly)
        expected_date = last_historical_date
        for f_date in forecaster.forecast_dates:
            # For Prophet's 'MS' frequency, the next date is the start of the next month
            expected_date = (expected_date + relativedelta(months=1)).replace(day=1)
            self.assertEqual(f_date, expected_date)


    def test_ensure_minimum_value(self):
        forecaster = ProphetForecaster(config=self.forecaster_config)
        forecaster.fit(self.sample_data)
        
        # Generate an initial forecast to see what values we get
        initial_forecast_df = forecaster.forecast(periods=3, frequency='M')
        
        if not initial_forecast_df.empty:
            # Set minimum to be slightly above the first forecasted value to ensure capping occurs
            potential_min_val = initial_forecast_df['value'].iloc[0] + 1.0 
        else: # Should not happen
            potential_min_val = 0.0

        forecaster.ensure_minimum = True
        forecaster.minimum_value = potential_min_val
        
        # Regenerate forecast with the minimum value constraint
        # Note: Prophet's own forecast doesn't have a floor. Our wrapper applies it.
        capped_forecast_df = forecaster.forecast(periods=3, frequency='M')
        
        self.assertTrue(all(capped_forecast_df['value'] >= potential_min_val))
        if forecaster.confidence_intervals is not None:
             self.assertTrue(all(forecaster.confidence_intervals['lower'] >= potential_min_val))
             # Upper bound should be >= the capped mean forecast value.
             # It's not directly capped by potential_min_val unless the mean itself is that low.
             self.assertTrue(all(forecaster.confidence_intervals['upper'] >= forecaster.confidence_intervals['value']))
             # And ensure value is indeed capped
             self.assertTrue(all(forecaster.confidence_intervals['value'] >= potential_min_val))


    def test_with_holidays(self):
        holiday_config = self.forecaster_config.copy()
        holiday_config['holidays'] = self.holidays_df
        
        forecaster = ProphetForecaster(config=holiday_config)
        # Check if holidays were correctly processed
        self.assertIsNotNone(forecaster.holidays)
        self.assertEqual(len(forecaster.holidays), 2)
        
        try:
            forecaster.fit(self.sample_data)
            self.assertTrue(forecaster.fitted)
            self.assertIsNotNone(forecaster.model)
            self.assertIsNotNone(forecaster.model.holidays) # Check if Prophet model itself registered holidays

            periods_to_forecast = 3
            forecast_df = forecaster.forecast(periods=periods_to_forecast, frequency='M')
            self.assertEqual(len(forecast_df), periods_to_forecast)
            # Further checks could involve comparing forecasts with/without holidays,
            # but that's more complex and depends on holiday impact.
            # For now, just ensure it runs and incorporates the holiday parameter.
        except Exception as e:
            self.fail(f"Test with holidays failed during fit/forecast: {e}")

    def test_predict_for_dates_future(self):
        forecaster = ProphetForecaster(config=self.forecaster_config)
        forecaster.fit(self.sample_data)
        
        last_hist_date = self.sample_data['date'].max()
        future_dates = [
            (last_hist_date + relativedelta(months=1)).replace(day=1),
            (last_hist_date + relativedelta(months=2)).replace(day=1),
            (last_hist_date + relativedelta(months=3)).replace(day=1)
        ]
        
        predictions = forecaster._predict_for_dates(np.array(future_dates))
        self.assertEqual(len(predictions), 3)
        self.assertFalse(np.isnan(predictions).any())

    def test_predict_for_dates_historical_and_future(self):
        forecaster = ProphetForecaster(config=self.forecaster_config)
        forecaster.fit(self.sample_data)

        first_hist_date = self.sample_data['date'].min()
        mid_hist_date = self.sample_data['date'].iloc[len(self.sample_data)//2]
        last_hist_date = self.sample_data['date'].max()
        
        request_dates = [
            first_hist_date, 
            mid_hist_date,
            (last_hist_date + relativedelta(months=1)).replace(day=1),
            (last_hist_date + relativedelta(months=3)).replace(day=1)
        ]
        
        # _predict_for_dates in ProphetForecaster should handle both historical and future dates
        # by calling model.predict() which can predict for any 'ds' values.
        predictions = forecaster._predict_for_dates(np.array(request_dates))
        
        self.assertEqual(len(predictions), 4) 
        self.assertFalse(np.isnan(predictions).any())
        
        # Optional: Check if historical predictions are somewhat close to actuals
        # This can be brittle, but for a simple linear growth it might be reasonable.
        # For example, prediction for first_hist_date should be close to self.sample_data['value'].iloc[0]
        # This depends on Prophet's fitting for in-sample dates.
        # self.assertAlmostEqual(predictions[0], self.sample_data['value'].iloc[0], delta=1.0) # Example, adjust delta

    def test_predict_for_dates_empty(self):
        forecaster = ProphetForecaster(config=self.forecaster_config)
        forecaster.fit(self.sample_data)

        # Test with empty array of dates
        predictions_empty = forecaster._predict_for_dates(np.array([]))
        self.assertEqual(len(predictions_empty), 0)


if __name__ == '__main__':
    unittest.main()
