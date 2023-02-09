Code
=====

.. _modules:

Modules
-------

Database Utility Functions
~~~~~~

.. autofunction:: read_sql
.. autofunction:: get_rt_pv_data
.. autofunction:: pv_forecaster_example.get_pv_meta
.. autofunction:: pv_forecaster_example.get_weather_history
.. autofunction:: pv_forecaster_example.check_weather_forecast
.. autofunction:: pv_forecaster_example.get_weather_forecast_all_raw
.. autofunction:: pv_forecaster_example.get_weather_forecast_all
.. autofunction:: pv_forecaster_example.get_ss_agg
.. autofunction:: pv_forecaster_example.get_slo_agg
.. autofunction:: pv_forecaster_example.write_rt_pv_data
.. autofunction:: pv_forecaster_example.write_rt_pv_forecast
.. autofunction:: pv_forecaster_example.write_ss_agg_forecast
.. autofunction:: pv_forecaster_example.write_slo_agg_forecast
.. autofunction:: pv_forecaster_example.write_ss_agg_clean
.. autofunction:: pv_forecaster_example.write_slo_agg_clean
.. autofunction:: pv_forecaster_example.write_wforecast_evaluation
.. autofunction:: pv_forecaster_example.get_weather_data

Data Preparation Functions
~~~~~~~~~~~

.. autofunction:: pv_forecaster_example.mask_night_hours
.. autofunction:: pv_forecaster_example.get_interactions
.. autofunction:: pv_forecaster_example.get_X
.. autofunction:: pv_forecaster_example.imput_missing_values
.. autofunction:: pv_forecaster_example.imput_missing_values_all
.. autofunction:: pv_forecaster_example.imput_train
.. autofunction:: pv_forecaster_example.imput_inference
.. autofunction:: pv_forecaster_example.get_ss_agg
.. autofunction:: pv_forecaster_example.get_slo_agg
.. autofunction:: pv_forecaster_example.create_base_addresses
.. autofunction:: pv_forecaster_example.create_requests
.. autofunction:: pv_forecaster_example.get_coordinates
.. autofunction:: pv_forecaster_example.add_weather_station


Forecasting Functions
~~~~~~

.. autofunction:: pv_forecaster_example.create_ts_idx
.. autofunction:: pv_forecaster_example.mask_night_hours
.. autofunction:: pv_forecaster_example.get_X_longterm
.. autofunction:: pv_forecaster_example.get_X_shortterm
.. autofunction:: pv_forecaster_example.get_forecasts
.. autofunction:: pv_forecaster_example.inference
.. autofunction:: pv_forecaster_example.get_ss_agg_forecasts
.. autofunction:: pv_forecaster_example.get_slo_agg_forecasts
.. autofunction:: pv_forecaster_example.add_PI_ss
.. autofunction:: pv_forecaster_example.add_PI_slo


Evaluation Functions
~~~~~~

.. autofunction:: pv_forecaster_example.evaluate_ss_forecasts
.. autofunction:: pv_forecaster_example.evaluate_slo_forecasts
.. autofunction:: pv_forecaster_example.calc_errors
.. autofunction:: pv_forecaster_example.calc_and_write_errors


.. _run:

Run
-------

The file for running the PV Forecaster application in real-time is *RunForecaster.py*.


