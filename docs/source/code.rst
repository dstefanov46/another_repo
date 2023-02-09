Code
=====

.. _modules:

Modules
-------

Database Utility Functions (module_name: DatabaseUtils)
~~~~~~

.. autofunction:: module_name.read_sql
.. autofunction:: module_name.get_rt_pv_data
.. autofunction:: module_name.get_pv_meta
.. autofunction:: module_name.get_weather_history
.. autofunction:: module_name.check_weather_forecast
.. autofunction:: module_name.get_weather_forecast_all_raw
.. autofunction:: module_name.get_weather_forecast_all
.. autofunction:: module_name.get_ss_agg
.. autofunction:: module_name.get_slo_agg
.. autofunction:: module_name.write_rt_pv_data
.. autofunction:: module_name.write_rt_pv_forecast
.. autofunction:: module_name.write_ss_agg_forecast
.. autofunction:: module_name.write_slo_agg_forecast
.. autofunction:: module_name.write_ss_agg_clean
.. autofunction:: module_name.write_slo_agg_clean
.. autofunction:: module_name.write_wforecast_evaluation
.. autofunction:: module_name.get_weather_data

Data Preparation Functions (module_name: DataPrep)
~~~~~~~~~~~

.. autofunction:: module_name.mask_night_hours
.. autofunction:: module_name.get_interactions
.. autofunction:: module_name.get_X
.. autofunction:: module_name.imput_missing_values
.. autofunction:: module_name.imput_missing_values_all
.. autofunction:: module_name.imput_train
.. autofunction:: module_name.imput_inference
.. autofunction:: module_name.get_ss_agg
.. autofunction:: module_name.get_slo_agg
.. autofunction:: module_name.create_base_addresses
.. autofunction:: module_name.create_requests
.. autofunction:: module_name.get_coordinates
.. autofunction:: module_name.add_weather_station


Forecasting Functions (module_name: Forecasting)
~~~~~~

.. autofunction:: module_name.create_ts_idx
.. autofunction:: module_name.mask_night_hours
.. autofunction:: module_name.get_X_longterm
.. autofunction:: module_name.get_X_shortterm
.. autofunction:: module_name.get_forecasts
.. autofunction:: module_name.inference
.. autofunction:: module_name.get_ss_agg_forecasts
.. autofunction:: module_name.get_slo_agg_forecasts
.. autofunction:: module_name.add_PI_ss
.. autofunction:: module_name.add_PI_slo


Evaluation Functions (module_name: Evaluation)
~~~~~~

.. autofunction:: module_name.evaluate_ss_forecasts
.. autofunction:: module_name.evaluate_slo_forecasts
.. autofunction:: module_name.calc_errors
.. autofunction:: module_name.calc_and_write_errors


.. _run:

Run
-------

The file for running the PV Forecaster application in real-time is *RunForecaster.py*.


