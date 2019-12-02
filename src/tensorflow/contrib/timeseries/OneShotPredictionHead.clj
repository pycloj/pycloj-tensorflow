(ns tensorflow.contrib.timeseries.OneShotPredictionHead
  "A time series head which exports a single stateless serving signature.

  The serving default signature exported by this head expects `times`, `values`,
  and any exogenous features, but no state. `values` has shape `[batch_size,
  filter_length, num_features]` and `times` has shape `[batch_size,
  total_length]`, where `total_length > filter_length`. Any exogenous features
  must have their shapes prefixed by the shape of the `times` feature.

  When serving, first performs filtering on the series up to `filter_length`
  starting from the default start state for the model, then computes predictions
  on the remainder of the series, returning them.

  Model state is neither accepted nor returned, so filtering must be performed
  each time predictions are requested when using this head.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce timeseries (import-module "tensorflow.contrib.timeseries"))

(defn OneShotPredictionHead 
  "A time series head which exports a single stateless serving signature.

  The serving default signature exported by this head expects `times`, `values`,
  and any exogenous features, but no state. `values` has shape `[batch_size,
  filter_length, num_features]` and `times` has shape `[batch_size,
  total_length]`, where `total_length > filter_length`. Any exogenous features
  must have their shapes prefixed by the shape of the `times` feature.

  When serving, first performs filtering on the series up to `filter_length`
  starting from the default start state for the model, then computes predictions
  on the remainder of the series, returning them.

  Model state is neither accepted nor returned, so filtering must be performed
  each time predictions are requested when using this head.
  "
  [ model state_manager optimizer input_statistics_generator name ]
  (py/call-attr timeseries "OneShotPredictionHead"  model state_manager optimizer input_statistics_generator name ))

(defn create-estimator-spec 
  "Performs basic error checking and returns an EstimatorSpec."
  [ self features mode labels ]
  (py/call-attr self "create_estimator_spec"  self features mode labels ))

(defn create-loss 
  "See `_Head`."
  [ self features mode logits labels ]
  (py/call-attr self "create_loss"  self features mode logits labels ))

(defn logits-dimension 
  "See `_Head`."
  [ self ]
    (py/call-attr self "logits_dimension"))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))
