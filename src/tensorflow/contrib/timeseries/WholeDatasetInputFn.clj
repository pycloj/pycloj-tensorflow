(ns tensorflow.contrib.timeseries.WholeDatasetInputFn
  "Supports passing a full time series to a model for evaluation/inference.

  Note that this `TimeSeriesInputFn` is not designed for high throughput, and
  should not be used for training. It allows for sequential evaluation on a full
  dataset (with sequential in-sample predictions), which then feeds naturally
  into `predict_continuation_input_fn` for making out-of-sample
  predictions. While this is useful for plotting and interactive use,
  `RandomWindowInputFn` is better suited to training and quantitative
  evaluation.
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

(defn WholeDatasetInputFn 
  "Supports passing a full time series to a model for evaluation/inference.

  Note that this `TimeSeriesInputFn` is not designed for high throughput, and
  should not be used for training. It allows for sequential evaluation on a full
  dataset (with sequential in-sample predictions), which then feeds naturally
  into `predict_continuation_input_fn` for making out-of-sample
  predictions. While this is useful for plotting and interactive use,
  `RandomWindowInputFn` is better suited to training and quantitative
  evaluation.
  "
  [ time_series_reader ]
  (py/call-attr timeseries "WholeDatasetInputFn"  time_series_reader ))

(defn create-batch 
  "A suitable `input_fn` for an `Estimator`'s `evaluate()`.

    Returns:
      A dictionary mapping feature names to `Tensors`, each shape
      prefixed by [1, data set size] (i.e. a batch size of 1).
    "
  [ self  ]
  (py/call-attr self "create_batch"  self  ))
