(ns tensorflow.contrib.timeseries.RandomWindowInputFn
  "Wraps a `TimeSeriesReader` to create random batches of windows.

  Tensors are first collected into sequential windows (in a windowing queue
  created by `tf.compat.v1.train.batch`, based on the order returned from
  `time_series_reader`), then these windows are randomly batched (in a
  `RandomShuffleQueue`), the Tensors returned by `create_batch` having shapes
  prefixed by [`batch_size`, `window_size`].

  This `TimeSeriesInputFn` is useful for both training and quantitative
  evaluation (but be sure to run several epochs for sequential models such as
  `StructuralEnsembleRegressor` to completely flush stale state left over from
  training). For qualitative evaluation or when preparing for predictions, use
  `WholeDatasetInputFn`.
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

(defn RandomWindowInputFn 
  "Wraps a `TimeSeriesReader` to create random batches of windows.

  Tensors are first collected into sequential windows (in a windowing queue
  created by `tf.compat.v1.train.batch`, based on the order returned from
  `time_series_reader`), then these windows are randomly batched (in a
  `RandomShuffleQueue`), the Tensors returned by `create_batch` having shapes
  prefixed by [`batch_size`, `window_size`].

  This `TimeSeriesInputFn` is useful for both training and quantitative
  evaluation (but be sure to run several epochs for sequential models such as
  `StructuralEnsembleRegressor` to completely flush stale state left over from
  training). For qualitative evaluation or when preparing for predictions, use
  `WholeDatasetInputFn`.
  "
  [time_series_reader window_size batch_size & {:keys [queue_capacity_multiplier shuffle_min_after_dequeue_multiplier discard_out_of_order discard_consecutive_batches_limit jitter num_threads shuffle_seed]
                       :or {shuffle_seed None}} ]
    (py/call-attr-kw timeseries "RandomWindowInputFn" [time_series_reader window_size batch_size] {:queue_capacity_multiplier queue_capacity_multiplier :shuffle_min_after_dequeue_multiplier shuffle_min_after_dequeue_multiplier :discard_out_of_order discard_out_of_order :discard_consecutive_batches_limit discard_consecutive_batches_limit :jitter jitter :num_threads num_threads :shuffle_seed shuffle_seed }))

(defn create-batch 
  "Create queues to window and batch time series data.

    Returns:
      A dictionary of Tensors corresponding to the output of `self._reader`
      (from the `time_series_reader` constructor argument), each with shapes
      prefixed by [`batch_size`, `window_size`].
    "
  [ self  ]
  (py/call-attr self "create_batch"  self  ))
