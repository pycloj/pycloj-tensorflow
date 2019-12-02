(ns tensorflow.contrib.timeseries.NumpyReader
  "A time series parser for feeding Numpy arrays to a `TimeSeriesInputFn`.

  Avoids embedding data in the graph as constants.
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
(defn NumpyReader 
  "A time series parser for feeding Numpy arrays to a `TimeSeriesInputFn`.

  Avoids embedding data in the graph as constants.
  "
  [data  & {:keys [read_num_records_hint]} ]
    (py/call-attr-kw timeseries "NumpyReader" [data] {:read_num_records_hint read_num_records_hint }))

(defn check-dataset-size 
  "Raise an error if the dataset is too small."
  [ self minimum_dataset_size ]
  (py/call-attr self "check_dataset_size"  self minimum_dataset_size ))

(defn read 
  "Returns a large chunk of the Numpy arrays for later re-chunking."
  [ self  ]
  (py/call-attr self "read"  self  ))

(defn read-full 
  "Returns `Tensor` versions of the full Numpy arrays."
  [ self  ]
  (py/call-attr self "read_full"  self  ))
