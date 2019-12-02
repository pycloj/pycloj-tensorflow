(ns tensorflow.contrib.timeseries.FilteringResults
  "Keys returned from evaluation/filtering."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce timeseries (import-module "tensorflow.contrib.timeseries"))

(defn FilteringResults 
  "Keys returned from evaluation/filtering."
  [  ]
  (py/call-attr timeseries "FilteringResults"  ))
