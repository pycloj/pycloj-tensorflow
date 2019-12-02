(ns tensorflow.contrib.timeseries.TrainEvalFeatures
  "Feature names used during training and evaluation."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce timeseries (import-module "tensorflow.contrib.timeseries"))

(defn TrainEvalFeatures 
  "Feature names used during training and evaluation."
  [  ]
  (py/call-attr timeseries "TrainEvalFeatures"  ))
