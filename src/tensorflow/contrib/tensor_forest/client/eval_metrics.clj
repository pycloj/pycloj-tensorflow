(ns tensorflow.contrib.tensor-forest.client.eval-metrics
  "A collection of functions to be used as evaluation metrics."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce eval-metrics (import-module "tensorflow.contrib.tensor_forest.client.eval_metrics"))

(defn get-metric 
  "Given a metric name, return the corresponding metric function."
  [ metric_name ]
  (py/call-attr eval-metrics "get_metric"  metric_name ))

(defn get-prediction-key 
  ""
  [ metric_name ]
  (py/call-attr eval-metrics "get_prediction_key"  metric_name ))
