(ns tensorflow.contrib.learn.PredictionKey
  "THIS CLASS IS DEPRECATED."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce learn (import-module "tensorflow.contrib.learn"))

(defn PredictionKey 
  "THIS CLASS IS DEPRECATED."
  [  ]
  (py/call-attr learn "PredictionKey"  ))
