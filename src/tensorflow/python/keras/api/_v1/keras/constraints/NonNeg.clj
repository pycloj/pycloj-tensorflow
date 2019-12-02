(ns tensorflow.python.keras.api.-v1.keras.constraints.NonNeg
  "Constrains the weights to be non-negative.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce constraints (import-module "tensorflow.python.keras.api._v1.keras.constraints"))

(defn NonNeg 
  "Constrains the weights to be non-negative.
  "
  [  ]
  (py/call-attr constraints "NonNeg"  ))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
