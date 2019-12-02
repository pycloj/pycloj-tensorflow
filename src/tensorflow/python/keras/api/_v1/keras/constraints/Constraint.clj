(ns tensorflow.python.keras.api.-v1.keras.constraints.Constraint
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce constraints (import-module "tensorflow.python.keras.api._v1.keras.constraints"))

(defn Constraint 
  ""
  [  ]
  (py/call-attr constraints "Constraint"  ))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
