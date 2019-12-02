(ns tensorflow.python.keras.api.-v1.keras.regularizers.Regularizer
  "Regularizer base class.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce regularizers (import-module "tensorflow.python.keras.api._v1.keras.regularizers"))

(defn Regularizer 
  "Regularizer base class.
  "
  [  ]
  (py/call-attr regularizers "Regularizer"  ))
