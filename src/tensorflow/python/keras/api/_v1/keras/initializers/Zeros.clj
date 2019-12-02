(ns tensorflow.python.keras.api.-v1.keras.initializers.zeros
  "Initializer that generates tensors initialized to 0."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce initializers (import-module "tensorflow.python.keras.api._v1.keras.initializers"))

(defn zeros 
  "Initializer that generates tensors initialized to 0."
  [ & {:keys [dtype]} ]
   (py/call-attr-kw initializers "zeros" [] {:dtype dtype }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
