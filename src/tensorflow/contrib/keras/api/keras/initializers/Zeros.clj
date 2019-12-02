(ns tensorflow.contrib.keras.api.keras.initializers.Zeros
  "Initializer that generates tensors initialized to 0."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce initializers (import-module "tensorflow.contrib.keras.api.keras.initializers"))

(defn Zeros 
  "Initializer that generates tensors initialized to 0."
  [ & {:keys [dtype]} ]
   (py/call-attr-kw initializers "Zeros" [] {:dtype dtype }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
