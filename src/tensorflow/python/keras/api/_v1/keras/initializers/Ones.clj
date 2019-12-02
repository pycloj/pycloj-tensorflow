(ns tensorflow.python.keras.api.-v1.keras.initializers.ones
  "Initializer that generates tensors initialized to 1."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce initializers (import-module "tensorflow.python.keras.api._v1.keras.initializers"))

(defn ones 
  "Initializer that generates tensors initialized to 1."
  [ & {:keys [dtype]} ]
   (py/call-attr-kw initializers "ones" [] {:dtype dtype }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
