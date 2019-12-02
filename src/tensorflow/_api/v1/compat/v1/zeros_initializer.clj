(ns tensorflow.-api.v1.compat.v1.zeros-initializer
  "Initializer that generates tensors initialized to 0."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v1 (import-module "tensorflow._api.v1.compat.v1"))

(defn zeros-initializer 
  "Initializer that generates tensors initialized to 0."
  [ & {:keys [dtype]} ]
   (py/call-attr-kw v1 "zeros_initializer" [] {:dtype dtype }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
