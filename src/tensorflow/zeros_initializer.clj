(ns tensorflow.zeros-initializer
  "Initializer that generates tensors initialized to 0."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tensorflow (import-module "tensorflow"))

(defn zeros-initializer 
  "Initializer that generates tensors initialized to 0."
  [ & {:keys [dtype]} ]
   (py/call-attr-kw tensorflow "zeros_initializer" [] {:dtype dtype }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
