(ns tensorflow.ones-initializer
  "Initializer that generates tensors initialized to 1."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tensorflow (import-module "tensorflow"))

(defn ones-initializer 
  "Initializer that generates tensors initialized to 1."
  [ & {:keys [dtype]} ]
   (py/call-attr-kw tensorflow "ones_initializer" [] {:dtype dtype }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
