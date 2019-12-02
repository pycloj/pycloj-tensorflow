(ns tensorflow.-api.v1.compat.v1.ones-initializer
  "Initializer that generates tensors initialized to 1."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v1 (import-module "tensorflow._api.v1.compat.v1"))

(defn ones-initializer 
  "Initializer that generates tensors initialized to 1."
  [ & {:keys [dtype]} ]
   (py/call-attr-kw v1 "ones_initializer" [] {:dtype dtype }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
