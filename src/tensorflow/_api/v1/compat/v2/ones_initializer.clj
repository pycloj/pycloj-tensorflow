(ns tensorflow.-api.v1.compat.v2.ones-initializer
  "Initializer that generates tensors initialized to 1."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v2 (import-module "tensorflow._api.v1.compat.v2"))

(defn ones-initializer 
  "Initializer that generates tensors initialized to 1."
  [  ]
  (py/call-attr v2 "ones_initializer"  ))

(defn get-config 
  "Returns the configuration of the initializer as a JSON-serializable dict.

    Returns:
      A JSON-serializable Python dict.
    "
  [ self  ]
  (py/call-attr self "get_config"  self  ))
