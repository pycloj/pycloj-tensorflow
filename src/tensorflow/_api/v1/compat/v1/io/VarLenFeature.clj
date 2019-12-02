(ns tensorflow.-api.v1.compat.v1.io.VarLenFeature
  "Configuration for parsing a variable-length input feature.

  Fields:
    dtype: Data type of input.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce io (import-module "tensorflow._api.v1.compat.v1.io"))

(defn VarLenFeature 
  "Configuration for parsing a variable-length input feature.

  Fields:
    dtype: Data type of input.
  "
  [ dtype ]
  (py/call-attr io "VarLenFeature"  dtype ))

(defn dtype 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "dtype"))
