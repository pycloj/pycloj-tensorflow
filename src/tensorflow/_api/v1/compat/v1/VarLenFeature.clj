(ns tensorflow.-api.v1.compat.v1.VarLenFeature
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
(defonce v1 (import-module "tensorflow._api.v1.compat.v1"))

(defn VarLenFeature 
  "Configuration for parsing a variable-length input feature.

  Fields:
    dtype: Data type of input.
  "
  [ dtype ]
  (py/call-attr v1 "VarLenFeature"  dtype ))

(defn dtype 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "dtype"))
