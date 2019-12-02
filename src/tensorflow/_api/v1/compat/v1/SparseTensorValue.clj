(ns tensorflow.-api.v1.compat.v1.SparseTensorValue
  "SparseTensorValue(indices, values, dense_shape)"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v1 (import-module "tensorflow._api.v1.compat.v1"))

(defn SparseTensorValue 
  "SparseTensorValue(indices, values, dense_shape)"
  [ indices values dense_shape ]
  (py/call-attr v1 "SparseTensorValue"  indices values dense_shape ))

(defn dense-shape 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "dense_shape"))

(defn indices 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "indices"))

(defn values 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "values"))
