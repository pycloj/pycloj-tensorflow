(ns tensorflow.SparseTensorValue
  "SparseTensorValue(indices, values, dense_shape)"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tensorflow (import-module "tensorflow"))

(defn SparseTensorValue 
  "SparseTensorValue(indices, values, dense_shape)"
  [ indices values dense_shape ]
  (py/call-attr tensorflow "SparseTensorValue"  indices values dense_shape ))

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
