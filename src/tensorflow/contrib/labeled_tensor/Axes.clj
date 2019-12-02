(ns tensorflow.contrib.labeled-tensor.Axes
  "Axis names and indices for a tensor.

  It is an ordered mapping, with keys given by axis name and values given
  by Axis objects. Duplicate axis names are not allowed.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce labeled-tensor (import-module "tensorflow.contrib.labeled_tensor"))

(defn Axes 
  "Axis names and indices for a tensor.

  It is an ordered mapping, with keys given by axis name and values given
  by Axis objects. Duplicate axis names are not allowed.
  "
  [ axes ]
  (py/call-attr labeled-tensor "Axes"  axes ))

(defn get 
  "D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None."
  [ self key default ]
  (py/call-attr self "get"  self key default ))

(defn items 
  "D.items() -> a set-like object providing a view on D's items"
  [ self  ]
  (py/call-attr self "items"  self  ))

(defn keys 
  "D.keys() -> a set-like object providing a view on D's keys"
  [ self  ]
  (py/call-attr self "keys"  self  ))

(defn remove 
  "Creates a new Axes object without the given axis."
  [ self axis_name ]
  (py/call-attr self "remove"  self axis_name ))

(defn values 
  "D.values() -> an object providing a view on D's values"
  [ self  ]
  (py/call-attr self "values"  self  ))
