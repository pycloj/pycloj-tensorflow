(ns tensorflow.-api.v1.compat.v1.io.FixedLenFeature
  "Configuration for parsing a fixed-length input feature.

  To treat sparse input as dense, provide a `default_value`; otherwise,
  the parse functions will fail on any examples missing this feature.

  Fields:
    shape: Shape of input data.
    dtype: Data type of input.
    default_value: Value to be used if an example is missing this feature. It
        must be compatible with `dtype` and of the specified `shape`.
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

(defn FixedLenFeature 
  "Configuration for parsing a fixed-length input feature.

  To treat sparse input as dense, provide a `default_value`; otherwise,
  the parse functions will fail on any examples missing this feature.

  Fields:
    shape: Shape of input data.
    dtype: Data type of input.
    default_value: Value to be used if an example is missing this feature. It
        must be compatible with `dtype` and of the specified `shape`.
  "
  [ shape dtype default_value ]
  (py/call-attr io "FixedLenFeature"  shape dtype default_value ))

(defn default-value 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "default_value"))

(defn dtype 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "dtype"))

(defn shape 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "shape"))
