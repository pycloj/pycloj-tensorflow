(ns tensorflow.contrib.labeled-tensor.FixedLenFeature
  "Configuration for parsing a fixed-length input feature.

  Fields:
    axes: A list of Axis objects or tuples (axis_name, axis_value),
      where `axis_name` is a string and `axis_value` is None (unknown size), an
      integer or a list of tick labels.
    dtype: Data type of input.
    default_value: Value to be used if an example is missing this feature. It
        must be compatible with `dtype`.
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

(defn FixedLenFeature 
  "Configuration for parsing a fixed-length input feature.

  Fields:
    axes: A list of Axis objects or tuples (axis_name, axis_value),
      where `axis_name` is a string and `axis_value` is None (unknown size), an
      integer or a list of tick labels.
    dtype: Data type of input.
    default_value: Value to be used if an example is missing this feature. It
        must be compatible with `dtype`.
  "
  [ axes dtype default_value ]
  (py/call-attr labeled-tensor "FixedLenFeature"  axes dtype default_value ))

(defn axes 
  ""
  [ self ]
    (py/call-attr self "axes"))

(defn default-value 
  ""
  [ self ]
    (py/call-attr self "default_value"))

(defn dtype 
  ""
  [ self ]
    (py/call-attr self "dtype"))
