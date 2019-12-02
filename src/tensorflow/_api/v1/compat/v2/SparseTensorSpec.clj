(ns tensorflow.-api.v1.compat.v2.SparseTensorSpec
  "Type specification for a `tf.SparseTensor`."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v2 (import-module "tensorflow._api.v1.compat.v2"))
(defn SparseTensorSpec 
  "Type specification for a `tf.SparseTensor`."
  [shape  & {:keys [dtype]} ]
    (py/call-attr-kw v2 "SparseTensorSpec" [shape] {:dtype dtype }))

(defn dtype 
  "The `tf.dtypes.DType` specified by this type for the SparseTensor."
  [ self ]
    (py/call-attr self "dtype"))

(defn is-compatible-with 
  "Returns true if `spec_or_value` is compatible with this TypeSpec."
  [ self spec_or_value ]
  (py/call-attr self "is_compatible_with"  self spec_or_value ))

(defn most-specific-compatible-type 
  "Returns the most specific TypeSpec compatible with `self` and `other`.

    Args:
      other: A `TypeSpec`.

    Raises:
      ValueError: If there is no TypeSpec that is compatible with both `self`
        and `other`.
    "
  [ self other ]
  (py/call-attr self "most_specific_compatible_type"  self other ))

(defn shape 
  "The `tf.TensorShape` specified by this type for the SparseTensor."
  [ self ]
    (py/call-attr self "shape"))

(defn value-type 
  ""
  [ self ]
    (py/call-attr self "value_type"))
