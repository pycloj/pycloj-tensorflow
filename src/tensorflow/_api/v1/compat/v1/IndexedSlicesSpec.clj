(ns tensorflow.-api.v1.compat.v1.IndexedSlicesSpec
  "Type specification for a `tf.IndexedSlices`."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v1 (import-module "tensorflow._api.v1.compat.v1"))

(defn IndexedSlicesSpec 
  "Type specification for a `tf.IndexedSlices`."
  [shape & {:keys [dtype indices_dtype dense_shape_dtype indices_shape]
                       :or {dense_shape_dtype None indices_shape None}} ]
    (py/call-attr-kw v1 "IndexedSlicesSpec" [shape] {:dtype dtype :indices_dtype indices_dtype :dense_shape_dtype dense_shape_dtype :indices_shape indices_shape }))

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

(defn value-type 
  ""
  [ self ]
    (py/call-attr self "value_type"))
