(ns tensorflow.-api.v1.compat.v2.data.DatasetSpec
  "Type specification for `tf.data.Dataset`."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce data (import-module "tensorflow._api.v1.compat.v2.data"))
(defn DatasetSpec 
  "Type specification for `tf.data.Dataset`."
  [element_spec  & {:keys [dataset_shape]} ]
    (py/call-attr-kw data "DatasetSpec" [element_spec] {:dataset_shape dataset_shape }))

(defn from-value 
  ""
  [ self value ]
  (py/call-attr self "from_value"  self value ))

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
