(ns tensorflow.-api.v1.compat.v1.TensorArraySpec
  "Type specification for a `tf.TensorArray`."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v1 (import-module "tensorflow._api.v1.compat.v1"))
(defn TensorArraySpec 
  "Type specification for a `tf.TensorArray`."
  [element_shape  & {:keys [dtype dynamic_size infer_shape]} ]
    (py/call-attr-kw v1 "TensorArraySpec" [element_shape] {:dtype dtype :dynamic_size dynamic_size :infer_shape infer_shape }))

(defn from-value 
  ""
  [ self value ]
  (py/call-attr self "from_value"  self value ))

(defn is-compatible-with 
  ""
  [ self other ]
  (py/call-attr self "is_compatible_with"  self other ))

(defn most-specific-compatible-type 
  ""
  [ self other ]
  (py/call-attr self "most_specific_compatible_type"  self other ))

(defn value-type 
  ""
  [ self ]
    (py/call-attr self "value_type"))
