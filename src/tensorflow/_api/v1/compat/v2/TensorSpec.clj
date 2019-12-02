(ns tensorflow.-api.v1.compat.v2.TensorSpec
  "Describes a tf.Tensor.

  Metadata for describing the `tf.Tensor` objects accepted or returned
  by some TensorFlow APIs.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v2 (import-module "tensorflow._api.v1.compat.v2"))

(defn TensorSpec 
  "Describes a tf.Tensor.

  Metadata for describing the `tf.Tensor` objects accepted or returned
  by some TensorFlow APIs.
  "
  [shape & {:keys [dtype name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "TensorSpec" [shape] {:dtype dtype :name name }))

(defn dtype 
  "Returns the `dtype` of elements in the tensor."
  [ self ]
    (py/call-attr self "dtype"))

(defn is-compatible-with 
  "Returns True if spec_or_tensor is compatible with this TensorSpec.

    Two tensors are considered compatible if they have the same dtype
    and their shapes are compatible (see `tf.TensorShape.is_compatible_with`).

    Args:
      spec_or_tensor: A tf.TensorSpec or a tf.Tensor

    Returns:
      True if spec_or_tensor is compatible with self.
    "
  [ self spec_or_tensor ]
  (py/call-attr self "is_compatible_with"  self spec_or_tensor ))

(defn most-specific-compatible-type 
  ""
  [ self other ]
  (py/call-attr self "most_specific_compatible_type"  self other ))

(defn name 
  "Returns the (optionally provided) name of the described tensor."
  [ self ]
    (py/call-attr self "name"))

(defn shape 
  "Returns the `TensorShape` that represents the shape of the tensor."
  [ self ]
    (py/call-attr self "shape"))

(defn value-type 
  ""
  [ self ]
    (py/call-attr self "value_type"))
