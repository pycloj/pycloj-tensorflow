(ns tensorflow.contrib.framework.BoundedTensorSpec
  "A `TensorSpec` that specifies minimum and maximum values.

  Example usage:
  ```python
  spec = tensor_spec.BoundedTensorSpec((1, 2, 3), tf.float32, 0, (5, 5, 5))
  tf_minimum = tf.convert_to_tensor(spec.minimum, dtype=spec.dtype)
  tf_maximum = tf.convert_to_tensor(spec.maximum, dtype=spec.dtype)
  ```

  Bounds are meant to be inclusive. This is especially important for
  integer types. The following spec will be satisfied by tensors
  with values in the set {0, 1, 2}:
  ```python
  spec = tensor_spec.BoundedTensorSpec((3, 5), tf.int32, 0, 2)
  ```
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce framework (import-module "tensorflow.contrib.framework"))

(defn BoundedTensorSpec 
  "A `TensorSpec` that specifies minimum and maximum values.

  Example usage:
  ```python
  spec = tensor_spec.BoundedTensorSpec((1, 2, 3), tf.float32, 0, (5, 5, 5))
  tf_minimum = tf.convert_to_tensor(spec.minimum, dtype=spec.dtype)
  tf_maximum = tf.convert_to_tensor(spec.maximum, dtype=spec.dtype)
  ```

  Bounds are meant to be inclusive. This is especially important for
  integer types. The following spec will be satisfied by tensors
  with values in the set {0, 1, 2}:
  ```python
  spec = tensor_spec.BoundedTensorSpec((3, 5), tf.int32, 0, 2)
  ```
  "
  [ shape dtype minimum maximum name ]
  (py/call-attr framework "BoundedTensorSpec"  shape dtype minimum maximum name ))

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

(defn maximum 
  "Returns a NumPy array specifying the maximum bounds (inclusive)."
  [ self ]
    (py/call-attr self "maximum"))

(defn minimum 
  "Returns a NumPy array specifying the minimum bounds (inclusive)."
  [ self ]
    (py/call-attr self "minimum"))

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
