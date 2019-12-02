(ns tensorflow.-api.v1.compat.v2.data.experimental.Optional
  "Wraps a value that may/may not be present at runtime.

  An `Optional` can represent the result of an operation that may fail as a
  value, rather than raising an exception and halting execution. For example,
  `tf.data.experimental.get_next_as_optional` returns an `Optional` that either
  contains the next value from a `tf.compat.v1.data.Iterator` if one exists, or
  a \"none\" value that indicates the end of the sequence has been reached.

  `Optional` can only be used by values that are convertible to `Tensor` or
  `CompositeTensor`.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.compat.v2.data.experimental"))

(defn Optional 
  "Wraps a value that may/may not be present at runtime.

  An `Optional` can represent the result of an operation that may fail as a
  value, rather than raising an exception and halting execution. For example,
  `tf.data.experimental.get_next_as_optional` returns an `Optional` that either
  contains the next value from a `tf.compat.v1.data.Iterator` if one exists, or
  a \"none\" value that indicates the end of the sequence has been reached.

  `Optional` can only be used by values that are convertible to `Tensor` or
  `CompositeTensor`.
  "
  [  ]
  (py/call-attr experimental "Optional"  ))

(defn from-value 
  "Returns an `Optional` that wraps the given value.

    Args:
      value: A value to wrap. The value must be convertible to `Tensor` or
        `CompositeTensor`.

    Returns:
      An `Optional` that wraps `value`.
    "
  [ self value ]
  (py/call-attr self "from_value"  self value ))

(defn get-value 
  "Returns the value wrapped by this optional.

    If this optional does not have a value (i.e. `self.has_value()` evaluates
    to `False`), this operation will raise `tf.errors.InvalidArgumentError`
    at runtime.

    Args:
      name: (Optional.) A name for the created operation.

    Returns:
      The wrapped value.
    "
  [ self name ]
  (py/call-attr self "get_value"  self name ))

(defn has-value 
  "Returns a tensor that evaluates to `True` if this optional has a value.

    Args:
      name: (Optional.) A name for the created operation.

    Returns:
      A scalar `tf.Tensor` of type `tf.bool`.
    "
  [ self name ]
  (py/call-attr self "has_value"  self name ))

(defn none-from-structure 
  "Returns an `Optional` that has no value.

    NOTE: This method takes an argument that defines the structure of the value
    that would be contained in the returned `Optional` if it had a value.

    Args:
      value_structure: A `Structure` object representing the structure of the
        components of this optional.

    Returns:
      An `Optional` that has no value.
    "
  [ self value_structure ]
  (py/call-attr self "none_from_structure"  self value_structure ))

(defn value-structure 
  "The structure of the components of this optional.

    Returns:
      A `Structure` object representing the structure of the components of this
        optional.
    "
  [ self ]
    (py/call-attr self "value_structure"))
