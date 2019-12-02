(ns tensorflow.-api.v1.compat.v1.Dimension
  "Represents the value of one dimension in a TensorShape."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v1 (import-module "tensorflow._api.v1.compat.v1"))

(defn Dimension 
  "Represents the value of one dimension in a TensorShape."
  [ value ]
  (py/call-attr v1 "Dimension"  value ))

(defn assert-is-compatible-with 
  "Raises an exception if `other` is not compatible with this Dimension.

    Args:
      other: Another Dimension.

    Raises:
      ValueError: If `self` and `other` are not compatible (see
        is_compatible_with).
    "
  [ self other ]
  (py/call-attr self "assert_is_compatible_with"  self other ))

(defn is-compatible-with 
  "Returns true if `other` is compatible with this Dimension.

    Two known Dimensions are compatible if they have the same value.
    An unknown Dimension is compatible with all other Dimensions.

    Args:
      other: Another Dimension.

    Returns:
      True if this Dimension and `other` are compatible.
    "
  [ self other ]
  (py/call-attr self "is_compatible_with"  self other ))

(defn merge-with 
  "Returns a Dimension that combines the information in `self` and `other`.

    Dimensions are combined as follows:

    ```python
    tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(n))     ==
    tf.compat.v1.Dimension(n)
    tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(None))  ==
    tf.compat.v1.Dimension(n)
    tf.compat.v1.Dimension(None).merge_with(tf.compat.v1.Dimension(n))     ==
    tf.compat.v1.Dimension(n)
    # equivalent to tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None).merge_with(tf.compat.v1.Dimension(None))

    # raises ValueError for n != m
    tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(m))
    ```

    Args:
      other: Another Dimension.

    Returns:
      A Dimension containing the combined information of `self` and
      `other`.

    Raises:
      ValueError: If `self` and `other` are not compatible (see
        is_compatible_with).
    "
  [ self other ]
  (py/call-attr self "merge_with"  self other ))

(defn value 
  "The value of this dimension, or None if it is unknown."
  [ self ]
    (py/call-attr self "value"))
