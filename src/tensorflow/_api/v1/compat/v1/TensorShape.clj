(ns tensorflow.-api.v1.compat.v1.TensorShape
  "Represents the shape of a `Tensor`.

  A `TensorShape` represents a possibly-partial shape specification for a
  `Tensor`. It may be one of the following:

  * *Fully-known shape:* has a known number of dimensions and a known size
    for each dimension. e.g. `TensorShape([16, 256])`
  * *Partially-known shape:* has a known number of dimensions, and an unknown
    size for one or more dimension. e.g. `TensorShape([None, 256])`
  * *Unknown shape:* has an unknown number of dimensions, and an unknown
    size in all dimensions. e.g. `TensorShape(None)`

  If a tensor is produced by an operation of type `\"Foo\"`, its shape
  may be inferred if there is a registered shape function for
  `\"Foo\"`. See [Shape
  functions](https://tensorflow.org/extend/adding_an_op#shape_functions_in_c)
  for details of shape functions and how to register them. Alternatively,
  the shape may be set explicitly using `tf.Tensor.set_shape`.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v1 (import-module "tensorflow._api.v1.compat.v1"))

(defn TensorShape 
  "Represents the shape of a `Tensor`.

  A `TensorShape` represents a possibly-partial shape specification for a
  `Tensor`. It may be one of the following:

  * *Fully-known shape:* has a known number of dimensions and a known size
    for each dimension. e.g. `TensorShape([16, 256])`
  * *Partially-known shape:* has a known number of dimensions, and an unknown
    size for one or more dimension. e.g. `TensorShape([None, 256])`
  * *Unknown shape:* has an unknown number of dimensions, and an unknown
    size in all dimensions. e.g. `TensorShape(None)`

  If a tensor is produced by an operation of type `\"Foo\"`, its shape
  may be inferred if there is a registered shape function for
  `\"Foo\"`. See [Shape
  functions](https://tensorflow.org/extend/adding_an_op#shape_functions_in_c)
  for details of shape functions and how to register them. Alternatively,
  the shape may be set explicitly using `tf.Tensor.set_shape`.
  "
  [ dims ]
  (py/call-attr v1 "TensorShape"  dims ))

(defn as-list 
  "Returns a list of integers or `None` for each dimension.

    Returns:
      A list of integers or `None` for each dimension.

    Raises:
      ValueError: If `self` is an unknown shape with an unknown rank.
    "
  [ self  ]
  (py/call-attr self "as_list"  self  ))

(defn as-proto 
  "Returns this shape as a `TensorShapeProto`."
  [ self  ]
  (py/call-attr self "as_proto"  self  ))

(defn assert-has-rank 
  "Raises an exception if `self` is not compatible with the given `rank`.

    Args:
      rank: An integer.

    Raises:
      ValueError: If `self` does not represent a shape with the given `rank`.
    "
  [ self rank ]
  (py/call-attr self "assert_has_rank"  self rank ))

(defn assert-is-compatible-with 
  "Raises exception if `self` and `other` do not represent the same shape.

    This method can be used to assert that there exists a shape that both
    `self` and `other` represent.

    Args:
      other: Another TensorShape.

    Raises:
      ValueError: If `self` and `other` do not represent the same shape.
    "
  [ self other ]
  (py/call-attr self "assert_is_compatible_with"  self other ))

(defn assert-is-fully-defined 
  "Raises an exception if `self` is not fully defined in every dimension.

    Raises:
      ValueError: If `self` does not have a known value for every dimension.
    "
  [ self  ]
  (py/call-attr self "assert_is_fully_defined"  self  ))

(defn assert-same-rank 
  "Raises an exception if `self` and `other` do not have compatible ranks.

    Args:
      other: Another `TensorShape`.

    Raises:
      ValueError: If `self` and `other` do not represent shapes with the
        same rank.
    "
  [ self other ]
  (py/call-attr self "assert_same_rank"  self other ))

(defn concatenate 
  "Returns the concatenation of the dimension in `self` and `other`.

    *N.B.* If either `self` or `other` is completely unknown,
    concatenation will discard information about the other shape. In
    future, we might support concatenation that preserves this
    information for use with slicing.

    Args:
      other: Another `TensorShape`.

    Returns:
      A `TensorShape` whose dimensions are the concatenation of the
      dimensions in `self` and `other`.
    "
  [ self other ]
  (py/call-attr self "concatenate"  self other ))

(defn dims 
  "Returns a list of Dimensions, or None if the shape is unspecified."
  [ self ]
    (py/call-attr self "dims"))

(defn is-compatible-with 
  "Returns True iff `self` is compatible with `other`.

    Two possibly-partially-defined shapes are compatible if there
    exists a fully-defined shape that both shapes can represent. Thus,
    compatibility allows the shape inference code to reason about
    partially-defined shapes. For example:

    * TensorShape(None) is compatible with all shapes.

    * TensorShape([None, None]) is compatible with all two-dimensional
      shapes, such as TensorShape([32, 784]), and also TensorShape(None). It is
      not compatible with, for example, TensorShape([None]) or
      TensorShape([None, None, None]).

    * TensorShape([32, None]) is compatible with all two-dimensional shapes
      with size 32 in the 0th dimension, and also TensorShape([None, None])
      and TensorShape(None). It is not compatible with, for example,
      TensorShape([32]), TensorShape([32, None, 1]) or TensorShape([64, None]).

    * TensorShape([32, 784]) is compatible with itself, and also
      TensorShape([32, None]), TensorShape([None, 784]), TensorShape([None,
      None]) and TensorShape(None). It is not compatible with, for example,
      TensorShape([32, 1, 784]) or TensorShape([None]).

    The compatibility relation is reflexive and symmetric, but not
    transitive. For example, TensorShape([32, 784]) is compatible with
    TensorShape(None), and TensorShape(None) is compatible with
    TensorShape([4, 4]), but TensorShape([32, 784]) is not compatible with
    TensorShape([4, 4]).

    Args:
      other: Another TensorShape.

    Returns:
      True iff `self` is compatible with `other`.

    "
  [ self other ]
  (py/call-attr self "is_compatible_with"  self other ))

(defn is-fully-defined 
  "Returns True iff `self` is fully defined in every dimension."
  [ self  ]
  (py/call-attr self "is_fully_defined"  self  ))

(defn merge-with 
  "Returns a `TensorShape` combining the information in `self` and `other`.

    The dimensions in `self` and `other` are merged elementwise,
    according to the rules defined for `Dimension.merge_with()`.

    Args:
      other: Another `TensorShape`.

    Returns:
      A `TensorShape` containing the combined information of `self` and
      `other`.

    Raises:
      ValueError: If `self` and `other` are not compatible.
    "
  [ self other ]
  (py/call-attr self "merge_with"  self other ))

(defn most-specific-compatible-shape 
  "Returns the most specific TensorShape compatible with `self` and `other`.

    * TensorShape([None, 1]) is the most specific TensorShape compatible with
      both TensorShape([2, 1]) and TensorShape([5, 1]). Note that
      TensorShape(None) is also compatible with above mentioned TensorShapes.

    * TensorShape([1, 2, 3]) is the most specific TensorShape compatible with
      both TensorShape([1, 2, 3]) and TensorShape([1, 2, 3]). There are more
      less specific TensorShapes compatible with above mentioned TensorShapes,
      e.g. TensorShape([1, 2, None]), TensorShape(None).

    Args:
      other: Another `TensorShape`.

    Returns:
      A `TensorShape` which is the most specific compatible shape of `self`
      and `other`.
    "
  [ self other ]
  (py/call-attr self "most_specific_compatible_shape"  self other ))

(defn ndims 
  "Deprecated accessor for `rank`."
  [ self ]
    (py/call-attr self "ndims"))

(defn num-elements 
  "Returns the total number of elements, or none for incomplete shapes."
  [ self  ]
  (py/call-attr self "num_elements"  self  ))

(defn rank 
  "Returns the rank of this shape, or None if it is unspecified."
  [ self ]
    (py/call-attr self "rank"))

(defn with-rank 
  "Returns a shape based on `self` with the given rank.

    This method promotes a completely unknown shape to one with a
    known rank.

    Args:
      rank: An integer.

    Returns:
      A shape that is at least as specific as `self` with the given rank.

    Raises:
      ValueError: If `self` does not represent a shape with the given `rank`.
    "
  [ self rank ]
  (py/call-attr self "with_rank"  self rank ))

(defn with-rank-at-least 
  "Returns a shape based on `self` with at least the given rank.

    Args:
      rank: An integer.

    Returns:
      A shape that is at least as specific as `self` with at least the given
      rank.

    Raises:
      ValueError: If `self` does not represent a shape with at least the given
        `rank`.
    "
  [ self rank ]
  (py/call-attr self "with_rank_at_least"  self rank ))

(defn with-rank-at-most 
  "Returns a shape based on `self` with at most the given rank.

    Args:
      rank: An integer.

    Returns:
      A shape that is at least as specific as `self` with at most the given
      rank.

    Raises:
      ValueError: If `self` does not represent a shape with at most the given
        `rank`.
    "
  [ self rank ]
  (py/call-attr self "with_rank_at_most"  self rank ))
