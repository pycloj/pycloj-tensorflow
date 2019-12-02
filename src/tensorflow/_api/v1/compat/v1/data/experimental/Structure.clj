(ns tensorflow.-api.v1.compat.v1.data.experimental.Structure
  "Specifies a TensorFlow value type.

  A `tf.TypeSpec` provides metadata describing an object accepted or returned
  by TensorFlow APIs.  Concrete subclasses, such as `tf.TensorSpec` and
  `tf.RaggedTensorSpec`, are used to describe different value types.

  For example, `tf.function`'s `input_signature` argument accepts a list
  (or nested structure) of `TypeSpec`s.

  Creating new subclasses of TypeSpec (outside of TensorFlow core) is not
  currently supported.  In particular, we may make breaking changes to the
  private methods and properties defined by this base class.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.compat.v1.data.experimental"))

(defn Structure 
  "Specifies a TensorFlow value type.

  A `tf.TypeSpec` provides metadata describing an object accepted or returned
  by TensorFlow APIs.  Concrete subclasses, such as `tf.TensorSpec` and
  `tf.RaggedTensorSpec`, are used to describe different value types.

  For example, `tf.function`'s `input_signature` argument accepts a list
  (or nested structure) of `TypeSpec`s.

  Creating new subclasses of TypeSpec (outside of TensorFlow core) is not
  currently supported.  In particular, we may make breaking changes to the
  private methods and properties defined by this base class.
  "
  [  ]
  (py/call-attr experimental "Structure"  ))

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
  "The Python type for values that are compatible with this TypeSpec."
  [ self ]
    (py/call-attr self "value_type"))
