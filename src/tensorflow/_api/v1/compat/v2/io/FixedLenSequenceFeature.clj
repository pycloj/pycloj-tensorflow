(ns tensorflow.-api.v1.compat.v2.io.FixedLenSequenceFeature
  "Configuration for parsing a variable-length input feature into a `Tensor`.

  The resulting `Tensor` of parsing a single `SequenceExample` or `Example` has
  a static `shape` of `[None] + shape` and the specified `dtype`.
  The resulting `Tensor` of parsing a `batch_size` many `Example`s has
  a static `shape` of `[batch_size, None] + shape` and the specified `dtype`.
  The entries in the `batch` from different `Examples` will be padded with
  `default_value` to the maximum length present in the `batch`.

  To treat a sparse input as dense, provide `allow_missing=True`; otherwise,
  the parse functions will fail on any examples missing this feature.

  Fields:
    shape: Shape of input data for dimension 2 and higher. First dimension is
      of variable length `None`.
    dtype: Data type of input.
    allow_missing: Whether to allow this feature to be missing from a feature
      list item. Is available only for parsing `SequenceExample` not for
      parsing `Examples`.
    default_value: Scalar value to be used to pad multiple `Example`s to their
      maximum length. Irrelevant for parsing a single `Example` or
      `SequenceExample`. Defaults to \"\" for dtype string and 0 otherwise
      (optional).
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce io (import-module "tensorflow._api.v1.compat.v2.io"))

(defn FixedLenSequenceFeature 
  "Configuration for parsing a variable-length input feature into a `Tensor`.

  The resulting `Tensor` of parsing a single `SequenceExample` or `Example` has
  a static `shape` of `[None] + shape` and the specified `dtype`.
  The resulting `Tensor` of parsing a `batch_size` many `Example`s has
  a static `shape` of `[batch_size, None] + shape` and the specified `dtype`.
  The entries in the `batch` from different `Examples` will be padded with
  `default_value` to the maximum length present in the `batch`.

  To treat a sparse input as dense, provide `allow_missing=True`; otherwise,
  the parse functions will fail on any examples missing this feature.

  Fields:
    shape: Shape of input data for dimension 2 and higher. First dimension is
      of variable length `None`.
    dtype: Data type of input.
    allow_missing: Whether to allow this feature to be missing from a feature
      list item. Is available only for parsing `SequenceExample` not for
      parsing `Examples`.
    default_value: Scalar value to be used to pad multiple `Example`s to their
      maximum length. Irrelevant for parsing a single `Example` or
      `SequenceExample`. Defaults to \"\" for dtype string and 0 otherwise
      (optional).
  "
  [shape dtype & {:keys [allow_missing default_value]
                       :or {default_value None}} ]
    (py/call-attr-kw io "FixedLenSequenceFeature" [shape dtype] {:allow_missing allow_missing :default_value default_value }))

(defn allow-missing 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "allow_missing"))

(defn default-value 
  "Alias for field number 3"
  [ self ]
    (py/call-attr self "default_value"))

(defn dtype 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "dtype"))

(defn shape 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "shape"))
