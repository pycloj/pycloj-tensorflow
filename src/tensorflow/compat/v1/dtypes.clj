(ns tensorflow.-api.v1.compat.v1.dtypes
  "Public API for tf.dtypes namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce dtypes (import-module "tensorflow._api.v1.compat.v1.dtypes"))

(defn as-dtype 
  "Converts the given `type_value` to a `DType`.

  Args:
    type_value: A value that can be converted to a `tf.DType` object. This may
      currently be a `tf.DType` object, a [`DataType`
      enum](https://www.tensorflow.org/code/tensorflow/core/framework/types.proto),
        a string type name, or a `numpy.dtype`.

  Returns:
    A `DType` corresponding to `type_value`.

  Raises:
    TypeError: If `type_value` cannot be converted to a `DType`.
  "
  [ type_value ]
  (py/call-attr dtypes "as_dtype"  type_value ))

(defn as-string 
  "Converts each entry in the given tensor to strings.

  Supports many numeric types and boolean.

  For Unicode, see the
  [https://www.tensorflow.org/tutorials/representation/unicode](Working with Unicode text)
  tutorial.

  Args:
    input: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`, `float32`, `float64`, `bool`.
    precision: An optional `int`. Defaults to `-1`.
      The post-decimal precision to use for floating point numbers.
      Only used if precision > -1.
    scientific: An optional `bool`. Defaults to `False`.
      Use scientific notation for floating point numbers.
    shortest: An optional `bool`. Defaults to `False`.
      Use shortest representation (either scientific or standard) for
      floating point numbers.
    width: An optional `int`. Defaults to `-1`.
      Pad pre-decimal numbers to this width.
      Applies to both floating point and integer numbers.
      Only used if width > -1.
    fill: An optional `string`. Defaults to `\"\"`.
      The value to pad if width > -1.  If empty, pads with spaces.
      Another typical value is '0'.  String cannot be longer than 1 character.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  "
  [input & {:keys [precision scientific shortest width fill name]
                       :or {name None}} ]
    (py/call-attr-kw dtypes "as_string" [input] {:precision precision :scientific scientific :shortest shortest :width width :fill fill :name name }))

(defn cast 
  "Casts a tensor to a new type.

  The operation casts `x` (in case of `Tensor`) or `x.values`
  (in case of `SparseTensor` or `IndexedSlices`) to `dtype`.

  For example:

  ```python
  x = tf.constant([1.8, 2.2], dtype=tf.float32)
  tf.dtypes.cast(x, tf.int32)  # [1, 2], dtype=tf.int32
  ```

  The operation supports data types (for `x` and `dtype`) of
  `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`, `int64`,
  `float16`, `float32`, `float64`, `complex64`, `complex128`, `bfloat16`.
  In case of casting from complex types (`complex64`, `complex128`) to real
  types, only the real part of `x` is returned. In case of casting from real
  types to complex types (`complex64`, `complex128`), the imaginary part of the
  returned value is set to `0`. The handling of complex types here matches the
  behavior of numpy.

  Args:
    x: A `Tensor` or `SparseTensor` or `IndexedSlices` of numeric type. It could
      be `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `int32`,
      `int64`, `float16`, `float32`, `float64`, `complex64`, `complex128`,
      `bfloat16`.
    dtype: The destination type. The list of supported dtypes is the same as
      `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` or `IndexedSlices` with same shape as `x` and
      same type as `dtype`.

  Raises:
    TypeError: If `x` cannot be cast to the `dtype`.
  "
  [ x dtype name ]
  (py/call-attr dtypes "cast"  x dtype name ))

(defn complex 
  "Converts two real numbers to a complex number.

  Given a tensor `real` representing the real part of a complex number, and a
  tensor `imag` representing the imaginary part of a complex number, this
  operation returns complex numbers elementwise of the form \\(a + bj\\), where
  *a* represents the `real` part and *b* represents the `imag` part.

  The input tensors `real` and `imag` must have the same shape.

  For example:

  ```python
  real = tf.constant([2.25, 3.25])
  imag = tf.constant([4.75, 5.75])
  tf.complex(real, imag)  # [[2.25 + 4.75j], [3.25 + 5.75j]]
  ```

  Args:
    real: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    imag: A `Tensor`. Must have the same type as `real`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64` or `complex128`.

  Raises: 
    TypeError: Real and imag must be correct types
  "
  [ real imag name ]
  (py/call-attr dtypes "complex"  real imag name ))

(defn saturate-cast 
  "Performs a safe saturating cast of `value` to `dtype`.

  This function casts the input to `dtype` without applying any scaling.  If
  there is a danger that values would over or underflow in the cast, this op
  applies the appropriate clamping before the cast.

  Args:
    value: A `Tensor`.
    dtype: The desired output `DType`.
    name: A name for the operation (optional).

  Returns:
    `value` safely cast to `dtype`.
  "
  [ value dtype name ]
  (py/call-attr dtypes "saturate_cast"  value dtype name ))
