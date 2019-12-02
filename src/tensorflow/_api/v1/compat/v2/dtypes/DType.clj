(ns tensorflow.-api.v1.compat.v2.dtypes.DType
  "Represents the type of the elements in a `Tensor`.

  The following `DType` objects are defined:

  * `tf.float16`: 16-bit half-precision floating-point.
  * `tf.float32`: 32-bit single-precision floating-point.
  * `tf.float64`: 64-bit double-precision floating-point.
  * `tf.bfloat16`: 16-bit truncated floating-point.
  * `tf.complex64`: 64-bit single-precision complex.
  * `tf.complex128`: 128-bit double-precision complex.
  * `tf.int8`: 8-bit signed integer.
  * `tf.uint8`: 8-bit unsigned integer.
  * `tf.uint16`: 16-bit unsigned integer.
  * `tf.uint32`: 32-bit unsigned integer.
  * `tf.uint64`: 64-bit unsigned integer.
  * `tf.int16`: 16-bit signed integer.
  * `tf.int32`: 32-bit signed integer.
  * `tf.int64`: 64-bit signed integer.
  * `tf.bool`: Boolean.
  * `tf.string`: String.
  * `tf.qint8`: Quantized 8-bit signed integer.
  * `tf.quint8`: Quantized 8-bit unsigned integer.
  * `tf.qint16`: Quantized 16-bit signed integer.
  * `tf.quint16`: Quantized 16-bit unsigned integer.
  * `tf.qint32`: Quantized 32-bit signed integer.
  * `tf.resource`: Handle to a mutable resource.
  * `tf.variant`: Values of arbitrary types.

  The `tf.as_dtype()` function converts numpy types and string type
  names to a `DType` object.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce dtypes (import-module "tensorflow._api.v1.compat.v2.dtypes"))

(defn DType 
  "Represents the type of the elements in a `Tensor`.

  The following `DType` objects are defined:

  * `tf.float16`: 16-bit half-precision floating-point.
  * `tf.float32`: 32-bit single-precision floating-point.
  * `tf.float64`: 64-bit double-precision floating-point.
  * `tf.bfloat16`: 16-bit truncated floating-point.
  * `tf.complex64`: 64-bit single-precision complex.
  * `tf.complex128`: 128-bit double-precision complex.
  * `tf.int8`: 8-bit signed integer.
  * `tf.uint8`: 8-bit unsigned integer.
  * `tf.uint16`: 16-bit unsigned integer.
  * `tf.uint32`: 32-bit unsigned integer.
  * `tf.uint64`: 64-bit unsigned integer.
  * `tf.int16`: 16-bit signed integer.
  * `tf.int32`: 32-bit signed integer.
  * `tf.int64`: 64-bit signed integer.
  * `tf.bool`: Boolean.
  * `tf.string`: String.
  * `tf.qint8`: Quantized 8-bit signed integer.
  * `tf.quint8`: Quantized 8-bit unsigned integer.
  * `tf.qint16`: Quantized 16-bit signed integer.
  * `tf.quint16`: Quantized 16-bit unsigned integer.
  * `tf.qint32`: Quantized 32-bit signed integer.
  * `tf.resource`: Handle to a mutable resource.
  * `tf.variant`: Values of arbitrary types.

  The `tf.as_dtype()` function converts numpy types and string type
  names to a `DType` object.
  "
  [ type_enum ]
  (py/call-attr dtypes "DType"  type_enum ))

(defn as-datatype-enum 
  "Returns a `types_pb2.DataType` enum value based on this `DType`."
  [ self ]
    (py/call-attr self "as_datatype_enum"))

(defn as-numpy-dtype 
  "Returns a `numpy.dtype` based on this `DType`."
  [ self ]
    (py/call-attr self "as_numpy_dtype"))

(defn base-dtype 
  "Returns a non-reference `DType` based on this `DType`."
  [ self ]
    (py/call-attr self "base_dtype"))

(defn is-bool 
  "Returns whether this is a boolean data type"
  [ self ]
    (py/call-attr self "is_bool"))

(defn is-compatible-with 
  "Returns True if the `other` DType will be converted to this DType.

    The conversion rules are as follows:

    ```python
    DType(T)       .is_compatible_with(DType(T))        == True
    ```

    Args:
      other: A `DType` (or object that may be converted to a `DType`).

    Returns:
      True if a Tensor of the `other` `DType` will be implicitly converted to
      this `DType`.
    "
  [ self other ]
  (py/call-attr self "is_compatible_with"  self other ))

(defn is-complex 
  "Returns whether this is a complex floating point type."
  [ self ]
    (py/call-attr self "is_complex"))

(defn is-floating 
  "Returns whether this is a (non-quantized, real) floating point type."
  [ self ]
    (py/call-attr self "is_floating"))

(defn is-integer 
  "Returns whether this is a (non-quantized) integer type."
  [ self ]
    (py/call-attr self "is_integer"))

(defn is-numpy-compatible 
  ""
  [ self ]
    (py/call-attr self "is_numpy_compatible"))

(defn is-quantized 
  "Returns whether this is a quantized data type."
  [ self ]
    (py/call-attr self "is_quantized"))

(defn is-unsigned 
  "Returns whether this type is unsigned.

    Non-numeric, unordered, and quantized types are not considered unsigned, and
    this function returns `False`.

    Returns:
      Whether a `DType` is unsigned.
    "
  [ self ]
    (py/call-attr self "is_unsigned"))

(defn limits 
  "Return intensity limits, i.e.

    (min, max) tuple, of the dtype.
    Args:
      clip_negative : bool, optional If True, clip the negative range (i.e.
        return 0 for min intensity) even if the image dtype allows negative
        values. Returns
      min, max : tuple Lower and upper intensity limits.
    "
  [ self ]
    (py/call-attr self "limits"))

(defn max 
  "Returns the maximum representable value in this data type.

    Raises:
      TypeError: if this is a non-numeric, unordered, or quantized type.

    "
  [ self ]
    (py/call-attr self "max"))

(defn min 
  "Returns the minimum representable value in this data type.

    Raises:
      TypeError: if this is a non-numeric, unordered, or quantized type.

    "
  [ self ]
    (py/call-attr self "min"))

(defn name 
  "Returns the string name for this `DType`."
  [ self ]
    (py/call-attr self "name"))

(defn real-dtype 
  "Returns the dtype correspond to this dtype's real part."
  [ self ]
    (py/call-attr self "real_dtype"))

(defn size 
  ""
  [ self ]
    (py/call-attr self "size"))
