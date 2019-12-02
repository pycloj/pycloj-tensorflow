(ns tensorflow.-api.v1.compat.v2
  "Bring in all of the public TensorFlow interface into this module."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v2 (import-module "tensorflow._api.v1.compat.v2"))

(defn Assert 
  "Asserts that the given condition is true.

  If `condition` evaluates to false, print the list of tensors in `data`.
  `summarize` determines how many entries of the tensors to print.

  NOTE: In graph mode, to ensure that Assert executes, one usually attaches
  a dependency:

  ```python
  # Ensure maximum element of x is smaller or equal to 1
  assert_op = tf.Assert(tf.less_equal(tf.reduce_max(x), 1.), [x])
  with tf.control_dependencies([assert_op]):
    ... code using x ...
  ```

  Args:
    condition: The condition to evaluate.
    data: The tensors to print out when condition is false.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).

  Returns:
    assert_op: An `Operation` that, when executed, raises a
    `tf.errors.InvalidArgumentError` if `condition` is not true.
    @compatibility(eager)
    returns None
    @end_compatibility

  Raises:
    @compatibility(eager)
    `tf.errors.InvalidArgumentError` if `condition` is not true
    @end_compatibility
  

  **NOTE** The output of this function should be used.  If it is not, a warning will be logged.  To mark the output as used, call its .mark_used() method."
  [ condition data summarize name ]
  (py/call-attr v2 "Assert"  condition data summarize name ))

(defn abs 
  "Computes the absolute value of a tensor.

  Given a tensor of integer or floating-point values, this operation returns a
  tensor of the same type, where each element contains the absolute value of the
  corresponding element in the input.

  Given a tensor `x` of complex numbers, this operation returns a tensor of type
  `float32` or `float64` that is the absolute value of each element in `x`. All
  elements in `x` must be complex numbers of the form \\(a + bj\\). The
  absolute value is computed as \\( \sqrt{a^2 + b^2}\\).  For example:
  ```python
  x = tf.constant([[-2.25 + 4.75j], [-3.25 + 5.75j]])
  tf.abs(x)  # [5.25594902, 6.60492229]
  ```

  Args:
    x: A `Tensor` or `SparseTensor` of type `float16`, `float32`, `float64`,
      `int32`, `int64`, `complex64` or `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` or `SparseTensor` the same size, type, and sparsity as `x` with
      absolute values.
    Note, for `complex64` or `complex128` input, the returned `Tensor` will be
      of type `float32` or `float64`, respectively.

    If `x` is a `SparseTensor`, returns
    `SparseTensor(x.indices, tf.math.abs(x.values, ...), x.dense_shape)`"
  [ x name ]
  (py/call-attr v2 "abs"  x name ))

(defn acos 
  "Computes acos of x element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr v2 "acos"  x name ))

(defn acosh 
  "Computes inverse hyperbolic cosine of x element-wise.

  Given an input tensor, the function computes inverse hyperbolic cosine of every element.
  Input range is `[1, inf]`. It returns `nan` if the input lies outside the range.

  ```python
  x = tf.constant([-2, -0.5, 1, 1.2, 200, 10000, float(\"inf\")])
  tf.math.acosh(x) ==> [nan nan 0. 0.62236255 5.9914584 9.903487 inf]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr v2 "acosh"  x name ))

(defn add 
  "Returns x + y element-wise.

  *NOTE*: `math.add` supports broadcasting. `AddN` does not. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`, `string`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr v2 "add"  x y name ))

(defn add-n 
  "Adds all input tensors element-wise.

  Converts `IndexedSlices` objects into dense tensors prior to adding.

  `tf.math.add_n` performs the same operation as `tf.math.accumulate_n`, but it
  waits for all of its inputs to be ready before beginning to sum.
  This buffering can result in higher memory consumption when inputs are ready
  at different times, since the minimum temporary storage required is
  proportional to the input size rather than the output size.

  This op does not [broadcast](
  https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html)
  its inputs. If you need broadcasting, use `tf.math.add` (or the `+` operator)
  instead.

  For example:

  ```python
  a = tf.constant([[3, 5], [4, 8]])
  b = tf.constant([[1, 6], [2, 9]])
  tf.math.add_n([a, b, a])  # [[7, 16], [10, 25]]
  ```

  Args:
    inputs: A list of `tf.Tensor` or `tf.IndexedSlices` objects, each with same
      shape and type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of same shape and type as the elements of `inputs`.

  Raises:
    ValueError: If `inputs` don't all have same shape and dtype or the shape
    cannot be inferred.
  "
  [ inputs name ]
  (py/call-attr v2 "add_n"  inputs name ))

(defn argmax 
  "Returns the index with the largest value across axes of a tensor.

  Note that in case of ties the identity of the return value is not guaranteed.

  For example:
  ```python
  A=tf.constant([2,20,30,3,6]) # Constant 1-D Tensor
  tf.math.argmax(A) # output 2 as index 2 (A[2]) is maximum in tensor A
  B=tf.constant([[2,20,30,3,6],[3,11,16,1,8],[14,45,23,5,27]])
  tf.math.argmax(B,0) # [2, 2, 0, 2, 2]
  tf.math.argmax(B,1) # [2, 2, 1]
  ```
   
  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`,
      `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`,
      `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`,
      `uint64`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      int32 or int64, must be in the range `-rank(input), rank(input))`.
      Describes which axis of the input Tensor to reduce across. For vectors,
      use axis = 0.
    output_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to
      `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_type`.

  Usage:
  ```python
  import tensorflow as tf
  a = [1, 10, 26.9, 2.8, 166.32, 62.3]
  b = tf.math.argmax(input = a)
  c = tf.keras.backend.eval(b)
  # c = 4
  # here a[4] = 166.32 which is the largest element of a across axis 0
  ```
  "
  [input axis & {:keys [output_type name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "argmax" [input axis] {:output_type output_type :name name }))

(defn argmin 
  "Returns the index with the smallest value across axes of a tensor.

  Note that in case of ties the identity of the return value is not guaranteed.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`,
      `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`,
      `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`,
      `uint64`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      int32 or int64, must be in the range `-rank(input), rank(input))`.
      Describes which axis of the input Tensor to reduce across. For vectors,
      use axis = 0.
    output_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to
      `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_type`.

  Usage:
  ```python
  import tensorflow as tf
  a = [1, 10, 26.9, 2.8, 166.32, 62.3]
  b = tf.math.argmin(input = a)
  c = tf.keras.backend.eval(b)
  # c = 0
  # here a[0] = 1 which is the smallest element of a across axis 0
  ```
  "
  [input axis & {:keys [output_type name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "argmin" [input axis] {:output_type output_type :name name }))

(defn argsort 
  "Returns the indices of a tensor that give its sorted order along an axis.

  For a 1D tensor, `tf.gather(values, tf.argsort(values))` is equivalent to
  `tf.sort(values)`. For higher dimensions, the output has the same shape as
  `values`, but along the given axis, values represent the index of the sorted
  element in that slice of the tensor at the given position.
  
  Usage:

  ```python
  import tensorflow as tf
  a = [1, 10, 26.9, 2.8, 166.32, 62.3]
  b = tf.argsort(a,axis=-1,direction='ASCENDING',stable=False,name=None)
  c = tf.keras.backend.eval(b)
  # Here, c = [0 3 1 2 5 4]
  ```

  Args:
    values: 1-D or higher numeric `Tensor`.
    axis: The axis along which to sort. The default is -1, which sorts the last
      axis.
    direction: The direction in which to sort the values (`'ASCENDING'` or
      `'DESCENDING'`).
    stable: If True, equal elements in the original tensor will not be
      re-ordered in the returned order. Unstable sort is not yet implemented,
      but will eventually be the default for performance reasons. If you require
      a stable order, pass `stable=True` for forwards compatibility.
    name: Optional name for the operation.

  Returns:
    An int32 `Tensor` with the same shape as `values`. The indices that would
        sort each slice of the given `values` along the given `axis`.

  Raises:
    ValueError: If axis is not a constant scalar, or the direction is invalid.
  "
  [values & {:keys [axis direction stable name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "argsort" [values] {:axis axis :direction direction :stable stable :name name }))

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
  (py/call-attr v2 "as_dtype"  type_value ))

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
    (py/call-attr-kw v2 "as_string" [input] {:precision precision :scientific scientific :shortest shortest :width width :fill fill :name name }))

(defn asin 
  "Computes the trignometric inverse sine of x element-wise.

  The `tf.math.asin` operation returns the inverse of `tf.math.sin`, such that
  if `y = tf.math.sin(x)` then, `x = tf.math.asin(y)`.

  **Note**: The output of `tf.math.asin` will lie within the invertible range 
  of sine, i.e [-pi/2, pi/2].

  For example:

  ```python
  # Note: [1.047, 0.785] ~= [(pi/3), (pi/4)]
  x = tf.constant([1.047, 0.785])
  y = tf.math.sin(x) # [0.8659266, 0.7068252]

  tf.math.asin(y) # [1.047, 0.785] = x
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr v2 "asin"  x name ))

(defn asinh 
  "Computes inverse hyperbolic sine of x element-wise.

    Given an input tensor, this function computes inverse hyperbolic sine
    for every element in the tensor. Both input and output has a range of
    `[-inf, inf]`.

    ```python
    x = tf.constant([-float(\"inf\"), -2, -0.5, 1, 1.2, 200, 10000, float(\"inf\")])
    tf.math.asinh(x) ==> [-inf -1.4436355 -0.4812118 0.8813736 1.0159732 5.991471 9.903487 inf]
    ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr v2 "asinh"  x name ))

(defn assert-equal 
  "Assert the condition `x == y` holds element-wise.

  This Op checks that `x[i] == y[i]` holds for every pair of (possibly
  broadcast) elements of `x` and `y`. If both `x` and `y` are empty, this is
  trivially satisfied.

  If `x` and `y` are not equal, `message`, as well as the first `summarize`
  entries of `x` and `y` are printed, and `InvalidArgumentError` is raised.

  Args:
    x:  Numeric `Tensor`.
    y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
    message: A string to prefix to the default message.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to \"assert_equal\".

  Returns:
    Op that raises `InvalidArgumentError` if `x == y` is False. This can be
      used with `tf.control_dependencies` inside of `tf.function`s to block
      followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility

  Raises:
    InvalidArgumentError: if the check can be performed immediately and
      `x == y` is False. The check can be performed immediately during eager
      execution or if `x` and `y` are statically known.
  "
  [ x y message summarize name ]
  (py/call-attr v2 "assert_equal"  x y message summarize name ))

(defn assert-greater 
  "Assert the condition `x > y` holds element-wise.

  This Op checks that `x[i] > y[i]` holds for every pair of (possibly
  broadcast) elements of `x` and `y`. If both `x` and `y` are empty, this is
  trivially satisfied.

  If `x` is not greater than `y` element-wise, `message`, as well as the first
  `summarize` entries of `x` and `y` are printed, and `InvalidArgumentError` is
  raised.

  Args:
    x:  Numeric `Tensor`.
    y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
    message: A string to prefix to the default message.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to \"assert_greater\".

  Returns:
    Op that raises `InvalidArgumentError` if `x > y` is False. This can be
      used with `tf.control_dependencies` inside of `tf.function`s to block
      followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility

  Raises:
    InvalidArgumentError: if the check can be performed immediately and
      `x > y` is False. The check can be performed immediately during eager
      execution or if `x` and `y` are statically known.
  "
  [ x y message summarize name ]
  (py/call-attr v2 "assert_greater"  x y message summarize name ))

(defn assert-less 
  "Assert the condition `x < y` holds element-wise.

  This Op checks that `x[i] < y[i]` holds for every pair of (possibly
  broadcast) elements of `x` and `y`. If both `x` and `y` are empty, this is
  trivially satisfied.

  If `x` is not less than `y` element-wise, `message`, as well as the first
  `summarize` entries of `x` and `y` are printed, and `InvalidArgumentError` is
  raised.

  Args:
    x:  Numeric `Tensor`.
    y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
    message: A string to prefix to the default message.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).  Defaults to \"assert_less\".

  Returns:
    Op that raises `InvalidArgumentError` if `x < y` is False.
    This can be used with `tf.control_dependencies` inside of `tf.function`s
    to block followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility

  Raises:
    InvalidArgumentError: if the check can be performed immediately and
      `x < y` is False. The check can be performed immediately during eager
      execution or if `x` and `y` are statically known.
  "
  [ x y message summarize name ]
  (py/call-attr v2 "assert_less"  x y message summarize name ))

(defn assert-rank 
  "Assert that `x` has rank equal to `rank`.

  This Op checks that the rank of `x` is equal to `rank`.

  If `x` has a different rank, `message`, as well as the shape of `x` are
  printed, and `InvalidArgumentError` is raised.

  Args:
    x: `Tensor`.
    rank: Scalar integer `Tensor`.
    message: A string to prefix to the default message.
    name: A name for this operation (optional). Defaults to
      \"assert_rank\".

  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank.
    If static checks determine `x` has correct rank, a `no_op` is returned.
    This can be used with `tf.control_dependencies` inside of `tf.function`s
    to block followup computation until the check has executed.
    @compatibility(eager)
    returns None
    @end_compatibility

  Raises:
    InvalidArgumentError: if the check can be performed immediately and
      `x` does not have rank `rank`. The check can be performed immediately
      during eager execution or if the shape of `x` is statically known.
  "
  [ x rank message name ]
  (py/call-attr v2 "assert_rank"  x rank message name ))

(defn atan 
  "Computes the trignometric inverse tangent of x element-wise.

  The `tf.math.atan` operation returns the inverse of `tf.math.tan`, such that
  if `y = tf.math.tan(x)` then, `x = tf.math.atan(y)`.

  **Note**: The output of `tf.math.atan` will lie within the invertible range 
  of tan, i.e (-pi/2, pi/2).

  For example:

  ```python
  # Note: [1.047, 0.785] ~= [(pi/3), (pi/4)]
  x = tf.constant([1.047, 0.785])
  y = tf.math.tan(x) # [1.731261, 0.99920404]

  tf.math.atan(y) # [1.047, 0.785] = x
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr v2 "atan"  x name ))

(defn atan2 
  "Computes arctangent of `y/x` element-wise, respecting signs of the arguments.

  This is the angle \( \theta \in [-\pi, \pi] \) such that
  \[ x = r \cos(\theta) \]
  and
  \[ y = r \sin(\theta) \]
  where \(r = \sqrt(x^2 + y^2) \).

  Args:
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    x: A `Tensor`. Must have the same type as `y`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `y`.
  "
  [ y x name ]
  (py/call-attr v2 "atan2"  y x name ))

(defn atanh 
  "Computes inverse hyperbolic tangent of x element-wise.

    Given an input tensor, this function computes inverse hyperbolic tangent
    for every element in the tensor. Input range is `[-1,1]` and output range is
    `[-inf, inf]`. If input is `-1`, output will be `-inf` and if the
    input is `1`, output will be `inf`. Values outside the range will have
    `nan` as output.

    ```python
    x = tf.constant([-float(\"inf\"), -1, -0.5, 1, 0, 0.5, 10, float(\"inf\")])
    tf.math.atanh(x) ==> [nan -inf -0.54930615 inf  0. 0.54930615 nan nan]
    ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr v2 "atanh"  x name ))

(defn batch-to-space 
  "BatchToSpace for N-D tensors of type T.

  This operation reshapes the \"batch\" dimension 0 into `M + 1` dimensions of
  shape `block_shape + [batch]`, interleaves these blocks back into the grid
  defined by the spatial dimensions `[1, ..., M]`, to obtain a result with the
  same rank as the input.  The spatial dimensions of this intermediate result
  are then optionally cropped according to `crops` to produce the output.  This
  is the reverse of SpaceToBatch.  See below for a precise description.

  Args:
    input: A `Tensor`. N-D with shape `input_shape = [batch] + spatial_shape +
      remaining_shape`, where spatial_shape has M dimensions.
    block_shape: A `Tensor`. Must be one of the following types: `int32`,
      `int64`. 1-D with shape `[M]`, all values must be >= 1. For backwards
      compatibility with TF 1.0, this parameter may be an int, in which case it
      is converted to `numpy.array([block_shape, block_shape],
      dtype=numpy.int64)`.
    crops: A `Tensor`. Must be one of the following types: `int32`, `int64`. 2-D
      with shape `[M, 2]`, all values must be >= 0. `crops[i] = [crop_start,
      crop_end]` specifies the amount to crop from input dimension `i + 1`,
      which corresponds to spatial dimension `i`.  It is required that
      `crop_start[i] + crop_end[i] <= block_shape[i] * input_shape[i + 1]`.
      This operation is equivalent to the following steps:
      1. Reshape `input` to `reshaped` of shape: [block_shape[0], ...,
        block_shape[M-1], batch / prod(block_shape), input_shape[1], ...,
        input_shape[N-1]]  2. Permute dimensions of `reshaped` to produce
        `permuted` of shape [batch / prod(block_shape),  input_shape[1],
        block_shape[0], ..., input_shape[M], block_shape[M-1],
        input_shape[M+1], ..., input_shape[N-1]]  3. Reshape `permuted` to
        produce `reshaped_permuted` of shape [batch / prod(block_shape),
        input_shape[1] * block_shape[0], ..., input_shape[M] * block_shape[M-1],
        input_shape[M+1], ..., input_shape[N-1]]  4. Crop the start and end of
        dimensions `[1, ..., M]` of `reshaped_permuted` according to `crops` to
        produce the
         output of shape: [batch / prod(block_shape),  input_shape[1] *
           block_shape[0] - crops[0,0] - crops[0,1], ..., input_shape[M] *
           block_shape[M-1] - crops[M-1,0] - crops[M-1,1],  input_shape[M+1],
           ..., input_shape[N-1]]
      Some examples:  (1) For the following input of shape `[4, 1, 1, 1]`,
          `block_shape = [2, 2]`, and `crops = [[0, 0], [0, 0]]`:  ``` [[[[1]]],
            [[[2]]], [[[3]]], [[[4]]]] ```
      The output tensor has shape `[1, 2, 2, 1]` and value:  ``` x = [[[[1],
        [2]], [[3], [4]]]] ```  (2) For the following input of shape `[4, 1, 1,
        3]`,
          `block_shape = [2, 2]`, and `crops = [[0, 0], [0, 0]]`:  ``` [[[1, 2,
            3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]]] ```
      The output tensor has shape `[1, 2, 2, 3]` and value:  ``` x = [[[[1, 2,
        3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]] ```  (3) For the following
        input of shape `[4, 2, 2, 1]`,
          `block_shape = [2, 2]`, and `crops = [[0, 0], [0, 0]]`:  ``` x =
            [[[[1], [3]], [[9], [11]]], [[[2], [4]], [[10], [12]]], [[[5], [7]],
            [[13], [15]]], [[[6], [8]], [[14], [16]]]] ```
      The output tensor has shape `[1, 4, 4, 1]` and value:  ``` x = [[[1],
        [2],  [3],  [4]], [[5],   [6],  [7],  [8]], [[9],  [10], [11],  [12]],
        [[13], [14], [15],  [16]]] ```  (4) For the following input of shape
        `[8, 1, 3, 1]`,
          `block_shape = [2, 2]`, and `crops = [[0, 0], [2, 0]]`:  ``` x =
            [[[[0], [1], [3]]], [[[0], [9], [11]]], [[[0], [2], [4]]], [[[0],
            [10], [12]]], [[[0], [5], [7]]], [[[0], [13], [15]]], [[[0], [6],
            [8]]], [[[0], [14], [16]]]] ```
      The output tensor has shape `[2, 2, 4, 1]` and value:  ``` x = [[[[1],
        [2],  [3],  [4]], [[5],   [6],  [7],  [8]]], [[[9],  [10], [11],  [12]],
        [[13], [14], [15],  [16]]]] ```
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input block_shape crops name ]
  (py/call-attr v2 "batch_to_space"  input block_shape crops name ))

(defn bitcast 
  "Bitcasts a tensor from one type to another without copying data.

  Given a tensor `input`, this operation returns a tensor that has the same buffer
  data as `input` with datatype `type`.

  If the input datatype `T` is larger than the output datatype `type` then the
  shape changes from [...] to [..., sizeof(`T`)/sizeof(`type`)].

  If `T` is smaller than `type`, the operator requires that the rightmost
  dimension be equal to sizeof(`type`)/sizeof(`T`). The shape then goes from
  [..., sizeof(`type`)/sizeof(`T`)] to [...].

  tf.bitcast() and tf.cast() work differently when real dtype is casted as a complex dtype
  (e.g. tf.complex64 or tf.complex128) as tf.cast() make imaginary part 0 while tf.bitcast()
  gives module error.
  For example,

  Example 1:
  ```python
  >>> a = [1., 2., 3.]
  >>> equality_bitcast = tf.bitcast(a,tf.complex128)
  tensorflow.python.framework.errors_impl.InvalidArgumentError: Cannot bitcast from float to complex128: shape [3] [Op:Bitcast]
  >>> equality_cast = tf.cast(a,tf.complex128)
  >>> print(equality_cast)
  tf.Tensor([1.+0.j 2.+0.j 3.+0.j], shape=(3,), dtype=complex128)
  ```
  Example 2:
  ```python
  >>> tf.bitcast(tf.constant(0xffffffff, dtype=tf.uint32), tf.uint8)
  <tf.Tensor: ... shape=(4,), dtype=uint8, numpy=array([255, 255, 255, 255], dtype=uint8)>
  ```
  Example 3:
  ```python
  >>> x = [1., 2., 3.]
  >>> y = [0., 2., 3.]
  >>> equality= tf.equal(x,y)
  >>> equality_cast = tf.cast(equality,tf.float32)
  >>> equality_bitcast = tf.bitcast(equality_cast,tf.uint8)
  >>> print(equality)
  tf.Tensor([False True True], shape=(3,), dtype=bool)
  >>> print(equality_cast)
  tf.Tensor([0. 1. 1.], shape=(3,), dtype=float32)
  >>> print(equality_bitcast)
  tf.Tensor(
  [[ 0 0 0 0]
   [ 0 0 128 63]
   [ 0 0 128 63]], shape=(3, 4), dtype=uint8)
  ```

  *NOTE*: Bitcast is implemented as a low-level cast, so machines with different
  endian orderings will give different results.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `uint32`, `uint64`, `int8`, `int16`, `complex64`, `complex128`, `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
    type: A `tf.DType` from: `tf.bfloat16, tf.half, tf.float32, tf.float64, tf.int64, tf.int32, tf.uint8, tf.uint16, tf.uint32, tf.uint64, tf.int8, tf.int16, tf.complex64, tf.complex128, tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `type`.
  "
  [ input type name ]
  (py/call-attr v2 "bitcast"  input type name ))
(defn boolean-mask 
  "Apply boolean mask to tensor.

  Numpy equivalent is `tensor[mask]`.

  ```python
  # 1-D example
  tensor = [0, 1, 2, 3]
  mask = np.array([True, False, True, False])
  boolean_mask(tensor, mask)  # [0, 2]
  ```

  In general, `0 < dim(mask) = K <= dim(tensor)`, and `mask`'s shape must match
  the first K dimensions of `tensor`'s shape.  We then have:
    `boolean_mask(tensor, mask)[i, j1,...,jd] = tensor[i1,...,iK,j1,...,jd]`
  where `(i1,...,iK)` is the ith `True` entry of `mask` (row-major order).
  The `axis` could be used with `mask` to indicate the axis to mask from.
  In that case, `axis + dim(mask) <= dim(tensor)` and `mask`'s shape must match
  the first `axis + dim(mask)` dimensions of `tensor`'s shape.

  See also: `tf.ragged.boolean_mask`, which can be applied to both dense and
  ragged tensors, and can be used if you need to preserve the masked dimensions
  of `tensor` (rather than flattening them, as `tf.boolean_mask` does).

  Args:
    tensor:  N-D tensor.
    mask:  K-D boolean tensor, K <= N and K must be known statically.
    axis:  A 0-D int Tensor representing the axis in `tensor` to mask from. By
      default, axis is 0 which will mask from the first dimension. Otherwise K +
      axis <= N.
    name:  A name for this operation (optional).

  Returns:
    (N-K+1)-dimensional tensor populated by entries in `tensor` corresponding
    to `True` values in `mask`.

  Raises:
    ValueError:  If shapes do not conform.

  Examples:

  ```python
  # 2-D example
  tensor = [[1, 2], [3, 4], [5, 6]]
  mask = np.array([True, False, True])
  boolean_mask(tensor, mask)  # [[1, 2], [5, 6]]
  ```
  "
  [tensor mask axis  & {:keys [name]} ]
    (py/call-attr-kw v2 "boolean_mask" [tensor mask axis] {:name name }))

(defn broadcast-dynamic-shape 
  "Computes the shape of a broadcast given symbolic shapes.

  When shape_x and shape_y are Tensors representing shapes (i.e. the result of
  calling tf.shape on another Tensor) this computes a Tensor which is the shape
  of the result of a broadcasting op applied in tensors of shapes shape_x and
  shape_y.

  For example, if shape_x is [1, 2, 3] and shape_y is [5, 1, 3], the result is a
  Tensor whose value is [5, 2, 3].

  This is useful when validating the result of a broadcasting operation when the
  tensors do not have statically known shapes.

  Args:
    shape_x: A rank 1 integer `Tensor`, representing the shape of x.
    shape_y: A rank 1 integer `Tensor`, representing the shape of y.

  Returns:
    A rank 1 integer `Tensor` representing the broadcasted shape.
  "
  [ shape_x shape_y ]
  (py/call-attr v2 "broadcast_dynamic_shape"  shape_x shape_y ))

(defn broadcast-static-shape 
  "Computes the shape of a broadcast given known shapes.

  When shape_x and shape_y are fully known TensorShapes this computes a
  TensorShape which is the shape of the result of a broadcasting op applied in
  tensors of shapes shape_x and shape_y.

  For example, if shape_x is [1, 2, 3] and shape_y is [5, 1, 3], the result is a
  TensorShape whose value is [5, 2, 3].

  This is useful when validating the result of a broadcasting operation when the
  tensors have statically known shapes.

  Args:
    shape_x: A `TensorShape`
    shape_y: A `TensorShape`

  Returns:
    A `TensorShape` representing the broadcasted shape.

  Raises:
    ValueError: If the two shapes can not be broadcasted.
  "
  [ shape_x shape_y ]
  (py/call-attr v2 "broadcast_static_shape"  shape_x shape_y ))

(defn broadcast-to 
  "Broadcast an array for a compatible shape.

  Broadcasting is the process of making arrays to have compatible shapes
  for arithmetic operations. Two shapes are compatible if for each
  dimension pair they are either equal or one of them is one. When trying
  to broadcast a Tensor to a shape, it starts with the trailing dimensions,
  and works its way forward.

  For example,

  ```python
  >>> x = tf.constant([1, 2, 3])
  >>> y = tf.broadcast_to(x, [3, 3])
  >>> sess.run(y)
  array([[1, 2, 3],
         [1, 2, 3],
         [1, 2, 3]], dtype=int32)
  ```

  In the above example, the input Tensor with the shape of `[1, 3]`
  is broadcasted to output Tensor with shape of `[3, 3]`.

  Args:
    input: A `Tensor`. A Tensor to broadcast.
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      An 1-D `int` Tensor. The shape of the desired output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input shape name ]
  (py/call-attr v2 "broadcast_to"  input shape name ))
(defn case 
  "Create a case operation.

  See also `tf.switch_case`.

  The `pred_fn_pairs` parameter is a list of pairs of size N.
  Each pair contains a boolean scalar tensor and a python callable that
  creates the tensors to be returned if the boolean evaluates to True.
  `default` is a callable generating a list of tensors. All the callables
  in `pred_fn_pairs` as well as `default` (if provided) should return the same
  number and types of tensors.

  If `exclusive==True`, all predicates are evaluated, and an exception is
  thrown if more than one of the predicates evaluates to `True`.
  If `exclusive==False`, execution stops at the first predicate which
  evaluates to True, and the tensors generated by the corresponding function
  are returned immediately. If none of the predicates evaluate to True, this
  operation returns the tensors generated by `default`.

  `tf.case` supports nested structures as implemented in
  `tf.contrib.framework.nest`. All of the callables must return the same
  (possibly nested) value structure of lists, tuples, and/or named tuples.
  Singleton lists and tuples form the only exceptions to this: when returned by
  a callable, they are implicitly unpacked to single values. This
  behavior is disabled by passing `strict=True`.

  @compatibility(v2)
  `pred_fn_pairs` could be a dictionary in v1. However, tf.Tensor and
  tf.Variable are no longer hashable in v2, so cannot be used as a key for a
  dictionary.  Please use a list or a tuple instead.
  @end_compatibility


  **Example 1:**

  Pseudocode:

  ```
  if (x < y) return 17;
  else return 23;
  ```

  Expressions:

  ```python
  f1 = lambda: tf.constant(17)
  f2 = lambda: tf.constant(23)
  r = tf.case([(tf.less(x, y), f1)], default=f2)
  ```

  **Example 2:**

  Pseudocode:

  ```
  if (x < y && x > z) raise OpError(\"Only one predicate may evaluate to True\");
  if (x < y) return 17;
  else if (x > z) return 23;
  else return -1;
  ```

  Expressions:

  ```python
  def f1(): return tf.constant(17)
  def f2(): return tf.constant(23)
  def f3(): return tf.constant(-1)
  r = tf.case([(tf.less(x, y), f1), (tf.greater(x, z), f2)],
           default=f3, exclusive=True)
  ```

  Args:
    pred_fn_pairs: List of pairs of a boolean scalar tensor and a callable which
      returns a list of tensors.
    default: Optional callable that returns a list of tensors.
    exclusive: True iff at most one predicate is allowed to evaluate to `True`.
    strict: A boolean that enables/disables 'strict' mode; see above.
    name: A name for this operation (optional).

  Returns:
    The tensors returned by the first pair whose predicate evaluated to True, or
    those returned by `default` if none does.

  Raises:
    TypeError: If `pred_fn_pairs` is not a list/tuple.
    TypeError: If `pred_fn_pairs` is a list but does not contain 2-tuples.
    TypeError: If `fns[i]` is not callable for any i, or `default` is not
               callable.
  "
  [pred_fn_pairs default  & {:keys [exclusive strict name]} ]
    (py/call-attr-kw v2 "case" [pred_fn_pairs default] {:exclusive exclusive :strict strict :name name }))

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
  (py/call-attr v2 "cast"  x dtype name ))

(defn clip-by-global-norm 
  "Clips values of multiple tensors by the ratio of the sum of their norms.

  Given a tuple or list of tensors `t_list`, and a clipping ratio `clip_norm`,
  this operation returns a list of clipped tensors `list_clipped`
  and the global norm (`global_norm`) of all tensors in `t_list`. Optionally,
  if you've already computed the global norm for `t_list`, you can specify
  the global norm with `use_norm`.

  To perform the clipping, the values `t_list[i]` are set to:

      t_list[i] * clip_norm / max(global_norm, clip_norm)

  where:

      global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))

  If `clip_norm > global_norm` then the entries in `t_list` remain as they are,
  otherwise they're all shrunk by the global ratio.

  If `global_norm == infinity` then the entries in `t_list` are all set to `NaN`
  to signal that an error occurred.

  Any of the entries of `t_list` that are of type `None` are ignored.

  This is the correct way to perform gradient clipping (for example, see
  [Pascanu et al., 2012](http://arxiv.org/abs/1211.5063)
  ([pdf](http://arxiv.org/pdf/1211.5063.pdf))).

  However, it is slower than `clip_by_norm()` because all the parameters must be
  ready before the clipping operation can be performed.

  Args:
    t_list: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.
    clip_norm: A 0-D (scalar) `Tensor` > 0. The clipping ratio.
    use_norm: A 0-D (scalar) `Tensor` of type `float` (optional). The global
      norm to use. If not provided, `global_norm()` is used to compute the norm.
    name: A name for the operation (optional).

  Returns:
    list_clipped: A list of `Tensors` of the same type as `list_t`.
    global_norm: A 0-D (scalar) `Tensor` representing the global norm.

  Raises:
    TypeError: If `t_list` is not a sequence.
  "
  [ t_list clip_norm use_norm name ]
  (py/call-attr v2 "clip_by_global_norm"  t_list clip_norm use_norm name ))

(defn clip-by-norm 
  "Clips tensor values to a maximum L2-norm.

  Given a tensor `t`, and a maximum clip value `clip_norm`, this operation
  normalizes `t` so that its L2-norm is less than or equal to `clip_norm`,
  along the dimensions given in `axes`. Specifically, in the default case
  where all dimensions are used for calculation, if the L2-norm of `t` is
  already less than or equal to `clip_norm`, then `t` is not modified. If
  the L2-norm is greater than `clip_norm`, then this operation returns a
  tensor of the same type and shape as `t` with its values set to:

  `t * clip_norm / l2norm(t)`

  In this case, the L2-norm of the output tensor is `clip_norm`.

  As another example, if `t` is a matrix and `axes == [1]`, then each row
  of the output will have L2-norm less than or equal to `clip_norm`. If
  `axes == [0]` instead, each column of the output will be clipped.

  This operation is typically used to clip gradients before applying them with
  an optimizer.

  Args:
    t: A `Tensor` or `IndexedSlices`.
    clip_norm: A 0-D (scalar) `Tensor` > 0. A maximum clipping value.
    axes: A 1-D (vector) `Tensor` of type int32 containing the dimensions
      to use for computing the L2-norm. If `None` (the default), uses all
      dimensions.
    name: A name for the operation (optional).

  Returns:
    A clipped `Tensor` or `IndexedSlices`.

  Raises:
    ValueError: If the clip_norm tensor is not a 0-D scalar tensor.
    TypeError: If dtype of the input is not a floating point or
      complex type.
  "
  [ t clip_norm axes name ]
  (py/call-attr v2 "clip_by_norm"  t clip_norm axes name ))

(defn clip-by-value 
  "Clips tensor values to a specified min and max.

  Given a tensor `t`, this operation returns a tensor of the same type and
  shape as `t` with its values clipped to `clip_value_min` and `clip_value_max`.
  Any values less than `clip_value_min` are set to `clip_value_min`. Any values
  greater than `clip_value_max` are set to `clip_value_max`.

  Note: `clip_value_min` needs to be smaller or equal to `clip_value_max` for
  correct results.

  For example:

  ```python
  A = tf.constant([[1, 20, 13], [3, 21, 13]])
  B = tf.clip_by_value(A, clip_value_min=0, clip_value_max=3) # [[1, 3, 3],[3, 3, 3]]
  C = tf.clip_by_value(A, clip_value_min=0., clip_value_max=3.) # throws `TypeError`
  as input and clip_values are of different dtype
  ```

  Args:
    t: A `Tensor` or `IndexedSlices`.
    clip_value_min: A 0-D (scalar) `Tensor`, or a `Tensor` with the same shape
      as `t`. The minimum value to clip by.
    clip_value_max: A 0-D (scalar) `Tensor`, or a `Tensor` with the same shape
      as `t`. The maximum value to clip by.
    name: A name for the operation (optional).

  Returns:
    A clipped `Tensor` or `IndexedSlices`.

  Raises:
    ValueError: If the clip tensors would trigger array broadcasting
      that would make the returned tensor larger than the input.
    TypeError: If dtype of the input is `int32` and dtype of
    the `clip_value_min' or `clip_value_max` is `float32`
  "
  [ t clip_value_min clip_value_max name ]
  (py/call-attr v2 "clip_by_value"  t clip_value_min clip_value_max name ))

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
  (py/call-attr v2 "complex"  real imag name ))
(defn concat 
  "Concatenates tensors along one dimension.

  Concatenates the list of tensors `values` along dimension `axis`.  If
  `values[i].shape = [D0, D1, ... Daxis(i), ...Dn]`, the concatenated
  result has shape

      [D0, D1, ... Raxis, ...Dn]

  where

      Raxis = sum(Daxis(i))

  That is, the data from the input tensors is joined along the `axis`
  dimension.

  The number of dimensions of the input tensors must match, and all dimensions
  except `axis` must be equal.

  For example:

  ```python
  t1 = [[1, 2, 3], [4, 5, 6]]
  t2 = [[7, 8, 9], [10, 11, 12]]
  tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
  tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

  # tensor t3 with shape [2, 3]
  # tensor t4 with shape [2, 3]
  tf.shape(tf.concat([t3, t4], 0))  # [4, 3]
  tf.shape(tf.concat([t3, t4], 1))  # [2, 6]
  ```
  As in Python, the `axis` could also be negative numbers. Negative `axis`
  are interpreted as counting from the end of the rank, i.e.,
   `axis + rank(values)`-th dimension.

  For example:

  ```python
  t1 = [[[1, 2], [2, 3]], [[4, 4], [5, 3]]]
  t2 = [[[7, 4], [8, 4]], [[2, 10], [15, 11]]]
  tf.concat([t1, t2], -1)
  ```

  would produce:

  ```python
  [[[ 1,  2,  7,  4],
    [ 2,  3,  8,  4]],

   [[ 4,  4,  2, 10],
    [ 5,  3, 15, 11]]]
  ```

  Note: If you are concatenating along a new axis consider using stack.
  E.g.

  ```python
  tf.concat([tf.expand_dims(t, axis) for t in tensors], axis)
  ```

  can be rewritten as

  ```python
  tf.stack(tensors, axis=axis)
  ```

  Args:
    values: A list of `Tensor` objects or a single `Tensor`.
    axis: 0-D `int32` `Tensor`.  Dimension along which to concatenate. Must be
      in the range `[-rank(values), rank(values))`. As in Python, indexing for
      axis is 0-based. Positive axis in the rage of `[0, rank(values))` refers
      to `axis`-th dimension. And negative axis refers to `axis +
      rank(values)`-th dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` resulting from concatenation of the input tensors.
  "
  [values axis  & {:keys [name]} ]
    (py/call-attr-kw v2 "concat" [values axis] {:name name }))

(defn cond 
  "Return `true_fn()` if the predicate `pred` is true else `false_fn()`.

  `true_fn` and `false_fn` both return lists of output tensors. `true_fn` and
  `false_fn` must have the same non-zero number and type of outputs.

  **WARNING**: Any Tensors or Operations created outside of `true_fn` and
  `false_fn` will be executed regardless of which branch is selected at runtime.

  Although this behavior is consistent with the dataflow model of TensorFlow,
  it has frequently surprised users who expected a lazier semantics.
  Consider the following simple program:

  ```python
  z = tf.multiply(a, b)
  result = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))
  ```

  If `x < y`, the `tf.add` operation will be executed and `tf.square`
  operation will not be executed. Since `z` is needed for at least one
  branch of the `cond`, the `tf.multiply` operation is always executed,
  unconditionally.

  Note that `cond` calls `true_fn` and `false_fn` *exactly once* (inside the
  call to `cond`, and not at all during `Session.run()`). `cond`
  stitches together the graph fragments created during the `true_fn` and
  `false_fn` calls with some additional graph nodes to ensure that the right
  branch gets executed depending on the value of `pred`.

  `tf.cond` supports nested structures as implemented in
  `tensorflow.python.util.nest`. Both `true_fn` and `false_fn` must return the
  same (possibly nested) value structure of lists, tuples, and/or named tuples.
  Singleton lists and tuples form the only exceptions to this: when returned by
  `true_fn` and/or `false_fn`, they are implicitly unpacked to single values.

  Note: It is illegal to \"directly\" use tensors created inside a cond branch
  outside it, e.g. by storing a reference to a branch tensor in the python
  state. If you need to use a tensor created in a branch function you should
  return it as an output of the branch function and use the output from
  `tf.cond` instead.

  Args:
    pred: A scalar determining whether to return the result of `true_fn` or
      `false_fn`.
    true_fn: The callable to be performed if pred is true.
    false_fn: The callable to be performed if pred is false.
    name: Optional name prefix for the returned tensors.

  Returns:
    Tensors returned by the call to either `true_fn` or `false_fn`. If the
    callables return a singleton list, the element is extracted from the list.

  Raises:
    TypeError: if `true_fn` or `false_fn` is not callable.
    ValueError: if `true_fn` and `false_fn` do not return the same number of
      tensors, or return tensors of different types.

  Example:

  ```python
  x = tf.constant(2)
  y = tf.constant(5)
  def f1(): return tf.multiply(x, 17)
  def f2(): return tf.add(y, 23)
  r = tf.cond(tf.less(x, y), f1, f2)
  # r is set to f1().
  # Operations in f2 (e.g., tf.add) are not executed.
  ```

  "
  [ pred true_fn false_fn name ]
  (py/call-attr v2 "cond"  pred true_fn false_fn name ))
(defn constant 
  "Creates a constant tensor.

  The resulting tensor is populated with values of type `dtype`, as
  specified by arguments `value` and (optionally) `shape` (see examples
  below).

  The argument `value` can be a constant value, or a list of values of type
  `dtype`. If `value` is a list, then the length of the list must be less
  than or equal to the number of elements implied by the `shape` argument (if
  specified). In the case where the list length is less than the number of
  elements specified by `shape`, the last element in the list will be used
  to fill the remaining entries.

  The argument `shape` is optional. If present, it specifies the dimensions of
  the resulting tensor. If not present, the shape of `value` is used.

  If the argument `dtype` is not specified, then the type is inferred from
  the type of `value`.

  For example:

  ```python
  # Constant 1-D Tensor populated with value list.
  tensor = tf.constant([1, 2, 3, 4, 5, 6]) => [1 2 3 4 5 6]

  # Constant 1-D Tensor populated with value list.
  tensor = tf.constant([1, 2, 3, 4, 5, 6], shape=(2,3))
       => [[1 2 3], [4 5 6]]

  # Constant 2-D tensor populated with scalar value -1.
  tensor = tf.constant(-1.0, shape=[2, 3]) => [[-1. -1. -1.]
                                               [-1. -1. -1.]]
  ```

  `tf.constant` differs from `tf.fill` in a few ways:

  *   `tf.constant` supports arbitrary constants, not just uniform scalar
      Tensors like `tf.fill`.
  *   `tf.constant` creates a `Const` node in the computation graph with the
      exact value at graph construction time. On the other hand, `tf.fill`
      creates an Op in the graph that is expanded at runtime.
  *   Because `tf.constant` only embeds constant values in the graph, it does
      not support dynamic shapes based on other runtime Tensors, whereas
      `tf.fill` does.

  Args:
    value:          A constant value (or list) of output type `dtype`.

    dtype:          The type of the elements of the resulting tensor.

    shape:          Optional dimensions of resulting tensor.

    name:           Optional name for the tensor.

  Returns:
    A Constant Tensor.

  Raises:
    TypeError: if shape is incorrectly specified or unsupported.
  "
  [value dtype shape  & {:keys [name]} ]
    (py/call-attr-kw v2 "constant" [value dtype shape] {:name name }))

(defn control-dependencies 
  "Wrapper for `Graph.control_dependencies()` using the default graph.

  See `tf.Graph.control_dependencies`
  for more details.

  When eager execution is enabled, any callable object in the `control_inputs`
  list will be called.

  Args:
    control_inputs: A list of `Operation` or `Tensor` objects which must be
      executed or computed before running the operations defined in the context.
      Can also be `None` to clear the control dependencies. If eager execution
      is enabled, any callable object in the `control_inputs` list will be
      called.

  Returns:
   A context manager that specifies control dependencies for all
   operations constructed within the context.
  "
  [ control_inputs ]
  (py/call-attr v2 "control_dependencies"  control_inputs ))

(defn convert-to-tensor 
  "Converts the given `value` to a `Tensor`.

  This function converts Python objects of various types to `Tensor`
  objects. It accepts `Tensor` objects, numpy arrays, Python lists,
  and Python scalars. For example:

  ```python
  import numpy as np

  def my_func(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return tf.matmul(arg, arg) + arg

  # The following calls are equivalent.
  value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
  value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
  value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
  ```

  This function can be useful when composing a new operation in Python
  (such as `my_func` in the example above). All standard Python op
  constructors apply this function to each of their Tensor-valued
  inputs, which allows those ops to accept numpy arrays, Python lists,
  and scalars in addition to `Tensor` objects.

  Note: This function diverges from default Numpy behavior for `float` and
    `string` types when `None` is present in a Python list or scalar. Rather
    than silently converting `None` values, an error will be thrown.

  Args:
    value: An object whose type has a registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the type
      is inferred from the type of `value`.
    dtype_hint: Optional element type for the returned tensor, used when dtype
      is None. In some cases, a caller may not have a dtype in mind when
      converting to a tensor, so dtype_hint can be used as a soft preference.
      If the conversion to `dtype_hint` is not possible, this argument has no
      effect.
    name: Optional name to use if a new `Tensor` is created.

  Returns:
    A `Tensor` based on `value`.

  Raises:
    TypeError: If no conversion function is registered for `value` to `dtype`.
    RuntimeError: If a registered conversion function returns an invalid value.
    ValueError: If the `value` is a tensor not of given `dtype` in graph mode.
  "
  [ value dtype dtype_hint name ]
  (py/call-attr v2 "convert_to_tensor"  value dtype dtype_hint name ))

(defn cos 
  "Computes cos of x element-wise.

    Given an input tensor, this function computes cosine of every
    element in the tensor. Input range is `(-inf, inf)` and
    output range is `[-1,1]`. If input lies outside the boundary, `nan`
    is returned.

    ```python
    x = tf.constant([-float(\"inf\"), -9, -0.5, 1, 1.2, 200, 10000, float(\"inf\")])
    tf.math.cos(x) ==> [nan -0.91113025 0.87758255 0.5403023 0.36235774 0.48718765 -0.95215535 nan]
    ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr v2 "cos"  x name ))

(defn cosh 
  "Computes hyperbolic cosine of x element-wise.

    Given an input tensor, this function computes hyperbolic cosine of every
    element in the tensor. Input range is `[-inf, inf]` and output range
    is `[1, inf]`.

    ```python
    x = tf.constant([-float(\"inf\"), -9, -0.5, 1, 1.2, 2, 10, float(\"inf\")])
    tf.math.cosh(x) ==> [inf 4.0515420e+03 1.1276259e+00 1.5430807e+00 1.8106556e+00 3.7621956e+00 1.1013233e+04 inf]
    ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr v2 "cosh"  x name ))

(defn cumsum 
  "Compute the cumulative sum of the tensor `x` along `axis`.

  By default, this op performs an inclusive cumsum, which means that the first
  element of the input is identical to the first element of the output:

  ```python
  tf.cumsum([a, b, c])  # [a, a + b, a + b + c]
  ```

  By setting the `exclusive` kwarg to `True`, an exclusive cumsum is performed
  instead:

  ```python
  tf.cumsum([a, b, c], exclusive=True)  # [0, a, a + b]
  ```

  By setting the `reverse` kwarg to `True`, the cumsum is performed in the
  opposite direction:

  ```python
  tf.cumsum([a, b, c], reverse=True)  # [a + b + c, b + c, c]
  ```

  This is more efficient than using separate `tf.reverse` ops.

  The `reverse` and `exclusive` kwargs can also be combined:

  ```python
  tf.cumsum([a, b, c], exclusive=True, reverse=True)  # [b + c, c, 0]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
      `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
      `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    axis: A `Tensor` of type `int32` (default: 0). Must be in the range
      `[-rank(x), rank(x))`.
    exclusive: If `True`, perform exclusive cumsum.
    reverse: A `bool` (default: False).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [x & {:keys [axis exclusive reverse name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "cumsum" [x] {:axis axis :exclusive exclusive :reverse reverse :name name }))

(defn custom-gradient 
  "Decorator to define a function with a custom gradient.

  This decorator allows fine grained control over the gradients of a sequence
  for operations.  This may be useful for multiple reasons, including providing
  a more efficient or numerically stable gradient for a sequence of operations.

  For example, consider the following function that commonly occurs in the
  computation of cross entropy and log likelihoods:

  ```python
  def log1pexp(x):
    return tf.math.log(1 + tf.exp(x))
  ```

  Due to numerical instability, the gradient this function evaluated at x=100 is
  NaN.  For example:

  ```python
  x = tf.constant(100.)
  y = log1pexp(x)
  dy = tf.gradients(y, x) # Will be NaN when evaluated.
  ```

  The gradient expression can be analytically simplified to provide numerical
  stability:

  ```python
  @tf.custom_gradient
  def log1pexp(x):
    e = tf.exp(x)
    def grad(dy):
      return dy * (1 - 1 / (1 + e))
    return tf.math.log(1 + e), grad
  ```

  With this definition, the gradient at x=100 will be correctly evaluated as
  1.0.

  See also `tf.RegisterGradient` which registers a gradient function for a
  primitive TensorFlow operation. `tf.custom_gradient` on the other hand allows
  for fine grained control over the gradient computation of a sequence of
  operations.

  Note that if the decorated function uses `Variable`s, the enclosing variable
  scope must be using `ResourceVariable`s.

  Args:
    f: function `f(*x)` that returns a tuple `(y, grad_fn)` where:
       - `x` is a sequence of `Tensor` inputs to the function.
       - `y` is a `Tensor` or sequence of `Tensor` outputs of applying
         TensorFlow operations in `f` to `x`.
       - `grad_fn` is a function with the signature `g(*grad_ys)` which returns
         a list of `Tensor`s - the derivatives of `Tensor`s in `y` with respect
         to the `Tensor`s in `x`.  `grad_ys` is a `Tensor` or sequence of
         `Tensor`s the same size as `y` holding the initial value gradients for
         each `Tensor` in `y`. In a pure mathematical sense, a vector-argument
         vector-valued function `f`'s derivatives should be its Jacobian matrix
         `J`. Here we are expressing the Jacobian `J` as a function `grad_fn`
         which defines how `J` will transform a vector `grad_ys` when
         left-multiplied with it (`grad_ys * J`). This functional representation
         of a matrix is convenient to use for chain-rule calculation
         (in e.g. the back-propagation algorithm).

         If `f` uses `Variable`s (that are not part of the
         inputs), i.e. through `get_variable`, then `grad_fn` should have
         signature `g(*grad_ys, variables=None)`, where `variables` is a list of
         the `Variable`s, and return a 2-tuple `(grad_xs, grad_vars)`, where
         `grad_xs` is the same as above, and `grad_vars` is a `list<Tensor>`
         with the derivatives of `Tensor`s in `y` with respect to the variables
         (that is, grad_vars has one Tensor per variable in variables).

  Returns:
    A function `h(x)` which returns the same value as `f(x)[0]` and whose
    gradient (as calculated by `tf.gradients`) is determined by `f(x)[1]`.
  "
  [ f ]
  (py/call-attr v2 "custom_gradient"  f ))

(defn device 
  "Specifies the device for ops created/executed in this context.

  `device_name` can be fully specified, as in \"/job:worker/task:1/device:cpu:0\",
  or partially specified, containing only a subset of the \"/\"-separated
  fields. Any fields which are specified override device annotations from outer
  scopes. For example:

  ```python
  with tf.device('/job:foo'):
    # ops created here have devices with /job:foo
    with tf.device('/job:bar/task:0/device:gpu:2'):
      # ops created here have the fully specified device above
    with tf.device('/device:gpu:1'):
      # ops created here have the device '/job:foo/device:gpu:1'
  ```

  Args:
    device_name: The device name to use in the context.

  Returns:
    A context manager that specifies the default device to use for newly
    created ops.

  Raises:
    RuntimeError: If a function is passed in.
  "
  [ device_name ]
  (py/call-attr v2 "device"  device_name ))

(defn divide 
  "Computes Python style division of `x` by `y`."
  [ x y name ]
  (py/call-attr v2 "divide"  x y name ))

(defn dynamic-partition 
  "Partitions `data` into `num_partitions` tensors using indices from `partitions`.

  For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
  becomes part of `outputs[partitions[js]]`.  The slices with `partitions[js] = i`
  are placed in `outputs[i]` in lexicographic order of `js`, and the first
  dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
  In detail,

  ```python
      outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]

      outputs[i] = pack([data[js, ...] for js if partitions[js] == i])
  ```

  `data.shape` must start with `partitions.shape`.

  For example:

  ```python
      # Scalar partitions.
      partitions = 1
      num_partitions = 2
      data = [10, 20]
      outputs[0] = []  # Empty with shape [0, 2]
      outputs[1] = [[10, 20]]

      # Vector partitions.
      partitions = [0, 0, 1, 1, 0]
      num_partitions = 2
      data = [10, 20, 30, 40, 50]
      outputs[0] = [10, 20, 50]
      outputs[1] = [30, 40]
  ```

  See `dynamic_stitch` for an example on how to merge partitions back.

  <div style=\"width:70%; margin:auto; margin-bottom:10px; margin-top:20px;\">
  <img style=\"width:100%\" src=\"https://www.tensorflow.org/images/DynamicPartition.png\" alt>
  </div>

  Args:
    data: A `Tensor`.
    partitions: A `Tensor` of type `int32`.
      Any shape.  Indices in the range `[0, num_partitions)`.
    num_partitions: An `int` that is `>= 1`.
      The number of partitions to output.
    name: A name for the operation (optional).

  Returns:
    A list of `num_partitions` `Tensor` objects with the same type as `data`.
  "
  [ data partitions num_partitions name ]
  (py/call-attr v2 "dynamic_partition"  data partitions num_partitions name ))

(defn dynamic-stitch 
  "Interleave the values from the `data` tensors into a single tensor.

  Builds a merged tensor such that

  ```python
      merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
  ```

  For example, if each `indices[m]` is scalar or vector, we have

  ```python
      # Scalar indices:
      merged[indices[m], ...] = data[m][...]

      # Vector indices:
      merged[indices[m][i], ...] = data[m][i, ...]
  ```

  Each `data[i].shape` must start with the corresponding `indices[i].shape`,
  and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
  must have `data[i].shape = indices[i].shape + constant`.  In terms of this
  `constant`, the output shape is

      merged.shape = [max(indices)] + constant

  Values are merged in order, so if an index appears in both `indices[m][i]` and
  `indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
  merged result. If you do not need this guarantee, ParallelDynamicStitch might
  perform better on some devices.

  For example:

  ```python
      indices[0] = 6
      indices[1] = [4, 1]
      indices[2] = [[5, 2], [0, 3]]
      data[0] = [61, 62]
      data[1] = [[41, 42], [11, 12]]
      data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
      merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
                [51, 52], [61, 62]]
  ```

  This method can be used to merge partitions created by `dynamic_partition`
  as illustrated on the following example:

  ```python
      # Apply function (increments x_i) on elements for which a certain condition
      # apply (x_i != -1 in this example).
      x=tf.constant([0.1, -1., 5.2, 4.3, -1., 7.4])
      condition_mask=tf.not_equal(x,tf.constant(-1.))
      partitioned_data = tf.dynamic_partition(
          x, tf.cast(condition_mask, tf.int32) , 2)
      partitioned_data[1] = partitioned_data[1] + 1.0
      condition_indices = tf.dynamic_partition(
          tf.range(tf.shape(x)[0]), tf.cast(condition_mask, tf.int32) , 2)
      x = tf.dynamic_stitch(condition_indices, partitioned_data)
      # Here x=[1.1, -1., 6.2, 5.3, -1, 8.4], the -1. values remain
      # unchanged.
  ```

  <div style=\"width:70%; margin:auto; margin-bottom:10px; margin-top:20px;\">
  <img style=\"width:100%\" src=\"https://www.tensorflow.org/images/DynamicStitch.png\" alt>
  </div>

  Args:
    indices: A list of at least 1 `Tensor` objects with type `int32`.
    data: A list with the same length as `indices` of `Tensor` objects with the same type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ indices data name ]
  (py/call-attr v2 "dynamic_stitch"  indices data name ))
(defn edit-distance 
  "Computes the Levenshtein distance between sequences.

  This operation takes variable-length sequences (`hypothesis` and `truth`),
  each provided as a `SparseTensor`, and computes the Levenshtein distance.
  You can normalize the edit distance by length of `truth` by setting
  `normalize` to true.

  For example, given the following input:

  ```python
  # 'hypothesis' is a tensor of shape `[2, 1]` with variable-length values:
  #   (0,0) = [\"a\"]
  #   (1,0) = [\"b\"]
  hypothesis = tf.SparseTensor(
      [[0, 0, 0],
       [1, 0, 0]],
      [\"a\", \"b\"],
      (2, 1, 1))

  # 'truth' is a tensor of shape `[2, 2]` with variable-length values:
  #   (0,0) = []
  #   (0,1) = [\"a\"]
  #   (1,0) = [\"b\", \"c\"]
  #   (1,1) = [\"a\"]
  truth = tf.SparseTensor(
      [[0, 1, 0],
       [1, 0, 0],
       [1, 0, 1],
       [1, 1, 0]],
      [\"a\", \"b\", \"c\", \"a\"],
      (2, 2, 2))

  normalize = True
  ```

  This operation would return the following:

  ```python
  # 'output' is a tensor of shape `[2, 2]` with edit distances normalized
  # by 'truth' lengths.
  output ==> [[inf, 1.0],  # (0,0): no truth, (0,1): no hypothesis
             [0.5, 1.0]]  # (1,0): addition, (1,1): no hypothesis
  ```

  Args:
    hypothesis: A `SparseTensor` containing hypothesis sequences.
    truth: A `SparseTensor` containing truth sequences.
    normalize: A `bool`. If `True`, normalizes the Levenshtein distance by
      length of `truth.`
    name: A name for the operation (optional).

  Returns:
    A dense `Tensor` with rank `R - 1`, where R is the rank of the
    `SparseTensor` inputs `hypothesis` and `truth`.

  Raises:
    TypeError: If either `hypothesis` or `truth` are not a `SparseTensor`.
  "
  [hypothesis truth  & {:keys [normalize name]} ]
    (py/call-attr-kw v2 "edit_distance" [hypothesis truth] {:normalize normalize :name name }))

(defn einsum 
  "Tensor contraction over specified indices and outer product.

  This function returns a tensor whose elements are defined by `equation`,
  which is written in a shorthand form inspired by the Einstein summation
  convention.  As an example, consider multiplying two matrices
  A and B to form a matrix C.  The elements of C are given by:

  ```
    C[i,k] = sum_j A[i,j] * B[j,k]
  ```

  The corresponding `equation` is:

  ```
    ij,jk->ik
  ```

  In general, the `equation` is obtained from the more familiar element-wise
  equation by
    1. removing variable names, brackets, and commas,
    2. replacing \"*\" with \",\",
    3. dropping summation signs, and
    4. moving the output to the right, and replacing \"=\" with \"->\".

  Many common operations can be expressed in this way.  For example:

  ```python
  # Matrix multiplication
  >>> einsum('ij,jk->ik', m0, m1)  # output[i,k] = sum_j m0[i,j] * m1[j, k]

  # Dot product
  >>> einsum('i,i->', u, v)  # output = sum_i u[i]*v[i]

  # Outer product
  >>> einsum('i,j->ij', u, v)  # output[i,j] = u[i]*v[j]

  # Transpose
  >>> einsum('ij->ji', m)  # output[j,i] = m[i,j]

  # Trace
  >>> einsum('ii', m)  # output[j,i] = trace(m) = sum_i m[i, i]

  # Batch matrix multiplication
  >>> einsum('aij,ajk->aik', s, t)  # out[a,i,k] = sum_j s[a,i,j] * t[a, j, k]
  ```

  To enable and control broadcasting, use an ellipsis.  For example, to do
  batch matrix multiplication, you could use:

  ```python
  >>> einsum('...ij,...jk->...ik', u, v)
  ```

  This function behaves like `numpy.einsum`, but does not support:

  * Subscripts where an axis appears more than once for a single input
    (e.g. `ijj,k->ik`) unless it is a trace (e.g. `ijji`).

  Args:
    equation: a `str` describing the contraction, in the same format as
      `numpy.einsum`.
    *inputs: the inputs to contract (each one a `Tensor`), whose shapes should
      be consistent with `equation`.
    name: A name for the operation (optional).

  Returns:
    The contracted `Tensor`, with shape determined by `equation`.

  Raises:
    ValueError: If
      - the format of `equation` is incorrect,
      - the number of inputs implied by `equation` does not match `len(inputs)`,
      - an axis appears in the output subscripts but not in any of the inputs,
      - the number of dimensions of an input differs from the number of
        indices in its subscript, or
      - the input shapes are inconsistent along a particular axis.
  "
  [ equation ]
  (py/call-attr v2 "einsum"  equation ))

(defn enable-v2-behavior 
  "Enables TensorFlow 2.x behaviors.

  This function can be called at the beginning of the program (before `Tensors`,
  `Graphs` or other structures have been created, and before devices have been
  initialized. It switches all global behaviors that are different between
  TensorFlow 1.x and 2.x to behave as intended for 2.x.

  This function is called in the main TensorFlow `__init__.py` file, user should
  not need to call it, except during complex migrations.
  "
  [  ]
  (py/call-attr v2 "enable_v2_behavior"  ))

(defn ensure-shape 
  "Updates the shape of a tensor and checks at runtime that the shape holds.

  For example:
  ```python
  x = tf.compat.v1.placeholder(tf.int32)
  print(x.shape)
  ==> TensorShape(None)
  y = x * 2
  print(y.shape)
  ==> TensorShape(None)

  y = tf.ensure_shape(y, (None, 3, 3))
  print(y.shape)
  ==> TensorShape([Dimension(None), Dimension(3), Dimension(3)])

  with tf.compat.v1.Session() as sess:
    # Raises tf.errors.InvalidArgumentError, because the shape (3,) is not
    # compatible with the shape (None, 3, 3)
    sess.run(y, feed_dict={x: [1, 2, 3]})

  ```

  NOTE: This differs from `Tensor.set_shape` in that it sets the static shape
  of the resulting tensor and enforces it at runtime, raising an error if the
  tensor's runtime shape is incompatible with the specified shape.
  `Tensor.set_shape` sets the static shape of the tensor without enforcing it
  at runtime, which may result in inconsistencies between the statically-known
  shape of tensors and the runtime value of tensors.

  Args:
    x: A `Tensor`.
    shape: A `TensorShape` representing the shape of this tensor, a
      `TensorShapeProto`, a list, a tuple, or None.
    name: A name for this operation (optional). Defaults to \"EnsureShape\".

  Returns:
    A `Tensor`. Has the same type and contents as `x`. At runtime, raises a
    `tf.errors.InvalidArgumentError` if `shape` is incompatible with the shape
    of `x`.
  "
  [ x shape name ]
  (py/call-attr v2 "ensure_shape"  x shape name ))

(defn equal 
  "Returns the truth value of (x == y) element-wise.

  Usage:

  ```python
  x = tf.constant([2, 4])
  y = tf.constant(2)
  tf.math.equal(x, y) ==> array([True, False])

  x = tf.constant([2, 4])
  y = tf.constant([2, 4])
  tf.math.equal(x, y) ==> array([True,  True])
  ```

  **NOTE**: `Equal` supports broadcasting. More about broadcasting [here](
  https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html)

  Args:
    x: A `Tensor` or `SparseTensor` or `IndexedSlices`.
    y: A `Tensor` or `SparseTensor` or `IndexedSlices`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type bool with the same size as that of x or y.
  "
  [ x y name ]
  (py/call-attr v2 "equal"  x y name ))

(defn executing-eagerly 
  "Returns True if the current thread has eager execution enabled.

  Eager execution is typically enabled via
  `tf.compat.v1.enable_eager_execution`, but may also be enabled within the
  context of a Python function via tf.contrib.eager.py_func.
  "
  [  ]
  (py/call-attr v2 "executing_eagerly"  ))

(defn exp 
  "Computes exponential of x element-wise.  \\(y = e^x\\).

    This function computes the exponential of every element in the input tensor.
    i.e. `exp(x)` or `e^(x)`, where `x` is the input tensor.
    `e` denotes Euler's number and is approximately equal to 2.718281.
    Output is positive for any real input.

    ```python
    x = tf.constant(2.0)
    tf.math.exp(x) ==> 7.389056

    x = tf.constant([2.0, 8.0])
    tf.math.exp(x) ==> array([7.389056, 2980.958], dtype=float32)
    ```

    For complex numbers, the exponential value is calculated as follows:

    ```
    e^(x+iy) = e^x * e^iy = e^x * (cos y + i sin y)
    ```

    Let's consider complex number 1+1j as an example.
    e^1 * (cos 1 + i sin 1) = 2.7182818284590 * (0.54030230586+0.8414709848j)

    ```python
    x = tf.constant(1 + 1j)
    tf.math.exp(x) ==> 1.4686939399158851+2.2873552871788423j
    ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr v2 "exp"  x name ))

(defn expand-dims 
  "Inserts a dimension of 1 into a tensor's shape.

  Given a tensor `input`, this operation inserts a dimension of 1 at the
  dimension index `axis` of `input`'s shape. The dimension index `axis` starts
  at zero; if you specify a negative number for `axis` it is counted backward
  from the end.

  This operation is useful if you want to add a batch dimension to a single
  element. For example, if you have a single image of shape `[height, width,
  channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
  which will make the shape `[1, height, width, channels]`.

  Other examples:

  ```python
  # 't' is a tensor of shape [2]
  tf.shape(tf.expand_dims(t, 0))  # [1, 2]
  tf.shape(tf.expand_dims(t, 1))  # [2, 1]
  tf.shape(tf.expand_dims(t, -1))  # [2, 1]

  # 't2' is a tensor of shape [2, 3, 5]
  tf.shape(tf.expand_dims(t2, 0))  # [1, 2, 3, 5]
  tf.shape(tf.expand_dims(t2, 2))  # [2, 3, 1, 5]
  tf.shape(tf.expand_dims(t2, 3))  # [2, 3, 5, 1]
  ```

  This operation requires that:

  `-1-input.dims() <= dim <= input.dims()`

  This operation is related to `squeeze()`, which removes dimensions of
  size 1.

  Args:
    input: A `Tensor`.
    axis: 0-D (scalar). Specifies the dimension index at which to expand the
      shape of `input`. Must be in the range `[-rank(input) - 1, rank(input)]`.
    name: The name of the output `Tensor` (optional).

  Returns:
    A `Tensor` with the same data as `input`, but its shape has an additional
    dimension of size 1 added.
  "
  [ input axis name ]
  (py/call-attr v2 "expand_dims"  input axis name ))

(defn extract-volume-patches 
  "Extract `patches` from `input` and put them in the \"depth\" output dimension. 3D extension of `extract_image_patches`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      5-D Tensor with shape `[batch, in_planes, in_rows, in_cols, depth]`.
    ksizes: A list of `ints` that has length `>= 5`.
      The size of the sliding window for each dimension of `input`.
    strides: A list of `ints` that has length `>= 5`.
      1-D of length 5. How far the centers of two consecutive patches are in
      `input`. Must be: `[1, stride_planes, stride_rows, stride_cols, 1]`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.

      We specify the size-related attributes as:

      ```python
            ksizes = [1, ksize_planes, ksize_rows, ksize_cols, 1]
            strides = [1, stride_planes, strides_rows, strides_cols, 1]
      ```
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input ksizes strides padding name ]
  (py/call-attr v2 "extract_volume_patches"  input ksizes strides padding name ))

(defn eye 
  "Construct an identity matrix, or a batch of matrices.

  ```python
  # Construct one identity matrix.
  tf.eye(2)
  ==> [[1., 0.],
       [0., 1.]]

  # Construct a batch of 3 identity matricies, each 2 x 2.
  # batch_identity[i, :, :] is a 2 x 2 identity matrix, i = 0, 1, 2.
  batch_identity = tf.eye(2, batch_shape=[3])

  # Construct one 2 x 3 \"identity\" matrix
  tf.eye(2, num_columns=3)
  ==> [[ 1.,  0.,  0.],
       [ 0.,  1.,  0.]]
  ```

  Args:
    num_rows: Non-negative `int32` scalar `Tensor` giving the number of rows
      in each batch matrix.
    num_columns: Optional non-negative `int32` scalar `Tensor` giving the number
      of columns in each batch matrix.  Defaults to `num_rows`.
    batch_shape:  A list or tuple of Python integers or a 1-D `int32` `Tensor`.
      If provided, the returned `Tensor` will have leading batch dimensions of
      this shape.
    dtype:  The type of an element in the resulting `Tensor`
    name:  A name for this `Op`.  Defaults to \"eye\".

  Returns:
    A `Tensor` of shape `batch_shape + [num_rows, num_columns]`
  "
  [num_rows num_columns batch_shape & {:keys [dtype name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "eye" [num_rows num_columns batch_shape] {:dtype dtype :name name }))

(defn fill 
  "Creates a tensor filled with a scalar value.

  This operation creates a tensor of shape `dims` and fills it with `value`.

  For example:

  ```
  # Output tensor has shape [2, 3].
  fill([2, 3], 9) ==> [[9, 9, 9]
                       [9, 9, 9]]
  ```

  `tf.fill` differs from `tf.constant` in a few ways:

  *   `tf.fill` only supports scalar contents, whereas `tf.constant` supports
      Tensor values.
  *   `tf.fill` creates an Op in the computation graph that constructs the
  actual
      Tensor value at runtime. This is in contrast to `tf.constant` which embeds
      the entire Tensor into the graph with a `Const` node.
  *   Because `tf.fill` evaluates at graph runtime, it supports dynamic shapes
      based on other runtime Tensors, unlike `tf.constant`.

  Args:
    dims: A `Tensor`. Must be one of the following types: `int32`, `int64`. 1-D.
      Represents the shape of the output tensor.
    value: A `Tensor`. 0-D (scalar). Value to fill the returned tensor.
      @compatibility(numpy) Equivalent to np.full @end_compatibility
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
  "
  [ dims value name ]
  (py/call-attr v2 "fill"  dims value name ))

(defn fingerprint 
  "Generates fingerprint values.

  Generates fingerprint values of `data`.

  Fingerprint op considers the first dimension of `data` as the batch dimension,
  and `output[i]` contains the fingerprint value generated from contents in
  `data[i, ...]` for all `i`.

  Fingerprint op writes fingerprint values as byte arrays. For example, the
  default method `farmhash64` generates a 64-bit fingerprint value at a time.
  This 8-byte value is written out as an `tf.uint8` array of size 8, in
  little-endian order.

  For example, suppose that `data` has data type `tf.int32` and shape (2, 3, 4),
  and that the fingerprint method is `farmhash64`. In this case, the output
  shape is (2, 8), where 2 is the batch dimension size of `data`, and 8 is the
  size of each fingerprint value in bytes. `output[0, :]` is generated from
  12 integers in `data[0, :, :]` and similarly `output[1, :]` is generated from
  other 12 integers in `data[1, :, :]`.

  Note that this op fingerprints the raw underlying buffer, and it does not
  fingerprint Tensor's metadata such as data type and/or shape. For example, the
  fingerprint values are invariant under reshapes and bitcasts as long as the
  batch dimension remain the same:

  ```python
  tf.fingerprint(data) == tf.fingerprint(tf.reshape(data, ...))
  tf.fingerprint(data) == tf.fingerprint(tf.bitcast(data, ...))
  ```

  For string data, one should expect `tf.fingerprint(data) !=
  tf.fingerprint(tf.string.reduce_join(data))` in general.

  Args:
    data: A `Tensor`. Must have rank 1 or higher.
    method: A `Tensor` of type `tf.string`. Fingerprint method used by this op.
      Currently available method is `farmhash64`.
    name: A name for the operation (optional).

  Returns:
    A two-dimensional `Tensor` of type `tf.uint8`. The first dimension equals to
    `data`'s first dimension, and the second dimension size depends on the
    fingerprint algorithm.
  "
  [data & {:keys [method name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "fingerprint" [data] {:method method :name name }))

(defn floor 
  "Returns element-wise largest integer not greater than x.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr v2 "floor"  x name ))

(defn foldl 
  "foldl on the list of tensors unpacked from `elems` on dimension 0.

  This foldl operator repeatedly applies the callable `fn` to a sequence
  of elements from first to last. The elements are made of the tensors
  unpacked from `elems` on dimension 0. The callable fn takes two tensors as
  arguments. The first argument is the accumulated value computed from the
  preceding invocation of fn, and the second is the value at the current
  position of `elems`. If `initializer` is None, `elems` must contain at least
  one element, and its first element is used as the initializer.

  Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
  of the result tensor is fn(initializer, values[0]).shape`.

  This method also allows multi-arity `elems` and output of `fn`.  If `elems`
  is a (possibly nested) list or tuple of tensors, then each of these tensors
  must have a matching first (unpack) dimension.  The signature of `fn` may
  match the structure of `elems`.  That is, if `elems` is
  `(t1, [t2, t3, [t4, t5]])`, then an appropriate signature for `fn` is:
  `fn = lambda (t1, [t2, t3, [t4, t5]]):`.

  Args:
    fn: The callable to be performed.
    elems: A tensor or (possibly nested) sequence of tensors, each of which will
      be unpacked along their first dimension.  The nested sequence of the
      resulting slices will be the first argument to `fn`.
    initializer: (optional) A tensor or (possibly nested) sequence of tensors,
      as the initial value for the accumulator.
    parallel_iterations: (optional) The number of iterations allowed to run in
      parallel.
    back_prop: (optional) True enables support for back propagation.
    swap_memory: (optional) True enables GPU-CPU memory swapping.
    name: (optional) Name prefix for the returned tensors.

  Returns:
    A tensor or (possibly nested) sequence of tensors, resulting from applying
    `fn` consecutively to the list of tensors unpacked from `elems`, from first
    to last.

  Raises:
    TypeError: if `fn` is not callable.

  Example:
    ```python
    elems = tf.constant([1, 2, 3, 4, 5, 6])
    sum = foldl(lambda a, x: a + x, elems)
    # sum == 21
    ```
  "
  [fn elems initializer & {:keys [parallel_iterations back_prop swap_memory name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "foldl" [fn elems initializer] {:parallel_iterations parallel_iterations :back_prop back_prop :swap_memory swap_memory :name name }))

(defn foldr 
  "foldr on the list of tensors unpacked from `elems` on dimension 0.

  This foldr operator repeatedly applies the callable `fn` to a sequence
  of elements from last to first. The elements are made of the tensors
  unpacked from `elems`. The callable fn takes two tensors as arguments.
  The first argument is the accumulated value computed from the preceding
  invocation of fn, and the second is the value at the current position of
  `elems`. If `initializer` is None, `elems` must contain at least one element,
  and its first element is used as the initializer.

  Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
  of the result tensor is `fn(initializer, values[0]).shape`.

  This method also allows multi-arity `elems` and output of `fn`.  If `elems`
  is a (possibly nested) list or tuple of tensors, then each of these tensors
  must have a matching first (unpack) dimension.  The signature of `fn` may
  match the structure of `elems`.  That is, if `elems` is
  `(t1, [t2, t3, [t4, t5]])`, then an appropriate signature for `fn` is:
  `fn = lambda (t1, [t2, t3, [t4, t5]]):`.

  Args:
    fn: The callable to be performed.
    elems: A tensor or (possibly nested) sequence of tensors, each of which will
      be unpacked along their first dimension.  The nested sequence of the
      resulting slices will be the first argument to `fn`.
    initializer: (optional) A tensor or (possibly nested) sequence of tensors,
      as the initial value for the accumulator.
    parallel_iterations: (optional) The number of iterations allowed to run in
      parallel.
    back_prop: (optional) True enables support for back propagation.
    swap_memory: (optional) True enables GPU-CPU memory swapping.
    name: (optional) Name prefix for the returned tensors.

  Returns:
    A tensor or (possibly nested) sequence of tensors, resulting from applying
    `fn` consecutively to the list of tensors unpacked from `elems`, from last
    to first.

  Raises:
    TypeError: if `fn` is not callable.

  Example:
    ```python
    elems = [1, 2, 3, 4, 5, 6]
    sum = foldr(lambda a, x: a + x, elems)
    # sum == 21
    ```
  "
  [fn elems initializer & {:keys [parallel_iterations back_prop swap_memory name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "foldr" [fn elems initializer] {:parallel_iterations parallel_iterations :back_prop back_prop :swap_memory swap_memory :name name }))

(defn function 
  "Creates a callable TensorFlow graph from a Python function.

  `function` constructs a callable that executes a TensorFlow graph
  (`tf.Graph`) created by tracing the TensorFlow operations in `func`.
  This allows the TensorFlow runtime to apply optimizations and exploit
  parallelism in the computation defined by `func`.

  _Example Usage_

  ```python
  def f(x, y):
    return tf.reduce_mean(tf.multiply(x ** 2, 3) + y)

  g = tf.function(f)

  x = tf.constant([[2.0, 3.0]])
  y = tf.constant([[3.0, -2.0]])

  # `f` and `g` will return the same value, but `g` will be executed as a
  # TensorFlow graph.
  assert f(x, y).numpy() == g(x, y).numpy()

  # Tensors and tf.Variables used by the Python function are captured in the
  # graph.
  @tf.function
  def h():
    return f(x, y)

  assert (h().numpy() == f(x, y).numpy()).all()

  # Data-dependent control flow is also captured in the graph. Supported
  # control flow statements include `if`, `for`, `while`, `break`, `continue`,
  # `return`.
  @tf.function
  def g(x):
    if tf.reduce_sum(x) > 0:
      return x * x
    else:
      return -x // 2

  # print and TensorFlow side effects are supported, but exercise caution when
  # using Python side effects like mutating objects, saving to files, etc.
  l = []

  @tf.function
  def g(x):
    for i in x:
      print(i)                              # Works
      tf.compat.v1.assign(v, i)                       # Works
      tf.compat.v1.py_func(lambda i: l.append(i))(i)  # Works
      l.append(i)                           # Caution! Doesn't work.
  ```

  Note that unlike other TensorFlow operations, we don't convert python
  numerical inputs to tensors. Moreover, a new graph is generated for each
  distinct python numerical value, for example calling `g(2)` and `g(3)` will
  generate two new graphs (while only one is generated if you call
  `g(tf.constant(2))` and `g(tf.constant(3))`). Therefore, python numerical
  inputs should be restricted to arguments that will have few distinct values,
  such as hyperparameters like the number of layers in a neural network. This
  allows TensorFlow to optimize each variant of the neural network.

  _Referencing `tf.Variable`s_

  The Python function `func` may reference stateful objects (such as
  `tf.Variable`).
  These are captured as implicit inputs to the callable returned by `function`.
  For example:

  ```python
  c = tf.Variable(0)

  @tf.function
  def f(x):
    c.assign_add(1)
    return x + tf.compat.v1.to_float(c)

  assert int(c) == 0
  assert f(1.0) == 2.0
  assert int(c) == 1
  assert f(1.0) == 3.0
  assert int(c) == 2
  ```

  `function` can be applied to methods of an object. For example:

  ```python
  class Dense(object):
    def __init__(self):
      self.W = tf.Variable(tf.compat.v1.glorot_uniform_initializer()((10, 10)))
      self.b = tf.Variable(tf.zeros(10))

    @tf.function
    def compute(self, x):
      return tf.matmul(x, self.W) + self.b

  d1 = Dense()
  d2 = Dense()
  x = tf.random.uniform((10, 10))
  # d1 and d2 are using distinct variables
  assert not (d1.compute(x).numpy() == d2.compute(x).numpy()).all()
  ```

  _Usage with `tf.keras`_

  The `call` methods of a `tf.keras.Model` subclass can be decorated with
  `function` in order to apply graph execution optimizations on it.
  For example:

  ```python
  class MyModel(tf.keras.Model):
    def __init__(self, keep_probability=0.2):
      super(MyModel, self).__init__()
      self.dense1 = tf.keras.layers.Dense(4)
      self.dense2 = tf.keras.layers.Dense(5)
      self.keep_probability = keep_probability

    @tf.function
    def call(self, inputs, training=True):
      y = self.dense2(self.dense1(inputs))
      if training:
        return tf.nn.dropout(y, self.keep_probability)
      else:
        return y

  model = MyModel()
  model(x, training=True)  # executes a graph, with dropout
  model(x, training=False) # executes a graph, without dropout
  ```

  _Input Signatures_

  `function` instantiates a separate graph for every unique set of input
  shapes and datatypes. For example, the following code snippet will result
  in three distinct graphs being traced, as each input has a different
  shape.

  ```python
  @tf.function
  def f(x): return tf.add(x, 1.)

  scalar = tf.constant(1.0)
  vector = tf.constant([1.0, 1.0])
  matrix = tf.constant([[3.0]])

  f(scalar)
  f(vector)
  f(matrix)
  ```

  An \"input signature\" can be optionally provided to `function` to control
  the graphs traced. The input signature specifies the shape and type of each
  `Tensor` argument to the function using a `tf.TensorSpec` object. For example,
  the following code snippet ensures that a single graph is created where the
  input `Tensor` is required to be a floating point tensor with no restrictions
  on shape.

  ```python
  @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
  def f(x): return tf.add(x, 1.)
  ```

  When an `input_signature` is specified, the callable will convert the inputs
  to the specified TensorSpecs.

  _Tracing and staging_

  When `autograph` is `True`, all Python control flow that depends on `Tensor`
  values is staged into a TensorFlow graph. When `autograph` is `False`, the
  function is traced and control flow is not allowed to depend on data.

  Note that `function` only stages TensorFlow operations, all Python code that
  `func` executes and does not depend on data will shape the _construction_ of
  the graph.
  For example, consider the following:

  ```python
  import numpy as np

  def add_noise():
    return tf.eye(5) + np.random.randn(5, 5)

  traced = tf.function(add_noise)
  ```

  `add_noise()` will return a different output every time it is invoked.
  However, `traced()` will return the same value every time it is called,
  since a particular random value generated by the `np.random.randn` call will
  be inserted in the traced/staged TensorFlow graph as a constant. In this
  particular example, replacing `np.random.randn(5, 5)` with
  `tf.random.normal((5, 5))` will result in the same behavior for `add_noise()`
  and `traced()`.

  _Python Side-Effects_

  A corollary of the previous discussion on tracing is the following: If a
  Python function `func` has Python side-effects, then executing `func` multiple
  times may not be semantically equivalent to executing `F = tf.function(func)`
  multiple times; this difference is due to the fact that `function` only
  captures the subgraph of TensorFlow operations that is constructed when `func`
  is invoked to trace a graph.

  The same is true if code with Python side effects is used inside control flow,
  such as a loop. If your code uses side effects that are not intended to
  control graph construction, wrap them inside `tf.compat.v1.py_func`.

  _Retracing_

  A single tf.function object might need to map to multiple computation graphs
  under the hood. This should be visible only as performance (tracing graphs has
  a nonzero computational and memory cost) but should not affect the correctness
  of the program. A traced function should return the same result as it would
  when run eagerly, assuming no unintended Python side-effects.

  Calling a `tf.function` with tensor arguments of different dtypes should lead
  to at least one computational graph per distinct set of dtypes. Alternatively,
  always calling a `tf.function` with tensor arguments of the same shapes and
  dtypes and the same non-tensor arguments should not lead to additional
  retracings of your function.

  Other than that, TensorFlow reserves the right to retrace functions as many
  times as needed, to ensure that traced functions behave as they would when run
  eagerly and to provide the best end-to-end performance. For example, the
  behavior of how many traces TensorFlow will do when the function is repeatedly
  called with different python scalars as arguments is left undefined to allow
  for future optimizations.

  To control the tracing behavior, use the following tools:
   - different `tf.function` objects are guaranteed to not share traces; and
   - specifying a signature or using concrete function objects returned from
     get_concrete_function() guarantees that only one function graph will be
     built.

  Args:
    func: function to be compiled. If `func` is None, returns a decorator that
      can be invoked with a single argument - `func`. The end result is
      equivalent to providing all the arguments up front. In other words,
      `tf.function(input_signature=...)(func)` is equivalent to
      `tf.function(func, input_signature=...)`. The former can be used to
      decorate Python functions, for example:
        @tf.function(input_signature=...)
        def foo(...): ...
    input_signature: A possibly nested sequence of `tf.TensorSpec` objects
      specifying the shapes and dtypes of the Tensors that will be supplied to
      this function. If `None`, a separate function is instantiated for each
      inferred input signature.  If input_signature is specified, every input to
      `func` must be a `Tensor`, and `func` cannot accept `**kwargs`.
    autograph: Whether autograph should be applied on `func` before tracing a
      graph. This allows for dynamic control flow (Python if's, loops etc.)
      in the traced graph. See https://www.tensorflow.org/guide/autograph for
        more information.
    experimental_autograph_options: Experimental knobs (in the form of a tuple
      of tensorflow.autograph.Feature values) to control behavior when
      autograph=True.
    experimental_relax_shapes: When true, argument shapes may be relaxed to
      avoid unecessary retracing.
    experimental_compile: If false, execute the function in a regular way. The
      function is optimized by some graph rewrite passes (some ops might be
      clustered into a single op) and interpreted by the standard TensorFlow
      executor, which dispatches op kernels one by one as they become
      executable. Set it to false when directly running a multi-device function
      on TPUs (e.g. two TPU cores, one TPU core and its host CPU). If True, the
      function is compiled directly by XLA (https://www.tensorflow.org/xla).
      XLA would fuse all the ops and emit more efficient code to run for some
      devices (e.g. TPU, XLA_GPU) and some use cases (e.g. dense tensor
      computation). It requires that the whole function is compilable by XLA
      (e.g. static tensor shape, a subset of operations, no string, compile-time
      constant input, etc). If None (default), compile the function with XLA
      when running on TPU and go through the regular function execution path
      when running on other devices. Note: TensorArrays on TPU don't work with
      standard TensorFlow executor.

  Returns:
     If `func` is not None, returns a callable that will execute the compiled
     function (and return zero or more `tf.Tensor` objects).
     If `func` is None, returns a decorator that, when invoked with a single
     `func` argument, returns a callable equivalent to the case above.

  Raises:
    TypeError: If `input_signature` is neither `None` nor a sequence of
      `TensorSpec` objects.
  "
  [func input_signature & {:keys [autograph experimental_autograph_options experimental_relax_shapes experimental_compile]
                       :or {experimental_autograph_options None experimental_compile None}} ]
    (py/call-attr-kw v2 "function" [func input_signature] {:autograph autograph :experimental_autograph_options experimental_autograph_options :experimental_relax_shapes experimental_relax_shapes :experimental_compile experimental_compile }))

(defn gather 
  "Gather slices from params axis axis according to indices.

  Gather slices from params axis `axis` according to `indices`.  `indices` must
  be an integer tensor of any dimension (usually 0-D or 1-D).

  For 0-D (scalar) `indices`:

  > `output`$$[p_0,          ..., p_{axis-1},        \hspace{5.1em}
  >            p_{axis + 1}, ..., p_{N-1}]$$ =\
  > `params`$$[p_0,          ..., p_{axis-1},        \hspace{1em}
  >            indices,                              \hspace{1em}
  >            p_{axis + 1}, ..., p_{N-1}]$$.

  For 1-D (vector) `indices` with `batch_dims=0`:

  > `output`$$[p_0,          ..., p_{axis-1},        \hspace{2.6em}
  >            i,                                    \hspace{2.6em}
  >            p_{axis + 1}, ..., p_{N-1}]$$ =\
  > `params`$$[p_0,          ..., p_{axis-1},        \hspace{1em}
  >            indices[i],                           \hspace{1em}
  >            p_{axis + 1}, ..., p_{N-1}]$$.

  In the general case, produces an output tensor where:

  $$\begin{align*}
  output[p_0,             &..., p_{axis-1},                       &
       &i_{B},           ..., i_{M-1},                          &
       p_{axis + 1},    &..., p_{N-1}]                          = \\
  params[p_0,             &..., p_{axis-1},                       &
       indices[p_0, ..., p_{B-1}, &i_{B}, ..., i_{M-1}],        &
       p_{axis + 1},    &..., p_{N-1}]
  \end{align*}$$

  Where $$N$$=`ndims(params)`, $$M$$=`ndims(indices)`, and $$B$$=`batch_dims`.
  Note that params.shape[:batch_dims] must be identical to
  indices.shape[:batch_dims].

  The shape of the output tensor is:

  > `output.shape = params.shape[:axis] + indices.shape[batch_dims:] +
  > params.shape[axis + 1:]`.

  Note that on CPU, if an out of bound index is found, an error is returned.
  On GPU, if an out of bound index is found, a 0 is stored in the corresponding
  output value.

  See also `tf.gather_nd`.

  <div style=\"width:70%; margin:auto; margin-bottom:10px; margin-top:20px;\">
  <img style=\"width:100%\" src=\"https://www.tensorflow.org/images/Gather.png\"
  alt>
  </div>

  Args:
    params: The `Tensor` from which to gather values. Must be at least rank
      `axis + 1`.
    indices: The index `Tensor`.  Must be one of the following types: `int32`,
      `int64`. Must be in range `[0, params.shape[axis])`.
    validate_indices: Deprecated, does nothing.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`. The
      `axis` in `params` to gather `indices` from. Must be greater than or equal
      to `batch_dims`.  Defaults to the first non-batch dimension. Supports
      negative indexes.
    batch_dims: An `integer`.  The number of batch dimensions.  Must be less
      than `rank(indices)`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `params`.
  "
  [params indices validate_indices axis & {:keys [batch_dims name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "gather" [params indices validate_indices axis] {:batch_dims batch_dims :name name }))

(defn gather-nd 
  "Gather slices from `params` into a Tensor with shape specified by `indices`.

  `indices` is an K-dimensional integer tensor, best thought of as a
  (K-1)-dimensional tensor of indices into `params`, where each element defines
  a slice of `params`:

      output[\\(i_0, ..., i_{K-2}\\)] = params[indices[\\(i_0, ..., i_{K-2}\\)]]

  Whereas in `tf.gather` `indices` defines slices into the first
  dimension of `params`, in `tf.gather_nd`, `indices` defines slices into the
  first `N` dimensions of `params`, where `N = indices.shape[-1]`.

  The last dimension of `indices` can be at most the rank of
  `params`:

      indices.shape[-1] <= params.rank

  The last dimension of `indices` corresponds to elements
  (if `indices.shape[-1] == params.rank`) or slices
  (if `indices.shape[-1] < params.rank`) along dimension `indices.shape[-1]`
  of `params`.  The output tensor has shape

      indices.shape[:-1] + params.shape[indices.shape[-1]:]

  Additionally both 'params' and 'indices' can have M leading batch
  dimensions that exactly match. In this case 'batch_dims' must be M.

  Note that on CPU, if an out of bound index is found, an error is returned.
  On GPU, if an out of bound index is found, a 0 is stored in the
  corresponding output value.

  Some examples below.

  Simple indexing into a matrix:

  ```python
      indices = [[0, 0], [1, 1]]
      params = [['a', 'b'], ['c', 'd']]
      output = ['a', 'd']
  ```

  Slice indexing into a matrix:

  ```python
      indices = [[1], [0]]
      params = [['a', 'b'], ['c', 'd']]
      output = [['c', 'd'], ['a', 'b']]
  ```

  Indexing into a 3-tensor:

  ```python
      indices = [[1]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [[['a1', 'b1'], ['c1', 'd1']]]


      indices = [[0, 1], [1, 0]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [['c0', 'd0'], ['a1', 'b1']]


      indices = [[0, 0, 1], [1, 0, 1]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = ['b0', 'b1']
  ```

  The examples below are for the case when only indices have leading extra
  dimensions. If both 'params' and 'indices' have leading batch dimensions, use
  the 'batch_dims' parameter to run gather_nd in batch mode.

  Batched indexing into a matrix:

  ```python
      indices = [[[0, 0]], [[0, 1]]]
      params = [['a', 'b'], ['c', 'd']]
      output = [['a'], ['b']]
  ```

  Batched slice indexing into a matrix:

  ```python
      indices = [[[1]], [[0]]]
      params = [['a', 'b'], ['c', 'd']]
      output = [[['c', 'd']], [['a', 'b']]]
  ```

  Batched indexing into a 3-tensor:

  ```python
      indices = [[[1]], [[0]]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [[[['a1', 'b1'], ['c1', 'd1']]],
                [[['a0', 'b0'], ['c0', 'd0']]]]

      indices = [[[0, 1], [1, 0]], [[0, 0], [1, 1]]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [[['c0', 'd0'], ['a1', 'b1']],
                [['a0', 'b0'], ['c1', 'd1']]]


      indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [['b0', 'b1'], ['d0', 'c1']]
  ```

  Examples with batched 'params' and 'indices':

  ```python
      batch_dims = 1
      indices = [[1], [0]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [['c0', 'd0'], ['a1', 'b1']]

      batch_dims = 1
      indices = [[[1]], [[0]]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [[['c0', 'd0']], [['a1', 'b1']]]

      batch_dims = 1
      indices = [[[1, 0]], [[0, 1]]]
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]]
      output = [['c0'], ['b1']]
  ```

  See also `tf.gather`.

  Args:
    params: A `Tensor`. The tensor from which to gather values.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Index tensor.
    name: A name for the operation (optional).
    batch_dims: An integer or a scalar 'Tensor'. The number of batch dimensions.

  Returns:
    A `Tensor`. Has the same type as `params`.
  "
  [params indices & {:keys [batch_dims name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "gather_nd" [params indices] {:batch_dims batch_dims :name name }))

(defn get-logger 
  "Return TF logger instance."
  [  ]
  (py/call-attr v2 "get_logger"  ))
(defn get-static-value 
  "Returns the constant value of the given tensor, if efficiently calculable.

  This function attempts to partially evaluate the given tensor, and
  returns its value as a numpy ndarray if this succeeds.

  Compatibility(V1): If `constant_value(tensor)` returns a non-`None` result, it
  will no longer be possible to feed a different value for `tensor`. This allows
  the result of this function to influence the graph that is constructed, and
  permits static shape optimizations.

  Args:
    tensor: The Tensor to be evaluated.
    partial: If True, the returned numpy array is allowed to have partially
      evaluated values. Values that can't be evaluated will be None.

  Returns:
    A numpy ndarray containing the constant value of the given `tensor`,
    or None if it cannot be calculated.

  Raises:
    TypeError: if tensor is not an ops.Tensor.
  "
  [tensor  & {:keys [partial]} ]
    (py/call-attr-kw v2 "get_static_value" [tensor] {:partial partial }))

(defn grad-pass-through 
  "Creates a grad-pass-through op with the forward behavior provided in f.

  Use this function to wrap any op, maintaining its behavior in the forward
  pass, but replacing the original op in the backward graph with an identity.
  For example:

  ```python
  x = tf.Variable(1.0, name=\"x\")
  z = tf.Variable(3.0, name=\"z\")

  with tf.GradientTape() as tape:
    # y will evaluate to 9.0
    y = tf.grad_pass_through(x.assign)(z**2)
  # grads will evaluate to 6.0
  grads = tape.gradient(y, z)
  ```

  Another example is a 'differentiable' moving average approximation, where
  gradients are allowed to flow into the last value fed to the moving average,
  but the moving average is still used for the forward pass:

  ```python
  x = ... # Some scalar value
  # A moving average object, we don't need to know how this is implemented
  moving_average = MovingAverage()
  with backprop.GradientTape() as tape:
    # mavg_x will evaluate to the current running average value
    mavg_x = tf.grad_pass_through(moving_average)(x)
  grads = tape.gradient(mavg_x, x) # grads will evaluate to 1.0
  ```

  Args:
    f: function `f(*x)` that returns a `Tensor` or nested structure of `Tensor`
      outputs.

  Returns:
   A function `h(x)` which returns the same values as `f(x)` and whose
   gradients are the same as those of an identity function.
  "
  [ f ]
  (py/call-attr v2 "grad_pass_through"  f ))

(defn gradients 
  "Constructs symbolic derivatives of sum of `ys` w.r.t. x in `xs`.

  `ys` and `xs` are each a `Tensor` or a list of tensors.  `grad_ys`
  is a list of `Tensor`, holding the gradients received by the
  `ys`. The list must be the same length as `ys`.

  `gradients()` adds ops to the graph to output the derivatives of `ys` with
  respect to `xs`.  It returns a list of `Tensor` of length `len(xs)` where
  each tensor is the `sum(dy/dx)` for y in `ys`.

  `grad_ys` is a list of tensors of the same length as `ys` that holds
  the initial gradients for each y in `ys`.  When `grad_ys` is None,
  we fill in a tensor of '1's of the shape of y for each y in `ys`.  A
  user can provide their own initial `grad_ys` to compute the
  derivatives using a different initial gradient for each y (e.g., if
  one wanted to weight the gradient differently for each value in
  each y).

  `stop_gradients` is a `Tensor` or a list of tensors to be considered constant
  with respect to all `xs`. These tensors will not be backpropagated through,
  as though they had been explicitly disconnected using `stop_gradient`.  Among
  other things, this allows computation of partial derivatives as opposed to
  total derivatives. For example:

  ```python
  a = tf.constant(0.)
  b = 2 * a
  g = tf.gradients(a + b, [a, b], stop_gradients=[a, b])
  ```

  Here the partial derivatives `g` evaluate to `[1.0, 1.0]`, compared to the
  total derivatives `tf.gradients(a + b, [a, b])`, which take into account the
  influence of `a` on `b` and evaluate to `[3.0, 1.0]`.  Note that the above is
  equivalent to:

  ```python
  a = tf.stop_gradient(tf.constant(0.))
  b = tf.stop_gradient(2 * a)
  g = tf.gradients(a + b, [a, b])
  ```

  `stop_gradients` provides a way of stopping gradient after the graph has
  already been constructed, as compared to `tf.stop_gradient` which is used
  during graph construction.  When the two approaches are combined,
  backpropagation stops at both `tf.stop_gradient` nodes and nodes in
  `stop_gradients`, whichever is encountered first.

  All integer tensors are considered constant with respect to all `xs`, as if
  they were included in `stop_gradients`.

  `unconnected_gradients` determines the value returned for each x in xs if it
  is unconnected in the graph to ys. By default this is None to safeguard
  against errors. Mathematically these gradients are zero which can be requested
  using the `'zero'` option. `tf.UnconnectedGradients` provides the
  following options and behaviors:

  ```python
  a = tf.ones([1, 2])
  b = tf.ones([3, 1])
  g1 = tf.gradients([b], [a], unnconnected_gradients='none')
  sess.run(g1)  # [None]

  g2 = tf.gradients([b], [a], unconnected_gradients='zero')
  sess.run(g2)  # [array([[0., 0.]], dtype=float32)]
  ```


  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    grad_ys: Optional. A `Tensor` or list of tensors the same size as
      `ys` and holding the gradients computed for each y in `ys`.
    name: Optional name to use for grouping all the gradient ops together.
      defaults to 'gradients'.
    gate_gradients: If True, add a tuple around the gradients returned
      for an operations.  This avoids some race conditions.
    aggregation_method: Specifies the method used to combine gradient terms.
      Accepted values are constants defined in the class `AggregationMethod`.
    stop_gradients: Optional. A `Tensor` or list of tensors not to differentiate
      through.
    unconnected_gradients: Optional. Specifies the gradient value returned when
      the given input tensors are unconnected. Accepted values are constants
      defined in the class `tf.UnconnectedGradients` and the default value is
      `none`.

  Returns:
    A list of `sum(dy/dx)` for each x in `xs`.

  Raises:
    LookupError: if one of the operations between `x` and `y` does not
      have a registered gradient function.
    ValueError: if the arguments are invalid.
    RuntimeError: if called in Eager mode.

  "
  [ys xs grad_ys & {:keys [name gate_gradients aggregation_method stop_gradients unconnected_gradients]
                       :or {aggregation_method None stop_gradients None}} ]
    (py/call-attr-kw v2 "gradients" [ys xs grad_ys] {:name name :gate_gradients gate_gradients :aggregation_method aggregation_method :stop_gradients stop_gradients :unconnected_gradients unconnected_gradients }))

(defn greater 
  "Returns the truth value of (x > y) element-wise.

  *NOTE*: `math.greater` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ x y name ]
  (py/call-attr v2 "greater"  x y name ))

(defn greater-equal 
  "Returns the truth value of (x >= y) element-wise.

  *NOTE*: `math.greater_equal` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ x y name ]
  (py/call-attr v2 "greater_equal"  x y name ))

(defn group 
  "Create an op that groups multiple operations.

  When this op finishes, all ops in `inputs` have finished. This op has no
  output.

  See also `tf.tuple` and
  `tf.control_dependencies`.

  Args:
    *inputs: Zero or more tensors to group.
    name: A name for this operation (optional).

  Returns:
    An Operation that executes all its inputs.

  Raises:
    ValueError: If an unknown keyword argument is provided.
  "
  [  ]
  (py/call-attr v2 "group"  ))

(defn guarantee-const 
  "Gives a guarantee to the TF runtime that the input tensor is a constant.

  The runtime is then free to make optimizations based on this.

  Only accepts value typed tensors as inputs and rejects resource variable handles
  as input.

  Returns the input tensor without modification.

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input name ]
  (py/call-attr v2 "guarantee_const"  input name ))

(defn hessians 
  "Constructs the Hessian of sum of `ys` with respect to `x` in `xs`.

  `hessians()` adds ops to the graph to output the Hessian matrix of `ys`
  with respect to `xs`.  It returns a list of `Tensor` of length `len(xs)`
  where each tensor is the Hessian of `sum(ys)`.

  The Hessian is a matrix of second-order partial derivatives of a scalar
  tensor (see https://en.wikipedia.org/wiki/Hessian_matrix for more details).

  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    name: Optional name to use for grouping all the gradient ops together.
      defaults to 'hessians'.
    colocate_gradients_with_ops: See `gradients()` documentation for details.
    gate_gradients: See `gradients()` documentation for details.
    aggregation_method: See `gradients()` documentation for details.

  Returns:
    A list of Hessian matrices of `sum(ys)` for each `x` in `xs`.

  Raises:
    LookupError: if one of the operations between `xs` and `ys` does not
      have a registered gradient function.
  "
  [ys xs & {:keys [gate_gradients aggregation_method name]
                       :or {aggregation_method None}} ]
    (py/call-attr-kw v2 "hessians" [ys xs] {:gate_gradients gate_gradients :aggregation_method aggregation_method :name name }))

(defn histogram-fixed-width 
  "Return histogram of values.

  Given the tensor `values`, this operation returns a rank 1 histogram counting
  the number of entries in `values` that fell into every bin.  The bins are
  equal width and determined by the arguments `value_range` and `nbins`.

  Args:
    values:  Numeric `Tensor`.
    value_range:  Shape [2] `Tensor` of same `dtype` as `values`.
      values <= value_range[0] will be mapped to hist[0],
      values >= value_range[1] will be mapped to hist[-1].
    nbins:  Scalar `int32 Tensor`.  Number of histogram bins.
    dtype:  dtype for returned histogram.
    name:  A name for this operation (defaults to 'histogram_fixed_width').

  Returns:
    A 1-D `Tensor` holding histogram of values.

  Raises:
    TypeError: If any unsupported dtype is provided.
    tf.errors.InvalidArgumentError: If value_range does not
        satisfy value_range[0] < value_range[1].

  Examples:

  ```python
  # Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
  nbins = 5
  value_range = [0.0, 5.0]
  new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]

  with tf.compat.v1.get_default_session() as sess:
    hist = tf.histogram_fixed_width(new_values, value_range, nbins=5)
    variables.global_variables_initializer().run()
    sess.run(hist) => [2, 1, 1, 0, 2]
  ```
  "
  [values value_range & {:keys [nbins dtype name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "histogram_fixed_width" [values value_range] {:nbins nbins :dtype dtype :name name }))

(defn histogram-fixed-width-bins 
  "Bins the given values for use in a histogram.

  Given the tensor `values`, this operation returns a rank 1 `Tensor`
  representing the indices of a histogram into which each element
  of `values` would be binned. The bins are equal width and
  determined by the arguments `value_range` and `nbins`.

  Args:
    values:  Numeric `Tensor`.
    value_range:  Shape [2] `Tensor` of same `dtype` as `values`.
      values <= value_range[0] will be mapped to hist[0],
      values >= value_range[1] will be mapped to hist[-1].
    nbins:  Scalar `int32 Tensor`.  Number of histogram bins.
    dtype:  dtype for returned histogram.
    name:  A name for this operation (defaults to 'histogram_fixed_width').

  Returns:
    A `Tensor` holding the indices of the binned values whose shape matches
    `values`.

  Raises:
    TypeError: If any unsupported dtype is provided.
    tf.errors.InvalidArgumentError: If value_range does not
        satisfy value_range[0] < value_range[1].

  Examples:

  ```python
  # Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
  nbins = 5
  value_range = [0.0, 5.0]
  new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]

  with tf.compat.v1.get_default_session() as sess:
    indices = tf.histogram_fixed_width_bins(new_values, value_range, nbins=5)
    variables.global_variables_initializer().run()
    sess.run(indices) # [0, 0, 1, 2, 4, 4]
  ```
  "
  [values value_range & {:keys [nbins dtype name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "histogram_fixed_width_bins" [values value_range] {:nbins nbins :dtype dtype :name name }))

(defn identity 
  "Return a tensor with the same shape and contents as input.

  For example:

  ```python
  import tensorflow as tf
  val0 = tf.ones((1,), dtype=tf.float32)
  a = tf.atan2(val0, val0)
  a_identity = tf.identity(a)
  print(a.numpy())          #[0.7853982]
  print(a_identity.numpy()) #[0.7853982]
  ```

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input name ]
  (py/call-attr v2 "identity"  input name ))

(defn identity-n 
  "Returns a list of tensors with the same shapes and contents as the input

  tensors.

  This op can be used to override the gradient for complicated functions. For
  example, suppose y = f(x) and we wish to apply a custom function g for backprop
  such that dx = g(dy). In Python,

  ```python
  with tf.get_default_graph().gradient_override_map(
      {'IdentityN': 'OverrideGradientWithG'}):
    y, _ = identity_n([f(x), x])

  @tf.RegisterGradient('OverrideGradientWithG')
  def ApplyG(op, dy, _):
    return [None, g(dy)]  # Do not backprop to f(x).
  ```

  Args:
    input: A list of `Tensor` objects.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `input`.
  "
  [ input name ]
  (py/call-attr v2 "identity_n"  input name ))

(defn import-graph-def 
  "Imports the graph from `graph_def` into the current default `Graph`. (deprecated arguments)

Warning: SOME ARGUMENTS ARE DEPRECATED: `(op_dict)`. They will be removed in a future version.
Instructions for updating:
Please file an issue at https://github.com/tensorflow/tensorflow/issues if you depend on this feature.

This function provides a way to import a serialized TensorFlow
[`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
protocol buffer, and extract individual objects in the `GraphDef` as
`tf.Tensor` and `tf.Operation` objects. Once extracted,
these objects are placed into the current default `Graph`. See
`tf.Graph.as_graph_def` for a way to create a `GraphDef`
proto.

Args:
  graph_def: A `GraphDef` proto containing operations to be imported into
    the default graph.
  input_map: A dictionary mapping input names (as strings) in `graph_def`
    to `Tensor` objects. The values of the named input tensors in the
    imported graph will be re-mapped to the respective `Tensor` values.
  return_elements: A list of strings containing operation names in
    `graph_def` that will be returned as `Operation` objects; and/or
    tensor names in `graph_def` that will be returned as `Tensor` objects.
  name: (Optional.) A prefix that will be prepended to the names in
    `graph_def`. Note that this does not apply to imported function names.
    Defaults to `\"import\"`.
  op_dict: (Optional.) Deprecated, do not use.
  producer_op_list: (Optional.) An `OpList` proto with the (possibly stripped)
    list of `OpDef`s used by the producer of the graph. If provided,
    unrecognized attrs for ops in `graph_def` that have their default value
    according to `producer_op_list` will be removed. This will allow some more
    `GraphDef`s produced by later binaries to be accepted by earlier binaries.

Returns:
  A list of `Operation` and/or `Tensor` objects from the imported graph,
  corresponding to the names in `return_elements`,
  and None if `returns_elements` is None.

Raises:
  TypeError: If `graph_def` is not a `GraphDef` proto,
    `input_map` is not a dictionary mapping strings to `Tensor` objects,
    or `return_elements` is not a list of strings.
  ValueError: If `input_map`, or `return_elements` contains names that
    do not appear in `graph_def`, or `graph_def` is not well-formed (e.g.
    it refers to an unknown tensor)."
  [ graph_def input_map return_elements name op_dict producer_op_list ]
  (py/call-attr v2 "import_graph_def"  graph_def input_map return_elements name op_dict producer_op_list ))

(defn init-scope 
  "A context manager that lifts ops out of control-flow scopes and function-building graphs.

  There is often a need to lift variable initialization ops out of control-flow
  scopes, function-building graphs, and gradient tapes. Entering an
  `init_scope` is a mechanism for satisfying these desiderata. In particular,
  entering an `init_scope` has three effects:

    (1) All control dependencies are cleared the moment the scope is entered;
        this is equivalent to entering the context manager returned from
        `control_dependencies(None)`, which has the side-effect of exiting
        control-flow scopes like `tf.cond` and `tf.while_loop`.

    (2) All operations that are created while the scope is active are lifted
        into the lowest context on the `context_stack` that is not building a
        graph function. Here, a context is defined as either a graph or an eager
        context. Every context switch, i.e., every installation of a graph as
        the default graph and every switch into eager mode, is logged in a
        thread-local stack called `context_switches`; the log entry for a
        context switch is popped from the stack when the context is exited.
        Entering an `init_scope` is equivalent to crawling up
        `context_switches`, finding the first context that is not building a
        graph function, and entering it. A caveat is that if graph mode is
        enabled but the default graph stack is empty, then entering an
        `init_scope` will simply install a fresh graph as the default one.

    (3) The gradient tape is paused while the scope is active.

  When eager execution is enabled, code inside an init_scope block runs with
  eager execution enabled even when defining graph functions via
  tf.contrib.eager.defun. For example:

  ```python
  tf.compat.v1.enable_eager_execution()

  @tf.contrib.eager.defun
  def func():
    # A defun-decorated function constructs TensorFlow graphs,
    # it does not execute eagerly.
    assert not tf.executing_eagerly()
    with tf.init_scope():
      # Initialization runs with eager execution enabled
      assert tf.executing_eagerly()
  ```

  Raises:
    RuntimeError: if graph state is incompatible with this initialization.
  "
  [  ]
  (py/call-attr v2 "init_scope"  ))

(defn is-tensor 
  "Checks whether `x` is a tensor or \"tensor-like\".

  If `is_tensor(x)` returns `True`, it is safe to assume that `x` is a tensor or
  can be converted to a tensor using `ops.convert_to_tensor(x)`.

  Args:
    x: A python object to check.

  Returns:
    `True` if `x` is a tensor or \"tensor-like\", `False` if not.
  "
  [ x ]
  (py/call-attr v2 "is_tensor"  x ))

(defn less 
  "Returns the truth value of (x < y) element-wise.

  *NOTE*: `math.less` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ x y name ]
  (py/call-attr v2 "less"  x y name ))

(defn less-equal 
  "Returns the truth value of (x <= y) element-wise.

  *NOTE*: `math.less_equal` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ x y name ]
  (py/call-attr v2 "less_equal"  x y name ))

(defn linspace 
  "Generates values in an interval.

  A sequence of `num` evenly-spaced values are generated beginning at `start`.
  If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
  so that the last one is exactly `stop`.

  For example:

  ```
  tf.linspace(10.0, 12.0, 3, name=\"linspace\") => [ 10.0  11.0  12.0]
  ```

  Args:
    start: A `Tensor`. Must be one of the following types: `bfloat16`, `float32`, `float64`.
      0-D tensor. First entry in the range.
    stop: A `Tensor`. Must have the same type as `start`.
      0-D tensor. Last entry in the range.
    num: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      0-D tensor. Number of values to generate.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `start`.
  "
  [ start stop num name ]
  (py/call-attr v2 "linspace"  start stop num name ))

(defn load-library 
  "Loads a TensorFlow plugin.

  \"library_location\" can be a path to a specific shared object, or a folder.
  If it is a folder, all shared objects that are named \"libtfkernel*\" will be
  loaded. When the library is loaded, kernels registered in the library via the
  `REGISTER_*` macros are made available in the TensorFlow process.

  Args:
    library_location: Path to the plugin or the folder of plugins.
      Relative or absolute filesystem path to a dynamic library file or folder.

  Returns:
    None

  Raises:
    OSError: When the file to be loaded is not found.
    RuntimeError: when unable to load the library.
  "
  [ library_location ]
  (py/call-attr v2 "load_library"  library_location ))

(defn load-op-library 
  "Loads a TensorFlow plugin, containing custom ops and kernels.

  Pass \"library_filename\" to a platform-specific mechanism for dynamically
  loading a library. The rules for determining the exact location of the
  library are platform-specific and are not documented here. When the
  library is loaded, ops and kernels registered in the library via the
  `REGISTER_*` macros are made available in the TensorFlow process. Note
  that ops with the same name as an existing op are rejected and not
  registered with the process.

  Args:
    library_filename: Path to the plugin.
      Relative or absolute filesystem path to a dynamic library file.

  Returns:
    A python module containing the Python wrappers for Ops defined in
    the plugin.

  Raises:
    RuntimeError: when unable to load the library or get the python wrappers.
  "
  [ library_filename ]
  (py/call-attr v2 "load_op_library"  library_filename ))

(defn logical-and 
  "Returns the truth value of x AND y element-wise.

  *NOTE*: `math.logical_and` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor` of type `bool`.
    y: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ x y name ]
  (py/call-attr v2 "logical_and"  x y name ))

(defn logical-not 
  "Returns the truth value of NOT x element-wise.

  Args:
    x: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ x name ]
  (py/call-attr v2 "logical_not"  x name ))

(defn logical-or 
  "Returns the truth value of x OR y element-wise.

  *NOTE*: `math.logical_or` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor` of type `bool`.
    y: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ x y name ]
  (py/call-attr v2 "logical_or"  x y name ))

(defn make-ndarray 
  "Create a numpy ndarray from a tensor.

  Create a numpy ndarray with the same shape and data as the tensor.

  Args:
    tensor: A TensorProto.

  Returns:
    A numpy array with the tensor contents.

  Raises:
    TypeError: if tensor has unsupported type.

  "
  [ tensor ]
  (py/call-attr v2 "make_ndarray"  tensor ))
(defn make-tensor-proto 
  "Create a TensorProto.

  In TensorFlow 2.0, representing tensors as protos should no longer be a
  common workflow. That said, this utility function is still useful for
  generating TF Serving request protos:

    request = tensorflow_serving.apis.predict_pb2.PredictRequest()
    request.model_spec.name = \"my_model\"
    request.model_spec.signature_name = \"serving_default\"
    request.inputs[\"images\"].CopyFrom(tf.make_tensor_proto(X_new))

  make_tensor_proto accepts \"values\" of a python scalar, a python list, a
  numpy ndarray, or a numpy scalar.

  If \"values\" is a python scalar or a python list, make_tensor_proto
  first convert it to numpy ndarray. If dtype is None, the
  conversion tries its best to infer the right numpy data
  type. Otherwise, the resulting numpy array has a compatible data
  type with the given dtype.

  In either case above, the numpy ndarray (either the caller provided
  or the auto converted) must have the compatible type with dtype.

  make_tensor_proto then converts the numpy array to a tensor proto.

  If \"shape\" is None, the resulting tensor proto represents the numpy
  array precisely.

  Otherwise, \"shape\" specifies the tensor's shape and the numpy array
  can not have more elements than what \"shape\" specifies.

  Args:
    values:         Values to put in the TensorProto.
    dtype:          Optional tensor_pb2 DataType value.
    shape:          List of integers representing the dimensions of tensor.
    verify_shape:   Boolean that enables verification of a shape of values.
    allow_broadcast:  Boolean that enables allowing scalars and 1 length vector
        broadcasting. Cannot be true when verify_shape is true.

  Returns:
    A `TensorProto`. Depending on the type, it may contain data in the
    \"tensor_content\" attribute, which is not directly useful to Python programs.
    To access the values you should convert the proto back to a numpy ndarray
    with `tf.make_ndarray(proto)`.

    If `values` is a `TensorProto`, it is immediately returned; `dtype` and
    `shape` are ignored.

  Raises:
    TypeError:  if unsupported types are provided.
    ValueError: if arguments have inappropriate values or if verify_shape is
     True and shape of values is not equals to a shape from the argument.

  "
  [values dtype shape  & {:keys [verify_shape allow_broadcast]} ]
    (py/call-attr-kw v2 "make_tensor_proto" [values dtype shape] {:verify_shape verify_shape :allow_broadcast allow_broadcast }))

(defn map-fn 
  "map on the list of tensors unpacked from `elems` on dimension 0.

  The simplest version of `map_fn` repeatedly applies the callable `fn` to a
  sequence of elements from first to last. The elements are made of the
  tensors unpacked from `elems`. `dtype` is the data type of the return
  value of `fn`. Users must provide `dtype` if it is different from
  the data type of `elems`.

  Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
  of the result tensor is `[values.shape[0]] + fn(values[0]).shape`.

  This method also allows multi-arity `elems` and output of `fn`.  If `elems`
  is a (possibly nested) list or tuple of tensors, then each of these tensors
  must have a matching first (unpack) dimension.  The signature of `fn` may
  match the structure of `elems`.  That is, if `elems` is
  `(t1, [t2, t3, [t4, t5]])`, then an appropriate signature for `fn` is:
  `fn = lambda (t1, [t2, t3, [t4, t5]]):`.

  Furthermore, `fn` may emit a different structure than its input.  For example,
  `fn` may look like: `fn = lambda t1: return (t1 + 1, t1 - 1)`.  In this case,
  the `dtype` parameter is not optional: `dtype` must be a type or (possibly
  nested) tuple of types matching the output of `fn`.

  To apply a functional operation to the nonzero elements of a SparseTensor
  one of the following methods is recommended. First, if the function is
  expressible as TensorFlow ops, use

  ```python
    result = SparseTensor(input.indices, fn(input.values), input.dense_shape)
  ```

  If, however, the function is not expressible as a TensorFlow op, then use

  ```python
  result = SparseTensor(
    input.indices, map_fn(fn, input.values), input.dense_shape)
  ```

  instead.

  When executing eagerly, map_fn does not execute in parallel even if
  `parallel_iterations` is set to a value > 1. You can still get the
  performance benefits of running a function in parallel by using the
  `tf.contrib.eager.defun` decorator,

  ```python
  # Assume the function being used in map_fn is fn.
  # To ensure map_fn calls fn in parallel, use the defun decorator.
  @tf.contrib.eager.defun
  def func(tensor):
    return tf.map_fn(fn, tensor)
  ```

  Note that if you use the defun decorator, any non-TensorFlow Python code
  that you may have written in your function won't get executed. See
  `tf.contrib.eager.defun` for more details. The recommendation would be to
  debug without defun but switch to defun to get performance benefits of
  running map_fn in parallel.

  Args:
    fn: The callable to be performed.  It accepts one argument, which will
      have the same (possibly nested) structure as `elems`.  Its output
      must have the same structure as `dtype` if one is provided, otherwise
      it must have the same structure as `elems`.
    elems: A tensor or (possibly nested) sequence of tensors, each of which
      will be unpacked along their first dimension.  The nested sequence
      of the resulting slices will be applied to `fn`.
    dtype: (optional) The output type(s) of `fn`.  If `fn` returns a structure
      of Tensors differing from the structure of `elems`, then `dtype` is not
      optional and must have the same structure as the output of `fn`.
    parallel_iterations: (optional) The number of iterations allowed to run
      in parallel. When graph building, the default value is 10. While executing
      eagerly, the default value is set to 1.
    back_prop: (optional) True enables support for back propagation.
    swap_memory: (optional) True enables GPU-CPU memory swapping.
    infer_shape: (optional) False disables tests for consistent output shapes.
    name: (optional) Name prefix for the returned tensors.

  Returns:
    A tensor or (possibly nested) sequence of tensors.  Each tensor packs the
    results of applying `fn` to tensors unpacked from `elems` along the first
    dimension, from first to last.

  Raises:
    TypeError: if `fn` is not callable or the structure of the output of
      `fn` and `dtype` do not match, or if elems is a SparseTensor.
    ValueError: if the lengths of the output of `fn` and `dtype` do not match.

  Examples:
    ```python
    elems = np.array([1, 2, 3, 4, 5, 6])
    squares = map_fn(lambda x: x * x, elems)
    # squares == [1, 4, 9, 16, 25, 36]
    ```

    ```python
    elems = (np.array([1, 2, 3]), np.array([-1, 1, -1]))
    alternate = map_fn(lambda x: x[0] * x[1], elems, dtype=tf.int64)
    # alternate == [-1, 2, -3]
    ```

    ```python
    elems = np.array([1, 2, 3])
    alternates = map_fn(lambda x: (x, -x), elems, dtype=(tf.int64, tf.int64))
    # alternates[0] == [1, 2, 3]
    # alternates[1] == [-1, -2, -3]
    ```
  "
  [fn elems dtype parallel_iterations & {:keys [back_prop swap_memory infer_shape name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "map_fn" [fn elems dtype parallel_iterations] {:back_prop back_prop :swap_memory swap_memory :infer_shape infer_shape :name name }))

(defn matmul 
  "Multiplies matrix `a` by matrix `b`, producing `a` * `b`.

  The inputs must, following any transpositions, be tensors of rank >= 2
  where the inner 2 dimensions specify valid matrix multiplication arguments,
  and any further outer dimensions match.

  Both matrices must be of the same type. The supported types are:
  `float16`, `float32`, `float64`, `int32`, `complex64`, `complex128`.

  Either matrix can be transposed or adjointed (conjugated and transposed) on
  the fly by setting one of the corresponding flag to `True`. These are `False`
  by default.

  If one or both of the matrices contain a lot of zeros, a more efficient
  multiplication algorithm can be used by setting the corresponding
  `a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.
  This optimization is only available for plain matrices (rank-2 tensors) with
  datatypes `bfloat16` or `float32`.

  For example:

  ```python
  # 2-D tensor `a`
  # [[1, 2, 3],
  #  [4, 5, 6]]
  a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])

  # 2-D tensor `b`
  # [[ 7,  8],
  #  [ 9, 10],
  #  [11, 12]]
  b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])

  # `a` * `b`
  # [[ 58,  64],
  #  [139, 154]]
  c = tf.matmul(a, b)


  # 3-D tensor `a`
  # [[[ 1,  2,  3],
  #   [ 4,  5,  6]],
  #  [[ 7,  8,  9],
  #   [10, 11, 12]]]
  a = tf.constant(np.arange(1, 13, dtype=np.int32),
                  shape=[2, 2, 3])

  # 3-D tensor `b`
  # [[[13, 14],
  #   [15, 16],
  #   [17, 18]],
  #  [[19, 20],
  #   [21, 22],
  #   [23, 24]]]
  b = tf.constant(np.arange(13, 25, dtype=np.int32),
                  shape=[2, 3, 2])

  # `a` * `b`
  # [[[ 94, 100],
  #   [229, 244]],
  #  [[508, 532],
  #   [697, 730]]]
  c = tf.matmul(a, b)

  # Since python >= 3.5 the @ operator is supported (see PEP 465).
  # In TensorFlow, it simply calls the `tf.matmul()` function, so the
  # following lines are equivalent:
  d = a @ b @ [[10.], [11.]]
  d = tf.matmul(tf.matmul(a, b), [[10.], [11.]])
  ```

  Args:
    a: `Tensor` of type `float16`, `float32`, `float64`, `int32`, `complex64`,
      `complex128` and rank > 1.
    b: `Tensor` with same type and rank as `a`.
    transpose_a: If `True`, `a` is transposed before multiplication.
    transpose_b: If `True`, `b` is transposed before multiplication.
    adjoint_a: If `True`, `a` is conjugated and transposed before
      multiplication.
    adjoint_b: If `True`, `b` is conjugated and transposed before
      multiplication.
    a_is_sparse: If `True`, `a` is treated as a sparse matrix.
    b_is_sparse: If `True`, `b` is treated as a sparse matrix.
    name: Name for the operation (optional).

  Returns:
    A `Tensor` of the same type as `a` and `b` where each inner-most matrix is
    the product of the corresponding matrices in `a` and `b`, e.g. if all
    transpose or adjoint attributes are `False`:

    `output`[..., i, j] = sum_k (`a`[..., i, k] * `b`[..., k, j]),
    for all indices i, j.

    Note: This is matrix product, not element-wise product.


  Raises:
    ValueError: If transpose_a and adjoint_a, or transpose_b and adjoint_b
      are both set to True.
  "
  [a b & {:keys [transpose_a transpose_b adjoint_a adjoint_b a_is_sparse b_is_sparse name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "matmul" [a b] {:transpose_a transpose_a :transpose_b transpose_b :adjoint_a adjoint_a :adjoint_b adjoint_b :a_is_sparse a_is_sparse :b_is_sparse b_is_sparse :name name }))

(defn matrix-square-root 
  "Computes the matrix square root of one or more square matrices:

  matmul(sqrtm(A), sqrtm(A)) = A

  The input matrix should be invertible. If the input matrix is real, it should
  have no eigenvalues which are real and negative (pairs of complex conjugate
  eigenvalues are allowed).

  The matrix square root is computed by first reducing the matrix to 
  quasi-triangular form with the real Schur decomposition. The square root 
  of the quasi-triangular matrix is then computed directly. Details of 
  the algorithm can be found in: Nicholas J. Higham, \"Computing real 
  square roots of a real matrix\", Linear Algebra Appl., 1987.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices. The output is a tensor of the same shape as the input
  containing the matrix square root for all input submatrices `[..., :, :]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`, `half`, `complex64`, `complex128`.
      Shape is `[..., M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input name ]
  (py/call-attr v2 "matrix_square_root"  input name ))

(defn maximum 
  "Returns the max of x and y (i.e. x > y ? x : y) element-wise.

  *NOTE*: `math.maximum` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr v2 "maximum"  x y name ))

(defn meshgrid 
  "Broadcasts parameters for evaluation on an N-D grid.

  Given N one-dimensional coordinate arrays `*args`, returns a list `outputs`
  of N-D coordinate arrays for evaluating expressions on an N-D grid.

  Notes:

  `meshgrid` supports cartesian ('xy') and matrix ('ij') indexing conventions.
  When the `indexing` argument is set to 'xy' (the default), the broadcasting
  instructions for the first two dimensions are swapped.

  Examples:

  Calling `X, Y = meshgrid(x, y)` with the tensors

  ```python
  x = [1, 2, 3]
  y = [4, 5, 6]
  X, Y = tf.meshgrid(x, y)
  # X = [[1, 2, 3],
  #      [1, 2, 3],
  #      [1, 2, 3]]
  # Y = [[4, 4, 4],
  #      [5, 5, 5],
  #      [6, 6, 6]]
  ```

  Args:
    *args: `Tensor`s with rank 1.
    **kwargs:
      - indexing: Either 'xy' or 'ij' (optional, default: 'xy').
      - name: A name for the operation (optional).

  Returns:
    outputs: A list of N `Tensor`s with rank N.

  Raises:
    TypeError: When no keyword arguments (kwargs) are passed.
    ValueError: When indexing keyword argument is not one of `xy` or `ij`.
  "
  [  ]
  (py/call-attr v2 "meshgrid"  ))

(defn minimum 
  "Returns the min of x and y (i.e. x < y ? x : y) element-wise.

  *NOTE*: `math.minimum` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr v2 "minimum"  x y name ))

(defn multiply 
  "Returns x * y element-wise.

  *NOTE*: `tf.multiply` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr v2 "multiply"  x y name ))

(defn negative 
  "Computes numerical negative value element-wise.

  I.e., \\(y = -x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.

    If `x` is a `SparseTensor`, returns
    `SparseTensor(x.indices, tf.math.negative(x.values, ...), x.dense_shape)`"
  [ x name ]
  (py/call-attr v2 "negative"  x name ))

(defn no-gradient 
  "Specifies that ops of type `op_type` is not differentiable.

  This function should *not* be used for operations that have a
  well-defined gradient that is not yet implemented.

  This function is only used when defining a new op type. It may be
  used for ops such as `tf.size()` that are not differentiable.  For
  example:

  ```python
  tf.no_gradient(\"Size\")
  ```

  The gradient computed for 'op_type' will then propagate zeros.

  For ops that have a well-defined gradient but are not yet implemented,
  no declaration should be made, and an error *must* be thrown if
  an attempt to request its gradient is made.

  Args:
    op_type: The string type of an operation. This corresponds to the
      `OpDef.name` field for the proto that defines the operation.

  Raises:
    TypeError: If `op_type` is not a string.

  "
  [ op_type ]
  (py/call-attr v2 "no_gradient"  op_type ))

(defn no-op 
  "Does nothing. Only useful as a placeholder for control edges.

  Args:
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ name ]
  (py/call-attr v2 "no_op"  name ))
(defn nondifferentiable-batch-function 
  "Batches the computation done by the decorated function.

  So, for example, in the following code

  ```python
  @batch_function(1, 2, 3)
  def layer(a):
    return tf.matmul(a, a)

  b = layer(w)
  ```

  if more than one session.run call is simultaneously trying to compute `b`
  the values of `w` will be gathered, non-deterministically concatenated
  along the first axis, and only one thread will run the computation. See the
  documentation of the `Batch` op for more details.

  Assumes that all arguments of the decorated function are Tensors which will
  be batched along their first dimension.

  SparseTensor is not supported. The return value of the decorated function
  must be a Tensor or a list/tuple of Tensors.

  Args:
    num_batch_threads: Number of scheduling threads for processing batches
     of work. Determines the number of batches processed in parallel.
    max_batch_size: Batch sizes will never be bigger than this.
    batch_timeout_micros: Maximum number of microseconds to wait before
     outputting an incomplete batch.
    allowed_batch_sizes: Optional list of allowed batch sizes. If left empty,
     does nothing. Otherwise, supplies a list of batch sizes, causing the op
     to pad batches up to one of those sizes. The entries must increase
     monotonically, and the final entry must equal max_batch_size.
    max_enqueued_batches: The maximum depth of the batch queue. Defaults to 10.
    autograph: Whether to use autograph to compile python and eager style code
     for efficient graph-mode execution.

  Returns:
    The decorated function will return the unbatched computation output Tensors.
  "
  [num_batch_threads max_batch_size batch_timeout_micros allowed_batch_sizes  & {:keys [max_enqueued_batches autograph]} ]
    (py/call-attr-kw v2 "nondifferentiable_batch_function" [num_batch_threads max_batch_size batch_timeout_micros allowed_batch_sizes] {:max_enqueued_batches max_enqueued_batches :autograph autograph }))

(defn norm 
  "Computes the norm of vectors, matrices, and tensors.

  This function can compute several different vector norms (the 1-norm, the
  Euclidean or 2-norm, the inf-norm, and in general the p-norm for p > 0) and
  matrix norms (Frobenius, 1-norm, 2-norm and inf-norm).

  Args:
    tensor: `Tensor` of types `float32`, `float64`, `complex64`, `complex128`
    ord: Order of the norm. Supported values are `'fro'`, `'euclidean'`,
      `1`, `2`, `np.inf` and any positive real number yielding the corresponding
      p-norm. Default is `'euclidean'` which is equivalent to Frobenius norm if
      `tensor` is a matrix and equivalent to 2-norm for vectors.
      Some restrictions apply:
        a) The Frobenius norm `'fro'` is not defined for vectors,
        b) If axis is a 2-tuple (matrix norm), only `'euclidean'`, '`fro'`, `1`,
           `2`, `np.inf` are supported.
      See the description of `axis` on how to compute norms for a batch of
      vectors or matrices stored in a tensor.
    axis: If `axis` is `None` (the default), the input is considered a vector
      and a single vector norm is computed over the entire set of values in the
      tensor, i.e. `norm(tensor, ord=ord)` is equivalent to
      `norm(reshape(tensor, [-1]), ord=ord)`.
      If `axis` is a Python integer, the input is considered a batch of vectors,
      and `axis` determines the axis in `tensor` over which to compute vector
      norms.
      If `axis` is a 2-tuple of Python integers it is considered a batch of
      matrices and `axis` determines the axes in `tensor` over which to compute
      a matrix norm.
      Negative indices are supported. Example: If you are passing a tensor that
      can be either a matrix or a batch of matrices at runtime, pass
      `axis=[-2,-1]` instead of `axis=None` to make sure that matrix norms are
      computed.
    keepdims: If True, the axis indicated in `axis` are kept with size 1.
      Otherwise, the dimensions in `axis` are removed from the output shape.
    name: The name of the op.

  Returns:
    output: A `Tensor` of the same type as tensor, containing the vector or
      matrix norms. If `keepdims` is True then the rank of output is equal to
      the rank of `tensor`. Otherwise, if `axis` is none the output is a scalar,
      if `axis` is an integer, the rank of `output` is one less than the rank
      of `tensor`, if `axis` is a 2-tuple the rank of `output` is two less
      than the rank of `tensor`.

  Raises:
    ValueError: If `ord` or `axis` is invalid.

  @compatibility(numpy)
  Mostly equivalent to numpy.linalg.norm.
  Not supported: ord <= 0, 2-norm for matrices, nuclear norm.
  Other differences:
    a) If axis is `None`, treats the flattened `tensor` as a vector
     regardless of rank.
    b) Explicitly supports 'euclidean' norm as the default, including for
     higher order tensors.
  @end_compatibility
  "
  [tensor & {:keys [ord axis keepdims name]
                       :or {axis None keepdims None name None}} ]
    (py/call-attr-kw v2 "norm" [tensor] {:ord ord :axis axis :keepdims keepdims :name name }))

(defn not-equal 
  "Returns the truth value of (x != y) element-wise.

  **NOTE**: `NotEqual` supports broadcasting. More about broadcasting [here](
  https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html)

  Args:
    x: A `Tensor` or `SparseTensor` or `IndexedSlices`.
    y: A `Tensor` or `SparseTensor` or `IndexedSlices`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type bool with the same size as that of x or y.
  "
  [ x y name ]
  (py/call-attr v2 "not_equal"  x y name ))

(defn numpy-function 
  "Wraps a python function and uses it as a TensorFlow op.

  Given a python function `func`, which takes numpy arrays as its
  arguments and returns numpy arrays as its outputs, wrap this function as an
  operation in a TensorFlow graph. The following snippet constructs a simple
  TensorFlow graph that invokes the `np.sinh()` NumPy function as a operation
  in the graph:

  ```python
  def my_func(x):
    # x will be a numpy array with the contents of the placeholder below
    return np.sinh(x)
  input = tf.compat.v1.placeholder(tf.float32)
  y = tf.compat.v1.numpy_function(my_func, [input], tf.float32)
  ```

  **N.B.** The `tf.compat.v1.numpy_function()` operation has the following known
  limitations:

  * The body of the function (i.e. `func`) will not be serialized in a
    `GraphDef`. Therefore, you should not use this function if you need to
    serialize your model and restore it in a different environment.

  * The operation must run in the same address space as the Python program
    that calls `tf.compat.v1.numpy_function()`. If you are using distributed
    TensorFlow, you
    must run a `tf.distribute.Server` in the same process as the program that
    calls
    `tf.compat.v1.numpy_function()` and you must pin the created operation to a device
    in that
    server (e.g. using `with tf.device():`).

  Args:
    func: A Python function, which accepts `ndarray` objects as arguments and
      returns a list of `ndarray` objects (or a single `ndarray`). This function
      must accept as many arguments as there are tensors in `inp`, and these
      argument types will match the corresponding `tf.Tensor` objects in `inp`.
      The returns `ndarray`s must match the number and types defined `Tout`.
      Important Note: Input and output numpy `ndarray`s of `func` are not
        guaranteed to be copies. In some cases their underlying memory will be
        shared with the corresponding TensorFlow tensors. In-place modification
        or storing `func` input or return values in python datastructures
        without explicit (np.)copy can have non-deterministic consequences.
    inp: A list of `Tensor` objects.
    Tout: A list or tuple of tensorflow data types or a single tensorflow data
      type if there is only one, indicating what `func` returns.
    stateful: (Boolean.) If True, the function should be considered stateful. If
      a function is stateless, when given the same input it will return the same
      output and have no observable side effects. Optimizations such as common
      subexpression elimination are only performed on stateless operations.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` or a single `Tensor` which `func` computes.
  "
  [ func inp Tout name ]
  (py/call-attr v2 "numpy_function"  func inp Tout name ))

(defn one-hot 
  "Returns a one-hot tensor.

  The locations represented by indices in `indices` take value `on_value`,
  while all other locations take value `off_value`.

  `on_value` and `off_value` must have matching data types. If `dtype` is also
  provided, they must be the same data type as specified by `dtype`.

  If `on_value` is not provided, it will default to the value `1` with type
  `dtype`

  If `off_value` is not provided, it will default to the value `0` with type
  `dtype`

  If the input `indices` is rank `N`, the output will have rank `N+1`. The
  new axis is created at dimension `axis` (default: the new axis is appended
  at the end).

  If `indices` is a scalar the output shape will be a vector of length `depth`

  If `indices` is a vector of length `features`, the output shape will be:

  ```
    features x depth if axis == -1
    depth x features if axis == 0
  ```

  If `indices` is a matrix (batch) with shape `[batch, features]`, the output
  shape will be:

  ```
    batch x features x depth if axis == -1
    batch x depth x features if axis == 1
    depth x batch x features if axis == 0
  ```

  If `indices` is a RaggedTensor, the 'axis' argument must be positive and refer
  to a non-ragged axis. The output will be equivalent to applying 'one_hot' on
  the values of the RaggedTensor, and creating a new RaggedTensor from the
  result.

  If `dtype` is not provided, it will attempt to assume the data type of
  `on_value` or `off_value`, if one or both are passed in. If none of
  `on_value`, `off_value`, or `dtype` are provided, `dtype` will default to the
  value `tf.float32`.

  Note: If a non-numeric data type output is desired (`tf.string`, `tf.bool`,
  etc.), both `on_value` and `off_value` _must_ be provided to `one_hot`.

  For example:

  ```python
  indices = [0, 1, 2]
  depth = 3
  tf.one_hot(indices, depth)  # output: [3 x 3]
  # [[1., 0., 0.],
  #  [0., 1., 0.],
  #  [0., 0., 1.]]

  indices = [0, 2, -1, 1]
  depth = 3
  tf.one_hot(indices, depth,
             on_value=5.0, off_value=0.0,
             axis=-1)  # output: [4 x 3]
  # [[5.0, 0.0, 0.0],  # one_hot(0)
  #  [0.0, 0.0, 5.0],  # one_hot(2)
  #  [0.0, 0.0, 0.0],  # one_hot(-1)
  #  [0.0, 5.0, 0.0]]  # one_hot(1)

  indices = [[0, 2], [1, -1]]
  depth = 3
  tf.one_hot(indices, depth,
             on_value=1.0, off_value=0.0,
             axis=-1)  # output: [2 x 2 x 3]
  # [[[1.0, 0.0, 0.0],   # one_hot(0)
  #   [0.0, 0.0, 1.0]],  # one_hot(2)
  #  [[0.0, 1.0, 0.0],   # one_hot(1)
  #   [0.0, 0.0, 0.0]]]  # one_hot(-1)

  indices = tf.ragged.constant([[0, 1], [2]])
  depth = 3
  tf.one_hot(indices, depth)  # output: [2 x None x 3]
  # [[[1., 0., 0.],
  #   [0., 1., 0.]],
  #  [[0., 0., 1.]]]
  ```

  Args:
    indices: A `Tensor` of indices.
    depth: A scalar defining the depth of the one hot dimension.
    on_value: A scalar defining the value to fill in output when `indices[j]
      = i`. (default: 1)
    off_value: A scalar defining the value to fill in output when `indices[j]
      != i`. (default: 0)
    axis: The axis to fill (default: -1, a new inner-most axis).
    dtype: The data type of the output tensor.
    name: A name for the operation (optional).

  Returns:
    output: The one-hot tensor.

  Raises:
    TypeError: If dtype of either `on_value` or `off_value` don't match `dtype`
    TypeError: If dtype of `on_value` and `off_value` don't match one another
  "
  [ indices depth on_value off_value axis dtype name ]
  (py/call-attr v2 "one_hot"  indices depth on_value off_value axis dtype name ))

(defn ones 
  "Creates a tensor with all elements set to 1.

  This operation returns a tensor of type `dtype` with shape `shape` and all
  elements set to 1.

  For example:

  ```python
  tf.ones([2, 3], tf.int32)  # [[1, 1, 1], [1, 1, 1]]
  ```

  Args:
    shape: A list of integers, a tuple of integers, or a 1-D `Tensor` of type
      `int32`.
    dtype: The type of an element in the resulting `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with all elements set to 1.
  "
  [shape & {:keys [dtype name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "ones" [shape] {:dtype dtype :name name }))

(defn ones-like 
  "Creates a tensor with all elements set to one.

  Given a single tensor (`tensor`), this operation returns a tensor of the
  same type and shape as `tensor` with all elements set to 1. Optionally,
  you can use `dtype` to specify a new type for the returned tensor.

  For example:

  ```python
  tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
  tf.ones_like(tensor)  # [[1, 1, 1], [1, 1, 1]]
  ```

  Args:
    input: A `Tensor`.
    dtype: A type for the returned `Tensor`. Must be `float16`, `float32`,
      `float64`, `int8`, `uint8`, `int16`, `uint16`, `int32`, `int64`,
      `complex64`, `complex128`, `bool` or `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with all elements set to one.
  "
  [ input dtype name ]
  (py/call-attr v2 "ones_like"  input dtype name ))

(defn pad 
  "Pads a tensor.

  This operation pads a `tensor` according to the `paddings` you specify.
  `paddings` is an integer tensor with shape `[n, 2]`, where n is the rank of
  `tensor`. For each dimension D of `input`, `paddings[D, 0]` indicates how
  many values to add before the contents of `tensor` in that dimension, and
  `paddings[D, 1]` indicates how many values to add after the contents of
  `tensor` in that dimension. If `mode` is \"REFLECT\" then both `paddings[D, 0]`
  and `paddings[D, 1]` must be no greater than `tensor.dim_size(D) - 1`. If
  `mode` is \"SYMMETRIC\" then both `paddings[D, 0]` and `paddings[D, 1]` must be
  no greater than `tensor.dim_size(D)`.

  The padded size of each dimension D of the output is:

  `paddings[D, 0] + tensor.dim_size(D) + paddings[D, 1]`

  For example:

  ```python
  t = tf.constant([[1, 2, 3], [4, 5, 6]])
  paddings = tf.constant([[1, 1,], [2, 2]])
  # 'constant_values' is 0.
  # rank of 't' is 2.
  tf.pad(t, paddings, \"CONSTANT\")  # [[0, 0, 0, 0, 0, 0, 0],
                                   #  [0, 0, 1, 2, 3, 0, 0],
                                   #  [0, 0, 4, 5, 6, 0, 0],
                                   #  [0, 0, 0, 0, 0, 0, 0]]

  tf.pad(t, paddings, \"REFLECT\")  # [[6, 5, 4, 5, 6, 5, 4],
                                  #  [3, 2, 1, 2, 3, 2, 1],
                                  #  [6, 5, 4, 5, 6, 5, 4],
                                  #  [3, 2, 1, 2, 3, 2, 1]]

  tf.pad(t, paddings, \"SYMMETRIC\")  # [[2, 1, 1, 2, 3, 3, 2],
                                    #  [2, 1, 1, 2, 3, 3, 2],
                                    #  [5, 4, 4, 5, 6, 6, 5],
                                    #  [5, 4, 4, 5, 6, 6, 5]]
  ```

  Args:
    tensor: A `Tensor`.
    paddings: A `Tensor` of type `int32`.
    mode: One of \"CONSTANT\", \"REFLECT\", or \"SYMMETRIC\" (case-insensitive)
    constant_values: In \"CONSTANT\" mode, the scalar pad value to use. Must be
      same type as `tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.

  Raises:
    ValueError: When mode is not one of \"CONSTANT\", \"REFLECT\", or \"SYMMETRIC\".
  "
  [tensor paddings & {:keys [mode constant_values name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "pad" [tensor paddings] {:mode mode :constant_values constant_values :name name }))
(defn parallel-stack 
  "Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor in parallel.

  Requires that the shape of inputs be known at graph construction time.

  Packs the list of tensors in `values` into a tensor with rank one higher than
  each tensor in `values`, by packing them along the first dimension.
  Given a list of length `N` of tensors of shape `(A, B, C)`; the `output`
  tensor will have the shape `(N, A, B, C)`.

  For example:

  ```python
  x = tf.constant([1, 4])
  y = tf.constant([2, 5])
  z = tf.constant([3, 6])
  tf.parallel_stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]]
  ```

  The difference between `stack` and `parallel_stack` is that `stack` requires
  all the inputs be computed before the operation will begin but doesn't require
  that the input shapes be known during graph construction.

  `parallel_stack` will copy pieces of the input into the output as they become
  available, in some situations this can provide a performance benefit.

  Unlike `stack`, `parallel_stack` does NOT support backpropagation.

  This is the opposite of unstack.  The numpy equivalent is

      tf.parallel_stack([x, y, z]) = np.asarray([x, y, z])

  Args:
    values: A list of `Tensor` objects with the same shape and type.
    name: A name for this operation (optional).

  Returns:
    output: A stacked `Tensor` with the same type as `values`.
  "
  [values  & {:keys [name]} ]
    (py/call-attr-kw v2 "parallel_stack" [values] {:name name }))

(defn pow 
  "Computes the power of one value to another.

  Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
  corresponding elements in `x` and `y`. For example:

  ```python
  x = tf.constant([[2, 2], [3, 3]])
  y = tf.constant([[8, 16], [2, 3]])
  tf.pow(x, y)  # [[256, 65536], [9, 27]]
  ```

  Args:
    x: A `Tensor` of type `float16`, `float32`, `float64`, `int32`, `int64`,
      `complex64`, or `complex128`.
    y: A `Tensor` of type `float16`, `float32`, `float64`, `int32`, `int64`,
      `complex64`, or `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`.
  "
  [ x y name ]
  (py/call-attr v2 "pow"  x y name ))

(defn print 
  "Print the specified inputs.

  A TensorFlow operator that prints the specified inputs to a desired
  output stream or logging level. The inputs may be dense or sparse Tensors,
  primitive python objects, data structures that contain tensors, and printable
  Python objects. Printed tensors will recursively show the first and last
  elements of each dimension to summarize.

  @compatibility(python2)
  In python 2.7, make sure to import the following:
  `from __future__ import print_function`
  @end_compatibility

  Example:
    Single-input usage:

    ```python
    tensor = tf.range(10)
    tf.print(tensor, output_stream=sys.stderr)
    ```

    (This prints \"[0 1 2 ... 7 8 9]\" to sys.stderr)

    Multi-input usage:

    ```python
    tensor = tf.range(10)
    tf.print(\"tensors:\", tensor, {2: tensor * 2}, output_stream=sys.stdout)
    ```

    (This prints \"tensors: [0 1 2 ... 7 8 9] {2: [0 2 4 ... 14 16 18]}\" to
    sys.stdout)

    Changing the input separator:
    ```python
    tensor_a = tf.range(2)
    tensor_b = tensor_a * 2
    tf.print(tensor_a, tensor_b, output_stream=sys.stderr, sep=',')
    ```

    (This prints \"[0 1],[0 2]\" to sys.stderr)

    Usage in a `tf.function`:

    ```python
    @tf.function
    def f():
        tensor = tf.range(10)
        tf.print(tensor, output_stream=sys.stderr)
        return tensor

    range_tensor = f()
    ```

    (This prints \"[0 1 2 ... 7 8 9]\" to sys.stderr)

  @compatibility(TF 1.x Graphs and Sessions)
  In graphs manually created outside of `tf.function`, this method returns
  the created TF operator that prints the data. To make sure the
  operator runs, users need to pass the produced op to
  `tf.compat.v1.Session`'s run method, or to use the op as a control
  dependency for executed ops by specifying
  `with tf.compat.v1.control_dependencies([print_op])`.
  @end_compatibility

    Compatibility usage in TF 1.x graphs:

    ```python
    sess = tf.compat.v1.Session()
    with sess.as_default():
        tensor = tf.range(10)
        print_op = tf.print(\"tensors:\", tensor, {2: tensor * 2},
                            output_stream=sys.stdout)
        with tf.control_dependencies([print_op]):
          tripled_tensor = tensor * 3
        sess.run(tripled_tensor)
    ```

    (This prints \"tensors: [0 1 2 ... 7 8 9] {2: [0 2 4 ... 14 16 18]}\" to
    sys.stdout)

  Note: In Jupyter notebooks and colabs, `tf.print` prints to the notebook
    cell outputs. It will not write to the notebook kernel's console logs.

  Args:
    *inputs: Positional arguments that are the inputs to print. Inputs in the
      printed output will be separated by spaces. Inputs may be python
      primitives, tensors, data structures such as dicts and lists that may
      contain tensors (with the data structures possibly nested in arbitrary
      ways), and printable python objects.
    output_stream: The output stream, logging level, or file to print to.
      Defaults to sys.stderr, but sys.stdout, tf.compat.v1.logging.info,
      tf.compat.v1.logging.warning, tf.compat.v1.logging.error,
      absl.logging.info, absl.logging.warning and absl.loogging,error are also
      supported. To print to a file, pass a string started with \"file://\"
      followed by the file path, e.g., \"file:///tmp/foo.out\".
    summarize: The first and last `summarize` elements within each dimension are
      recursively printed per Tensor. If None, then the first 3 and last 3
      elements of each dimension are printed for each tensor. If set to -1, it
      will print all elements of every tensor.
    sep: The string to use to separate the inputs. Defaults to \" \".
    end: End character that is appended at the end the printed string.
      Defaults to the newline character.
    name: A name for the operation (optional).

  Returns:
    None when executing eagerly. During graph tracing this returns
    a TF operator that prints the specified inputs in the specified output
    stream or logging level. This operator will be automatically executed
    except inside of `tf.compat.v1` graphs and sessions.

  Raises:
    ValueError: If an unsupported output stream is specified.
  "
  [  ]
  (py/call-attr v2 "print"  ))

(defn py-function 
  "Wraps a python function into a TensorFlow op that executes it eagerly.

  This function allows expressing computations in a TensorFlow graph as
  Python functions. In particular, it wraps a Python function `func`
  in a once-differentiable TensorFlow operation that executes it with eager
  execution enabled. As a consequence, `tf.py_function` makes it
  possible to express control flow using Python constructs (`if`, `while`,
  `for`, etc.), instead of TensorFlow control flow constructs (`tf.cond`,
  `tf.while_loop`). For example, you might use `tf.py_function` to
  implement the log huber function:

  ```python
  def log_huber(x, m):
    if tf.abs(x) <= m:
      return x**2
    else:
      return m**2 * (1 - 2 * tf.math.log(m) + tf.math.log(x**2))

  x = tf.compat.v1.placeholder(tf.float32)
  m = tf.compat.v1.placeholder(tf.float32)

  y = tf.py_function(func=log_huber, inp=[x, m], Tout=tf.float32)
  dy_dx = tf.gradients(y, x)[0]

  with tf.compat.v1.Session() as sess:
    # The session executes `log_huber` eagerly. Given the feed values below,
    # it will take the first branch, so `y` evaluates to 1.0 and
    # `dy_dx` evaluates to 2.0.
    y, dy_dx = sess.run([y, dy_dx], feed_dict={x: 1.0, m: 2.0})
  ```

  You can also use `tf.py_function` to debug your models at runtime
  using Python tools, i.e., you can isolate portions of your code that
  you want to debug, wrap them in Python functions and insert `pdb` tracepoints
  or print statements as desired, and wrap those functions in
  `tf.py_function`.

  For more information on eager execution, see the
  [Eager guide](https://tensorflow.org/guide/eager).

  `tf.py_function` is similar in spirit to `tf.compat.v1.py_func`, but unlike
  the latter, the former lets you use TensorFlow operations in the wrapped
  Python function. In particular, while `tf.compat.v1.py_func` only runs on CPUs
  and
  wraps functions that take NumPy arrays as inputs and return NumPy arrays as
  outputs, `tf.py_function` can be placed on GPUs and wraps functions
  that take Tensors as inputs, execute TensorFlow operations in their bodies,
  and return Tensors as outputs.

  Like `tf.compat.v1.py_func`, `tf.py_function` has the following limitations
  with respect to serialization and distribution:

  * The body of the function (i.e. `func`) will not be serialized in a
    `GraphDef`. Therefore, you should not use this function if you need to
    serialize your model and restore it in a different environment.

  * The operation must run in the same address space as the Python program
    that calls `tf.py_function()`. If you are using distributed
    TensorFlow, you must run a `tf.distribute.Server` in the same process as the
    program that calls `tf.py_function()` and you must pin the created
    operation to a device in that server (e.g. using `with tf.device():`).


  Args:
    func: A Python function which accepts a list of `Tensor` objects having
      element types that match the corresponding `tf.Tensor` objects in `inp`
      and returns a list of `Tensor` objects (or a single `Tensor`, or `None`)
      having element types that match the corresponding values in `Tout`.
    inp: A list of `Tensor` objects.
    Tout: A list or tuple of tensorflow data types or a single tensorflow data
      type if there is only one, indicating what `func` returns; an empty list
      if no value is returned (i.e., if the return value is `None`).
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` or a single `Tensor` which `func` computes; an empty list
    if `func` returns None.
  "
  [ func inp Tout name ]
  (py/call-attr v2 "py_function"  func inp Tout name ))

(defn range 
  "Creates a sequence of numbers.

  Creates a sequence of numbers that begins at `start` and extends by
  increments of `delta` up to but not including `limit`.

  The dtype of the resulting tensor is inferred from the inputs unless
  it is provided explicitly.

  Like the Python builtin `range`, `start` defaults to 0, so that
  `range(n) = range(0, n)`.

  For example:

  ```python
  start = 3
  limit = 18
  delta = 3
  tf.range(start, limit, delta)  # [3, 6, 9, 12, 15]

  start = 3
  limit = 1
  delta = -0.5
  tf.range(start, limit, delta)  # [3, 2.5, 2, 1.5]

  limit = 5
  tf.range(limit)  # [0, 1, 2, 3, 4]
  ```

  Args:
    start: A 0-D `Tensor` (scalar). Acts as first entry in the range if `limit`
      is not None; otherwise, acts as range limit and first entry defaults to 0.
    limit: A 0-D `Tensor` (scalar). Upper limit of sequence, exclusive. If None,
      defaults to the value of `start` while the first entry of the range
      defaults to 0.
    delta: A 0-D `Tensor` (scalar). Number that increments `start`. Defaults to
      1.
    dtype: The type of the elements of the resulting tensor.
    name: A name for the operation. Defaults to \"range\".

  Returns:
    An 1-D `Tensor` of type `dtype`.

  @compatibility(numpy)
  Equivalent to np.arange
  @end_compatibility
  "
  [start limit & {:keys [delta dtype name]
                       :or {dtype None}} ]
    (py/call-attr-kw v2 "range" [start limit] {:delta delta :dtype dtype :name name }))

(defn rank 
  "Returns the rank of a tensor.

  Returns a 0-D `int32` `Tensor` representing the rank of `input`.

  For example:

  ```python
  # shape of tensor 't' is [2, 2, 3]
  t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
  tf.rank(t)  # 3
  ```

  **Note**: The rank of a tensor is not the same as the rank of a matrix. The
  rank of a tensor is the number of indices required to uniquely select each
  element of the tensor. Rank is also known as \"order\", \"degree\", or \"ndims.\"

  Args:
    input: A `Tensor` or `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.

  @compatibility(numpy)
  Equivalent to np.ndim
  @end_compatibility
  "
  [ input name ]
  (py/call-attr v2 "rank"  input name ))

(defn realdiv 
  "Returns x / y element-wise for real types.

  If `x` and `y` are reals, this will return the floating-point division.

  *NOTE*: `Div` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr v2 "realdiv"  x y name ))

(defn recompute-grad 
  "An eager-compatible version of recompute_grad.

  For f(*args, **kwargs), this supports gradients with respect to args, or to
  gradients with respect to any variables residing in the kwarg 'variables'.
  Note that for keras layer and model objects, this is handled automatically.

  Warning: If `f` was originally a tf.keras Model or Layer object, `g` will not
  be able to access the member variables of that object, because `g` returns
  through the wrapper function `inner`.  When recomputing gradients through
  objects that inherit from keras, we suggest keeping a reference to the
  underlying object around for the purpose of accessing these variables.

  Args:
    f: function `f(*x)` that returns a `Tensor` or sequence of `Tensor` outputs.

  Returns:
   A function `g` that wraps `f`, but which recomputes `f` on the backwards
   pass of a gradient call.
  "
  [ f ]
  (py/call-attr v2 "recompute_grad"  f ))

(defn reduce-all 
  "Computes the \"logical and\" of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  x = tf.constant([[True,  True], [False, False]])
  tf.reduce_all(x)  # False
  tf.reduce_all(x, 0)  # [False, False]
  tf.reduce_all(x, 1)  # [True, False]
  ```

  Args:
    input_tensor: The boolean tensor to reduce.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.all
  @end_compatibility
  "
  [input_tensor axis & {:keys [keepdims name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "reduce_all" [input_tensor axis] {:keepdims keepdims :name name }))

(defn reduce-any 
  "Computes the \"logical or\" of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  x = tf.constant([[True,  True], [False, False]])
  tf.reduce_any(x)  # True
  tf.reduce_any(x, 0)  # [True, True]
  tf.reduce_any(x, 1)  # [True, False]
  ```

  Args:
    input_tensor: The boolean tensor to reduce.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.any
  @end_compatibility
  "
  [input_tensor axis & {:keys [keepdims name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "reduce_any" [input_tensor axis] {:keepdims keepdims :name name }))

(defn reduce-logsumexp 
  "Computes log(sum(exp(elements across dimensions of a tensor))).

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.

  This function is more numerically stable than log(sum(exp(input))). It avoids
  overflows caused by taking the exp of large inputs and underflows caused by
  taking the log of small inputs.

  For example:

  ```python
  x = tf.constant([[0., 0., 0.], [0., 0., 0.]])
  tf.reduce_logsumexp(x)  # log(6)
  tf.reduce_logsumexp(x, 0)  # [log(2), log(2), log(2)]
  tf.reduce_logsumexp(x, 1)  # [log(3), log(3)]
  tf.reduce_logsumexp(x, 1, keepdims=True)  # [[log(3)], [log(3)]]
  tf.reduce_logsumexp(x, [0, 1])  # log(6)
  ```

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.
  "
  [input_tensor axis & {:keys [keepdims name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "reduce_logsumexp" [input_tensor axis] {:keepdims keepdims :name name }))

(defn reduce-max 
  "Computes the maximum of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  Args:
    input_tensor: The tensor to reduce. Should have real numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.max
  @end_compatibility
  "
  [input_tensor axis & {:keys [keepdims name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "reduce_max" [input_tensor axis] {:keepdims keepdims :name name }))

(defn reduce-mean 
  "Computes the mean of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  x = tf.constant([[1., 1.], [2., 2.]])
  tf.reduce_mean(x)  # 1.5
  tf.reduce_mean(x, 0)  # [1.5, 1.5]
  tf.reduce_mean(x, 1)  # [1.,  2.]
  ```

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.mean

  Please note that `np.mean` has a `dtype` parameter that could be used to
  specify the output type. By default this is `dtype=float64`. On the other
  hand, `tf.reduce_mean` has an aggressive type inference from `input_tensor`,
  for example:

  ```python
  x = tf.constant([1, 0, 1, 0])
  tf.reduce_mean(x)  # 0
  y = tf.constant([1., 0., 1., 0.])
  tf.reduce_mean(y)  # 0.5
  ```

  @end_compatibility
  "
  [input_tensor axis & {:keys [keepdims name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "reduce_mean" [input_tensor axis] {:keepdims keepdims :name name }))

(defn reduce-min 
  "Computes the minimum of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  Args:
    input_tensor: The tensor to reduce. Should have real numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.min
  @end_compatibility
  "
  [input_tensor axis & {:keys [keepdims name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "reduce_min" [input_tensor axis] {:keepdims keepdims :name name }))

(defn reduce-prod 
  "Computes the product of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor.

  @compatibility(numpy)
  Equivalent to np.prod
  @end_compatibility
  "
  [input_tensor axis & {:keys [keepdims name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "reduce_prod" [input_tensor axis] {:keepdims keepdims :name name }))

(defn reduce-sum 
  "Computes the sum of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` is None, all dimensions are reduced, and a
  tensor with a single element is returned.

  For example:

  ```python
  x = tf.constant([[1, 1, 1], [1, 1, 1]])
  tf.reduce_sum(x)  # 6
  tf.reduce_sum(x, 0)  # [2, 2, 2]
  tf.reduce_sum(x, 1)  # [3, 3]
  tf.reduce_sum(x, 1, keepdims=True)  # [[3], [3]]
  tf.reduce_sum(x, [0, 1])  # 6
  ```

  Args:
    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If `None` (the default), reduces all
      dimensions. Must be in the range `[-rank(input_tensor),
      rank(input_tensor))`.
    keepdims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    The reduced tensor, of the same dtype as the input_tensor.

  @compatibility(numpy)
  Equivalent to np.sum apart the fact that numpy upcast uint8 and int32 to
  int64 while tensorflow returns the same dtype as the input.
  @end_compatibility
  "
  [input_tensor axis & {:keys [keepdims name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "reduce_sum" [input_tensor axis] {:keepdims keepdims :name name }))
(defn register-tensor-conversion-function 
  "Registers a function for converting objects of `base_type` to `Tensor`.

  The conversion function must have the following signature:

  ```python
      def conversion_func(value, dtype=None, name=None, as_ref=False):
        # ...
  ```

  It must return a `Tensor` with the given `dtype` if specified. If the
  conversion function creates a new `Tensor`, it should use the given
  `name` if specified. All exceptions will be propagated to the caller.

  The conversion function may return `NotImplemented` for some
  inputs. In this case, the conversion process will continue to try
  subsequent conversion functions.

  If `as_ref` is true, the function must return a `Tensor` reference,
  such as a `Variable`.

  NOTE: The conversion functions will execute in order of priority,
  followed by order of registration. To ensure that a conversion function
  `F` runs before another conversion function `G`, ensure that `F` is
  registered with a smaller priority than `G`.

  Args:
    base_type: The base type or tuple of base types for all objects that
      `conversion_func` accepts.
    conversion_func: A function that converts instances of `base_type` to
      `Tensor`.
    priority: Optional integer that indicates the priority for applying this
      conversion function. Conversion functions with smaller priority values run
      earlier than conversion functions with larger priority values. Defaults to
      100.

  Raises:
    TypeError: If the arguments do not have the appropriate type.
  "
  [base_type conversion_func  & {:keys [priority]} ]
    (py/call-attr-kw v2 "register_tensor_conversion_function" [base_type conversion_func] {:priority priority }))

(defn repeat 
  "Repeat elements of `input`

  Args:
    input: An `N`-dimensional Tensor.
    repeats: An 1-D `int` Tensor. The number of repetitions for each element.
      repeats is broadcasted to fit the shape of the given axis. `len(repeats)`
      must equal `input.shape[axis]` if axis is not None.
    axis: An int. The axis along which to repeat values. By default (axis=None),
      use the flattened input array, and return a flat output array.
    name: A name for the operation.

  Returns:
    A Tensor which has the same shape as `input`, except along the given axis.
      If axis is None then the output array is flattened to match the flattened
      input array.
  #### Examples:
    ```python
    >>> repeat(['a', 'b', 'c'], repeats=[3, 0, 2], axis=0)
    ['a', 'a', 'a', 'c', 'c']
    >>> repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=0)
    [[1, 2], [1, 2], [3, 4], [3, 4], [3, 4]]
    >>> repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=1)
    [[1, 1, 2, 2, 2], [3, 3, 4, 4, 4]]
    >>> repeat(3, repeats=4)
    [3, 3, 3, 3]
    >>> repeat([[1,2], [3,4]], repeats=2)
    [1, 1, 2, 2, 3, 3, 4, 4]
    ```
  "
  [ input repeats axis name ]
  (py/call-attr v2 "repeat"  input repeats axis name ))

(defn required-space-to-batch-paddings 
  "Calculate padding required to make block_shape divide input_shape.

  This function can be used to calculate a suitable paddings argument for use
  with space_to_batch_nd and batch_to_space_nd.

  Args:
    input_shape: int32 Tensor of shape [N].
    block_shape: int32 Tensor of shape [N].
    base_paddings: Optional int32 Tensor of shape [N, 2].  Specifies the minimum
      amount of padding to use.  All elements must be >= 0.  If not specified,
      defaults to 0.
    name: string.  Optional name prefix.

  Returns:
    (paddings, crops), where:

    `paddings` and `crops` are int32 Tensors of rank 2 and shape [N, 2]
    satisfying:

        paddings[i, 0] = base_paddings[i, 0].
        0 <= paddings[i, 1] - base_paddings[i, 1] < block_shape[i]
        (input_shape[i] + paddings[i, 0] + paddings[i, 1]) % block_shape[i] == 0

        crops[i, 0] = 0
        crops[i, 1] = paddings[i, 1] - base_paddings[i, 1]

  Raises: ValueError if called with incompatible shapes.
  "
  [ input_shape block_shape base_paddings name ]
  (py/call-attr v2 "required_space_to_batch_paddings"  input_shape block_shape base_paddings name ))

(defn reshape 
  "Reshapes a tensor.

  Given `tensor`, this operation returns a tensor that has the same values
  as `tensor` with shape `shape`.

  If one component of `shape` is the special value -1, the size of that
  dimension is computed so that the total size remains constant.  In particular,
  a `shape` of `[-1]` flattens into 1-D.  At most one component of `shape` can
  be -1.

  If `shape` is 1-D or higher, then the operation returns a tensor with shape
  `shape` filled with the values of `tensor`. In this case, the number of
  elements implied by `shape` must be the same as the number of elements in
  `tensor`.

  For example:

  ```
  # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
  # tensor 't' has shape [9]
  reshape(t, [3, 3]) ==> [[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]]

  # tensor 't' is [[[1, 1], [2, 2]],
  #                [[3, 3], [4, 4]]]
  # tensor 't' has shape [2, 2, 2]
  reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                          [3, 3, 4, 4]]

  # tensor 't' is [[[1, 1, 1],
  #                 [2, 2, 2]],
  #                [[3, 3, 3],
  #                 [4, 4, 4]],
  #                [[5, 5, 5],
  #                 [6, 6, 6]]]
  # tensor 't' has shape [3, 2, 3]
  # pass '[-1]' to flatten 't'
  reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]

  # -1 can also be used to infer the shape

  # -1 is inferred to be 9:
  reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [4, 4, 4, 5, 5, 5, 6, 6, 6]]
  # -1 is inferred to be 2:
  reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [4, 4, 4, 5, 5, 5, 6, 6, 6]]
  # -1 is inferred to be 3:
  reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
                                [2, 2, 2],
                                [3, 3, 3]],
                               [[4, 4, 4],
                                [5, 5, 5],
                                [6, 6, 6]]]

  # tensor 't' is [7]
  # shape `[]` reshapes to a scalar
  reshape(t, []) ==> 7
  ```

  Args:
    tensor: A `Tensor`.
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Defines the shape of the output tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  "
  [ tensor shape name ]
  (py/call-attr v2 "reshape"  tensor shape name ))

(defn reverse 
  "Reverses specific dimensions of a tensor.

  NOTE `tf.reverse` has now changed behavior in preparation for 1.0.
  `tf.reverse_v2` is currently an alias that will be deprecated before TF 1.0.

  Given a `tensor`, and a `int32` tensor `axis` representing the set of
  dimensions of `tensor` to reverse. This operation reverses each dimension
  `i` for which there exists `j` s.t. `axis[j] == i`.

  `tensor` can have up to 8 dimensions. The number of dimensions specified
  in `axis` may be 0 or more entries. If an index is specified more than
  once, a InvalidArgument error is raised.

  For example:

  ```
  # tensor 't' is [[[[ 0,  1,  2,  3],
  #                  [ 4,  5,  6,  7],
  #                  [ 8,  9, 10, 11]],
  #                 [[12, 13, 14, 15],
  #                  [16, 17, 18, 19],
  #                  [20, 21, 22, 23]]]]
  # tensor 't' shape is [1, 2, 3, 4]

  # 'dims' is [3] or 'dims' is [-1]
  reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                          [ 7,  6,  5,  4],
                          [ 11, 10, 9, 8]],
                         [[15, 14, 13, 12],
                          [19, 18, 17, 16],
                          [23, 22, 21, 20]]]]

  # 'dims' is '[1]' (or 'dims' is '[-3]')
  reverse(t, dims) ==> [[[[12, 13, 14, 15],
                          [16, 17, 18, 19],
                          [20, 21, 22, 23]
                         [[ 0,  1,  2,  3],
                          [ 4,  5,  6,  7],
                          [ 8,  9, 10, 11]]]]

  # 'dims' is '[2]' (or 'dims' is '[-2]')
  reverse(t, dims) ==> [[[[8, 9, 10, 11],
                          [4, 5, 6, 7],
                          [0, 1, 2, 3]]
                         [[20, 21, 22, 23],
                          [16, 17, 18, 19],
                          [12, 13, 14, 15]]]]
  ```

  Args:
    tensor: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `uint16`, `int16`, `int32`, `int64`, `bool`, `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`, `string`.
      Up to 8-D.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      1-D. The indices of the dimensions to reverse. Must be in the range
      `[-rank(tensor), rank(tensor))`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  "
  [ tensor axis name ]
  (py/call-attr v2 "reverse"  tensor axis name ))

(defn reverse-sequence 
  "Reverses variable length slices.

  This op first slices `input` along the dimension `batch_axis`, and for each
  slice `i`, reverses the first `seq_lengths[i]` elements along
  the dimension `seq_axis`.

  The elements of `seq_lengths` must obey `seq_lengths[i] <= input.dims[seq_dim]`,
  and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.

  The output slice `i` along dimension `batch_axis` is then given by input
  slice `i`, with the first `seq_lengths[i]` slices along dimension
  `seq_axis` reversed.

  For example:

  ```
  # Given this:
  batch_dim = 0
  seq_dim = 1
  input.dims = (4, 8, ...)
  seq_lengths = [7, 2, 3, 5]

  # then slices of input are reversed on seq_dim, but only up to seq_lengths:
  output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
  output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
  output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
  output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]

  # while entries past seq_lens are copied through:
  output[0, 7:, :, ...] = input[0, 7:, :, ...]
  output[1, 2:, :, ...] = input[1, 2:, :, ...]
  output[2, 3:, :, ...] = input[2, 3:, :, ...]
  output[3, 2:, :, ...] = input[3, 2:, :, ...]
  ```

  In contrast, if:

  ```
  # Given this:
  batch_dim = 2
  seq_dim = 0
  input.dims = (8, ?, 4, ...)
  seq_lengths = [7, 2, 3, 5]

  # then slices of input are reversed on seq_dim, but only up to seq_lengths:
  output[0:7, :, 0, :, ...] = input[7:0:-1, :, 0, :, ...]
  output[0:2, :, 1, :, ...] = input[2:0:-1, :, 1, :, ...]
  output[0:3, :, 2, :, ...] = input[3:0:-1, :, 2, :, ...]
  output[0:5, :, 3, :, ...] = input[5:0:-1, :, 3, :, ...]

  # while entries past seq_lens are copied through:
  output[7:, :, 0, :, ...] = input[7:, :, 0, :, ...]
  output[2:, :, 1, :, ...] = input[2:, :, 1, :, ...]
  output[3:, :, 2, :, ...] = input[3:, :, 2, :, ...]
  output[2:, :, 3, :, ...] = input[2:, :, 3, :, ...]
  ```

  Args:
    input: A `Tensor`. The input to reverse.
    seq_lengths: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      1-D with length `input.dims(batch_dim)` and
      `max(seq_lengths) <= input.dims(seq_dim)`
    seq_axis: An `int`. The dimension which is partially reversed.
    batch_axis: An optional `int`. Defaults to `0`.
      The dimension along which reversal is performed.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input seq_lengths seq_axis batch_axis name ]
  (py/call-attr v2 "reverse_sequence"  input seq_lengths seq_axis batch_axis name ))

(defn roll 
  "Rolls the elements of a tensor along an axis.

  The elements are shifted positively (towards larger indices) by the offset of
  `shift` along the dimension of `axis`. Negative `shift` values will shift
  elements in the opposite direction. Elements that roll passed the last position
  will wrap around to the first and vice versa. Multiple shifts along multiple
  axes may be specified.

  For example:

  ```
  # 't' is [0, 1, 2, 3, 4]
  roll(t, shift=2, axis=0) ==> [3, 4, 0, 1, 2]

  # shifting along multiple dimensions
  # 't' is [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
  roll(t, shift=[1, -2], axis=[0, 1]) ==> [[7, 8, 9, 5, 6], [2, 3, 4, 0, 1]]

  # shifting along the same axis multiple times
  # 't' is [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
  roll(t, shift=[2, -3], axis=[1, 1]) ==> [[1, 2, 3, 4, 0], [6, 7, 8, 9, 5]]
  ```

  Args:
    input: A `Tensor`.
    shift: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Dimension must be 0-D or 1-D. `shift[i]` specifies the number of places by which
      elements are shifted positively (towards larger indices) along the dimension
      specified by `axis[i]`. Negative shifts will roll the elements in the opposite
      direction.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Dimension must be 0-D or 1-D. `axis[i]` specifies the dimension that the shift
      `shift[i]` should occur. If the same axis is referenced more than once, the
      total shift for that axis will be the sum of all the shifts that belong to that
      axis.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input shift axis name ]
  (py/call-attr v2 "roll"  input shift axis name ))

(defn round 
  "Rounds the values of a tensor to the nearest integer, element-wise.

  Rounds half to even.  Also known as bankers rounding. If you want to round
  according to the current system rounding mode use tf::cint.
  For example:

  ```python
  x = tf.constant([0.9, 2.5, 2.3, 1.5, -4.5])
  tf.round(x)  # [ 1.0, 2.0, 2.0, 2.0, -4.0 ]
  ```

  Args:
    x: A `Tensor` of type `float16`, `float32`, `float64`, `int32`, or `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of same shape and type as `x`.
  "
  [ x name ]
  (py/call-attr v2 "round"  x name ))

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
  (py/call-attr v2 "saturate_cast"  value dtype name ))

(defn scalar-mul 
  "Multiplies a scalar times a `Tensor` or `IndexedSlices` object.

  Intended for use in gradient code which might deal with `IndexedSlices`
  objects, which are easy to multiply by a scalar but more expensive to
  multiply with arbitrary tensors.

  Args:
    scalar: A 0-D scalar `Tensor`. Must have known shape.
    x: A `Tensor` or `IndexedSlices` to be scaled.
    name: A name for the operation (optional).

  Returns:
    `scalar * x` of the same type (`Tensor` or `IndexedSlices`) as `x`.

  Raises:
    ValueError: if scalar is not a 0-D `scalar`.
  "
  [ scalar x name ]
  (py/call-attr v2 "scalar_mul"  scalar x name ))

(defn scan 
  "scan on the list of tensors unpacked from `elems` on dimension 0.

  The simplest version of `scan` repeatedly applies the callable `fn` to a
  sequence of elements from first to last. The elements are made of the tensors
  unpacked from `elems` on dimension 0. The callable fn takes two tensors as
  arguments. The first argument is the accumulated value computed from the
  preceding invocation of fn, and the second is the value at the current
  position of `elems`. If `initializer` is None, `elems` must contain at least
  one element, and its first element is used as the initializer.

  Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
  of the result tensor is `[len(values)] + fn(initializer, values[0]).shape`.
  If reverse=True, it's fn(initializer, values[-1]).shape.

  This method also allows multi-arity `elems` and accumulator.  If `elems`
  is a (possibly nested) list or tuple of tensors, then each of these tensors
  must have a matching first (unpack) dimension.  The second argument of
  `fn` must match the structure of `elems`.

  If no `initializer` is provided, the output structure and dtypes of `fn`
  are assumed to be the same as its input; and in this case, the first
  argument of `fn` must match the structure of `elems`.

  If an `initializer` is provided, then the output of `fn` must have the same
  structure as `initializer`; and the first argument of `fn` must match
  this structure.

  For example, if `elems` is `(t1, [t2, t3])` and `initializer` is
  `[i1, i2]` then an appropriate signature for `fn` in `python2` is:
  `fn = lambda (acc_p1, acc_p2), (t1, [t2, t3]):` and `fn` must return a list,
  `[acc_n1, acc_n2]`.  An alternative correct signature for `fn`, and the
   one that works in `python3`, is:
  `fn = lambda a, t:`, where `a` and `t` correspond to the input tuples.

  Args:
    fn: The callable to be performed.  It accepts two arguments.  The first will
      have the same structure as `initializer` if one is provided, otherwise it
      will have the same structure as `elems`.  The second will have the same
      (possibly nested) structure as `elems`.  Its output must have the same
      structure as `initializer` if one is provided, otherwise it must have the
      same structure as `elems`.
    elems: A tensor or (possibly nested) sequence of tensors, each of which will
      be unpacked along their first dimension.  The nested sequence of the
      resulting slices will be the first argument to `fn`.
    initializer: (optional) A tensor or (possibly nested) sequence of tensors,
      initial value for the accumulator, and the expected output type of `fn`.
    parallel_iterations: (optional) The number of iterations allowed to run in
      parallel.
    back_prop: (optional) True enables support for back propagation.
    swap_memory: (optional) True enables GPU-CPU memory swapping.
    infer_shape: (optional) False disables tests for consistent output shapes.
    reverse: (optional) True scans the tensor last to first (instead of first to
      last).
    name: (optional) Name prefix for the returned tensors.

  Returns:
    A tensor or (possibly nested) sequence of tensors.  Each tensor packs the
    results of applying `fn` to tensors unpacked from `elems` along the first
    dimension, and the previous accumulator value(s), from first to last (or
    last to first, if `reverse=True`).

  Raises:
    TypeError: if `fn` is not callable or the structure of the output of
      `fn` and `initializer` do not match.
    ValueError: if the lengths of the output of `fn` and `initializer`
      do not match.

  Examples:
    ```python
    elems = np.array([1, 2, 3, 4, 5, 6])
    sum = scan(lambda a, x: a + x, elems)
    # sum == [1, 3, 6, 10, 15, 21]
    sum = scan(lambda a, x: a + x, elems, reverse=True)
    # sum == [21, 20, 18, 15, 11, 6]
    ```

    ```python
    elems = np.array([1, 2, 3, 4, 5, 6])
    initializer = np.array(0)
    sum_one = scan(
        lambda a, x: x[0] - x[1] + a, (elems + 1, elems), initializer)
    # sum_one == [1, 2, 3, 4, 5, 6]
    ```

    ```python
    elems = np.array([1, 0, 0, 0, 0, 0])
    initializer = (np.array(0), np.array(1))
    fibonaccis = scan(lambda a, _: (a[1], a[0] + a[1]), elems, initializer)
    # fibonaccis == ([1, 1, 2, 3, 5, 8], [1, 2, 3, 5, 8, 13])
    ```
  "
  [fn elems initializer & {:keys [parallel_iterations back_prop swap_memory infer_shape reverse name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "scan" [fn elems initializer] {:parallel_iterations parallel_iterations :back_prop back_prop :swap_memory swap_memory :infer_shape infer_shape :reverse reverse :name name }))

(defn scatter-nd 
  "Scatter `updates` into a new tensor according to `indices`.

  Creates a new tensor by applying sparse `updates` to individual values or
  slices within a tensor (initially zero for numeric, empty for string) of
  the given `shape` according to indices.  This operator is the inverse of the
  `tf.gather_nd` operator which extracts values or slices from a given tensor.

  This operation is similar to tensor_scatter_add, except that the tensor is
  zero-initialized. Calling `tf.scatter_nd(indices, values, shape)` is identical
  to `tensor_scatter_add(tf.zeros(shape, values.dtype), indices, values)`

  If `indices` contains duplicates, then their updates are accumulated (summed).

  **WARNING**: The order in which updates are applied is nondeterministic, so the
  output will be nondeterministic if `indices` contains duplicates -- because
  of some numerical approximation issues, numbers summed in different order
  may yield different results.

  `indices` is an integer tensor containing indices into a new tensor of shape
  `shape`.  The last dimension of `indices` can be at most the rank of `shape`:

      indices.shape[-1] <= shape.rank

  The last dimension of `indices` corresponds to indices into elements
  (if `indices.shape[-1] = shape.rank`) or slices
  (if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
  `shape`.  `updates` is a tensor with shape

      indices.shape[:-1] + shape[indices.shape[-1]:]

  The simplest form of scatter is to insert individual elements in a tensor by
  index. For example, say we want to insert 4 scattered elements in a rank-1
  tensor with 8 elements.

  <div style=\"width:70%; margin:auto; margin-bottom:10px; margin-top:20px;\">
  <img style=\"width:100%\" src=\"https://www.tensorflow.org/images/ScatterNd1.png\" alt>
  </div>

  In Python, this scatter operation would look like this:

  ```python
      indices = tf.constant([[4], [3], [1], [7]])
      updates = tf.constant([9, 10, 11, 12])
      shape = tf.constant([8])
      scatter = tf.scatter_nd(indices, updates, shape)
      with tf.Session() as sess:
        print(sess.run(scatter))
  ```

  The resulting tensor would look like this:

      [0, 11, 0, 10, 9, 0, 0, 12]

  We can also, insert entire slices of a higher rank tensor all at once. For
  example, if we wanted to insert two slices in the first dimension of a
  rank-3 tensor with two matrices of new values.

  <div style=\"width:70%; margin:auto; margin-bottom:10px; margin-top:20px;\">
  <img style=\"width:100%\" src=\"https://www.tensorflow.org/images/ScatterNd2.png\" alt>
  </div>

  In Python, this scatter operation would look like this:

  ```python
      indices = tf.constant([[0], [2]])
      updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                              [7, 7, 7, 7], [8, 8, 8, 8]],
                             [[5, 5, 5, 5], [6, 6, 6, 6],
                              [7, 7, 7, 7], [8, 8, 8, 8]]])
      shape = tf.constant([4, 4, 4])
      scatter = tf.scatter_nd(indices, updates, shape)
      with tf.Session() as sess:
        print(sess.run(scatter))
  ```

  The resulting tensor would look like this:

      [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
       [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
       [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
       [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]

  Note that on CPU, if an out of bound index is found, an error is returned.
  On GPU, if an out of bound index is found, the index is ignored.

  Args:
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Index tensor.
    updates: A `Tensor`. Updates to scatter into output.
    shape: A `Tensor`. Must have the same type as `indices`.
      1-D. The shape of the resulting tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `updates`.
  "
  [ indices updates shape name ]
  (py/call-attr v2 "scatter_nd"  indices updates shape name ))

(defn searchsorted 
  "Searches input tensor for values on the innermost dimension.

  A 2-D example:

  ```
    sorted_sequence = [[0, 3, 9, 9, 10],
                       [1, 2, 3, 4, 5]]
    values = [[2, 4, 9],
              [0, 2, 6]]

    result = searchsorted(sorted_sequence, values, side=\"left\")

    result == [[1, 2, 2],
               [0, 1, 5]]

    result = searchsorted(sorted_sequence, values, side=\"right\")

    result == [[1, 2, 4],
               [0, 2, 5]]
  ```

  Args:
    sorted_sequence: N-D `Tensor` containing a sorted sequence.
    values: N-D `Tensor` containing the search values.
    side: 'left' or 'right'; 'left' corresponds to lower_bound and 'right' to
      upper_bound.
    out_type: The output type (`int32` or `int64`).  Default is `tf.int32`.
    name: Optional name for the operation.

  Returns:
    An N-D `Tensor` the size of values containing the result of applying either
    lower_bound or upper_bound (depending on side) to each value.  The result
    is not a global index to the entire `Tensor`, but the index in the last
    dimension.

  Raises:
    ValueError: If the last dimension of `sorted_sequence >= 2^31-1` elements.
                If the total size of values exceeds `2^31 - 1` elements.
                If the first `N-1` dimensions of the two tensors don't match.
  "
  [sorted_sequence values & {:keys [side out_type name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "searchsorted" [sorted_sequence values] {:side side :out_type out_type :name name }))

(defn sequence-mask 
  "Returns a mask tensor representing the first N positions of each cell.

  If `lengths` has shape `[d_1, d_2, ..., d_n]` the resulting tensor `mask` has
  dtype `dtype` and shape `[d_1, d_2, ..., d_n, maxlen]`, with

  ```
  mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])
  ```

  Examples:

  ```python
  tf.sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                  #  [True, True, True, False, False],
                                  #  [True, True, False, False, False]]

  tf.sequence_mask([[1, 3],[2,0]])  # [[[True, False, False],
                                    #   [True, True, True]],
                                    #  [[True, True, False],
                                    #   [False, False, False]]]
  ```

  Args:
    lengths: integer tensor, all its values <= maxlen.
    maxlen: scalar integer tensor, size of last dimension of returned tensor.
      Default is the maximum value in `lengths`.
    dtype: output type of the resulting tensor.
    name: name of the op.

  Returns:
    A mask tensor of shape `lengths.shape + (maxlen,)`, cast to specified dtype.
  Raises:
    ValueError: if `maxlen` is not a scalar.
  "
  [lengths maxlen & {:keys [dtype name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "sequence_mask" [lengths maxlen] {:dtype dtype :name name }))

(defn shape 
  "Returns the shape of a tensor.

  This operation returns a 1-D integer tensor representing the shape of `input`.

  For example:

  ```python
  t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
  tf.shape(t)  # [2, 2, 3]
  ```

  Args:
    input: A `Tensor` or `SparseTensor`.
    out_type: (Optional) The specified output type of the operation (`int32` or
      `int64`). Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  "
  [input & {:keys [out_type name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "shape" [input] {:out_type out_type :name name }))

(defn shape-n 
  "Returns shape of tensors.

  Args:
    input: A list of at least 1 `Tensor` object with the same type.
    out_type: The specified output type of the operation (`int32` or `int64`).
      Defaults to `tf.int32`(optional).
    name: A name for the operation (optional).

  Returns:
    A list with the same length as `input` of `Tensor` objects with
      type `out_type`.
  "
  [input & {:keys [out_type name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "shape_n" [input] {:out_type out_type :name name }))

(defn sigmoid 
  "Computes sigmoid of `x` element-wise.

  Specifically, `y = 1 / (1 + exp(-x))`.

  Args:
    x: A Tensor with type `float16`, `float32`, `float64`, `complex64`, or
      `complex128`.
    name: A name for the operation (optional).

  Returns:
    A Tensor with the same type as `x`.

  @compatibility(scipy)
  Equivalent to scipy.special.expit
  @end_compatibility
  "
  [ x name ]
  (py/call-attr v2 "sigmoid"  x name ))

(defn sign 
  "Returns an element-wise indication of the sign of a number.

  `y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.

  For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.

    If `x` is a `SparseTensor`, returns
    `SparseTensor(x.indices, tf.math.sign(x.values, ...), x.dense_shape)`"
  [ x name ]
  (py/call-attr v2 "sign"  x name ))

(defn sin 
  "Computes sine of x element-wise.

    Given an input tensor, this function computes sine of every
    element in the tensor. Input range is `(-inf, inf)` and
    output range is `[-1,1]`.

    ```python
    x = tf.constant([-float(\"inf\"), -9, -0.5, 1, 1.2, 200, 10, float(\"inf\")])
    tf.math.sin(x) ==> [nan -0.4121185 -0.47942555 0.84147096 0.9320391 -0.87329733 -0.54402107 nan]
    ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr v2 "sin"  x name ))

(defn sinh 
  "Computes hyperbolic sine of x element-wise.

    Given an input tensor, this function computes hyperbolic sine of every
    element in the tensor. Input range is `[-inf,inf]` and output range
    is `[-inf,inf]`.

    ```python
    x = tf.constant([-float(\"inf\"), -9, -0.5, 1, 1.2, 2, 10, float(\"inf\")])
    tf.math.sinh(x) ==> [-inf -4.0515420e+03 -5.2109528e-01 1.1752012e+00 1.5094614e+00 3.6268604e+00 1.1013232e+04 inf]
    ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr v2 "sinh"  x name ))

(defn size 
  ""
  [input & {:keys [out_type name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "size" [input] {:out_type out_type :name name }))

(defn slice 
  "Extracts a slice from a tensor.

  This operation extracts a slice of size `size` from a tensor `input_` starting
  at the location specified by `begin`. The slice `size` is represented as a
  tensor shape, where `size[i]` is the number of elements of the 'i'th dimension
  of `input_` that you want to slice. The starting location (`begin`) for the
  slice is represented as an offset in each dimension of `input_`. In other
  words, `begin[i]` is the offset into the i'th dimension of `input_` that you
  want to slice from.

  Note that `tf.Tensor.__getitem__` is typically a more pythonic way to
  perform slices, as it allows you to write `foo[3:7, :-2]` instead of
  `tf.slice(foo, [3, 0], [4, foo.get_shape()[1]-2])`.

  `begin` is zero-based; `size` is one-based. If `size[i]` is -1,
  all remaining elements in dimension i are included in the
  slice. In other words, this is equivalent to setting:

  `size[i] = input_.dim_size(i) - begin[i]`

  This operation requires that:

  `0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n]`

  For example:

  ```python
  t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                   [[3, 3, 3], [4, 4, 4]],
                   [[5, 5, 5], [6, 6, 6]]])
  tf.slice(t, [1, 0, 0], [1, 1, 3])  # [[[3, 3, 3]]]
  tf.slice(t, [1, 0, 0], [1, 2, 3])  # [[[3, 3, 3],
                                     #   [4, 4, 4]]]
  tf.slice(t, [1, 0, 0], [2, 1, 3])  # [[[3, 3, 3]],
                                     #  [[5, 5, 5]]]
  ```

  Args:
    input_: A `Tensor`.
    begin: An `int32` or `int64` `Tensor`.
    size: An `int32` or `int64` `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` the same type as `input_`.
  "
  [ input_ begin size name ]
  (py/call-attr v2 "slice"  input_ begin size name ))

(defn sort 
  "Sorts a tensor.
  
  Usage:

  ```python
  import tensorflow as tf
  a = [1, 10, 26.9, 2.8, 166.32, 62.3]
  b = tf.sort(a,axis=-1,direction='ASCENDING',name=None)
  c = tf.keras.backend.eval(b)
  # Here, c = [  1.     2.8   10.    26.9   62.3  166.32]
  ```
  
  Args:
    values: 1-D or higher numeric `Tensor`.
    axis: The axis along which to sort. The default is -1, which sorts the last
      axis.
    direction: The direction in which to sort the values (`'ASCENDING'` or
      `'DESCENDING'`).
    name: Optional name for the operation.

  Returns:
    A `Tensor` with the same dtype and shape as `values`, with the elements
        sorted along the given `axis`.

  Raises:
    ValueError: If axis is not a constant scalar, or the direction is invalid.
  "
  [values & {:keys [axis direction name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "sort" [values] {:axis axis :direction direction :name name }))

(defn space-to-batch 
  "SpaceToBatch for N-D tensors of type T.

  This operation divides \"spatial\" dimensions `[1, ..., M]` of the input into a
  grid of blocks of shape `block_shape`, and interleaves these blocks with the
  \"batch\" dimension (0) such that in the output, the spatial dimensions
  `[1, ..., M]` correspond to the position within the grid, and the batch
  dimension combines both the position within a spatial block and the original
  batch position.  Prior to division into blocks, the spatial dimensions of the
  input are optionally zero padded according to `paddings`.  See below for a
  precise description.

  Args:
    input: A `Tensor`.
      N-D with shape `input_shape = [batch] + spatial_shape + remaining_shape`,
      where spatial_shape has `M` dimensions.
    block_shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      1-D with shape `[M]`, all values must be >= 1.
    paddings: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2-D with shape `[M, 2]`, all values must be >= 0.
        `paddings[i] = [pad_start, pad_end]` specifies the padding for input dimension
        `i + 1`, which corresponds to spatial dimension `i`.  It is required that
        `block_shape[i]` divides `input_shape[i + 1] + pad_start + pad_end`.

      This operation is equivalent to the following steps:

      1. Zero-pad the start and end of dimensions `[1, ..., M]` of the
         input according to `paddings` to produce `padded` of shape `padded_shape`.

      2. Reshape `padded` to `reshaped_padded` of shape:

           [batch] +
           [padded_shape[1] / block_shape[0],
             block_shape[0],
            ...,
            padded_shape[M] / block_shape[M-1],
            block_shape[M-1]] +
           remaining_shape

      3. Permute dimensions of `reshaped_padded` to produce
         `permuted_reshaped_padded` of shape:

           block_shape +
           [batch] +
           [padded_shape[1] / block_shape[0],
            ...,
            padded_shape[M] / block_shape[M-1]] +
           remaining_shape

      4. Reshape `permuted_reshaped_padded` to flatten `block_shape` into the batch
         dimension, producing an output tensor of shape:

           [batch * prod(block_shape)] +
           [padded_shape[1] / block_shape[0],
            ...,
            padded_shape[M] / block_shape[M-1]] +
           remaining_shape

      Some examples:

      (1) For the following input of shape `[1, 2, 2, 1]`, `block_shape = [2, 2]`, and
          `paddings = [[0, 0], [0, 0]]`:

      ```
      x = [[[[1], [2]], [[3], [4]]]]
      ```

      The output tensor has shape `[4, 1, 1, 1]` and value:

      ```
      [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
      ```

      (2) For the following input of shape `[1, 2, 2, 3]`, `block_shape = [2, 2]`, and
          `paddings = [[0, 0], [0, 0]]`:

      ```
      x = [[[[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]]]]
      ```

      The output tensor has shape `[4, 1, 1, 3]` and value:

      ```
      [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
      ```

      (3) For the following input of shape `[1, 4, 4, 1]`, `block_shape = [2, 2]`, and
          `paddings = [[0, 0], [0, 0]]`:

      ```
      x = [[[[1],   [2],  [3],  [4]],
            [[5],   [6],  [7],  [8]],
            [[9],  [10], [11],  [12]],
            [[13], [14], [15],  [16]]]]
      ```

      The output tensor has shape `[4, 2, 2, 1]` and value:

      ```
      x = [[[[1], [3]], [[9], [11]]],
           [[[2], [4]], [[10], [12]]],
           [[[5], [7]], [[13], [15]]],
           [[[6], [8]], [[14], [16]]]]
      ```

      (4) For the following input of shape `[2, 2, 4, 1]`, block_shape = `[2, 2]`, and
          paddings = `[[0, 0], [2, 0]]`:

      ```
      x = [[[[1],   [2],  [3],  [4]],
            [[5],   [6],  [7],  [8]]],
           [[[9],  [10], [11],  [12]],
            [[13], [14], [15],  [16]]]]
      ```

      The output tensor has shape `[8, 1, 3, 1]` and value:

      ```
      x = [[[[0], [1], [3]]], [[[0], [9], [11]]],
           [[[0], [2], [4]]], [[[0], [10], [12]]],
           [[[0], [5], [7]]], [[[0], [13], [15]]],
           [[[0], [6], [8]]], [[[0], [14], [16]]]]
      ```

      Among others, this operation is useful for reducing atrous convolution into
      regular convolution.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input block_shape paddings name ]
  (py/call-attr v2 "space_to_batch"  input block_shape paddings name ))

(defn space-to-batch-nd 
  "SpaceToBatch for N-D tensors of type T.

  This operation divides \"spatial\" dimensions `[1, ..., M]` of the input into a
  grid of blocks of shape `block_shape`, and interleaves these blocks with the
  \"batch\" dimension (0) such that in the output, the spatial dimensions
  `[1, ..., M]` correspond to the position within the grid, and the batch
  dimension combines both the position within a spatial block and the original
  batch position.  Prior to division into blocks, the spatial dimensions of the
  input are optionally zero padded according to `paddings`.  See below for a
  precise description.

  Args:
    input: A `Tensor`.
      N-D with shape `input_shape = [batch] + spatial_shape + remaining_shape`,
      where spatial_shape has `M` dimensions.
    block_shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      1-D with shape `[M]`, all values must be >= 1.
    paddings: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2-D with shape `[M, 2]`, all values must be >= 0.
        `paddings[i] = [pad_start, pad_end]` specifies the padding for input dimension
        `i + 1`, which corresponds to spatial dimension `i`.  It is required that
        `block_shape[i]` divides `input_shape[i + 1] + pad_start + pad_end`.

      This operation is equivalent to the following steps:

      1. Zero-pad the start and end of dimensions `[1, ..., M]` of the
         input according to `paddings` to produce `padded` of shape `padded_shape`.

      2. Reshape `padded` to `reshaped_padded` of shape:

           [batch] +
           [padded_shape[1] / block_shape[0],
             block_shape[0],
            ...,
            padded_shape[M] / block_shape[M-1],
            block_shape[M-1]] +
           remaining_shape

      3. Permute dimensions of `reshaped_padded` to produce
         `permuted_reshaped_padded` of shape:

           block_shape +
           [batch] +
           [padded_shape[1] / block_shape[0],
            ...,
            padded_shape[M] / block_shape[M-1]] +
           remaining_shape

      4. Reshape `permuted_reshaped_padded` to flatten `block_shape` into the batch
         dimension, producing an output tensor of shape:

           [batch * prod(block_shape)] +
           [padded_shape[1] / block_shape[0],
            ...,
            padded_shape[M] / block_shape[M-1]] +
           remaining_shape

      Some examples:

      (1) For the following input of shape `[1, 2, 2, 1]`, `block_shape = [2, 2]`, and
          `paddings = [[0, 0], [0, 0]]`:

      ```
      x = [[[[1], [2]], [[3], [4]]]]
      ```

      The output tensor has shape `[4, 1, 1, 1]` and value:

      ```
      [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
      ```

      (2) For the following input of shape `[1, 2, 2, 3]`, `block_shape = [2, 2]`, and
          `paddings = [[0, 0], [0, 0]]`:

      ```
      x = [[[[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]]]]
      ```

      The output tensor has shape `[4, 1, 1, 3]` and value:

      ```
      [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
      ```

      (3) For the following input of shape `[1, 4, 4, 1]`, `block_shape = [2, 2]`, and
          `paddings = [[0, 0], [0, 0]]`:

      ```
      x = [[[[1],   [2],  [3],  [4]],
            [[5],   [6],  [7],  [8]],
            [[9],  [10], [11],  [12]],
            [[13], [14], [15],  [16]]]]
      ```

      The output tensor has shape `[4, 2, 2, 1]` and value:

      ```
      x = [[[[1], [3]], [[9], [11]]],
           [[[2], [4]], [[10], [12]]],
           [[[5], [7]], [[13], [15]]],
           [[[6], [8]], [[14], [16]]]]
      ```

      (4) For the following input of shape `[2, 2, 4, 1]`, block_shape = `[2, 2]`, and
          paddings = `[[0, 0], [2, 0]]`:

      ```
      x = [[[[1],   [2],  [3],  [4]],
            [[5],   [6],  [7],  [8]]],
           [[[9],  [10], [11],  [12]],
            [[13], [14], [15],  [16]]]]
      ```

      The output tensor has shape `[8, 1, 3, 1]` and value:

      ```
      x = [[[[0], [1], [3]]], [[[0], [9], [11]]],
           [[[0], [2], [4]]], [[[0], [10], [12]]],
           [[[0], [5], [7]]], [[[0], [13], [15]]],
           [[[0], [6], [8]]], [[[0], [14], [16]]]]
      ```

      Among others, this operation is useful for reducing atrous convolution into
      regular convolution.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input block_shape paddings name ]
  (py/call-attr v2 "space_to_batch_nd"  input block_shape paddings name ))

(defn split 
  "Splits a tensor into sub tensors.

  If `num_or_size_splits` is an integer, then `value` is split along dimension
  `axis` into `num_split` smaller tensors. This requires that `num_split` evenly
  divides `value.shape[axis]`.

  If `num_or_size_splits` is a 1-D Tensor (or list), we call it `size_splits`
  and `value` is split into `len(size_splits)` elements. The shape of the `i`-th
  element has the same size as the `value` except along dimension `axis` where
  the size is `size_splits[i]`.

  For example:

  ```python
  # 'value' is a tensor with shape [5, 30]
  # Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1
  split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
  tf.shape(split0)  # [5, 4]
  tf.shape(split1)  # [5, 15]
  tf.shape(split2)  # [5, 11]
  # Split 'value' into 3 tensors along dimension 1
  split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)
  tf.shape(split0)  # [5, 10]
  ```

  Args:
    value: The `Tensor` to split.
    num_or_size_splits: Either an integer indicating the number of splits along
      split_dim or a 1-D integer `Tensor` or Python list containing the sizes of
      each output tensor along split_dim. If a scalar then it must evenly divide
      `value.shape[axis]`; otherwise the sum of sizes along the split dimension
      must match that of the `value`.
    axis: An integer or scalar `int32` `Tensor`. The dimension along which to
      split. Must be in the range `[-rank(value), rank(value))`. Defaults to 0.
    num: Optional, used to specify the number of outputs when it cannot be
      inferred from the shape of `size_splits`.
    name: A name for the operation (optional).

  Returns:
    if `num_or_size_splits` is a scalar returns `num_or_size_splits` `Tensor`
    objects; if `num_or_size_splits` is a 1-D Tensor returns
    `num_or_size_splits.get_shape[0]` `Tensor` objects resulting from splitting
    `value`.

  Raises:
    ValueError: If `num` is unspecified and cannot be inferred.
  "
  [value num_or_size_splits & {:keys [axis num name]
                       :or {num None}} ]
    (py/call-attr-kw v2 "split" [value num_or_size_splits] {:axis axis :num num :name name }))

(defn sqrt 
  "Computes square root of x element-wise.

  I.e., \\(y = \sqrt{x} = x^{1/2}\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.

    If `x` is a `SparseTensor`, returns
    `SparseTensor(x.indices, tf.math.sqrt(x.values, ...), x.dense_shape)`"
  [ x name ]
  (py/call-attr v2 "sqrt"  x name ))

(defn square 
  "Computes square of x element-wise.

  I.e., \\(y = x * x = x^2\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.

    If `x` is a `SparseTensor`, returns
    `SparseTensor(x.indices, tf.math.square(x.values, ...), x.dense_shape)`"
  [ x name ]
  (py/call-attr v2 "square"  x name ))

(defn squeeze 
  "Removes dimensions of size 1 from the shape of a tensor.

  Given a tensor `input`, this operation returns a tensor of the same type with
  all dimensions of size 1 removed. If you don't want to remove all size 1
  dimensions, you can remove specific size 1 dimensions by specifying
  `axis`.

  For example:

  ```python
  # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
  tf.shape(tf.squeeze(t))  # [2, 3]
  ```

  Or, to remove specific size 1 dimensions:

  ```python
  # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
  tf.shape(tf.squeeze(t, [2, 4]))  # [1, 2, 3, 1]
  ```

  Unlike the older op `tf.compat.v1.squeeze`, this op does not accept a
  deprecated `squeeze_dims` argument.

  Note: if `input` is a `tf.RaggedTensor`, then this operation takes `O(N)`
  time, where `N` is the number of elements in the squeezed dimensions.

  Args:
    input: A `Tensor`. The `input` to squeeze.
    axis: An optional list of `ints`. Defaults to `[]`. If specified, only
      squeezes the dimensions listed. The dimension index starts at 0. It is an
      error to squeeze a dimension that is not 1. Must be in the range
      `[-rank(input), rank(input))`. Must be specified if `input` is a
      `RaggedTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    Contains the same data as `input`, but has one or more dimensions of
    size 1 removed.

  Raises:
    ValueError: The input cannot be converted to a tensor, or the specified
      axis cannot be squeezed.
  "
  [ input axis name ]
  (py/call-attr v2 "squeeze"  input axis name ))
(defn stack 
  "Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor.

  Packs the list of tensors in `values` into a tensor with rank one higher than
  each tensor in `values`, by packing them along the `axis` dimension.
  Given a list of length `N` of tensors of shape `(A, B, C)`;

  if `axis == 0` then the `output` tensor will have the shape `(N, A, B, C)`.
  if `axis == 1` then the `output` tensor will have the shape `(A, N, B, C)`.
  Etc.

  For example:

  ```python
  x = tf.constant([1, 4])
  y = tf.constant([2, 5])
  z = tf.constant([3, 6])
  tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
  tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
  ```

  This is the opposite of unstack.  The numpy equivalent is

  ```python
  tf.stack([x, y, z]) = np.stack([x, y, z])
  ```

  Args:
    values: A list of `Tensor` objects with the same shape and type.
    axis: An `int`. The axis to stack along. Defaults to the first dimension.
      Negative values wrap around, so the valid range is `[-(R+1), R+1)`.
    name: A name for this operation (optional).

  Returns:
    output: A stacked `Tensor` with the same type as `values`.

  Raises:
    ValueError: If `axis` is out of the range [-(R+1), R+1).
  "
  [values  & {:keys [axis name]} ]
    (py/call-attr-kw v2 "stack" [values] {:axis axis :name name }))

(defn stop-gradient 
  "Stops gradient computation.

  When executed in a graph, this op outputs its input tensor as-is.

  When building ops to compute gradients, this op prevents the contribution of
  its inputs to be taken into account.  Normally, the gradient generator adds ops
  to a graph to compute the derivatives of a specified 'loss' by recursively
  finding out inputs that contributed to its computation.  If you insert this op
  in the graph it inputs are masked from the gradient generator.  They are not
  taken into account for computing gradients.

  This is useful any time you want to compute a value with TensorFlow but need
  to pretend that the value was a constant. Some examples include:

  *  The *EM* algorithm where the *M-step* should not involve backpropagation
     through the output of the *E-step*.
  *  Contrastive divergence training of Boltzmann machines where, when
     differentiating the energy function, the training must not backpropagate
     through the graph that generated the samples from the model.
  *  Adversarial training, where no backprop should happen through the adversarial
     example generation process.

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input name ]
  (py/call-attr v2 "stop_gradient"  input name ))

(defn strided-slice 
  "Extracts a strided slice of a tensor (generalized python array indexing).

  **Instead of calling this op directly most users will want to use the
  NumPy-style slicing syntax (e.g. `tensor[..., 3:4:-1, tf.newaxis, 3]`), which
  is supported via `tf.Tensor.__getitem__` and `tf.Variable.__getitem__`.**
  The interface of this op is a low-level encoding of the slicing syntax.

  Roughly speaking, this op extracts a slice of size `(end-begin)/stride`
  from the given `input_` tensor. Starting at the location specified by `begin`
  the slice continues by adding `stride` to the index until all dimensions are
  not less than `end`.
  Note that a stride can be negative, which causes a reverse slice.

  Given a Python slice `input[spec0, spec1, ..., specn]`,
  this function will be called as follows.

  `begin`, `end`, and `strides` will be vectors of length n.
  n in general is not equal to the rank of the `input_` tensor.

  In each mask field (`begin_mask`, `end_mask`, `ellipsis_mask`,
  `new_axis_mask`, `shrink_axis_mask`) the ith bit will correspond to
  the ith spec.

  If the ith bit of `begin_mask` is set, `begin[i]` is ignored and
  the fullest possible range in that dimension is used instead.
  `end_mask` works analogously, except with the end range.

  `foo[5:,:,:3]` on a 7x8x9 tensor is equivalent to `foo[5:7,0:8,0:3]`.
  `foo[::-1]` reverses a tensor with shape 8.

  If the ith bit of `ellipsis_mask` is set, as many unspecified dimensions
  as needed will be inserted between other dimensions. Only one
  non-zero bit is allowed in `ellipsis_mask`.

  For example `foo[3:5,...,4:5]` on a shape 10x3x3x10 tensor is
  equivalent to `foo[3:5,:,:,4:5]` and
  `foo[3:5,...]` is equivalent to `foo[3:5,:,:,:]`.

  If the ith bit of `new_axis_mask` is set, then `begin`,
  `end`, and `stride` are ignored and a new length 1 dimension is
  added at this point in the output tensor.

  For example,
  `foo[:4, tf.newaxis, :2]` would produce a shape `(4, 1, 2)` tensor.

  If the ith bit of `shrink_axis_mask` is set, it implies that the ith
  specification shrinks the dimensionality by 1, taking on the value at index
  `begin[i]`. `end[i]` and `strides[i]` are ignored in this case. For example in
  Python one might do `foo[:, 3, :]` which would result in `shrink_axis_mask`
  equal to 2.


  NOTE: `begin` and `end` are zero-indexed.
  `strides` entries must be non-zero.


  ```python
  t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                   [[3, 3, 3], [4, 4, 4]],
                   [[5, 5, 5], [6, 6, 6]]])
  tf.strided_slice(t, [1, 0, 0], [2, 1, 3], [1, 1, 1])  # [[[3, 3, 3]]]
  tf.strided_slice(t, [1, 0, 0], [2, 2, 3], [1, 1, 1])  # [[[3, 3, 3],
                                                        #   [4, 4, 4]]]
  tf.strided_slice(t, [1, -1, 0], [2, -3, 3], [1, -1, 1])  # [[[4, 4, 4],
                                                           #   [3, 3, 3]]]
  ```

  Args:
    input_: A `Tensor`.
    begin: An `int32` or `int64` `Tensor`.
    end: An `int32` or `int64` `Tensor`.
    strides: An `int32` or `int64` `Tensor`.
    begin_mask: An `int32` mask.
    end_mask: An `int32` mask.
    ellipsis_mask: An `int32` mask.
    new_axis_mask: An `int32` mask.
    shrink_axis_mask: An `int32` mask.
    var: The variable corresponding to `input_` or None
    name: A name for the operation (optional).

  Returns:
    A `Tensor` the same type as `input`.
  "
  [input_ begin end strides & {:keys [begin_mask end_mask ellipsis_mask new_axis_mask shrink_axis_mask var name]
                       :or {var None name None}} ]
    (py/call-attr-kw v2 "strided_slice" [input_ begin end strides] {:begin_mask begin_mask :end_mask end_mask :ellipsis_mask ellipsis_mask :new_axis_mask new_axis_mask :shrink_axis_mask shrink_axis_mask :var var :name name }))

(defn subtract 
  "Returns x - y element-wise.

  *NOTE*: `Subtract` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr v2 "subtract"  x y name ))
(defn switch-case 
  "Create a switch/case operation, i.e. an integer-indexed conditional.

  See also `tf.case`.

  This op can be substantially more efficient than `tf.case` when exactly one
  branch will be selected. `tf.switch_case` is more like a C++ switch/case
  statement than `tf.case`, which is more like an if/elif/elif/else chain.

  The `branch_fns` parameter is either a list
  of (int, callable) pairs, or simply a list of callables (in which case the
  index is implicitly the key). The `branch_index` `Tensor` is used to select an
  element in `branch_fns` with matching `int` key, falling back to `default`
  if none match, or `max(keys)` if no `default` is provided. The keys must form
  a contiguous set from `0` to `len(branch_fns) - 1`.

  `tf.switch_case` supports nested structures as implemented in `tf.nest`. All
  callables must return the same (possibly nested) value structure of lists,
  tuples, and/or named tuples.

  @compatibility(v2)
  `branch_fns` could be a dictionary in v1. However, tf.Tensor and
  tf.Variable are no longer hashable in v2, so cannot be used as a key for a
  dictionary.  Please use a list or a tuple instead.
  @end_compatibility

  **Example:**

  Pseudocode:

  ```c++
  switch (branch_index) {  // c-style switch
    case 0: return 17;
    case 1: return 31;
    default: return -1;
  }
  ```
  or
  ```python
  branches = {0: lambda: 17, 1: lambda: 31}
  branches.get(branch_index, lambda: -1)()
  ```

  Expressions:

  ```python
  def f1(): return tf.constant(17)
  def f2(): return tf.constant(31)
  def f3(): return tf.constant(-1)
  r = tf.switch_case(branch_index, branch_fns={0: f1, 1: f2}, default=f3)
  # Equivalent: tf.switch_case(branch_index, branch_fns={0: f1, 1: f2, 2: f3})
  ```

  Args:
    branch_index: An int Tensor specifying which of `branch_fns` should be
      executed.
    branch_fns: A `list` of (int, callable) pairs, or simply a list of
    callables (in which case the index serves as the key). Each callable must
    return a matching structure of tensors.
    default: Optional callable that returns a structure of tensors.
    name: A name for this operation (optional).

  Returns:
    The tensors returned by the callable identified by `branch_index`, or those
    returned by `default` if no key matches and `default` was provided, or those
    returned by the max-keyed `branch_fn` if no `default` is provided.

  Raises:
    TypeError: If `branch_fns` is not a list/dictionary.
    TypeError: If `branch_fns` is a list but does not contain 2-tuples or
               callables.
    TypeError: If `fns[i]` is not callable for any i, or `default` is not
               callable.
  "
  [branch_index branch_fns default  & {:keys [name]} ]
    (py/call-attr-kw v2 "switch_case" [branch_index branch_fns default] {:name name }))

(defn tan 
  "Computes tan of x element-wise.

    Given an input tensor, this function computes tangent of every
    element in the tensor. Input range is `(-inf, inf)` and
    output range is `(-inf, inf)`. If input lies outside the boundary, `nan`
    is returned.

    ```python
    x = tf.constant([-float(\"inf\"), -9, -0.5, 1, 1.2, 200, 10000, float(\"inf\")])
    tf.math.tan(x) ==> [nan 0.45231566 -0.5463025 1.5574077 2.572152 -1.7925274 0.32097113 nan]
    ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr v2 "tan"  x name ))

(defn tanh 
  "Computes hyperbolic tangent of `x` element-wise.

    Given an input tensor, this function computes hyperbolic tangent of every
    element in the tensor. Input range is `[-inf, inf]` and
    output range is `[-1,1]`.

    ```python
    x = tf.constant([-float(\"inf\"), -5, -0.5, 1, 1.2, 2, 3, float(\"inf\")])
    tf.math.tanh(x) ==> [-1. -0.99990916 -0.46211717 0.7615942 0.8336547 0.9640276 0.9950547 1.]
    ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.

    If `x` is a `SparseTensor`, returns
    `SparseTensor(x.indices, tf.math.tanh(x.values, ...), x.dense_shape)`"
  [ x name ]
  (py/call-attr v2 "tanh"  x name ))

(defn tensor-scatter-nd-add 
  "Adds sparse `updates` to an existing tensor according to `indices`.

  This operation creates a new tensor by adding sparse `updates` to the passed
  in `tensor`.
  This operation is very similar to `tf.scatter_nd_add`, except that the updates
  are added onto an existing tensor (as opposed to a variable). If the memory
  for the existing tensor cannot be re-used, a copy is made and updated.

  `indices` is an integer tensor containing indices into a new tensor of shape
  `shape`.  The last dimension of `indices` can be at most the rank of `shape`:

      indices.shape[-1] <= shape.rank

  The last dimension of `indices` corresponds to indices into elements
  (if `indices.shape[-1] = shape.rank`) or slices
  (if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
  `shape`.  `updates` is a tensor with shape

      indices.shape[:-1] + shape[indices.shape[-1]:]

  The simplest form of tensor_scatter_add is to add individual elements to a
  tensor by index. For example, say we want to add 4 elements in a rank-1
  tensor with 8 elements.

  In Python, this scatter add operation would look like this:

  ```python
      indices = tf.constant([[4], [3], [1], [7]])
      updates = tf.constant([9, 10, 11, 12])
      tensor = tf.ones([8], dtype=tf.int32)
      updated = tf.tensor_scatter_add(tensor, indices, updates)
      with tf.Session() as sess:
        print(sess.run(scatter))
  ```

  The resulting tensor would look like this:

      [1, 12, 1, 11, 10, 1, 1, 13]

  We can also, insert entire slices of a higher rank tensor all at once. For
  example, if we wanted to insert two slices in the first dimension of a
  rank-3 tensor with two matrices of new values.

  In Python, this scatter add operation would look like this:

  ```python
      indices = tf.constant([[0], [2]])
      updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                              [7, 7, 7, 7], [8, 8, 8, 8]],
                             [[5, 5, 5, 5], [6, 6, 6, 6],
                              [7, 7, 7, 7], [8, 8, 8, 8]]])
      tensor = tf.ones([4, 4, 4])
      updated = tf.tensor_scatter_add(tensor, indices, updates)
      with tf.Session() as sess:
        print(sess.run(scatter))
  ```

  The resulting tensor would look like this:

      [[[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
       [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
       [[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
       [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]

  Note that on CPU, if an out of bound index is found, an error is returned.
  On GPU, if an out of bound index is found, the index is ignored.

  Args:
    tensor: A `Tensor`. Tensor to copy/update.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Index tensor.
    updates: A `Tensor`. Must have the same type as `tensor`.
      Updates to scatter into output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  "
  [ tensor indices updates name ]
  (py/call-attr v2 "tensor_scatter_nd_add"  tensor indices updates name ))

(defn tensor-scatter-nd-sub 
  "Subtracts sparse `updates` from an existing tensor according to `indices`.

  This operation creates a new tensor by subtracting sparse `updates` from the
  passed in `tensor`.
  This operation is very similar to `tf.scatter_nd_sub`, except that the updates
  are subtracted from an existing tensor (as opposed to a variable). If the memory
  for the existing tensor cannot be re-used, a copy is made and updated.

  `indices` is an integer tensor containing indices into a new tensor of shape
  `shape`.  The last dimension of `indices` can be at most the rank of `shape`:

      indices.shape[-1] <= shape.rank

  The last dimension of `indices` corresponds to indices into elements
  (if `indices.shape[-1] = shape.rank`) or slices
  (if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
  `shape`.  `updates` is a tensor with shape

      indices.shape[:-1] + shape[indices.shape[-1]:]

  The simplest form of tensor_scatter_sub is to subtract individual elements
  from a tensor by index. For example, say we want to insert 4 scattered elements
  in a rank-1 tensor with 8 elements.

  In Python, this scatter subtract operation would look like this:

  ```python
      indices = tf.constant([[4], [3], [1], [7]])
      updates = tf.constant([9, 10, 11, 12])
      tensor = tf.ones([8], dtype=tf.int32)
      updated = tf.tensor_scatter_sub(tensor, indices, updates)
      with tf.Session() as sess:
        print(sess.run(scatter))
  ```

  The resulting tensor would look like this:

      [1, -10, 1, -9, -8, 1, 1, -11]

  We can also, insert entire slices of a higher rank tensor all at once. For
  example, if we wanted to insert two slices in the first dimension of a
  rank-3 tensor with two matrices of new values.

  In Python, this scatter add operation would look like this:

  ```python
      indices = tf.constant([[0], [2]])
      updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                              [7, 7, 7, 7], [8, 8, 8, 8]],
                             [[5, 5, 5, 5], [6, 6, 6, 6],
                              [7, 7, 7, 7], [8, 8, 8, 8]]])
      tensor = tf.ones([4, 4, 4])
      updated = tf.tensor_scatter_sub(tensor, indices, updates)
      with tf.Session() as sess:
        print(sess.run(scatter))
  ```

  The resulting tensor would look like this:

      [[[-4, -4, -4, -4], [-5, -5, -5, -5], [-6, -6, -6, -6], [-7, -7, -7, -7]],
       [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
       [[-4, -4, -4, -4], [-5, -5, -5, -5], [-6, -6, -6, -6], [-7, -7, -7, -7]],
       [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]

  Note that on CPU, if an out of bound index is found, an error is returned.
  On GPU, if an out of bound index is found, the index is ignored.

  Args:
    tensor: A `Tensor`. Tensor to copy/update.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Index tensor.
    updates: A `Tensor`. Must have the same type as `tensor`.
      Updates to scatter into output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  "
  [ tensor indices updates name ]
  (py/call-attr v2 "tensor_scatter_nd_sub"  tensor indices updates name ))

(defn tensor-scatter-nd-update 
  "Scatter `updates` into an existing tensor according to `indices`.

  This operation creates a new tensor by applying sparse `updates` to the passed
  in `tensor`.
  This operation is very similar to `tf.scatter_nd`, except that the updates are
  scattered onto an existing tensor (as opposed to a zero-tensor). If the memory
  for the existing tensor cannot be re-used, a copy is made and updated.

  If `indices` contains duplicates, then their updates are accumulated (summed).

  **WARNING**: The order in which updates are applied is nondeterministic, so the
  output will be nondeterministic if `indices` contains duplicates -- because
  of some numerical approximation issues, numbers summed in different order
  may yield different results.

  `indices` is an integer tensor containing indices into a new tensor of shape
  `shape`.  The last dimension of `indices` can be at most the rank of `shape`:

      indices.shape[-1] <= shape.rank

  The last dimension of `indices` corresponds to indices into elements
  (if `indices.shape[-1] = shape.rank`) or slices
  (if `indices.shape[-1] < shape.rank`) along dimension `indices.shape[-1]` of
  `shape`.  `updates` is a tensor with shape

      indices.shape[:-1] + shape[indices.shape[-1]:]

  The simplest form of scatter is to insert individual elements in a tensor by
  index. For example, say we want to insert 4 scattered elements in a rank-1
  tensor with 8 elements.

  <div style=\"width:70%; margin:auto; margin-bottom:10px; margin-top:20px;\">
  <img style=\"width:100%\" src=\"https://www.tensorflow.org/images/ScatterNd1.png\" alt>
  </div>

  In Python, this scatter operation would look like this:

  ```python
      indices = tf.constant([[4], [3], [1], [7]])
      updates = tf.constant([9, 10, 11, 12])
      tensor = tf.ones([8], dtype=tf.int32)
      updated = tf.tensor_scatter_update(tensor, indices, updates)
      with tf.Session() as sess:
        print(sess.run(scatter))
  ```

  The resulting tensor would look like this:

      [1, 11, 1, 10, 9, 1, 1, 12]

  We can also, insert entire slices of a higher rank tensor all at once. For
  example, if we wanted to insert two slices in the first dimension of a
  rank-3 tensor with two matrices of new values.

  In Python, this scatter operation would look like this:

  ```python
      indices = tf.constant([[0], [2]])
      updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                              [7, 7, 7, 7], [8, 8, 8, 8]],
                             [[5, 5, 5, 5], [6, 6, 6, 6],
                              [7, 7, 7, 7], [8, 8, 8, 8]]])
      tensor = tf.ones([4, 4, 4])
      updated = tf.tensor_scatter_update(tensor, indices, updates)
      with tf.Session() as sess:
        print(sess.run(scatter))
  ```

  The resulting tensor would look like this:

      [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
       [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
       [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
       [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]

  Note that on CPU, if an out of bound index is found, an error is returned.
  On GPU, if an out of bound index is found, the index is ignored.

  Args:
    tensor: A `Tensor`. Tensor to copy/update.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Index tensor.
    updates: A `Tensor`. Must have the same type as `tensor`.
      Updates to scatter into output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  "
  [ tensor indices updates name ]
  (py/call-attr v2 "tensor_scatter_nd_update"  tensor indices updates name ))

(defn tensordot 
  "Tensor contraction of a and b along specified axes and outer product.

  Tensordot (also known as tensor contraction) sums the product of elements
  from `a` and `b` over the indices specified by `a_axes` and `b_axes`.
  The lists `a_axes` and `b_axes` specify those pairs of axes along which to
  contract the tensors. The axis `a_axes[i]` of `a` must have the same dimension
  as axis `b_axes[i]` of `b` for all `i` in `range(0, len(a_axes))`. The lists
  `a_axes` and `b_axes` must have identical length and consist of unique
  integers that specify valid axes for each of the tensors. Additionally
  outer product is supported by passing `axes=0`.

  This operation corresponds to `numpy.tensordot(a, b, axes)`.

  Example 1: When `a` and `b` are matrices (order 2), the case `axes = 1`
  is equivalent to matrix multiplication.

  Example 2: When `a` and `b` are matrices (order 2), the case
  `axes = [[1], [0]]` is equivalent to matrix multiplication.

  Example 3: When `a` and `b` are matrices (order 2), the case `axes=0` gives
  the outer product, a tensor of order 4.

  Example 4: Suppose that \\(a_{ijk}\\) and \\(b_{lmn}\\) represent two
  tensors of order 3. Then, `contract(a, b, [[0], [2]])` is the order 4 tensor
  \\(c_{jklm}\\) whose entry
  corresponding to the indices \\((j,k,l,m)\\) is given by:

  \\( c_{jklm} = \sum_i a_{ijk} b_{lmi} \\).

  In general, `order(c) = order(a) + order(b) - 2*len(axes[0])`.

  Args:
    a: `Tensor` of type `float32` or `float64`.
    b: `Tensor` with the same type as `a`.
    axes: Either a scalar `N`, or a list or an `int32` `Tensor` of shape [2, k].
      If axes is a scalar, sum over the last N axes of a and the first N axes of
      b in order. If axes is a list or `Tensor` the first and second row contain
      the set of unique integers specifying axes along which the contraction is
      computed, for `a` and `b`, respectively. The number of axes for `a` and
      `b` must be equal. If `axes=0`, computes the outer product between `a` and
      `b`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type as `a`.

  Raises:
    ValueError: If the shapes of `a`, `b`, and `axes` are incompatible.
    IndexError: If the values in axes exceed the rank of the corresponding
      tensor.
  "
  [ a b axes name ]
  (py/call-attr v2 "tensordot"  a b axes name ))

(defn tile 
  "Constructs a tensor by tiling a given tensor.

  This operation creates a new tensor by replicating `input` `multiples` times.
  The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
  and the values of `input` are replicated `multiples[i]` times along the 'i'th
  dimension. For example, tiling `[a b c d]` by `[2]` produces
  `[a b c d a b c d]`.

  Args:
    input: A `Tensor`. 1-D or higher.
    multiples: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      1-D. Length must be the same as the number of dimensions in `input`
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input multiples name ]
  (py/call-attr v2 "tile"  input multiples name ))

(defn timestamp 
  "Provides the time since epoch in seconds.

  Returns the timestamp as a `float64` for seconds since the Unix epoch.

  Note: the timestamp is computed when the op is executed, not when it is added
  to the graph.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float64`.
  "
  [ name ]
  (py/call-attr v2 "timestamp"  name ))
(defn transpose 
  "Transposes `a`.

  Permutes the dimensions according to `perm`.

  The returned tensor's dimension i will correspond to the input dimension
  `perm[i]`. If `perm` is not given, it is set to (n-1...0), where n is
  the rank of the input tensor. Hence by default, this operation performs a
  regular matrix transpose on 2-D input Tensors. If conjugate is True and
  `a.dtype` is either `complex64` or `complex128` then the values of `a`
  are conjugated and transposed.

  @compatibility(numpy)
  In `numpy` transposes are memory-efficient constant time operations as they
  simply return a new view of the same data with adjusted `strides`.

  TensorFlow does not support strides, so `transpose` returns a new tensor with
  the items permuted.
  @end_compatibility

  For example:

  ```python
  x = tf.constant([[1, 2, 3], [4, 5, 6]])
  tf.transpose(x)  # [[1, 4]
                   #  [2, 5]
                   #  [3, 6]]

  # Equivalently
  tf.transpose(x, perm=[1, 0])  # [[1, 4]
                                #  [2, 5]
                                #  [3, 6]]

  # If x is complex, setting conjugate=True gives the conjugate transpose
  x = tf.constant([[1 + 1j, 2 + 2j, 3 + 3j],
                   [4 + 4j, 5 + 5j, 6 + 6j]])
  tf.transpose(x, conjugate=True)  # [[1 - 1j, 4 - 4j],
                                   #  [2 - 2j, 5 - 5j],
                                   #  [3 - 3j, 6 - 6j]]

  # 'perm' is more useful for n-dimensional tensors, for n > 2
  x = tf.constant([[[ 1,  2,  3],
                    [ 4,  5,  6]],
                   [[ 7,  8,  9],
                    [10, 11, 12]]])

  # Take the transpose of the matrices in dimension-0
  # (this common operation has a shorthand `linalg.matrix_transpose`)
  tf.transpose(x, perm=[0, 2, 1])  # [[[1,  4],
                                   #   [2,  5],
                                   #   [3,  6]],
                                   #  [[7, 10],
                                   #   [8, 11],
                                   #   [9, 12]]]
  ```

  Args:
    a: A `Tensor`.
    perm: A permutation of the dimensions of `a`.
    conjugate: Optional bool. Setting it to `True` is mathematically equivalent
      to tf.math.conj(tf.transpose(input)).
    name: A name for the operation (optional).

  Returns:
    A transposed `Tensor`.
  "
  [a perm  & {:keys [conjugate name]} ]
    (py/call-attr-kw v2 "transpose" [a perm] {:conjugate conjugate :name name }))

(defn truediv 
  "Divides x / y elementwise (using Python 3 division operator semantics).

  NOTE: Prefer using the Tensor operator or tf.divide which obey Python
  division operator semantics.

  This function forces Python 3 division operator semantics where all integer
  arguments are cast to floating types first.   This op is generated by normal
  `x / y` division in Python 3 and in Python 2.7 with
  `from __future__ import division`.  If you want integer division that rounds
  down, use `x // y` or `tf.math.floordiv`.

  `x` and `y` must have the same numeric type.  If the inputs are floating
  point, the output will have the same type.  If the inputs are integral, the
  inputs are cast to `float32` for `int8` and `int16` and `float64` for `int32`
  and `int64` (matching the behavior of Numpy).

  Args:
    x: `Tensor` numerator of numeric type.
    y: `Tensor` denominator of numeric type.
    name: A name for the operation (optional).

  Returns:
    `x / y` evaluated in floating point.

  Raises:
    TypeError: If `x` and `y` have different dtypes.
  "
  [ x y name ]
  (py/call-attr v2 "truediv"  x y name ))

(defn truncatediv 
  "Returns x / y element-wise for integer types.

  Truncation designates that negative numbers will round fractional quantities
  toward zero. I.e. -7 / 5 = -1. This matches C semantics but it is different
  than Python semantics. See `FloorDiv` for a division function that matches
  Python Semantics.

  *NOTE*: `truncatediv` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr v2 "truncatediv"  x y name ))

(defn truncatemod 
  "Returns element-wise remainder of division. This emulates C semantics in that

  the result here is consistent with a truncating divide. E.g. `truncate(x / y) *
  y + truncate_mod(x, y) = x`.

  *NOTE*: `truncatemod` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`, `bfloat16`, `half`, `float32`, `float64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr v2 "truncatemod"  x y name ))

(defn tuple 
  "Group tensors together.

  This creates a tuple of tensors with the same values as the `tensors`
  argument, except that the value of each tensor is only returned after the
  values of all tensors have been computed.

  `control_inputs` contains additional ops that have to finish before this op
  finishes, but whose outputs are not returned.

  This can be used as a \"join\" mechanism for parallel computations: all the
  argument tensors can be computed in parallel, but the values of any tensor
  returned by `tuple` are only available after all the parallel computations
  are done.

  See also `tf.group` and
  `tf.control_dependencies`.

  Args:
    tensors: A list of `Tensor`s or `IndexedSlices`, some entries can be `None`.
    control_inputs: List of additional ops to finish before returning.
    name: (optional) A name to use as a `name_scope` for the operation.

  Returns:
    Same as `tensors`.

  Raises:
    ValueError: If `tensors` does not contain any `Tensor` or `IndexedSlices`.
    TypeError: If `control_inputs` is not a list of `Operation` or `Tensor`
      objects.

  "
  [ tensors control_inputs name ]
  (py/call-attr v2 "tuple"  tensors control_inputs name ))

(defn unique 
  "Finds unique elements in a 1-D tensor.

  This operation returns a tensor `y` containing all of the unique elements of `x`
  sorted in the same order that they occur in `x`. This operation also returns a
  tensor `idx` the same size as `x` that contains the index of each value of `x`
  in the unique output `y`. In other words:

  `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

  For example:

  ```
  # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
  y, idx = unique(x)
  y ==> [1, 2, 4, 7, 8]
  idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
  ```

  Args:
    x: A `Tensor`. 1-D.
    out_idx: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, idx).

    y: A `Tensor`. Has the same type as `x`.
    idx: A `Tensor` of type `out_idx`.
  "
  [x & {:keys [out_idx name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "unique" [x] {:out_idx out_idx :name name }))

(defn unique-with-counts 
  "Finds unique elements in a 1-D tensor.

  This operation returns a tensor `y` containing all of the unique elements of `x`
  sorted in the same order that they occur in `x`. This operation also returns a
  tensor `idx` the same size as `x` that contains the index of each value of `x`
  in the unique output `y`. Finally, it returns a third tensor `count` that
  contains the count of each element of `y` in `x`. In other words:

  `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

  For example:

  ```
  # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
  y, idx, count = unique_with_counts(x)
  y ==> [1, 2, 4, 7, 8]
  idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
  count ==> [2, 1, 3, 1, 2]
  ```

  Args:
    x: A `Tensor`. 1-D.
    out_idx: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, idx, count).

    y: A `Tensor`. Has the same type as `x`.
    idx: A `Tensor` of type `out_idx`.
    count: A `Tensor` of type `out_idx`.
  "
  [x & {:keys [out_idx name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "unique_with_counts" [x] {:out_idx out_idx :name name }))

(defn unravel-index 
  "Converts an array of flat indices into a tuple of coordinate arrays.

  
  Example:

  ```
  y = tf.unravel_index(indices=[2, 5, 7], dims=[3, 3])
  # 'dims' represent a hypothetical (3, 3) tensor of indices:
  # [[0, 1, *2*],
  #  [3, 4, *5*],
  #  [6, *7*, 8]]
  # For each entry from 'indices', this operation returns
  # its coordinates (marked with '*'), such as
  # 2 ==> (0, 2)
  # 5 ==> (1, 2)
  # 7 ==> (2, 1)
  y ==> [[0, 1, 2], [2, 2, 1]]
  ```

  @compatibility(numpy)
  Equivalent to np.unravel_index
  @end_compatibility

  Args:
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      An 0-D or 1-D `int` Tensor whose elements are indices into the
      flattened version of an array of dimensions dims.
    dims: A `Tensor`. Must have the same type as `indices`.
      An 1-D `int` Tensor. The shape of the array to use for unraveling
      indices.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `indices`.
  "
  [ indices dims name ]
  (py/call-attr v2 "unravel_index"  indices dims name ))
(defn unstack 
  "Unpacks the given dimension of a rank-`R` tensor into rank-`(R-1)` tensors.

  Unpacks `num` tensors from `value` by chipping it along the `axis` dimension.
  If `num` is not specified (the default), it is inferred from `value`'s shape.
  If `value.shape[axis]` is not known, `ValueError` is raised.

  For example, given a tensor of shape `(A, B, C, D)`;

  If `axis == 0` then the i'th tensor in `output` is the slice
    `value[i, :, :, :]` and each tensor in `output` will have shape `(B, C, D)`.
    (Note that the dimension unpacked along is gone, unlike `split`).

  If `axis == 1` then the i'th tensor in `output` is the slice
    `value[:, i, :, :]` and each tensor in `output` will have shape `(A, C, D)`.
  Etc.

  This is the opposite of stack.

  Args:
    value: A rank `R > 0` `Tensor` to be unstacked.
    num: An `int`. The length of the dimension `axis`. Automatically inferred if
      `None` (the default).
    axis: An `int`. The axis to unstack along. Defaults to the first dimension.
      Negative values wrap around, so the valid range is `[-R, R)`.
    name: A name for the operation (optional).

  Returns:
    The list of `Tensor` objects unstacked from `value`.

  Raises:
    ValueError: If `num` is unspecified and cannot be inferred.
    ValueError: If `axis` is out of the range [-R, R).
  "
  [value num  & {:keys [axis name]} ]
    (py/call-attr-kw v2 "unstack" [value num] {:axis axis :name name }))

(defn variable-creator-scope 
  "Scope which defines a variable creation function to be used by variable().

  variable_creator is expected to be a function with the following signature:

  ```
    def variable_creator(next_creator, **kwargs)
  ```

  The creator is supposed to eventually call the next_creator to create a
  variable if it does want to create a variable and not call Variable or
  ResourceVariable directly. This helps make creators composable. A creator may
  choose to create multiple variables, return already existing variables, or
  simply register that a variable was created and defer to the next creators in
  line. Creators can also modify the keyword arguments seen by the next
  creators.

  Custom getters in the variable scope will eventually resolve down to these
  custom creators when they do create variables.

  The valid keyword arguments in kwds are:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called. In
        that case, `dtype` must be specified. (Note that initializer functions
        from init_ops.py must first be bound to a shape before being used here.)
      trainable: If `True`, the default, GradientTapes automatically watch
        uses of this Variable.
      validate_shape: If `False`, allows the variable to be initialized with a
        value of unknown shape. If `True`, the default, the shape of
        `initial_value` must be known.
      caching_device: Optional device string describing where the Variable
        should be cached for reading.  Defaults to the Variable's device.
        If not `None`, caches on another device.  Typical use is to cache
        on the device where the Ops using the Variable reside, to deduplicate
        copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      dtype: If set, initial_value will be converted to the given type.
        If `None`, either the datatype will be kept (if `initial_value` is
        a Tensor), or `convert_to_tensor` will decide.
      constraint: A constraint function to be applied to the variable after
        updates by some algorithms.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses
        when to synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.

  This set may grow over time, so it's important the signature of creators is as
  mentioned above.

  Args:
    variable_creator: the passed creator

  Yields:
    A scope in which the creator is active
  "
  [ variable_creator ]
  (py/call-attr v2 "variable_creator_scope"  variable_creator ))

(defn vectorized-map 
  "Parallel map on the list of tensors unpacked from `elems` on dimension 0.


  This method works similar to tf.map_fn but is optimized to run much faster,
  possibly with a much larger memory footprint. The speedups are obtained by
  vectorization (see https://arxiv.org/pdf/1903.04243.pdf). The idea behind
  vectorization is to semantically launch all the invocations of `fn` in
  parallel and fuse corresponding operations across all these invocations. This
  fusion is done statically at graph generation time and the generated code is
  often similar in performance to a manually fused version.

  Because `tf.vectorized_map` fully parallelizes the batch, this method will
  generally be significantly faster than using `tf.map_fn`, especially in eager
  mode. However this is an experimental feature and currently has a lot of
  limitations:
    - There should be no data dependency between the different semantic
      invocations of `fn`, i.e. it should be safe to map the elements of the
      inputs in any order.
    - Stateful kernels may mostly not be supported since these often imply a
      data dependency. We do support a limited set of such stateful kernels
      though (like RandomFoo, Variable operations like reads, etc).
    - `fn` has limited support for control flow operations. `tf.cond` in
      particular is not supported.
    - `fn` should return nested structure of Tensors or Operations. However
      if an Operation is returned, it should have zero outputs.
    - The shape and dtype of any intermediate or output tensors in the
      computation of `fn` should not depend on the input to `fn`.

  Args:
    fn: The callable to be performed. It accepts one argument, which will have
      the same (possibly nested) structure as `elems`, and returns a possibly
      nested structure of Tensors and Operations, which may be different than
      the structure of `elems`.
    elems: A tensor or (possibly nested) sequence of tensors, each of which will
      be unpacked along their first dimension. The nested sequence of the
      resulting slices will be mapped over by `fn`.

  Returns:
    A tensor or (possibly nested) sequence of tensors. Each tensor packs the
    results of applying fn to tensors unpacked from elems along the first
    dimension, from first to last.

  Examples:
  ```python
  def outer_product(a):
    return tf.tensordot(a, a, 0)

  batch_size = 100
  a = tf.ones((batch_size, 32, 32))
  c = tf.vectorized_map(outer_product, a)
  assert c.shape == (batch_size, 32, 32, 32, 32)
  ```

  ```python
  # Computing per-example gradients

  batch_size = 10
  num_features = 32
  layer = tf.keras.layers.Dense(1)

  def model_fn(arg):
    with tf.GradientTape() as g:
      inp, label = arg
      inp = tf.expand_dims(inp, 0)
      label = tf.expand_dims(label, 0)
      prediction = layer(inp)
      loss = tf.nn.l2_loss(label - prediction)
    return g.gradient(loss, (layer.kernel, layer.bias))

  inputs = tf.random_uniform([batch_size, num_features])
  labels = tf.random_uniform([batch_size, 1])
  per_example_gradients = tf.vectorized_map(model_fn, (inputs, labels))
  assert per_example_gradients[0].shape == (batch_size, num_features, 1)
  assert per_example_gradients[1].shape == (batch_size, 1)
  ```
  "
  [ fn elems ]
  (py/call-attr v2 "vectorized_map"  fn elems ))

(defn where 
  "Return the elements, either from `x` or `y`, depending on the `condition`.

  If both `x` and `y` are None, then this operation returns the coordinates of
  true elements of `condition`.  The coordinates are returned in a 2-D tensor
  where the first dimension (rows) represents the number of true elements, and
  the second dimension (columns) represents the coordinates of the true
  elements. Keep in mind, the shape of the output tensor can vary depending on
  how many true values there are in input. Indices are output in row-major
  order.

  If both non-None, `condition`, `x` and `y` must be broadcastable to the same
  shape.

  The `condition` tensor acts as a mask that chooses, based on the value at each
  element, whether the corresponding element / row in the output should be taken
  from `x` (if true) or `y` (if false).

  Args:
    condition: A `Tensor` of type `bool`
    x: A Tensor which is of the same type as `y`, and may be broadcastable with
      `condition` and `y`.
    y: A Tensor which is of the same type as `x`, and may be broadcastable with
      `condition` and `x`.
    name: A name of the operation (optional).

  Returns:
    A `Tensor` with the same type as `x` and `y`, and shape that
      is broadcast from `condition`, `x`, and `y`, if `x`, `y` are non-None.
    A `Tensor` with shape `(num_true, dim_size(condition))`.

  Raises:
    ValueError: When exactly one of `x` or `y` is non-None.
  "
  [ condition x y name ]
  (py/call-attr v2 "where"  condition x y name ))

(defn while-loop 
  "Repeat `body` while the condition `cond` is true.

  `cond` is a callable returning a boolean scalar tensor. `body` is a callable
  returning a (possibly nested) tuple, namedtuple or list of tensors of the same
  arity (length and structure) and types as `loop_vars`. `loop_vars` is a
  (possibly nested) tuple, namedtuple or list of tensors that is passed to both
  `cond` and `body`. `cond` and `body` both take as many arguments as there are
  `loop_vars`.

  In addition to regular Tensors or IndexedSlices, the body may accept and
  return TensorArray objects.  The flows of the TensorArray objects will
  be appropriately forwarded between loops and during gradient calculations.

  Note that `while_loop` calls `cond` and `body` *exactly once* (inside the
  call to `while_loop`, and not at all during `Session.run()`). `while_loop`
  stitches together the graph fragments created during the `cond` and `body`
  calls with some additional graph nodes to create the graph flow that
  repeats `body` until `cond` returns false.

  For correctness, `tf.while_loop()` strictly enforces shape invariants for
  the loop variables. A shape invariant is a (possibly partial) shape that
  is unchanged across the iterations of the loop. An error will be raised
  if the shape of a loop variable after an iteration is determined to be more
  general than or incompatible with its shape invariant. For example, a shape
  of [11, None] is more general than a shape of [11, 17], and [11, 21] is not
  compatible with [11, 17]. By default (if the argument `shape_invariants` is
  not specified), it is assumed that the initial shape of each tensor in
  `loop_vars` is the same in every iteration. The `shape_invariants` argument
  allows the caller to specify a less specific shape invariant for each loop
  variable, which is needed if the shape varies between iterations. The
  `tf.Tensor.set_shape`
  function may also be used in the `body` function to indicate that
  the output loop variable has a particular shape. The shape invariant for
  SparseTensor and IndexedSlices are treated specially as follows:

  a) If a loop variable is a SparseTensor, the shape invariant must be
  TensorShape([r]) where r is the rank of the dense tensor represented
  by the sparse tensor. It means the shapes of the three tensors of the
  SparseTensor are ([None], [None, r], [r]). NOTE: The shape invariant here
  is the shape of the SparseTensor.dense_shape property. It must be the shape of
  a vector.

  b) If a loop variable is an IndexedSlices, the shape invariant must be
  a shape invariant of the values tensor of the IndexedSlices. It means
  the shapes of the three tensors of the IndexedSlices are (shape, [shape[0]],
  [shape.ndims]).

  `while_loop` implements non-strict semantics, enabling multiple iterations
  to run in parallel. The maximum number of parallel iterations can be
  controlled by `parallel_iterations`, which gives users some control over
  memory consumption and execution order. For correct programs, `while_loop`
  should return the same result for any parallel_iterations > 0.

  For training, TensorFlow stores the tensors that are produced in the
  forward inference and are needed in back propagation. These tensors are a
  main source of memory consumption and often cause OOM errors when training
  on GPUs. When the flag swap_memory is true, we swap out these tensors from
  GPU to CPU. This for example allows us to train RNN models with very long
  sequences and large batches.

  Args:
    cond: A callable that represents the termination condition of the loop.
    body: A callable that represents the loop body.
    loop_vars: A (possibly nested) tuple, namedtuple or list of numpy array,
      `Tensor`, and `TensorArray` objects.
    shape_invariants: The shape invariants for the loop variables.
    parallel_iterations: The number of iterations allowed to run in parallel. It
      must be a positive integer.
    back_prop: Whether backprop is enabled for this while loop.
    swap_memory: Whether GPU-CPU memory swap is enabled for this loop.
    maximum_iterations: Optional maximum number of iterations of the while loop
      to run.  If provided, the `cond` output is AND-ed with an additional
      condition ensuring the number of iterations executed is no greater than
      `maximum_iterations`.
    name: Optional name prefix for the returned tensors.

  Returns:
    The output tensors for the loop variables after the loop. The return value
      has the same structure as `loop_vars`.

  Raises:
    TypeError: if `cond` or `body` is not callable.
    ValueError: if `loop_vars` is empty.

  Example:

  ```python
  i = tf.constant(0)
  c = lambda i: tf.less(i, 10)
  b = lambda i: tf.add(i, 1)
  r = tf.while_loop(c, b, [i])
  ```

  Example with nesting and a namedtuple:

  ```python
  import collections
  Pair = collections.namedtuple('Pair', 'j, k')
  ijk_0 = (tf.constant(0), Pair(tf.constant(1), tf.constant(2)))
  c = lambda i, p: i < 10
  b = lambda i, p: (i + 1, Pair((p.j + p.k), (p.j - p.k)))
  ijk_final = tf.while_loop(c, b, ijk_0)
  ```

  Example using shape_invariants:

  ```python
  i0 = tf.constant(0)
  m0 = tf.ones([2, 2])
  c = lambda i, m: i < 10
  b = lambda i, m: [i+1, tf.concat([m, m], axis=0)]
  tf.while_loop(
      c, b, loop_vars=[i0, m0],
      shape_invariants=[i0.get_shape(), tf.TensorShape([None, 2])])
  ```

  Example which demonstrates non-strict semantics: In the following
  example, the final value of the counter `i` does not depend on `x`. So
  the `while_loop` can increment the counter parallel to updates of `x`.
  However, because the loop counter at one loop iteration depends
  on the value at the previous iteration, the loop counter itself cannot
  be incremented in parallel. Hence if we just want the final value of the
  counter (which we print on the line `print(sess.run(i))`), then
  `x` will never be incremented, but the counter will be updated on a
  single thread. Conversely, if we want the value of the output (which we
  print on the line `print(sess.run(out).shape)`), then the counter may be
  incremented on its own thread, while `x` can be incremented in
  parallel on a separate thread. In the extreme case, it is conceivable
  that the thread incrementing the counter runs until completion before
  `x` is incremented even a single time. The only thing that can never
  happen is that the thread updating `x` can never get ahead of the
  counter thread because the thread incrementing `x` depends on the value
  of the counter.

  ```python
  import tensorflow as tf

  n = 10000
  x = tf.constant(list(range(n)))
  c = lambda i, x: i < n
  b = lambda i, x: (tf.compat.v1.Print(i + 1, [i]), tf.compat.v1.Print(x + 1,
  [i], \"x:\"))
  i, out = tf.while_loop(c, b, (0, x))
  with tf.compat.v1.Session() as sess:
      print(sess.run(i))  # prints [0] ... [9999]

      # The following line may increment the counter and x in parallel.
      # The counter thread may get ahead of the other thread, but not the
      # other way around. So you may see things like
      # [9996] x:[9987]
      # meaning that the counter thread is on iteration 9996,
      # while the other thread is on iteration 9987
      print(sess.run(out).shape)
  ```

  "
  [cond body loop_vars shape_invariants & {:keys [parallel_iterations back_prop swap_memory maximum_iterations name]
                       :or {maximum_iterations None name None}} ]
    (py/call-attr-kw v2 "while_loop" [cond body loop_vars shape_invariants] {:parallel_iterations parallel_iterations :back_prop back_prop :swap_memory swap_memory :maximum_iterations maximum_iterations :name name }))

(defn zeros 
  "Creates a tensor with all elements set to zero.

  This operation returns a tensor of type `dtype` with shape `shape` and
  all elements set to zero.

  For example:

  ```python
  tf.zeros([3, 4], tf.int32)  # [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
  ```

  Args:
    shape: A list of integers, a tuple of integers, or a 1-D `Tensor` of type
      `int32`.
    dtype: The type of an element in the resulting `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with all elements set to zero.
  "
  [shape & {:keys [dtype name]
                       :or {name None}} ]
    (py/call-attr-kw v2 "zeros" [shape] {:dtype dtype :name name }))

(defn zeros-like 
  "Creates a tensor with all elements set to zero.

  Given a single tensor (`tensor`), this operation returns a tensor of the
  same type and shape as `tensor` with all elements set to zero. Optionally,
  you can use `dtype` to specify a new type for the returned tensor.

  For example:

  ```python
  tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
  tf.zeros_like(tensor)  # [[0, 0, 0], [0, 0, 0]] with dtype=int32

  If dtype of input `tensor` is `float32`, then the output is also of `float32`
  tensor = tf.constant([[1.0, 2.0, 3.0], [4, 5, 6]])
  tf.zeros_like(tensor)  # [[0., 0., 0.], [0., 0., 0.]] with dtype=floa32

  If you want to specify desired dtype of output `tensor`, then specify it in
  the op tensor = tf.constant([[1.0, 2.0, 3.0], [4, 5, 6]])
  tf.zeros_like(tensor,dtype=tf.int32)  # [[0, 0, 0], [0, 0, 0]] with
  dtype=int32
  ```

  Args:
    input: A `Tensor`.
    dtype: A type for the returned `Tensor`. Must be `float16`, `float32`,
      `float64`, `int8`, `uint8`, `int16`, `uint16`, `int32`, `int64`,
      `complex64`, `complex128`, `bool` or `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with all elements set to zero.
  "
  [ input dtype name ]
  (py/call-attr v2 "zeros_like"  input dtype name ))
