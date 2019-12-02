(ns tensorflow.contrib.quantization
  "Ops for building quantized models."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce quantization (import-module "tensorflow.contrib.quantization"))

(defn Abs 
  "Computes the absolute value of a tensor.

  Given a tensor `x`, this operation returns a tensor containing the absolute
  value of each element in `x`. For example, if x is an input element and y is
  an output element, this operation computes \\(y = |x|\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int8`, `int16`, `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Abs"  x name ))

(defn AccumulateNV2 
  "Returns the element-wise sum of a list of tensors.

  `tf.accumulate_n_v2` performs the same operation as `tf.add_n`, but does not
  wait for all of its inputs to be ready before beginning to sum. This can
  save memory if inputs are ready at different times, since minimum temporary
  storage is proportional to the output size rather than the inputs size.

  Unlike the original `accumulate_n`, `accumulate_n_v2` is differentiable.

  Returns a `Tensor` of same shape and type as the elements of `inputs`.

  Args:
    inputs: A list of at least 1 `Tensor` objects with the same type in: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A list of `Tensor` objects, each with same shape and type.
    shape: A `tf.TensorShape` or list of `ints`.
      Shape of elements of `inputs`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `inputs`.
  "
  [ inputs shape name ]
  (py/call-attr quantization "AccumulateNV2"  inputs shape name ))

(defn Acos 
  "Computes acos of x element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Acos"  x name ))

(defn Acosh 
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
  (py/call-attr quantization "Acosh"  x name ))

(defn Add 
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
  (py/call-attr quantization "Add"  x y name ))

(defn AddN 
  "Add all input tensors element wise.

    Inputs must be of same size and shape.

    ```python
    x = [9, 7, 10]
    tf.math.add_n(x) ==> 26
    ```

  Args:
    inputs: A list of at least 1 `Tensor` objects with the same type in: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`, `variant`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `inputs`.
  "
  [ inputs name ]
  (py/call-attr quantization "AddN"  inputs name ))

(defn AddV2 
  "Returns x + y element-wise.

  *NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "AddV2"  x y name ))

(defn All 
  "Computes the \"logical and\" of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor` of type `bool`. The tensor to reduce.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce. Must be in the range
      `[-rank(input), rank(input))`.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [input axis & {:keys [keep_dims name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "All" [input axis] {:keep_dims keep_dims :name name }))

(defn Angle 
  "Returns the argument of a complex number.

  Given a tensor `input` of complex numbers, this operation returns a tensor of
  type `float` that is the argument of each element in `input`. All elements in
  `input` must be complex numbers of the form \\(a + bj\\), where *a*
  is the real part and *b* is the imaginary part.

  The argument returned by this operation is of the form \\(atan2(b, a)\\).

  For example:

  ```
  # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.angle(input) ==> [2.0132, 1.056]
  ```

  @compatibility(numpy)
  Equivalent to np.angle.
  @end_compatibility

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
    Tout: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  "
  [input & {:keys [Tout name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Angle" [input] {:Tout Tout :name name }))

(defn Any 
  "Computes the \"logical or\" of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor` of type `bool`. The tensor to reduce.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce. Must be in the range
      `[-rank(input), rank(input))`.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [input axis & {:keys [keep_dims name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Any" [input axis] {:keep_dims keep_dims :name name }))

(defn ApproximateEqual 
  "Returns the truth value of abs(x-y) < tolerance element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    tolerance: An optional `float`. Defaults to `1e-05`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [x y & {:keys [tolerance name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "ApproximateEqual" [x y] {:tolerance tolerance :name name }))

(defn ArgMax 
  "Returns the index with the largest value across dimensions of a tensor.

  Note that in case of ties the identity of the return value is not guaranteed.

  Usage:
    ```python
    import tensorflow as tf
    a = [1, 10, 26.9, 2.8, 166.32, 62.3]
    b = tf.math.argmax(input = a)
    c = tf.keras.backend.eval(b)
    # c = 4
    # here a[4] = 166.32 which is the largest element of a across axis 0
    ```

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    dimension: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      int32 or int64, must be in the range `[-rank(input), rank(input))`.
      Describes which dimension of the input Tensor to reduce across. For vectors,
      use dimension = 0.
    output_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_type`.
  "
  [input dimension & {:keys [output_type name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "ArgMax" [input dimension] {:output_type output_type :name name }))

(defn ArgMin 
  "Returns the index with the smallest value across dimensions of a tensor.

  Note that in case of ties the identity of the return value is not guaranteed.

  Usage:
    ```python
    import tensorflow as tf
    a = [1, 10, 26.9, 2.8, 166.32, 62.3]
    b = tf.math.argmin(input = a)
    c = tf.keras.backend.eval(b)
    # c = 0
    # here a[0] = 1 which is the smallest element of a across axis 0
    ```

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    dimension: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      int32 or int64, must be in the range `[-rank(input), rank(input))`.
      Describes which dimension of the input Tensor to reduce across. For vectors,
      use dimension = 0.
    output_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_type`.
  "
  [input dimension & {:keys [output_type name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "ArgMin" [input dimension] {:output_type output_type :name name }))

(defn Asin 
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
  (py/call-attr quantization "Asin"  x name ))

(defn Asinh 
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
  (py/call-attr quantization "Asinh"  x name ))

(defn Atan 
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
  (py/call-attr quantization "Atan"  x name ))

(defn Atan2 
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
  (py/call-attr quantization "Atan2"  y x name ))

(defn Atanh 
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
  (py/call-attr quantization "Atanh"  x name ))

(defn AvgPool 
  "Performs average pooling on the input.

  Each entry in `output` is the mean of the corresponding size `ksize`
  window in `value`.

  Args:
    value: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the sliding window for each dimension of `value`.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of `value`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
  "
  [value ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "AvgPool" [value ksize strides padding] {:data_format data_format :name name }))

(defn AvgPool3D 
  "Performs 3D average pooling on the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NDHWC\", \"NCDHW\"`. Defaults to `\"NDHWC\"`.
      The data format of the input and output data. With the
      default format \"NDHWC\", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCDHW\", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "AvgPool3D" [input ksize strides padding] {:data_format data_format :name name }))

(defn AvgPool3DGrad 
  "Computes gradients of average pooling function.

  Args:
    orig_input_shape: A `Tensor` of type `int32`.
      The original input dimensions.
    grad: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      Output backprop of shape `[batch, depth, rows, cols, channels]`.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NDHWC\", \"NCDHW\"`. Defaults to `\"NDHWC\"`.
      The data format of the input and output data. With the
      default format \"NDHWC\", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCDHW\", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  "
  [orig_input_shape grad ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "AvgPool3DGrad" [orig_input_shape grad ksize strides padding] {:data_format data_format :name name }))

(defn AvgPoolGrad 
  "Computes gradients of the average pooling function.

  Args:
    orig_input_shape: A `Tensor` of type `int32`.
      1-D.  Shape of the original input to `avg_pool`.
    grad: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t.
      the output of `avg_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the sliding window for each dimension of the input.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the input.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  "
  [orig_input_shape grad ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "AvgPoolGrad" [orig_input_shape grad ksize strides padding] {:data_format data_format :name name }))

(defn BatchMatMul 
  "Multiplies slices of two tensors in batches.

  Multiplies all slices of `Tensor` `x` and `y` (each slice can be
  viewed as an element of a batch), and arranges the individual results
  in a single output tensor of the same batch size. Each of the
  individual slices can optionally be adjointed (to adjoint a matrix
  means to transpose and conjugate it) before multiplication by setting
  the `adj_x` or `adj_y` flag to `True`, which are by default `False`.

  The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
  and `[..., r_y, c_y]`.

  The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:

      r_o = c_x if adj_x else r_x
      c_o = r_y if adj_y else c_y

  It is computed as:

      output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
      2-D or higher with shape `[..., r_x, c_x]`.
    y: A `Tensor`. Must have the same type as `x`.
      2-D or higher with shape `[..., r_y, c_y]`.
    adj_x: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `x`. Defaults to `False`.
    adj_y: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `y`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [x y & {:keys [adj_x adj_y name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "BatchMatMul" [x y] {:adj_x adj_x :adj_y adj_y :name name }))

(defn BatchMatMulV2 
  "Multiplies slices of two tensors in batches.

  Multiplies all slices of `Tensor` `x` and `y` (each slice can be
  viewed as an element of a batch), and arranges the individual results
  in a single output tensor of the same batch size. Each of the
  individual slices can optionally be adjointed (to adjoint a matrix
  means to transpose and conjugate it) before multiplication by setting
  the `adj_x` or `adj_y` flag to `True`, which are by default `False`.

  The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
  and `[..., r_y, c_y]`.

  The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:

      r_o = c_x if adj_x else r_x
      c_o = r_y if adj_y else c_y

  It is computed as:

      output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])

  *NOTE*: `BatchMatMulV2` supports broadcasting in the batch dimensions. More
  about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
      2-D or higher with shape `[..., r_x, c_x]`.
    y: A `Tensor`. Must have the same type as `x`.
      2-D or higher with shape `[..., r_y, c_y]`.
    adj_x: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `x`. Defaults to `False`.
    adj_y: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `y`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [x y & {:keys [adj_x adj_y name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "BatchMatMulV2" [x y] {:adj_x adj_x :adj_y adj_y :name name }))

(defn BatchNormWithGlobalNormalization 
  "Batch normalization.

  This op is deprecated. Prefer `tf.nn.batch_normalization`.

  Args:
    t: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A 4D input Tensor.
    m: A `Tensor`. Must have the same type as `t`.
      A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    v: A `Tensor`. Must have the same type as `t`.
      A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    beta: A `Tensor`. Must have the same type as `t`.
      A 1D beta Tensor with size matching the last dimension of t.
      An offset to be added to the normalized tensor.
    gamma: A `Tensor`. Must have the same type as `t`.
      A 1D gamma Tensor with size matching the last dimension of t.
      If \"scale_after_normalization\" is true, this tensor will be multiplied
      with the normalized tensor.
    variance_epsilon: A `float`. A small float number to avoid dividing by 0.
    scale_after_normalization: A `bool`.
      A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `t`.
  "
  [ t m v beta gamma variance_epsilon scale_after_normalization name ]
  (py/call-attr quantization "BatchNormWithGlobalNormalization"  t m v beta gamma variance_epsilon scale_after_normalization name ))

(defn BatchNormWithGlobalNormalizationGrad 
  "Gradients for batch normalization.

  This op is deprecated. See `tf.nn.batch_normalization`.

  Args:
    t: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A 4D input Tensor.
    m: A `Tensor`. Must have the same type as `t`.
      A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    v: A `Tensor`. Must have the same type as `t`.
      A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    gamma: A `Tensor`. Must have the same type as `t`.
      A 1D gamma Tensor with size matching the last dimension of t.
      If \"scale_after_normalization\" is true, this Tensor will be multiplied
      with the normalized Tensor.
    backprop: A `Tensor`. Must have the same type as `t`. 4D backprop Tensor.
    variance_epsilon: A `float`. A small float number to avoid dividing by 0.
    scale_after_normalization: A `bool`.
      A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (dx, dm, dv, db, dg).

    dx: A `Tensor`. Has the same type as `t`.
    dm: A `Tensor`. Has the same type as `t`.
    dv: A `Tensor`. Has the same type as `t`.
    db: A `Tensor`. Has the same type as `t`.
    dg: A `Tensor`. Has the same type as `t`.
  "
  [ t m v gamma backprop variance_epsilon scale_after_normalization name ]
  (py/call-attr quantization "BatchNormWithGlobalNormalizationGrad"  t m v gamma backprop variance_epsilon scale_after_normalization name ))

(defn BesselI0e 
  "Computes the Bessel i0e function of `x` element-wise.

  Exponentially scaled modified Bessel function of order 0 defined as
  `bessel_i0e(x) = exp(-abs(x)) bessel_i0(x)`.

  This function is faster and numerically stabler than `bessel_i0(x)`.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "BesselI0e"  x name ))

(defn BesselI1e 
  "Computes the Bessel i1e function of `x` element-wise.

  Exponentially scaled modified Bessel function of order 0 defined as
  `bessel_i1e(x) = exp(-abs(x)) bessel_i1(x)`.

  This function is faster and numerically stabler than `bessel_i1(x)`.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "BesselI1e"  x name ))

(defn Betainc 
  "Compute the regularized incomplete beta integral \\(I_x(a, b)\\).

  The regularized incomplete beta integral is defined as:


  \\(I_x(a, b) = \frac{B(x; a, b)}{B(a, b)}\\)

  where


  \\(B(x; a, b) = \int_0^x t^{a-1} (1 - t)^{b-1} dt\\)


  is the incomplete beta function and \\(B(a, b)\\) is the *complete*
  beta function.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    b: A `Tensor`. Must have the same type as `a`.
    x: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  "
  [ a b x name ]
  (py/call-attr quantization "Betainc"  a b x name ))

(defn BiasAdd 
  "Adds `bias` to `value`.

  This is a special case of `tf.add` where `bias` is restricted to be 1-D.
  Broadcasting is supported, so `value` may have any number of dimensions.

  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Any number of dimensions.
    bias: A `Tensor`. Must have the same type as `value`.
      1-D with size the last dimension of `value`.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the bias tensor will be added to the last dimension
      of the value tensor.
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
      The tensor will be added to \"in_channels\", the third-to-the-last
          dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
  "
  [value bias & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "BiasAdd" [value bias] {:data_format data_format :name name }))

(defn BiasAddGrad 
  "The backward operation for \"BiasAdd\" on the \"bias\" tensor.

  It accumulates all the values from out_backprop into the feature dimension.
  For NHWC data format, the feature dimension is the last. For NCHW data format,
  the feature dimension is the third-to-last.

  Args:
    out_backprop: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Any number of dimensions.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the bias tensor will be added to the last dimension
      of the value tensor.
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
      The tensor will be added to \"in_channels\", the third-to-the-last
          dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `out_backprop`.
  "
  [out_backprop & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "BiasAddGrad" [out_backprop] {:data_format data_format :name name }))

(defn BiasAddV1 
  "Adds `bias` to `value`.

  This is a deprecated version of BiasAdd and will be soon removed.

  This is a special case of `tf.add` where `bias` is restricted to be 1-D.
  Broadcasting is supported, so `value` may have any number of dimensions.

  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Any number of dimensions.
    bias: A `Tensor`. Must have the same type as `value`.
      1-D with size the last dimension of `value`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
  "
  [ value bias name ]
  (py/call-attr quantization "BiasAddV1"  value bias name ))

(defn Bincount 
  "Counts the number of occurrences of each value in an integer array.

  Outputs a vector with length `size` and the same dtype as `weights`. If
  `weights` are empty, then index `i` stores the number of times the value `i` is
  counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
  the value in `weights` at each index where the corresponding value in `arr` is
  `i`.

  Values in `arr` outside of the range [0, size) are ignored.

  Args:
    arr: A `Tensor` of type `int32`. int32 `Tensor`.
    size: A `Tensor` of type `int32`. non-negative int32 scalar `Tensor`.
    weights: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      is an int32, int64, float32, or float64 `Tensor` with the same
      shape as `arr`, or a length-0 `Tensor`, in which case it acts as all weights
      equal to 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `weights`.
  "
  [ arr size weights name ]
  (py/call-attr quantization "Bincount"  arr size weights name ))

(defn Bucketize 
  "Bucketizes 'input' based on 'boundaries'.

  For example, if the inputs are
      boundaries = [0, 10, 100]
      input = [[-5, 10000]
               [150,   10]
               [5,    100]]

  then the output will be
      output = [[0, 3]
                [3, 2]
                [1, 3]]

  Args:
    input: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      Any shape of Tensor contains with int or float type.
    boundaries: A list of `floats`.
      A sorted list of floats gives the boundary of the buckets.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  "
  [ input boundaries name ]
  (py/call-attr quantization "Bucketize"  input boundaries name ))

(defn Cast 
  "Cast x of type SrcT to y of DstT.

  Args:
    x: A `Tensor`.
    DstT: A `tf.DType`.
    Truncate: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `DstT`.
  "
  [x DstT & {:keys [Truncate name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Cast" [x DstT] {:Truncate Truncate :name name }))

(defn Ceil 
  "Returns element-wise smallest integer not less than x.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Ceil"  x name ))

(defn ClipByValue 
  "Clips tensor values to a specified min and max.

  Given a tensor `t`, this operation returns a tensor of the same type and
  shape as `t` with its values clipped to `clip_value_min` and `clip_value_max`.
  Any values less than `clip_value_min` are set to `clip_value_min`. Any values
  greater than `clip_value_max` are set to `clip_value_max`.

  Args:
    t: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A `Tensor`.
    clip_value_min: A `Tensor`. Must have the same type as `t`.
      A 0-D (scalar) `Tensor`, or a `Tensor` with the same shape
      as `t`. The minimum value to clip by.
    clip_value_max: A `Tensor`. Must have the same type as `t`.
      A 0-D (scalar) `Tensor`, or a `Tensor` with the same shape
      as `t`. The maximum value to clip by.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `t`.
  "
  [ t clip_value_min clip_value_max name ]
  (py/call-attr quantization "ClipByValue"  t clip_value_min clip_value_max name ))

(defn CompareAndBitpack 
  "Compare values of `input` to `threshold` and pack resulting bits into a `uint8`.

  Each comparison returns a boolean `true` (if `input_value > threshold`)
  or and `false` otherwise.

  This operation is useful for Locality-Sensitive-Hashing (LSH) and other
  algorithms that use hashing approximations of cosine and `L2` distances;
  codes can be generated from an input via:

  ```python
  codebook_size = 50
  codebook_bits = codebook_size * 32
  codebook = tf.get_variable('codebook', [x.shape[-1].value, codebook_bits],
                             dtype=x.dtype,
                             initializer=tf.orthogonal_initializer())
  codes = compare_and_threshold(tf.matmul(x, codebook), threshold=0.)
  codes = tf.bitcast(codes, tf.int32)  # go from uint8 to int32
  # now codes has shape x.shape[:-1] + [codebook_size]
  ```

  **NOTE**: Currently, the innermost dimension of the tensor must be divisible
  by 8.

  Given an `input` shaped `[s0, s1, ..., s_n]`, the output is
  a `uint8` tensor shaped `[s0, s1, ..., s_n / 8]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `bool`, `half`, `float32`, `float64`, `int8`, `int16`, `int32`, `int64`.
      Values to compare against `threshold` and bitpack.
    threshold: A `Tensor`. Must have the same type as `input`.
      Threshold to compare against.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8`.
  "
  [ input threshold name ]
  (py/call-attr quantization "CompareAndBitpack"  input threshold name ))

(defn Complex 
  "Converts two real numbers to a complex number.

  Given a tensor `real` representing the real part of a complex number, and a
  tensor `imag` representing the imaginary part of a complex number, this
  operation returns complex numbers elementwise of the form \\(a + bj\\), where
  *a* represents the `real` part and *b* represents the `imag` part.

  The input tensors `real` and `imag` must have the same shape.

  For example:

  ```
  # tensor 'real' is [2.25, 3.25]
  # tensor `imag` is [4.75, 5.75]
  tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
  ```

  Args:
    real: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    imag: A `Tensor`. Must have the same type as `real`.
    Tout: An optional `tf.DType` from: `tf.complex64, tf.complex128`. Defaults to `tf.complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  "
  [real imag & {:keys [Tout name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Complex" [real imag] {:Tout Tout :name name }))

(defn ComplexAbs 
  "Computes the complex absolute value of a tensor.

  Given a tensor `x` of complex numbers, this operation returns a tensor of type
  `float` or `double` that is the absolute value of each element in `x`. All
  elements in `x` must be complex numbers of the form \\(a + bj\\). The absolute
  value is computed as \\( \sqrt{a^2 + b^2}\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
    Tout: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  "
  [x & {:keys [Tout name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "ComplexAbs" [x] {:Tout Tout :name name }))

(defn Conj 
  "Returns the complex conjugate of a complex number.

  Given a tensor `input` of complex numbers, this operation returns a tensor of
  complex numbers that are the complex conjugate of each element in `input`. The
  complex numbers in `input` must be of the form \\(a + bj\\), where *a* is the
  real part and *b* is the imaginary part.

  The complex conjugate returned by this operation is of the form \\(a - bj\\).

  For example:

  ```
  # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`, `variant`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input name ]
  (py/call-attr quantization "Conj"  input name ))

(defn Conv2D 
  "Computes a 2-D convolution given 4-D `input` and `filter` tensors.

  Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
  and a filter / kernel tensor of shape
  `[filter_height, filter_width, in_channels, out_channels]`, this op
  performs the following:

  1. Flattens the filter to a 2-D matrix with shape
     `[filter_height * filter_width * in_channels, output_channels]`.
  2. Extracts image patches from the input tensor to form a *virtual*
     tensor of shape `[batch, out_height, out_width,
     filter_height * filter_width * in_channels]`.
  3. For each patch, right-multiplies the filter matrix and the image patch
     vector.

  In detail, with the default NHWC format,

      output[b, i, j, k] =
          sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                          filter[di, dj, q, k]

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      A 4-D tensor. The dimension order is interpreted according to the value
      of `data_format`, see below for details.
    filter: A `Tensor`. Must have the same type as `input`.
      A 4-D tensor of shape
      `[filter_height, filter_width, in_channels, out_channels]`
    strides: A list of `ints`.
      1-D tensor of length 4.  The stride of the sliding window for each
      dimension of `input`. The dimension order is determined by the value of
      `data_format`, see below for details.
    padding: A `string` from: `\"SAME\", \"VALID\", \"EXPLICIT\"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    explicit_paddings: An optional list of `ints`. Defaults to `[]`.
      If `padding` is `\"EXPLICIT\"`, the list of explicit padding amounts. For the ith
      dimension, the amount of padding inserted before and after the dimension is
      `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
      `padding` is not `\"EXPLICIT\"`, `explicit_paddings` must be empty.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, channels, height, width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each
      filter element on that dimension. The dimension order is determined by the
      value of `data_format`, see above for details. Dilations in the batch and
      depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input filter strides padding & {:keys [use_cudnn_on_gpu explicit_paddings data_format dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Conv2D" [input filter strides padding] {:use_cudnn_on_gpu use_cudnn_on_gpu :explicit_paddings explicit_paddings :data_format data_format :dilations dilations :name name }))

(defn Conv2DBackpropFilter 
  "Computes the gradients of convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
    filter_sizes: A `Tensor` of type `int32`.
      An integer vector representing the tensor shape of `filter`,
      where `filter` is a 4-D
      `[filter_height, filter_width, in_channels, out_channels]` tensor.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution. Must be in the same order as the dimension specified with
      format.
    padding: A `string` from: `\"SAME\", \"VALID\", \"EXPLICIT\"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    explicit_paddings: An optional list of `ints`. Defaults to `[]`.
      If `padding` is `\"EXPLICIT\"`, the list of explicit padding amounts. For the ith
      dimension, the amount of padding inserted before and after the dimension is
      `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
      `padding` is not `\"EXPLICIT\"`, `explicit_paddings` must be empty.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input filter_sizes out_backprop strides padding & {:keys [use_cudnn_on_gpu explicit_paddings data_format dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Conv2DBackpropFilter" [input filter_sizes out_backprop strides padding] {:use_cudnn_on_gpu use_cudnn_on_gpu :explicit_paddings explicit_paddings :data_format data_format :dilations dilations :name name }))

(defn Conv2DBackpropInput 
  "Computes the gradients of convolution with respect to the input.

  Args:
    input_sizes: A `Tensor` of type `int32`.
      An integer vector representing the shape of `input`,
      where `input` is a 4-D `[batch, height, width, channels]` tensor.
    filter: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape
      `[filter_height, filter_width, in_channels, out_channels]`.
    out_backprop: A `Tensor`. Must have the same type as `filter`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution. Must be in the same order as the dimension specified with
      format.
    padding: A `string` from: `\"SAME\", \"VALID\", \"EXPLICIT\"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    explicit_paddings: An optional list of `ints`. Defaults to `[]`.
      If `padding` is `\"EXPLICIT\"`, the list of explicit padding amounts. For the ith
      dimension, the amount of padding inserted before and after the dimension is
      `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
      `padding` is not `\"EXPLICIT\"`, `explicit_paddings` must be empty.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `filter`.
  "
  [input_sizes filter out_backprop strides padding & {:keys [use_cudnn_on_gpu explicit_paddings data_format dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Conv2DBackpropInput" [input_sizes filter out_backprop strides padding] {:use_cudnn_on_gpu use_cudnn_on_gpu :explicit_paddings explicit_paddings :data_format data_format :dilations dilations :name name }))

(defn Conv3D 
  "Computes a 3-D convolution given 5-D `input` and `filter` tensors.

  In signal processing, cross-correlation is a measure of similarity of
  two waveforms as a function of a time-lag applied to one of them. This
  is also known as a sliding dot product or sliding inner-product.

  Our Conv3D implements a form of cross-correlation.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      Shape `[batch, in_depth, in_height, in_width, in_channels]`.
    filter: A `Tensor`. Must have the same type as `input`.
      Shape `[filter_depth, filter_height, filter_width, in_channels,
      out_channels]`. `in_channels` must match between `input` and `filter`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NDHWC\", \"NCDHW\"`. Defaults to `\"NDHWC\"`.
      The data format of the input and output data. With the
      default format \"NDHWC\", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCDHW\", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1, 1]`.
      1-D tensor of length 5.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each
      filter element on that dimension. The dimension order is determined by the
      value of `data_format`, see above for details. Dilations in the batch and
      depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input filter strides padding & {:keys [data_format dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Conv3D" [input filter strides padding] {:data_format data_format :dilations dilations :name name }))

(defn Conv3DBackpropFilter 
  "Computes the gradients of 3-D convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      Shape `[batch, depth, rows, cols, in_channels]`.
    filter: A `Tensor`. Must have the same type as `input`.
      Shape `[depth, rows, cols, in_channels, out_channels]`.
      `in_channels` must match between `input` and `filter`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
      out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1, 1]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input filter out_backprop strides padding & {:keys [dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Conv3DBackpropFilter" [input filter out_backprop strides padding] {:dilations dilations :name name }))

(defn Conv3DBackpropFilterV2 
  "Computes the gradients of 3-D convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      Shape `[batch, depth, rows, cols, in_channels]`.
    filter_sizes: A `Tensor` of type `int32`.
      An integer vector representing the tensor shape of `filter`,
      where `filter` is a 5-D
      `[filter_depth, filter_height, filter_width, in_channels, out_channels]`
      tensor.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
      out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NDHWC\", \"NCDHW\"`. Defaults to `\"NDHWC\"`.
      The data format of the input and output data. With the
      default format \"NDHWC\", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCDHW\", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1, 1]`.
      1-D tensor of length 5.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each
      filter element on that dimension. The dimension order is determined by the
      value of `data_format`, see above for details. Dilations in the batch and
      depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input filter_sizes out_backprop strides padding & {:keys [data_format dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Conv3DBackpropFilterV2" [input filter_sizes out_backprop strides padding] {:data_format data_format :dilations dilations :name name }))

(defn Conv3DBackpropInput 
  "Computes the gradients of 3-D convolution with respect to the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      Shape `[batch, depth, rows, cols, in_channels]`.
    filter: A `Tensor`. Must have the same type as `input`.
      Shape `[depth, rows, cols, in_channels, out_channels]`.
      `in_channels` must match between `input` and `filter`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
      out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1, 1]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input filter out_backprop strides padding & {:keys [dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Conv3DBackpropInput" [input filter out_backprop strides padding] {:dilations dilations :name name }))

(defn Conv3DBackpropInputV2 
  "Computes the gradients of 3-D convolution with respect to the input.

  Args:
    input_sizes: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      An integer vector representing the tensor shape of `input`,
      where `input` is a 5-D
      `[batch, depth, rows, cols, in_channels]` tensor.
    filter: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      Shape `[depth, rows, cols, in_channels, out_channels]`.
      `in_channels` must match between `input` and `filter`.
    out_backprop: A `Tensor`. Must have the same type as `filter`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
      out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NDHWC\", \"NCDHW\"`. Defaults to `\"NDHWC\"`.
      The data format of the input and output data. With the
      default format \"NDHWC\", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCDHW\", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1, 1]`.
      1-D tensor of length 5.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each
      filter element on that dimension. The dimension order is determined by the
      value of `data_format`, see above for details. Dilations in the batch and
      depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `filter`.
  "
  [input_sizes filter out_backprop strides padding & {:keys [data_format dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Conv3DBackpropInputV2" [input_sizes filter out_backprop strides padding] {:data_format data_format :dilations dilations :name name }))

(defn Cos 
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
  (py/call-attr quantization "Cos"  x name ))

(defn Cosh 
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
  (py/call-attr quantization "Cosh"  x name ))

(defn Cross 
  "Compute the pairwise cross product.

  `a` and `b` must be the same shape; they can either be simple 3-element vectors,
  or any shape where the innermost dimension is 3. In the latter case, each pair
  of corresponding 3-element vectors is cross-multiplied independently.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      A tensor containing 3-element vectors.
    b: A `Tensor`. Must have the same type as `a`.
      Another tensor, of same type and shape as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  "
  [ a b name ]
  (py/call-attr quantization "Cross"  a b name ))

(defn Cumprod 
  "Compute the cumulative product of the tensor `x` along `axis`.

  By default, this op performs an inclusive cumprod, which means that the first
  element of the input is identical to the first element of the output:

  ```python
  tf.cumprod([a, b, c])  # => [a, a * b, a * b * c]
  ```

  By setting the `exclusive` kwarg to `True`, an exclusive cumprod is
  performed instead:

  ```python
  tf.cumprod([a, b, c], exclusive=True)  # => [1, a, a * b]
  ```

  By setting the `reverse` kwarg to `True`, the cumprod is performed in the
  opposite direction:

  ```python
  tf.cumprod([a, b, c], reverse=True)  # => [a * b * c, b * c, c]
  ```

  This is more efficient than using separate `tf.reverse` ops.

  The `reverse` and `exclusive` kwargs can also be combined:

  ```python
  tf.cumprod([a, b, c], exclusive=True, reverse=True)  # => [b * c, c, 1]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A `Tensor`. Must be one of the following types: `float32`, `float64`,
      `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
      `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `Tensor` of type `int32` (default: 0). Must be in the range
      `[-rank(x), rank(x))`.
    exclusive: An optional `bool`. Defaults to `False`.
      If `True`, perform exclusive cumprod.
    reverse: An optional `bool`. Defaults to `False`.
      A `bool` (default: False).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [x axis & {:keys [exclusive reverse name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Cumprod" [x axis] {:exclusive exclusive :reverse reverse :name name }))

(defn Cumsum 
  "Compute the cumulative sum of the tensor `x` along `axis`.

  By default, this op performs an inclusive cumsum, which means that the first
  element of the input is identical to the first element of the output:

  ```python
  tf.cumsum([a, b, c])  # => [a, a + b, a + b + c]
  ```

  By setting the `exclusive` kwarg to `True`, an exclusive cumsum is
  performed instead:

  ```python
  tf.cumsum([a, b, c], exclusive=True)  # => [0, a, a + b]
  ```

  By setting the `reverse` kwarg to `True`, the cumsum is performed in the
  opposite direction:

  ```python
  tf.cumsum([a, b, c], reverse=True)  # => [a + b + c, b + c, c]
  ```

  This is more efficient than using separate `tf.reverse` ops.

  The `reverse` and `exclusive` kwargs can also be combined:

  ```python
  tf.cumsum([a, b, c], exclusive=True, reverse=True)  # => [b + c, c, 0]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A `Tensor`. Must be one of the following types: `float32`, `float64`,
      `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
      `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `Tensor` of type `int32` (default: 0). Must be in the range
      `[-rank(x), rank(x))`.
    exclusive: An optional `bool`. Defaults to `False`.
      If `True`, perform exclusive cumsum.
    reverse: An optional `bool`. Defaults to `False`.
      A `bool` (default: False).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [x axis & {:keys [exclusive reverse name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Cumsum" [x axis] {:exclusive exclusive :reverse reverse :name name }))

(defn CumulativeLogsumexp 
  "Compute the cumulative product of the tensor `x` along `axis`.

  By default, this op performs an inclusive cumulative log-sum-exp,
  which means that the first
  element of the input is identical to the first element of the output:
  ```python
  tf.math.cumulative_logsumexp([a, b, c])  # => [a, log(exp(a) + exp(b)), log(exp(a) + exp(b) + exp(c))]
  ```

  By setting the `exclusive` kwarg to `True`, an exclusive cumulative log-sum-exp is
  performed instead:
  ```python
  tf.cumulative_logsumexp([a, b, c], exclusive=True)  # => [-inf, a, log(exp(a) * exp(b))]
  ```
  Note that the neutral element of the log-sum-exp operation is `-inf`,
  however, for performance reasons, the minimal value representable by the
  floating point type is used instead.

  By setting the `reverse` kwarg to `True`, the cumulative log-sum-exp is performed in the
  opposite direction.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      A `Tensor`. Must be one of the following types: `float16`, `float32`, `float64`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `Tensor` of type `int32` (default: 0). Must be in the range
      `[-rank(x), rank(x))`.
    exclusive: An optional `bool`. Defaults to `False`.
      If `True`, perform exclusive cumulative log-sum-exp.
    reverse: An optional `bool`. Defaults to `False`.
      A `bool` (default: False).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [x axis & {:keys [exclusive reverse name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "CumulativeLogsumexp" [x axis] {:exclusive exclusive :reverse reverse :name name }))

(defn DataFormatDimMap 
  "Returns the dimension index in the destination data format given the one in

  the source data format.

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A Tensor with each element as a dimension index in source data format.
      Must be in the range [-4, 4).
    src_format: An optional `string`. Defaults to `\"NHWC\"`.
      source data format.
    dst_format: An optional `string`. Defaults to `\"NCHW\"`.
      destination data format.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [x & {:keys [src_format dst_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "DataFormatDimMap" [x] {:src_format src_format :dst_format dst_format :name name }))

(defn DataFormatVecPermute 
  "Returns the permuted vector/tensor in the destination data format given the

  one in the source data format.

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Vector of size 4 or Tensor of shape (4, 2) in source data format.
    src_format: An optional `string`. Defaults to `\"NHWC\"`.
      source data format.
    dst_format: An optional `string`. Defaults to `\"NCHW\"`.
      destination data format.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [x & {:keys [src_format dst_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "DataFormatVecPermute" [x] {:src_format src_format :dst_format dst_format :name name }))

(defn DepthwiseConv2dNative 
  "Computes a 2-D depthwise convolution given 4-D `input` and `filter` tensors.

  Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
  and a filter / kernel tensor of shape
  `[filter_height, filter_width, in_channels, channel_multiplier]`, containing
  `in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
  a different filter to each input channel (expanding from 1 channel to
  `channel_multiplier` channels for each), then concatenates the results
  together. Thus, the output has `in_channels * channel_multiplier` channels.

  ```
  for k in 0..in_channels-1
    for q in 0..channel_multiplier-1
      output[b, i, j, k * channel_multiplier + q] =
        sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                          filter[di, dj, k, q]
  ```

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    filter: A `Tensor`. Must have the same type as `input`.
    strides: A list of `ints`.
      1-D of length 4.  The stride of the sliding window for each dimension
      of `input`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, channels, height, width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input filter strides padding & {:keys [data_format dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "DepthwiseConv2dNative" [input filter strides padding] {:data_format data_format :dilations dilations :name name }))

(defn DepthwiseConv2dNativeBackpropFilter 
  "Computes the gradients of depthwise convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape based on `data_format`.  For example, if
      `data_format` is 'NHWC' then `input` is a 4-D `[batch, in_height,
      in_width, in_channels]` tensor.
    filter_sizes: A `Tensor` of type `int32`.
      An integer vector representing the tensor shape of `filter`,
      where `filter` is a 4-D
      `[filter_height, filter_width, in_channels, depthwise_multiplier]` tensor.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape  based on `data_format`.
      For example, if `data_format` is 'NHWC' then
      out_backprop shape is `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, channels, height, width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input filter_sizes out_backprop strides padding & {:keys [data_format dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "DepthwiseConv2dNativeBackpropFilter" [input filter_sizes out_backprop strides padding] {:data_format data_format :dilations dilations :name name }))

(defn DepthwiseConv2dNativeBackpropInput 
  "Computes the gradients of depthwise convolution with respect to the input.

  Args:
    input_sizes: A `Tensor` of type `int32`.
      An integer vector representing the shape of `input`, based
      on `data_format`.  For example, if `data_format` is 'NHWC' then
       `input` is a 4-D `[batch, height, width, channels]` tensor.
    filter: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape
      `[filter_height, filter_width, in_channels, depthwise_multiplier]`.
    out_backprop: A `Tensor`. Must have the same type as `filter`.
      4-D with shape  based on `data_format`.
      For example, if `data_format` is 'NHWC' then
      out_backprop shape is `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, channels, height, width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `filter`.
  "
  [input_sizes filter out_backprop strides padding & {:keys [data_format dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "DepthwiseConv2dNativeBackpropInput" [input_sizes filter out_backprop strides padding] {:data_format data_format :dilations dilations :name name }))

(defn Digamma 
  "Computes Psi, the derivative of Lgamma (the log of the absolute value of

  `Gamma(x)`), element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Digamma"  x name ))

(defn Dilation2D 
  "Computes the grayscale dilation of 4-D `input` and 3-D `filter` tensors.

  The `input` tensor has shape `[batch, in_height, in_width, depth]` and the
  `filter` tensor has shape `[filter_height, filter_width, depth]`, i.e., each
  input channel is processed independently of the others with its own structuring
  function. The `output` tensor has shape
  `[batch, out_height, out_width, depth]`. The spatial dimensions of the output
  tensor depend on the `padding` algorithm. We currently only support the default
  \"NHWC\" `data_format`.

  In detail, the grayscale morphological 2-D dilation is the max-sum correlation
  (for consistency with `conv2d`, we use unmirrored filters):

      output[b, y, x, c] =
         max_{dy, dx} input[b,
                            strides[1] * y + rates[1] * dy,
                            strides[2] * x + rates[2] * dx,
                            c] +
                      filter[dy, dx, c]

  Max-pooling is a special case when the filter has size equal to the pooling
  kernel size and contains all zeros.

  Note on duality: The dilation of `input` by the `filter` is equal to the
  negation of the erosion of `-input` by the reflected `filter`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      4-D with shape `[batch, in_height, in_width, depth]`.
    filter: A `Tensor`. Must have the same type as `input`.
      3-D with shape `[filter_height, filter_width, depth]`.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the input
      tensor. Must be: `[1, stride_height, stride_width, 1]`.
    rates: A list of `ints` that has length `>= 4`.
      The input stride for atrous morphological dilation. Must be:
      `[1, rate_height, rate_width, 1]`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input filter strides rates padding name ]
  (py/call-attr quantization "Dilation2D"  input filter strides rates padding name ))

(defn Dilation2DBackpropFilter 
  "Computes the gradient of morphological 2-D dilation with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      4-D with shape `[batch, in_height, in_width, depth]`.
    filter: A `Tensor`. Must have the same type as `input`.
      3-D with shape `[filter_height, filter_width, depth]`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, depth]`.
    strides: A list of `ints` that has length `>= 4`.
      1-D of length 4. The stride of the sliding window for each dimension of
      the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
    rates: A list of `ints` that has length `>= 4`.
      1-D of length 4. The input stride for atrous morphological dilation.
      Must be: `[1, rate_height, rate_width, 1]`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input filter out_backprop strides rates padding name ]
  (py/call-attr quantization "Dilation2DBackpropFilter"  input filter out_backprop strides rates padding name ))

(defn Dilation2DBackpropInput 
  "Computes the gradient of morphological 2-D dilation with respect to the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      4-D with shape `[batch, in_height, in_width, depth]`.
    filter: A `Tensor`. Must have the same type as `input`.
      3-D with shape `[filter_height, filter_width, depth]`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, depth]`.
    strides: A list of `ints` that has length `>= 4`.
      1-D of length 4. The stride of the sliding window for each dimension of
      the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
    rates: A list of `ints` that has length `>= 4`.
      1-D of length 4. The input stride for atrous morphological dilation.
      Must be: `[1, rate_height, rate_width, 1]`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input filter out_backprop strides rates padding name ]
  (py/call-attr quantization "Dilation2DBackpropInput"  input filter out_backprop strides rates padding name ))

(defn Div 
  "Returns x / y element-wise.

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
  (py/call-attr quantization "Div"  x y name ))

(defn DivNoNan 
  "Returns 0 if the denominator is zero.

  
  *NOTE*: `DivNoNan` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "DivNoNan"  x y name ))

(defn Elu 
  "Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.

  See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
  ](http://arxiv.org/abs/1511.07289)

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  "
  [ features name ]
  (py/call-attr quantization "Elu"  features name ))

(defn EluGrad 
  "Computes gradients for the exponential linear (Elu) operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      The backpropagated gradients to the corresponding Elu operation.
    outputs: A `Tensor`. Must have the same type as `gradients`.
      The outputs of the corresponding Elu operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  "
  [ gradients outputs name ]
  (py/call-attr quantization "EluGrad"  gradients outputs name ))

(defn Equal 
  "Returns the truth value of (x == y) element-wise.

  *NOTE*: `Equal` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  ```python
  x = tf.constant([2, 4])
  y = tf.constant(2)
  tf.math.equal(x, y) ==> array([True, False])

  x = tf.constant([2, 4])
  y = tf.constant([2, 4])
  tf.math.equal(x, y) ==> array([True,  True])
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `quint8`, `qint8`, `qint32`, `string`, `bool`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    incompatible_shape_error: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [x y & {:keys [incompatible_shape_error name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Equal" [x y] {:incompatible_shape_error incompatible_shape_error :name name }))

(defn Erf 
  "Computes the Gauss error function of `x` element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Erf"  x name ))

(defn Erfc 
  "Computes the complementary error function of `x` element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Erfc"  x name ))

(defn EuclideanNorm 
  "Computes the euclidean norm of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The tensor to reduce.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce. Must be in the range
      `[-rank(input), rank(input))`.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input axis & {:keys [keep_dims name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "EuclideanNorm" [input axis] {:keep_dims keep_dims :name name }))

(defn Exp 
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
  (py/call-attr quantization "Exp"  x name ))

(defn Expm1 
  "Computes `exp(x) - 1` element-wise.

    i.e. `exp(x) - 1` or `e^(x) - 1`, where `x` is the input tensor.
    `e` denotes Euler's number and is approximately equal to 2.718281.

    ```python
    x = tf.constant(2.0)
    tf.math.expm1(x) ==> 6.389056

    x = tf.constant([2.0, 8.0])
    tf.math.expm1(x) ==> array([6.389056, 2979.958], dtype=float32)

    x = tf.constant(1 + 1j)
    tf.math.expm1(x) ==> (0.46869393991588515+2.2873552871788423j)
    ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Expm1"  x name ))

(defn Floor 
  "Returns element-wise largest integer not greater than x.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Floor"  x name ))

(defn FloorDiv 
  "Returns x // y element-wise.

  *NOTE*: `floor_div` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "FloorDiv"  x y name ))

(defn FloorMod 
  "Returns element-wise remainder of division. When `x < 0` xor `y < 0` is

  true, this follows Python semantics in that the result here is consistent
  with a flooring divide. E.g. `floor(x / y) * y + mod(x, y) = x`.

  *NOTE*: `math.floormod` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`, `bfloat16`, `half`, `float32`, `float64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "FloorMod"  x y name ))

(defn FractionalAvgPool 
  "Performs fractional average pooling on the input.

  Fractional average pooling is similar to Fractional max pooling in the pooling
  region generation step. The only difference is that after pooling regions are
  generated, a mean operation is performed instead of a max operation in each
  pooling region.

  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      4-D with shape `[batch, height, width, channels]`.
    pooling_ratio: A list of `floats` that has length `>= 4`.
      Pooling ratio for each dimension of `value`, currently only
      supports row and col dimension and should be >= 1.0. For example, a valid
      pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements
      must be 1.0 because we don't allow pooling on batch and channels
      dimensions. 1.44 and 1.73 are pooling ratio on height and width dimensions
      respectively.
    pseudo_random: An optional `bool`. Defaults to `False`.
      When set to True, generates the pooling sequence in a
      pseudorandom fashion, otherwise, in a random fashion. Check paper [Benjamin
      Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071) for
      difference between pseudorandom and random.
    overlapping: An optional `bool`. Defaults to `False`.
      When set to True, it means when pooling, the values at the boundary
      of adjacent pooling cells are used by both cells. For example:

      `index  0  1  2  3  4`

      `value  20 5  16 3  7`

      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
      The result would be [41/3, 26/3] for fractional avg pooling.
    deterministic: An optional `bool`. Defaults to `False`.
      When set to True, a fixed pooling region will be used when
      iterating over a FractionalAvgPool node in the computation graph. Mainly used
      in unit test to make FractionalAvgPool deterministic.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, row_pooling_sequence, col_pooling_sequence).

    output: A `Tensor`. Has the same type as `value`.
    row_pooling_sequence: A `Tensor` of type `int64`.
    col_pooling_sequence: A `Tensor` of type `int64`.
  "
  [value pooling_ratio & {:keys [pseudo_random overlapping deterministic seed seed2 name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "FractionalAvgPool" [value pooling_ratio] {:pseudo_random pseudo_random :overlapping overlapping :deterministic deterministic :seed seed :seed2 seed2 :name name }))

(defn FractionalAvgPoolGrad 
  "Computes gradient of the FractionalAvgPool function.

  Unlike FractionalMaxPoolGrad, we don't need to find arg_max for
  FractionalAvgPoolGrad, we just need to evenly back-propagate each element of
  out_backprop to those indices that form the same pooling cell. Therefore, we
  just need to know the shape of original input tensor, instead of the whole
  tensor.

  Args:
    orig_input_tensor_shape: A `Tensor` of type `int64`.
      Original input tensor shape for `fractional_avg_pool`
    out_backprop: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      4-D with shape `[batch, height, width, channels]`.  Gradients
      w.r.t. the output of `fractional_avg_pool`.
    row_pooling_sequence: A `Tensor` of type `int64`.
      row pooling sequence, form pooling region with
      col_pooling_sequence.
    col_pooling_sequence: A `Tensor` of type `int64`.
      column pooling sequence, form pooling region with
      row_pooling sequence.
    overlapping: An optional `bool`. Defaults to `False`.
      When set to True, it means when pooling, the values at the boundary
      of adjacent pooling cells are used by both cells. For example:

      `index  0  1  2  3  4`

      `value  20 5  16 3  7`

      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
      The result would be [41/3, 26/3] for fractional avg pooling.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `out_backprop`.
  "
  [orig_input_tensor_shape out_backprop row_pooling_sequence col_pooling_sequence & {:keys [overlapping name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "FractionalAvgPoolGrad" [orig_input_tensor_shape out_backprop row_pooling_sequence col_pooling_sequence] {:overlapping overlapping :name name }))

(defn FractionalMaxPool 
  "Performs fractional max pooling on the input.

  Fractional max pooling is slightly different than regular max pooling.  In
  regular max pooling, you downsize an input set by taking the maximum value of
  smaller N x N subsections of the set (often 2x2), and try to reduce the set by
  a factor of N, where N is an integer.  Fractional max pooling, as you might
  expect from the word \"fractional\", means that the overall reduction ratio N
  does not have to be an integer.

  The sizes of the pooling regions are generated randomly but are fairly uniform.
  For example, let's look at the height dimension, and the constraints on the
  list of rows that will be pool boundaries.

  First we define the following:

  1.  input_row_length : the number of rows from the input set
  2.  output_row_length : which will be smaller than the input
  3.  alpha = input_row_length / output_row_length : our reduction ratio
  4.  K = floor(alpha)
  5.  row_pooling_sequence : this is the result list of pool boundary rows

  Then, row_pooling_sequence should satisfy:

  1.  a[0] = 0 : the first value of the sequence is 0
  2.  a[end] = input_row_length : the last value of the sequence is the size
  3.  K <= (a[i+1] - a[i]) <= K+1 : all intervals are K or K+1 size
  4.  length(row_pooling_sequence) = output_row_length+1

  For more details on fractional max pooling, see this paper:
  [Benjamin Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071)

  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      4-D with shape `[batch, height, width, channels]`.
    pooling_ratio: A list of `floats` that has length `>= 4`.
      Pooling ratio for each dimension of `value`, currently only
      supports row and col dimension and should be >= 1.0. For example, a valid
      pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements
      must be 1.0 because we don't allow pooling on batch and channels
      dimensions. 1.44 and 1.73 are pooling ratio on height and width dimensions
      respectively.
    pseudo_random: An optional `bool`. Defaults to `False`.
      When set to True, generates the pooling sequence in a
      pseudorandom fashion, otherwise, in a random fashion. Check paper [Benjamin
      Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071) for
      difference between pseudorandom and random.
    overlapping: An optional `bool`. Defaults to `False`.
      When set to True, it means when pooling, the values at the boundary
      of adjacent pooling cells are used by both cells. For example:

      `index  0  1  2  3  4`

      `value  20 5  16 3  7`

      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
      The result would be [20, 16] for fractional max pooling.
    deterministic: An optional `bool`. Defaults to `False`.
      When set to True, a fixed pooling region will be used when
      iterating over a FractionalMaxPool node in the computation graph. Mainly used
      in unit test to make FractionalMaxPool deterministic.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, row_pooling_sequence, col_pooling_sequence).

    output: A `Tensor`. Has the same type as `value`.
    row_pooling_sequence: A `Tensor` of type `int64`.
    col_pooling_sequence: A `Tensor` of type `int64`.
  "
  [value pooling_ratio & {:keys [pseudo_random overlapping deterministic seed seed2 name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "FractionalMaxPool" [value pooling_ratio] {:pseudo_random pseudo_random :overlapping overlapping :deterministic deterministic :seed seed :seed2 seed2 :name name }))

(defn FractionalMaxPoolGrad 
  "Computes gradient of the FractionalMaxPool function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      Original input for `fractional_max_pool`
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      Original output for `fractional_max_pool`
    out_backprop: A `Tensor`. Must have the same type as `orig_input`.
      4-D with shape `[batch, height, width, channels]`.  Gradients
      w.r.t. the output of `fractional_max_pool`.
    row_pooling_sequence: A `Tensor` of type `int64`.
      row pooling sequence, form pooling region with
      col_pooling_sequence.
    col_pooling_sequence: A `Tensor` of type `int64`.
      column pooling sequence, form pooling region with
      row_pooling sequence.
    overlapping: An optional `bool`. Defaults to `False`.
      When set to True, it means when pooling, the values at the boundary
      of adjacent pooling cells are used by both cells. For example:

      `index  0  1  2  3  4`

      `value  20 5  16 3  7`

      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
      The result would be [20, 16] for fractional max pooling.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  "
  [orig_input orig_output out_backprop row_pooling_sequence col_pooling_sequence & {:keys [overlapping name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "FractionalMaxPoolGrad" [orig_input orig_output out_backprop row_pooling_sequence col_pooling_sequence] {:overlapping overlapping :name name }))

(defn FusedBatchNorm 
  "Batch normalization.

  Note that the size of 4D Tensors are defined by either \"NHWC\" or \"NCHW\".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`.
      A 4D Tensor for input data.
    scale: A `Tensor`. Must have the same type as `x`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    offset: A `Tensor`. Must have the same type as `x`.
      A 1D Tensor for offset, to shift to the normalized x.
    mean: A `Tensor`. Must have the same type as `x`.
      A 1D Tensor for population mean. Used for inference only;
      must be empty for training.
    variance: A `Tensor`. Must have the same type as `x`.
      A 1D Tensor for population variance. Used for inference only;
      must be empty for training.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      The data format for x and y. Either \"NHWC\" (default) or \"NCHW\".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, batch_mean, batch_variance, reserve_space_1, reserve_space_2).

    y: A `Tensor`. Has the same type as `x`.
    batch_mean: A `Tensor`. Has the same type as `x`.
    batch_variance: A `Tensor`. Has the same type as `x`.
    reserve_space_1: A `Tensor`. Has the same type as `x`.
    reserve_space_2: A `Tensor`. Has the same type as `x`.
  "
  [x scale offset mean variance & {:keys [epsilon data_format is_training name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "FusedBatchNorm" [x scale offset mean variance] {:epsilon epsilon :data_format data_format :is_training is_training :name name }))

(defn FusedBatchNormGrad 
  "Gradient for batch normalization.

  Note that the size of 4D Tensors are defined by either \"NHWC\" or \"NCHW\".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    y_backprop: A `Tensor`. Must be one of the following types: `float32`.
      A 4D Tensor for the gradient with respect to y.
    x: A `Tensor`. Must have the same type as `y_backprop`.
      A 4D Tensor for input data.
    scale: A `Tensor`. Must have the same type as `y_backprop`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    reserve_space_1: A `Tensor`. Must have the same type as `y_backprop`.
      When is_training is True, a 1D Tensor for the computed batch
      mean to be reused in gradient computation. When is_training is
      False, a 1D Tensor for the population mean to be reused in both
      1st and 2nd order gradient computation.
    reserve_space_2: A `Tensor`. Must have the same type as `y_backprop`.
      When is_training is True, a 1D Tensor for the computed batch
      variance (inverted variance in the cuDNN case) to be reused in
      gradient computation. When is_training is False, a 1D Tensor
      for the population variance to be reused in both 1st and 2nd
      order gradient computation.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      The data format for y_backprop, x, x_backprop.
      Either \"NHWC\" (default) or \"NCHW\".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (x_backprop, scale_backprop, offset_backprop, reserve_space_3, reserve_space_4).

    x_backprop: A `Tensor`. Has the same type as `y_backprop`.
    scale_backprop: A `Tensor`. Has the same type as `y_backprop`.
    offset_backprop: A `Tensor`. Has the same type as `y_backprop`.
    reserve_space_3: A `Tensor`. Has the same type as `y_backprop`.
    reserve_space_4: A `Tensor`. Has the same type as `y_backprop`.
  "
  [y_backprop x scale reserve_space_1 reserve_space_2 & {:keys [epsilon data_format is_training name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "FusedBatchNormGrad" [y_backprop x scale reserve_space_1 reserve_space_2] {:epsilon epsilon :data_format data_format :is_training is_training :name name }))

(defn FusedBatchNormGradV2 
  "Gradient for batch normalization.

  Note that the size of 4D Tensors are defined by either \"NHWC\" or \"NCHW\".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    y_backprop: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      A 4D Tensor for the gradient with respect to y.
    x: A `Tensor`. Must have the same type as `y_backprop`.
      A 4D Tensor for input data.
    scale: A `Tensor` of type `float32`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    reserve_space_1: A `Tensor`. Must be one of the following types: `float32`.
      When is_training is True, a 1D Tensor for the computed batch
      mean to be reused in gradient computation. When is_training is
      False, a 1D Tensor for the population mean to be reused in both
      1st and 2nd order gradient computation.
    reserve_space_2: A `Tensor`. Must have the same type as `reserve_space_1`.
      When is_training is True, a 1D Tensor for the computed batch
      variance (inverted variance in the cuDNN case) to be reused in
      gradient computation. When is_training is False, a 1D Tensor
      for the population variance to be reused in both 1st and 2nd
      order gradient computation.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      The data format for y_backprop, x, x_backprop.
      Either \"NHWC\" (default) or \"NCHW\".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (x_backprop, scale_backprop, offset_backprop, reserve_space_3, reserve_space_4).

    x_backprop: A `Tensor`. Has the same type as `y_backprop`.
    scale_backprop: A `Tensor`. Has the same type as `reserve_space_1`.
    offset_backprop: A `Tensor`. Has the same type as `reserve_space_1`.
    reserve_space_3: A `Tensor`. Has the same type as `reserve_space_1`.
    reserve_space_4: A `Tensor`. Has the same type as `reserve_space_1`.
  "
  [y_backprop x scale reserve_space_1 reserve_space_2 & {:keys [epsilon data_format is_training name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "FusedBatchNormGradV2" [y_backprop x scale reserve_space_1 reserve_space_2] {:epsilon epsilon :data_format data_format :is_training is_training :name name }))

(defn FusedBatchNormGradV3 
  "Gradient for batch normalization.

  Note that the size of 4D Tensors are defined by either \"NHWC\" or \"NCHW\".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    y_backprop: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      A 4D Tensor for the gradient with respect to y.
    x: A `Tensor`. Must have the same type as `y_backprop`.
      A 4D Tensor for input data.
    scale: A `Tensor` of type `float32`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    reserve_space_1: A `Tensor`. Must be one of the following types: `float32`.
      When is_training is True, a 1D Tensor for the computed batch
      mean to be reused in gradient computation. When is_training is
      False, a 1D Tensor for the population mean to be reused in both
      1st and 2nd order gradient computation.
    reserve_space_2: A `Tensor`. Must have the same type as `reserve_space_1`.
      When is_training is True, a 1D Tensor for the computed batch
      variance (inverted variance in the cuDNN case) to be reused in
      gradient computation. When is_training is False, a 1D Tensor
      for the population variance to be reused in both 1st and 2nd
      order gradient computation.
    reserve_space_3: A `Tensor`. Must have the same type as `reserve_space_1`.
      When is_training is True, a 1D Tensor for some intermediate results to be reused
      in gradient computation. When is_training is False, a dummy empty Tensor will be
      created.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      The data format for y_backprop, x, x_backprop.
      Either \"NHWC\" (default) or \"NCHW\".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (x_backprop, scale_backprop, offset_backprop, reserve_space_4, reserve_space_5).

    x_backprop: A `Tensor`. Has the same type as `y_backprop`.
    scale_backprop: A `Tensor`. Has the same type as `reserve_space_1`.
    offset_backprop: A `Tensor`. Has the same type as `reserve_space_1`.
    reserve_space_4: A `Tensor`. Has the same type as `reserve_space_1`.
    reserve_space_5: A `Tensor`. Has the same type as `reserve_space_1`.
  "
  [y_backprop x scale reserve_space_1 reserve_space_2 reserve_space_3 & {:keys [epsilon data_format is_training name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "FusedBatchNormGradV3" [y_backprop x scale reserve_space_1 reserve_space_2 reserve_space_3] {:epsilon epsilon :data_format data_format :is_training is_training :name name }))

(defn FusedBatchNormV2 
  "Batch normalization.

  Note that the size of 4D Tensors are defined by either \"NHWC\" or \"NCHW\".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      A 4D Tensor for input data.
    scale: A `Tensor`. Must be one of the following types: `float32`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    offset: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for offset, to shift to the normalized x.
    mean: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for population mean. Used for inference only;
      must be empty for training.
    variance: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for population variance. Used for inference only;
      must be empty for training.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      The data format for x and y. Either \"NHWC\" (default) or \"NCHW\".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, batch_mean, batch_variance, reserve_space_1, reserve_space_2).

    y: A `Tensor`. Has the same type as `x`.
    batch_mean: A `Tensor`. Has the same type as `scale`.
    batch_variance: A `Tensor`. Has the same type as `scale`.
    reserve_space_1: A `Tensor`. Has the same type as `scale`.
    reserve_space_2: A `Tensor`. Has the same type as `scale`.
  "
  [x scale offset mean variance & {:keys [epsilon data_format is_training name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "FusedBatchNormV2" [x scale offset mean variance] {:epsilon epsilon :data_format data_format :is_training is_training :name name }))

(defn FusedBatchNormV3 
  "Batch normalization.

  Note that the size of 4D Tensors are defined by either \"NHWC\" or \"NCHW\".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      A 4D Tensor for input data.
    scale: A `Tensor`. Must be one of the following types: `float32`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    offset: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for offset, to shift to the normalized x.
    mean: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for population mean. Used for inference only;
      must be empty for training.
    variance: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for population variance. Used for inference only;
      must be empty for training.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      The data format for x and y. Either \"NHWC\" (default) or \"NCHW\".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, batch_mean, batch_variance, reserve_space_1, reserve_space_2, reserve_space_3).

    y: A `Tensor`. Has the same type as `x`.
    batch_mean: A `Tensor`. Has the same type as `scale`.
    batch_variance: A `Tensor`. Has the same type as `scale`.
    reserve_space_1: A `Tensor`. Has the same type as `scale`.
    reserve_space_2: A `Tensor`. Has the same type as `scale`.
    reserve_space_3: A `Tensor`. Has the same type as `scale`.
  "
  [x scale offset mean variance & {:keys [epsilon data_format is_training name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "FusedBatchNormV3" [x scale offset mean variance] {:epsilon epsilon :data_format data_format :is_training is_training :name name }))

(defn FusedPadConv2D 
  "Performs a padding as a preprocess during a convolution.

  Similar to FusedResizeAndPadConv2d, this op allows for an optimized
  implementation where the spatial padding transformation stage is fused with the
  im2col lookup, but in this case without the bilinear filtering required for
  resizing. Fusing the padding prevents the need to write out the intermediate
  results as whole tensors, reducing memory pressure, and we can get some latency
  gains by merging the transformation calculations.
  The data_format attribute for Conv2D isn't supported by this op, and 'NHWC'
  order is used instead.
  Internally this op uses a single per-graph scratch buffer, which means that it
  will block if multiple versions are being run in parallel. This is because this
  operator is primarily an optimization to minimize memory usage.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
    paddings: A `Tensor` of type `int32`.
      A two-column matrix specifying the padding sizes. The number of
      rows must be the same as the rank of `input`.
    filter: A `Tensor`. Must have the same type as `input`. 4-D with shape
      `[filter_height, filter_width, in_channels, out_channels]`.
    mode: A `string` from: `\"REFLECT\", \"SYMMETRIC\"`.
    strides: A list of `ints`.
      1-D of length 4.  The stride of the sliding window for each dimension
      of `input`. Must be in the same order as the dimension specified with format.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input paddings filter mode strides padding name ]
  (py/call-attr quantization "FusedPadConv2D"  input paddings filter mode strides padding name ))

(defn FusedResizeAndPadConv2D 
  "Performs a resize and padding as a preprocess during a convolution.

  It's often possible to do spatial transformations more efficiently as part of
  the packing stage of a convolution, so this op allows for an optimized
  implementation where these stages are fused together. This prevents the need to
  write out the intermediate results as whole tensors, reducing memory pressure,
  and we can get some latency gains by merging the transformation calculations.
  The data_format attribute for Conv2D isn't supported by this op, and defaults to
  'NHWC' order.
  Internally this op uses a single per-graph scratch buffer, which means that it
  will block if multiple versions are being run in parallel. This is because this
  operator is primarily an optimization to minimize memory usage.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
    size: A `Tensor` of type `int32`.
      A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
      new size for the images.
    paddings: A `Tensor` of type `int32`.
      A two-column matrix specifying the padding sizes. The number of
      rows must be the same as the rank of `input`.
    filter: A `Tensor`. Must have the same type as `input`. 4-D with shape
      `[filter_height, filter_width, in_channels, out_channels]`.
    mode: A `string` from: `\"REFLECT\", \"SYMMETRIC\"`.
    strides: A list of `ints`.
      1-D of length 4.  The stride of the sliding window for each dimension
      of `input`. Must be in the same order as the dimension specified with format.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    resize_align_corners: An optional `bool`. Defaults to `False`.
      If true, the centers of the 4 corner pixels of the input and output tensors are
      aligned, preserving the values at the corner pixels. Defaults to false.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input size paddings filter mode strides padding & {:keys [resize_align_corners name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "FusedResizeAndPadConv2D" [input size paddings filter mode strides padding] {:resize_align_corners resize_align_corners :name name }))

(defn Greater 
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
  (py/call-attr quantization "Greater"  x y name ))

(defn GreaterEqual 
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
  (py/call-attr quantization "GreaterEqual"  x y name ))

(defn HistogramFixedWidth 
  "Return histogram of values.

  Given the tensor `values`, this operation returns a rank 1 histogram counting
  the number of entries in `values` that fall into every bin.  The bins are
  equal width and determined by the arguments `value_range` and `nbins`.

  ```python
  # Bins will be:  (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
  nbins = 5
  value_range = [0.0, 5.0]
  new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]

  with tf.get_default_session() as sess:
    hist = tf.histogram_fixed_width(new_values, value_range, nbins=5)
    variables.global_variables_initializer().run()
    sess.run(hist) => [2, 1, 1, 0, 2]
  ```

  Args:
    values: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      Numeric `Tensor`.
    value_range: A `Tensor`. Must have the same type as `values`.
      Shape [2] `Tensor` of same `dtype` as `values`.
      values <= value_range[0] will be mapped to hist[0],
      values >= value_range[1] will be mapped to hist[-1].
    nbins: A `Tensor` of type `int32`.
      Scalar `int32 Tensor`.  Number of histogram bins.
    dtype: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  "
  [values value_range nbins & {:keys [dtype name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "HistogramFixedWidth" [values value_range nbins] {:dtype dtype :name name }))

(defn Igamma 
  "Compute the lower regularized incomplete Gamma function `P(a, x)`.

  The lower regularized incomplete Gamma function is defined as:


  \\(P(a, x) = gamma(a, x) / Gamma(a) = 1 - Q(a, x)\\)

  where

  \\(gamma(a, x) = \\int_{0}^{x} t^{a-1} exp(-t) dt\\)

  is the lower incomplete Gamma function.

  Note, above `Q(a, x)` (`Igammac`) is the upper regularized complete
  Gamma function.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    x: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  "
  [ a x name ]
  (py/call-attr quantization "Igamma"  a x name ))

(defn IgammaGradA 
  "Computes the gradient of `igamma(a, x)` wrt `a`.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    x: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  "
  [ a x name ]
  (py/call-attr quantization "IgammaGradA"  a x name ))

(defn Igammac 
  "Compute the upper regularized incomplete Gamma function `Q(a, x)`.

  The upper regularized incomplete Gamma function is defined as:

  \\(Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)\\)

  where

  \\(Gamma(a, x) = int_{x}^{\infty} t^{a-1} exp(-t) dt\\)

  is the upper incomplete Gama function.

  Note, above `P(a, x)` (`Igamma`) is the lower regularized complete
  Gamma function.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    x: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  "
  [ a x name ]
  (py/call-attr quantization "Igammac"  a x name ))

(defn Imag 
  "Returns the imaginary part of a complex number.

  Given a tensor `input` of complex numbers, this operation returns a tensor of
  type `float` that is the imaginary part of each element in `input`. All
  elements in `input` must be complex numbers of the form \\(a + bj\\), where *a*
  is the real part and *b* is the imaginary part returned by this operation.

  For example:

  ```
  # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.imag(input) ==> [4.75, 5.75]
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
    Tout: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  "
  [input & {:keys [Tout name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Imag" [input] {:Tout Tout :name name }))

(defn InTopK 
  "Says whether the targets are in the top `K` predictions.

  This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
  prediction for the target class is among the top `k` predictions among
  all predictions for example `i`. Note that the behavior of `InTopK` differs
  from the `TopK` op in its handling of ties; if multiple classes have the
  same prediction value and straddle the top-`k` boundary, all of those
  classes are considered to be in the top `k`.

  More formally, let

    \\(predictions_i\\) be the predictions for all classes for example `i`,
    \\(targets_i\\) be the target class for example `i`,
    \\(out_i\\) be the output for example `i`,

  $$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$

  Args:
    predictions: A `Tensor` of type `float32`.
      A `batch_size` x `classes` tensor.
    targets: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `batch_size` vector of class ids.
    k: An `int`. Number of top elements to look at for computing precision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ predictions targets k name ]
  (py/call-attr quantization "InTopK"  predictions targets k name ))

(defn InTopKV2 
  "Says whether the targets are in the top `K` predictions.

  This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
  prediction for the target class is among the top `k` predictions among
  all predictions for example `i`. Note that the behavior of `InTopK` differs
  from the `TopK` op in its handling of ties; if multiple classes have the
  same prediction value and straddle the top-`k` boundary, all of those
  classes are considered to be in the top `k`.

  More formally, let

    \\(predictions_i\\) be the predictions for all classes for example `i`,
    \\(targets_i\\) be the target class for example `i`,
    \\(out_i\\) be the output for example `i`,

  $$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$

  Args:
    predictions: A `Tensor` of type `float32`.
      A `batch_size` x `classes` tensor.
    targets: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `batch_size` vector of class ids.
    k: A `Tensor`. Must have the same type as `targets`.
      Number of top elements to look at for computing precision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ predictions targets k name ]
  (py/call-attr quantization "InTopKV2"  predictions targets k name ))

(defn Inv 
  "Computes the reciprocal of x element-wise.

  I.e., \\(y = 1 / x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Inv"  x name ))

(defn InvGrad 
  "Computes the gradient for the inverse of `x` wrt its input.

  Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
  is the corresponding input gradient.

  Args:
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    dy: A `Tensor`. Must have the same type as `y`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `y`.
  "
  [ y dy name ]
  (py/call-attr quantization "InvGrad"  y dy name ))

(defn IsFinite 
  "Returns which elements of x are finite.

  @compatibility(numpy)
  Equivalent to np.isfinite
  @end_compatibility

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ x name ]
  (py/call-attr quantization "IsFinite"  x name ))

(defn IsInf 
  "Returns which elements of x are Inf.

  @compatibility(numpy)
  Equivalent to np.isinf
  @end_compatibility

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ x name ]
  (py/call-attr quantization "IsInf"  x name ))

(defn IsNan 
  "Returns which elements of x are NaN.

  @compatibility(numpy)
  Equivalent to np.isnan
  @end_compatibility

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ x name ]
  (py/call-attr quantization "IsNan"  x name ))

(defn L2Loss 
  "L2 Loss.

  Computes half the L2 norm of a tensor without the `sqrt`:

      output = sum(t ** 2) / 2

  Args:
    t: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      Typically 2-D, but may have any dimensions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `t`.
  "
  [ t name ]
  (py/call-attr quantization "L2Loss"  t name ))

(defn LRN 
  "Local Response Normalization.

  The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
  dimension), and each vector is normalized independently.  Within a given vector,
  each component is divided by the weighted, squared sum of inputs within
  `depth_radius`.  In detail,

      sqr_sum[a, b, c, d] =
          sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
      output = input / (bias + alpha * sqr_sum) ** beta

  For details, see [Krizhevsky et al., ImageNet classification with deep
  convolutional neural networks (NIPS 2012)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      4-D.
    depth_radius: An optional `int`. Defaults to `5`.
      0-D.  Half-width of the 1-D normalization window.
    bias: An optional `float`. Defaults to `1`.
      An offset (usually positive to avoid dividing by 0).
    alpha: An optional `float`. Defaults to `1`.
      A scale factor, usually positive.
    beta: An optional `float`. Defaults to `0.5`. An exponent.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input & {:keys [depth_radius bias alpha beta name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "LRN" [input] {:depth_radius depth_radius :bias bias :alpha alpha :beta beta :name name }))

(defn LRNGrad 
  "Gradients for Local Response Normalization.

  Args:
    input_grads: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      4-D with shape `[batch, height, width, channels]`.
    input_image: A `Tensor`. Must have the same type as `input_grads`.
      4-D with shape `[batch, height, width, channels]`.
    output_image: A `Tensor`. Must have the same type as `input_grads`.
      4-D with shape `[batch, height, width, channels]`.
    depth_radius: An optional `int`. Defaults to `5`. A depth radius.
    bias: An optional `float`. Defaults to `1`.
      An offset (usually > 0 to avoid dividing by 0).
    alpha: An optional `float`. Defaults to `1`.
      A scale factor, usually positive.
    beta: An optional `float`. Defaults to `0.5`. An exponent.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input_grads`.
  "
  [input_grads input_image output_image & {:keys [depth_radius bias alpha beta name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "LRNGrad" [input_grads input_image output_image] {:depth_radius depth_radius :bias bias :alpha alpha :beta beta :name name }))

(defn LeakyRelu 
  "Computes rectified linear: `max(features, features * alpha)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    alpha: An optional `float`. Defaults to `0.2`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  "
  [features & {:keys [alpha name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "LeakyRelu" [features] {:alpha alpha :name name }))

(defn LeakyReluGrad 
  "Computes rectified linear gradients for a LeakyRelu operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      The backpropagated gradients to the corresponding LeakyRelu operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding LeakyRelu operation,
      OR the outputs of that operation (both work equivalently).
    alpha: An optional `float`. Defaults to `0.2`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  "
  [gradients features & {:keys [alpha name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "LeakyReluGrad" [gradients features] {:alpha alpha :name name }))

(defn Less 
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
  (py/call-attr quantization "Less"  x y name ))

(defn LessEqual 
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
  (py/call-attr quantization "LessEqual"  x y name ))

(defn Lgamma 
  "Computes the log of the absolute value of `Gamma(x)` element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Lgamma"  x name ))

(defn LinSpace 
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
  (py/call-attr quantization "LinSpace"  start stop num name ))

(defn Log 
  "Computes natural logarithm of x element-wise.

  I.e., \\(y = \log_e x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Log"  x name ))

(defn Log1p 
  "Computes natural logarithm of (1 + x) element-wise.

  I.e., \\(y = \log_e (1 + x)\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Log1p"  x name ))

(defn LogSoftmax 
  "Computes log softmax activations.

  For each batch `i` and class `j` we have

      logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))

  Args:
    logits: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      2-D with shape `[batch_size, num_classes]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `logits`.
  "
  [ logits name ]
  (py/call-attr quantization "LogSoftmax"  logits name ))

(defn LogicalAnd 
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
  (py/call-attr quantization "LogicalAnd"  x y name ))

(defn LogicalNot 
  "Returns the truth value of NOT x element-wise.

  Args:
    x: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ x name ]
  (py/call-attr quantization "LogicalNot"  x name ))

(defn LogicalOr 
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
  (py/call-attr quantization "LogicalOr"  x y name ))

(defn MatMul 
  "Multiply the matrix \"a\" by the matrix \"b\".

  The inputs must be two-dimensional matrices and the inner dimension of
  \"a\" (after being transposed if transpose_a is true) must match the
  outer dimension of \"b\" (after being transposed if transposed_b is
  true).

  *Note*: The default kernel implementation for MatMul on GPUs uses
  cublas.

  Args:
    a: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    b: A `Tensor`. Must have the same type as `a`.
    transpose_a: An optional `bool`. Defaults to `False`.
      If true, \"a\" is transposed before multiplication.
    transpose_b: An optional `bool`. Defaults to `False`.
      If true, \"b\" is transposed before multiplication.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  "
  [a b & {:keys [transpose_a transpose_b name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "MatMul" [a b] {:transpose_a transpose_a :transpose_b transpose_b :name name }))

(defn Max 
  "Computes the maximum of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The tensor to reduce.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce. Must be in the range
      `[-rank(input), rank(input))`.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input axis & {:keys [keep_dims name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Max" [input axis] {:keep_dims keep_dims :name name }))

(defn MaxPool 
  "Performs max pooling on the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `qint8`.
      4-D input to pool over.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\", \"NCHW_VECT_C\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "MaxPool" [input ksize strides padding] {:data_format data_format :name name }))

(defn MaxPool3D 
  "Performs 3D max pooling on the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NDHWC\", \"NCDHW\"`. Defaults to `\"NDHWC\"`.
      The data format of the input and output data. With the
      default format \"NDHWC\", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCDHW\", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "MaxPool3D" [input ksize strides padding] {:data_format data_format :name name }))

(defn MaxPool3DGrad 
  "Computes gradients of max pooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      Output backprop of shape `[batch, depth, rows, cols, channels]`.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NDHWC\", \"NCDHW\"`. Defaults to `\"NDHWC\"`.
      The data format of the input and output data. With the
      default format \"NDHWC\", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCDHW\", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  "
  [orig_input orig_output grad ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "MaxPool3DGrad" [orig_input orig_output grad ksize strides padding] {:data_format data_format :name name }))

(defn MaxPool3DGradGrad 
  "Computes second-order gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must have the same type as `orig_input`.
      Output backprop of shape `[batch, depth, rows, cols, channels]`.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NDHWC\", \"NCDHW\"`. Defaults to `\"NDHWC\"`.
      The data format of the input and output data. With the
      default format \"NDHWC\", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCDHW\", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  "
  [orig_input orig_output grad ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "MaxPool3DGradGrad" [orig_input orig_output grad ksize strides padding] {:data_format data_format :name name }))

(defn MaxPoolGrad 
  "Computes gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must have the same type as `orig_input`.
      4-D.  Gradients w.r.t. the output of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  "
  [orig_input orig_output grad ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "MaxPoolGrad" [orig_input orig_output grad ksize strides padding] {:data_format data_format :name name }))

(defn MaxPoolGradGrad 
  "Computes second-order gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must have the same type as `orig_input`.
      4-D.  Gradients of gradients w.r.t. the input of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  "
  [orig_input orig_output grad ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "MaxPoolGradGrad" [orig_input orig_output grad ksize strides padding] {:data_format data_format :name name }))

(defn MaxPoolGradGradV2 
  "Computes second-order gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must have the same type as `orig_input`.
      4-D.  Gradients of gradients w.r.t. the input of `max_pool`.
    ksize: A `Tensor` of type `int32`.
      The size of the window for each dimension of the input tensor.
    strides: A `Tensor` of type `int32`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  "
  [orig_input orig_output grad ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "MaxPoolGradGradV2" [orig_input orig_output grad ksize strides padding] {:data_format data_format :name name }))

(defn MaxPoolGradGradWithArgmax 
  "Computes second-order gradients of the maxpooling function.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input.
    grad: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the
      input of `max_pool`.
    argmax: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The indices of the maximum values chosen for each output of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    include_batch_in_index: An optional `bool`. Defaults to `False`.
      Whether to include batch dimension in flattened index of `argmax`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input grad argmax ksize strides padding & {:keys [include_batch_in_index name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "MaxPoolGradGradWithArgmax" [input grad argmax ksize strides padding] {:include_batch_in_index include_batch_in_index :name name }))

(defn MaxPoolGradV2 
  "Computes gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must have the same type as `orig_input`.
      4-D.  Gradients w.r.t. the output of `max_pool`.
    ksize: A `Tensor` of type `int32`.
      The size of the window for each dimension of the input tensor.
    strides: A `Tensor` of type `int32`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  "
  [orig_input orig_output grad ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "MaxPoolGradV2" [orig_input orig_output grad ksize strides padding] {:data_format data_format :name name }))

(defn MaxPoolGradWithArgmax 
  "Computes gradients of the maxpooling function.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input.
    grad: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the
      output of `max_pool`.
    argmax: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The indices of the maximum values chosen for each output of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    include_batch_in_index: An optional `bool`. Defaults to `False`.
      Whether to include batch dimension in flattened index of `argmax`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input grad argmax ksize strides padding & {:keys [include_batch_in_index name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "MaxPoolGradWithArgmax" [input grad argmax ksize strides padding] {:include_batch_in_index include_batch_in_index :name name }))

(defn MaxPoolV2 
  "Performs max pooling on the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `qint8`.
      4-D input to pool over.
    ksize: A `Tensor` of type `int32`.
      The size of the window for each dimension of the input tensor.
    strides: A `Tensor` of type `int32`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\", \"NCHW_VECT_C\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "MaxPoolV2" [input ksize strides padding] {:data_format data_format :name name }))

(defn MaxPoolWithArgmax 
  "Performs max pooling on the input and outputs both max values and indices.

  The indices in `argmax` are flattened, so that a maximum value at position
  `[b, y, x, c]` becomes flattened index:
  `(y * width + x) * channels + c` if `include_batch_in_index` is False;
  `((b * height + y) * width + x) * channels + c` if `include_batch_in_index` is True.

  The indices returned are always in `[0, height) x [0, width)` before flattening,
  even if padding is involved and the mathematically correct answer is outside
  (either negative or too large).  This is a bug, but fixing it is difficult to do
  in a safe backwards compatible way, especially due to flattening.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      4-D with shape `[batch, height, width, channels]`.  Input to pool over.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    Targmax: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
    include_batch_in_index: An optional `bool`. Defaults to `False`.
      Whether to include batch dimension in flattened index of `argmax`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, argmax).

    output: A `Tensor`. Has the same type as `input`.
    argmax: A `Tensor` of type `Targmax`.
  "
  [input ksize strides padding & {:keys [Targmax include_batch_in_index name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "MaxPoolWithArgmax" [input ksize strides padding] {:Targmax Targmax :include_batch_in_index include_batch_in_index :name name }))

(defn Maximum 
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
  (py/call-attr quantization "Maximum"  x y name ))

(defn Mean 
  "Computes the mean of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The tensor to reduce.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce. Must be in the range
      `[-rank(input), rank(input))`.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input axis & {:keys [keep_dims name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Mean" [input axis] {:keep_dims keep_dims :name name }))

(defn Min 
  "Computes the minimum of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The tensor to reduce.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce. Must be in the range
      `[-rank(input), rank(input))`.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input axis & {:keys [keep_dims name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Min" [input axis] {:keep_dims keep_dims :name name }))

(defn Minimum 
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
  (py/call-attr quantization "Minimum"  x y name ))

(defn Mod 
  "Returns element-wise remainder of division. This emulates C semantics in that

  the result here is consistent with a truncating divide. E.g.
  `tf.truncatediv(x, y) * y + truncate_mod(x, y) = x`.

  *NOTE*: `Mod` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`, `half`, `half`, `bfloat16`, `float32`, `float64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "Mod"  x y name ))

(defn Mul 
  "Returns x * y element-wise.

  *NOTE*: `Multiply` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "Mul"  x y name ))

(defn MulNoNan 
  "Returns x * y element-wise. Returns zero if y is zero, even if x if infinite or NaN.

  *NOTE*: `MulNoNan` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "MulNoNan"  x y name ))

(defn Neg 
  "Computes numerical negative value element-wise.

  I.e., \\(y = -x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Neg"  x name ))

(defn NextAfter 
  "Returns the next representable value of `x1` in the direction of `x2`, element-wise.

  This operation returns the same result as the C++ std::nextafter function.

  It can also return a subnormal number.

  @compatibility(cpp)
  Equivalent to C++ std::nextafter function.
  @end_compatibility

  Args:
    x1: A `Tensor`. Must be one of the following types: `float64`, `float32`.
    x2: A `Tensor`. Must have the same type as `x1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x1`.
  "
  [ x1 x2 name ]
  (py/call-attr quantization "NextAfter"  x1 x2 name ))

(defn NotEqual 
  "Returns the truth value of (x != y) element-wise.

  *NOTE*: `NotEqual` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `quint8`, `qint8`, `qint32`, `string`, `bool`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    incompatible_shape_error: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [x y & {:keys [incompatible_shape_error name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "NotEqual" [x y] {:incompatible_shape_error incompatible_shape_error :name name }))

(defn NthElement 
  "Finds values of the `n`-th order statistic for the last dimension.

  If the input is a vector (rank-1), finds the entries which is the nth-smallest
  value in the vector and outputs their values as scalar tensor.

  For matrices (resp. higher rank input), computes the entries which is the
  nth-smallest value in each row (resp. vector along the last dimension). Thus,

      values.shape = input.shape[:-1]

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      1-D or higher with last dimension at least `n+1`.
    n: A `Tensor` of type `int32`.
      0-D. Position of sorted vector to select along the last dimension (along
      each row for matrices). Valid range of n is `[0, input.shape[:-1])`
    reverse: An optional `bool`. Defaults to `False`.
      When set to True, find the nth-largest value in the vector and vice
      versa.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input n & {:keys [reverse name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "NthElement" [input n] {:reverse reverse :name name }))

(defn Polygamma 
  "Compute the polygamma function \\(\psi^{(n)}(x)\\).

  The polygamma function is defined as:


  \\(\psi^{(a)}(x) = \frac{d^a}{dx^a} \psi(x)\\)

  where \\(\psi(x)\\) is the digamma function.
  The polygamma function is defined only for non-negative integer orders \\a\\.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    x: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  "
  [ a x name ]
  (py/call-attr quantization "Polygamma"  a x name ))

(defn Pow 
  "Computes the power of one value to another.

  Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
  corresponding elements in `x` and `y`. For example:

  ```
  # tensor 'x' is [[2, 2]], [3, 3]]
  # tensor 'y' is [[8, 16], [2, 3]]
  tf.pow(x, y) ==> [[256, 65536], [9, 27]]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `float32`, `half`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "Pow"  x y name ))

(defn Prod 
  "Computes the product of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The tensor to reduce.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce. Must be in the range
      `[-rank(input), rank(input))`.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input axis & {:keys [keep_dims name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Prod" [input axis] {:keep_dims keep_dims :name name }))

(defn QuantizeDownAndShrinkRange 
  "Convert the quantized 'input' tensor into a lower-precision 'output', using the

  actual distribution of the values to maximize the usage of the lower bit depth
  and adjusting the output min and max ranges accordingly.

  [input_min, input_max] are scalar floats that specify the range for the float
  interpretation of the 'input' data. For example, if input_min is -1.0f and
  input_max is 1.0f, and we are dealing with quint16 quantized data, then a 0
  value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.

  This operator tries to squeeze as much precision as possible into an output with
  a lower bit depth by calculating the actual min and max values found in the
  data. For example, maybe that quint16 input has no values lower than 16,384 and
  none higher than 49,152. That means only half the range is actually needed, all
  the float interpretations are between -0.5f and 0.5f, so if we want to compress
  the data into a quint8 output, we can use that range rather than the theoretical
  -1.0f to 1.0f that is suggested by the input min and max.

  In practice, this is most useful for taking output from operations like
  QuantizedMatMul that can produce higher bit-depth outputs than their inputs and
  may have large potential output ranges, but in practice have a distribution of
  input values that only uses a small fraction of the possible range. By feeding
  that output into this operator, we can reduce it from 32 bits down to 8 with
  minimal loss of accuracy.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    input_min: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    input_max: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`.
      The type of the output. Should be a lower bit depth than Tinput.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).

    output: A `Tensor` of type `out_type`.
    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  "
  [ input input_min input_max out_type name ]
  (py/call-attr quantization "QuantizeDownAndShrinkRange"  input input_min input_max out_type name ))

(defn QuantizedAdd 
  "Returns x + y element-wise, working on quantized buffers.

  Args:
    x: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    y: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_x: A `Tensor` of type `float32`.
      The float value that the lowest quantized `x` value represents.
    max_x: A `Tensor` of type `float32`.
      The float value that the highest quantized `x` value represents.
    min_y: A `Tensor` of type `float32`.
      The float value that the lowest quantized `y` value represents.
    max_y: A `Tensor` of type `float32`.
      The float value that the highest quantized `y` value represents.
    Toutput: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (z, min_z, max_z).

    z: A `Tensor` of type `Toutput`.
    min_z: A `Tensor` of type `float32`.
    max_z: A `Tensor` of type `float32`.
  "
  [x y min_x max_x min_y max_y & {:keys [Toutput name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedAdd" [x y min_x max_x min_y max_y] {:Toutput Toutput :name name }))

(defn QuantizedAvgPool 
  "Produces the average pool of the input tensor for quantized types.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      4-D with shape `[batch, height, width, channels]`.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    ksize: A list of `ints`.
      The size of the window for each dimension of the input tensor.
      The length must be 4 to match the number of dimensions of the input.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      tensor.  The length must be 4 to match the number of dimensions of the input.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor`. Has the same type as `input`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [ input min_input max_input ksize strides padding name ]
  (py/call-attr quantization "QuantizedAvgPool"  input min_input max_input ksize strides padding name ))

(defn QuantizedBatchNormWithGlobalNormalization 
  "Quantized Batch normalization.

  This op is deprecated and will be removed in the future. Prefer
  `tf.nn.batch_normalization`.

  Args:
    t: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A 4D input Tensor.
    t_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized input.
    t_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized input.
    m: A `Tensor`. Must have the same type as `t`.
      A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    m_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized mean.
    m_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized mean.
    v: A `Tensor`. Must have the same type as `t`.
      A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    v_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized variance.
    v_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized variance.
    beta: A `Tensor`. Must have the same type as `t`.
      A 1D beta Tensor with size matching the last dimension of t.
      An offset to be added to the normalized tensor.
    beta_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized offset.
    beta_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized offset.
    gamma: A `Tensor`. Must have the same type as `t`.
      A 1D gamma Tensor with size matching the last dimension of t.
      If \"scale_after_normalization\" is true, this tensor will be multiplied
      with the normalized tensor.
    gamma_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized gamma.
    gamma_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized gamma.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`.
    variance_epsilon: A `float`. A small float number to avoid dividing by 0.
    scale_after_normalization: A `bool`.
      A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (result, result_min, result_max).

    result: A `Tensor` of type `out_type`.
    result_min: A `Tensor` of type `float32`.
    result_max: A `Tensor` of type `float32`.
  "
  [ t t_min t_max m m_min m_max v v_min v_max beta beta_min beta_max gamma gamma_min gamma_max out_type variance_epsilon scale_after_normalization name ]
  (py/call-attr quantization "QuantizedBatchNormWithGlobalNormalization"  t t_min t_max m m_min m_max v v_min v_max beta beta_min beta_max gamma gamma_min gamma_max out_type variance_epsilon scale_after_normalization name ))

(defn QuantizedBiasAdd 
  "Adds Tensor 'bias' to Tensor 'input' for Quantized types.

  Broadcasts the values of bias on dimensions 0..N-2 of 'input'.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A 1D bias Tensor with size matching the last dimension of 'input'.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    min_bias: A `Tensor` of type `float32`.
      The float value that the lowest quantized bias value represents.
    max_bias: A `Tensor` of type `float32`.
      The float value that the highest quantized bias value represents.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_out, max_out).

    output: A `Tensor` of type `out_type`.
    min_out: A `Tensor` of type `float32`.
    max_out: A `Tensor` of type `float32`.
  "
  [ input bias min_input max_input min_bias max_bias out_type name ]
  (py/call-attr quantization "QuantizedBiasAdd"  input bias min_input max_input min_bias max_bias out_type name ))

(defn QuantizedConv2D 
  "Computes a 2D convolution given quantized 4D input and filter tensors.

  The inputs are quantized tensors where the lowest value represents the real
  number of the associated minimum, and the highest represents the maximum.
  This means that you can only interpret the quantized output in the same way, by
  taking the returned minimum and maximum values into account.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      filter's input_depth dimension must match input's depth dimensions.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    min_filter: A `Tensor` of type `float32`.
      The float value that the lowest quantized filter value represents.
    max_filter: A `Tensor` of type `float32`.
      The float value that the highest quantized filter value represents.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      tensor.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each
      filter element on that dimension. The dimension order is determined by the
      value of `data_format`, see above for details. Dilations in the batch and
      depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedConv2D" [input filter min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :name name }))

(defn QuantizedConv2DAndRelu 
  "TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations padding_list name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedConv2DAndRelu" [input filter min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name }))

(defn QuantizedConv2DAndReluAndRequantize 
  "TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding & {:keys [out_type dilations padding_list name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedConv2DAndReluAndRequantize" [input filter min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name }))

(defn QuantizedConv2DAndRequantize 
  "TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint8`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding & {:keys [out_type dilations padding_list name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedConv2DAndRequantize" [input filter min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name }))

(defn QuantizedConv2DPerChannel 
  "Computes QuantizedConv2D per channel.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original filter tensor.
    min_input: A `Tensor` of type `float32`.
      The minimum value of the input tensor
    max_input: A `Tensor` of type `float32`.
      The maximum value of the input tensor.
    min_filter: A `Tensor` of type `float32`.
      The minimum value of the filter tensor.
    max_filter: A `Tensor` of type `float32`.
      The maximum value of the filter tensor.
    strides: A list of `ints`. list of stride values.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
      The quantized type of output tensor that needs to be converted.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      list of dilation values.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedConv2DPerChannel" [input filter min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :name name }))

(defn QuantizedConv2DWithBias 
  "TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor` of type `float32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter bias min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations padding_list name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedConv2DWithBias" [input filter bias min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name }))

(defn QuantizedConv2DWithBiasAndRelu 
  "TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor` of type `float32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter bias min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations padding_list name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedConv2DWithBiasAndRelu" [input filter bias min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name }))

(defn QuantizedConv2DWithBiasAndReluAndRequantize 
  "TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding & {:keys [out_type dilations padding_list name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedConv2DWithBiasAndReluAndRequantize" [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name }))

(defn QuantizedConv2DWithBiasAndRequantize 
  "TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint8`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding & {:keys [out_type dilations padding_list name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedConv2DWithBiasAndRequantize" [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name }))

(defn QuantizedConv2DWithBiasSignedSumAndReluAndRequantize 
  "TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    summand: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_summand: A `Tensor` of type `float32`.
    max_summand: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output summand min_summand max_summand strides padding & {:keys [out_type dilations padding_list name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedConv2DWithBiasSignedSumAndReluAndRequantize" [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output summand min_summand max_summand strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name }))

(defn QuantizedConv2DWithBiasSumAndRelu 
  "TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor` of type `float32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    summand: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter bias min_input max_input min_filter max_filter summand strides padding & {:keys [out_type dilations padding_list name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedConv2DWithBiasSumAndRelu" [input filter bias min_input max_input min_filter max_filter summand strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name }))

(defn QuantizedConv2DWithBiasSumAndReluAndRequantize 
  "TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    summand: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_summand: A `Tensor` of type `float32`.
    max_summand: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output summand min_summand max_summand strides padding & {:keys [out_type dilations padding_list name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedConv2DWithBiasSumAndReluAndRequantize" [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output summand min_summand max_summand strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name }))

(defn QuantizedDepthwiseConv2D 
  "Computes quantized depthwise Conv2D.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original filter tensor.
    min_input: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    min_filter: A `Tensor` of type `float32`.
      The float value that the minimum quantized filter value represents.
    max_filter: A `Tensor` of type `float32`.
      The float value that the maximum quantized filter value represents.
    strides: A list of `ints`. List of stride values.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
      The type of the output.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      List of dilation values.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedDepthwiseConv2D" [input filter min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :name name }))

(defn QuantizedDepthwiseConv2DWithBias 
  "Computes quantized depthwise Conv2D with Bias.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original filter tensor.
    bias: A `Tensor` of type `float32`. The original bias tensor.
    min_input: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    min_filter: A `Tensor` of type `float32`.
      The float value that the minimum quantized filter value represents.
    max_filter: A `Tensor` of type `float32`.
      The float value that the maximum quantized filter value represents.
    strides: A list of `ints`. List of stride values.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
      The type of the output.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      List of dilation values.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter bias min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedDepthwiseConv2DWithBias" [input filter bias min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :name name }))

(defn QuantizedDepthwiseConv2DWithBiasAndRelu 
  "Computes quantized depthwise Conv2D with Bias and Relu.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original filter tensor.
    bias: A `Tensor` of type `float32`. The original bias tensor.
    min_input: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    min_filter: A `Tensor` of type `float32`.
      The float value that the minimum quantized filter value represents.
    max_filter: A `Tensor` of type `float32`.
      The float value that the maximum quantized filter value represents.
    strides: A list of `ints`. List of stride values.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
      The type of the output.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      List of dilation values.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter bias min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedDepthwiseConv2DWithBiasAndRelu" [input filter bias min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :name name }))

(defn QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize 
  "Computes quantized depthwise Conv2D with Bias, Relu and Requantize.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original filter tensor.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
      The original bias tensor.
    min_input: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    min_filter: A `Tensor` of type `float32`.
      The float value that the minimum quantized filter value represents.
    max_filter: A `Tensor` of type `float32`.
      The float value that the maximum quantized filter value represents.
    min_freezed_output: A `Tensor` of type `float32`.
      The minimum float value of the output tensor.
    max_freezed_output: A `Tensor` of type `float32`.
      The maximum float value of the output tensor.
    strides: A list of `ints`. List of stride values.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
      The type of the output.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      List of dilation values.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding & {:keys [out_type dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize" [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding] {:out_type out_type :dilations dilations :name name }))

(defn QuantizedMatMul 
  "Perform a quantized matrix multiplication of  `a` by the matrix `b`.

  The inputs must be two-dimensional matrices and the inner dimension of
  `a` (after being transposed if `transpose_a` is non-zero) must match the
  outer dimension of `b` (after being transposed if `transposed_b` is
  non-zero).

  Args:
    a: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      Must be a two-dimensional tensor.
    b: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      Must be a two-dimensional tensor.
    min_a: A `Tensor` of type `float32`.
      The float value that the lowest quantized `a` value represents.
    max_a: A `Tensor` of type `float32`.
      The float value that the highest quantized `a` value represents.
    min_b: A `Tensor` of type `float32`.
      The float value that the lowest quantized `b` value represents.
    max_b: A `Tensor` of type `float32`.
      The float value that the highest quantized `b` value represents.
    Toutput: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    transpose_a: An optional `bool`. Defaults to `False`.
      If true, `a` is transposed before multiplication.
    transpose_b: An optional `bool`. Defaults to `False`.
      If true, `b` is transposed before multiplication.
    Tactivation: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
      The type of output produced by activation function
      following this operation.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (out, min_out, max_out).

    out: A `Tensor` of type `Toutput`.
    min_out: A `Tensor` of type `float32`.
    max_out: A `Tensor` of type `float32`.
  "
  [a b min_a max_a min_b max_b & {:keys [Toutput transpose_a transpose_b Tactivation name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedMatMul" [a b min_a max_a min_b max_b] {:Toutput Toutput :transpose_a transpose_a :transpose_b transpose_b :Tactivation Tactivation :name name }))

(defn QuantizedMatMulWithBias 
  "Performs a quantized matrix multiplication of `a` by the matrix `b` with bias
add.

  The inputs must be two-dimensional matrices and 1D bias vector. And the inner
  dimension of `a` (after being transposed if `transpose_a` is non-zero) must
  match the outer dimension of `b` (after being transposed if `transposed_b` is
  non-zero). Then do broadcast add operation with bias values on the matrix
  mulplication result. The bias size must match inner dimension of `b`.

  Args:
    a: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A matrix to be multiplied. Must be a two-dimensional tensor of type `quint8`.
    b: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A matrix to be multiplied and must be a two-dimensional tensor of type `qint8`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
      A 1D bias tensor with size matching inner dimension of `b` (after being
      transposed if `transposed_b` is non-zero).
    min_a: A `Tensor` of type `float32`.
      The float value that the lowest quantized `a` value represents.
    max_a: A `Tensor` of type `float32`.
      The float value that the highest quantized `a` value represents.
    min_b: A `Tensor` of type `float32`.
      The float value that the lowest quantized `b` value represents.
    max_b: A `Tensor` of type `float32`.
      The float value that the highest quantized `b` value represents.
    Toutput: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    transpose_a: An optional `bool`. Defaults to `False`.
      If true, `a` is transposed before multiplication.
    transpose_b: An optional `bool`. Defaults to `False`.
      If true, `b` is transposed before multiplication.
    input_quant_mode: An optional `string` from: `\"MIN_FIRST\", \"SCALED\"`. Defaults to `\"MIN_FIRST\"`.
      Input data quantization mode. Either MIN_FIRST(default) or SCALED.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (out, min_out, max_out).

    out: A `Tensor` of type `Toutput`.
    min_out: A `Tensor` of type `float32`.
    max_out: A `Tensor` of type `float32`.
  "
  [a b bias min_a max_a min_b max_b & {:keys [Toutput transpose_a transpose_b input_quant_mode name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedMatMulWithBias" [a b bias min_a max_a min_b max_b] {:Toutput Toutput :transpose_a transpose_a :transpose_b transpose_b :input_quant_mode input_quant_mode :name name }))

(defn QuantizedMatMulWithBiasAndRelu 
  "Perform a quantized matrix multiplication of  `a` by the matrix `b` with bias
add and relu fusion.

  The inputs must be two-dimensional matrices and 1D bias vector. And the inner
  dimension of `a` (after being transposed if `transpose_a` is non-zero) must
  match the outer dimension of `b` (after being transposed if `transposed_b` is
  non-zero). Then do broadcast add operation with bias values on the matrix
  mulplication result. The bias size must match inner dimension of `b`. Then do
  relu activation to get non-negative result.

  Args:
    a: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A matrix to be multiplied. Must be a two-dimensional tensor of type `quint8`.
    b: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A matrix to be multiplied and must be a two-dimensional tensor of type `qint8`.
    bias: A `Tensor` of type `float32`.
      A 1D bias tensor with size matching with inner dimension of `b` (after being
      transposed if `transposed_b` is non-zero).
    min_a: A `Tensor` of type `float32`.
      The float value that the lowest quantized `a` value represents.
    max_a: A `Tensor` of type `float32`.
      The float value that the highest quantized `a` value represents.
    min_b: A `Tensor` of type `float32`.
      The float value that the lowest quantized `b` value represents.
    max_b: A `Tensor` of type `float32`.
      The float value that the highest quantized `b` value represents.
    Toutput: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    transpose_a: An optional `bool`. Defaults to `False`.
      If true, `a` is transposed before multiplication.
    transpose_b: An optional `bool`. Defaults to `False`.
      If true, `b` is transposed before multiplication.
    input_quant_mode: An optional `string` from: `\"MIN_FIRST\", \"SCALED\"`. Defaults to `\"MIN_FIRST\"`.
      Input data quantization mode. Either MIN_FIRST(default) or SCALED.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (out, min_out, max_out).

    out: A `Tensor` of type `Toutput`.
    min_out: A `Tensor` of type `float32`.
    max_out: A `Tensor` of type `float32`.
  "
  [a b bias min_a max_a min_b max_b & {:keys [Toutput transpose_a transpose_b input_quant_mode name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedMatMulWithBiasAndRelu" [a b bias min_a max_a min_b max_b] {:Toutput Toutput :transpose_a transpose_a :transpose_b transpose_b :input_quant_mode input_quant_mode :name name }))

(defn QuantizedMatMulWithBiasAndReluAndRequantize 
  "Perform a quantized matrix multiplication of  `a` by the matrix `b` with bias
add and relu and requantize fusion.

  The inputs must be two-dimensional matrices and 1D bias vector. And the inner
  dimension of `a` (after being transposed if `transpose_a` is non-zero) must
  match the outer dimension of `b` (after being transposed if `transposed_b` is
  non-zero). Then do broadcast add operation with bias values on the matrix
  mulplication result. The bias size must match inner dimension of `b`.  Then do
  relu activation to get non-negative result. Then do requantize operation to get
  final uint8 result.

  Args:
    a: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A matrix to be multiplied. Must be a two-dimensional tensor of type `quint8`.
    b: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A matrix to be multiplied and must be a two-dimensional tensor of type `qint8`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
      A 1D bias tensor with size matching with inner dimension of `b` (after being
      transposed if `transposed_b` is non-zero).
    min_a: A `Tensor` of type `float32`.
      The float value that the lowest quantized `a` value represents.
    max_a: A `Tensor` of type `float32`.
      The float value that the highest quantized `a` value represents.
    min_b: A `Tensor` of type `float32`.
      The float value that the lowest quantized `b` value represents.
    max_b: A `Tensor` of type `float32`.
      The float value that the highest quantized `b` value represents.
    min_freezed_output: A `Tensor` of type `float32`.
      The float value that the highest quantized output value after requantize.
    max_freezed_output: A `Tensor` of type `float32`.
    Toutput: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    transpose_a: An optional `bool`. Defaults to `False`.
      If true, `a` is transposed before multiplication.
    transpose_b: An optional `bool`. Defaults to `False`.
      If true, `b` is transposed before multiplication.
    input_quant_mode: An optional `string` from: `\"MIN_FIRST\", \"SCALED\"`. Defaults to `\"MIN_FIRST\"`.
      Input data quantization mode. Either MIN_FIRST(default) or SCALED.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (out, min_out, max_out).

    out: A `Tensor` of type `Toutput`.
    min_out: A `Tensor` of type `float32`.
    max_out: A `Tensor` of type `float32`.
  "
  [a b bias min_a max_a min_b max_b min_freezed_output max_freezed_output & {:keys [Toutput transpose_a transpose_b input_quant_mode name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedMatMulWithBiasAndReluAndRequantize" [a b bias min_a max_a min_b max_b min_freezed_output max_freezed_output] {:Toutput Toutput :transpose_a transpose_a :transpose_b transpose_b :input_quant_mode input_quant_mode :name name }))

(defn QuantizedMaxPool 
  "Produces the max pool of the input tensor for quantized types.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The 4D (batch x rows x cols x depth) Tensor to MaxReduce over.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    ksize: A list of `ints`.
      The size of the window for each dimension of the input tensor.
      The length must be 4 to match the number of dimensions of the input.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      tensor. The length must be 4 to match the number of dimensions of the input.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor`. Has the same type as `input`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [ input min_input max_input ksize strides padding name ]
  (py/call-attr quantization "QuantizedMaxPool"  input min_input max_input ksize strides padding name ))

(defn QuantizedMul 
  "Returns x * y element-wise, working on quantized buffers.

  Args:
    x: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    y: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_x: A `Tensor` of type `float32`.
      The float value that the lowest quantized `x` value represents.
    max_x: A `Tensor` of type `float32`.
      The float value that the highest quantized `x` value represents.
    min_y: A `Tensor` of type `float32`.
      The float value that the lowest quantized `y` value represents.
    max_y: A `Tensor` of type `float32`.
      The float value that the highest quantized `y` value represents.
    Toutput: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (z, min_z, max_z).

    z: A `Tensor` of type `Toutput`.
    min_z: A `Tensor` of type `float32`.
    max_z: A `Tensor` of type `float32`.
  "
  [x y min_x max_x min_y max_y & {:keys [Toutput name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedMul" [x y min_x max_x min_y max_y] {:Toutput Toutput :name name }))

(defn QuantizedRelu 
  "Computes Quantized Rectified Linear: `max(features, 0)`

  Args:
    features: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_features: A `Tensor` of type `float32`.
      The float value that the lowest quantized value represents.
    max_features: A `Tensor` of type `float32`.
      The float value that the highest quantized value represents.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (activations, min_activations, max_activations).

    activations: A `Tensor` of type `out_type`.
    min_activations: A `Tensor` of type `float32`.
    max_activations: A `Tensor` of type `float32`.
  "
  [features min_features max_features & {:keys [out_type name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedRelu" [features min_features max_features] {:out_type out_type :name name }))

(defn QuantizedRelu6 
  "Computes Quantized Rectified Linear 6: `min(max(features, 0), 6)`

  Args:
    features: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_features: A `Tensor` of type `float32`.
      The float value that the lowest quantized value represents.
    max_features: A `Tensor` of type `float32`.
      The float value that the highest quantized value represents.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (activations, min_activations, max_activations).

    activations: A `Tensor` of type `out_type`.
    min_activations: A `Tensor` of type `float32`.
    max_activations: A `Tensor` of type `float32`.
  "
  [features min_features max_features & {:keys [out_type name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedRelu6" [features min_features max_features] {:out_type out_type :name name }))

(defn QuantizedReluX 
  "Computes Quantized Rectified Linear X: `min(max(features, 0), max_value)`

  Args:
    features: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    max_value: A `Tensor` of type `float32`.
    min_features: A `Tensor` of type `float32`.
      The float value that the lowest quantized value represents.
    max_features: A `Tensor` of type `float32`.
      The float value that the highest quantized value represents.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (activations, min_activations, max_activations).

    activations: A `Tensor` of type `out_type`.
    min_activations: A `Tensor` of type `float32`.
    max_activations: A `Tensor` of type `float32`.
  "
  [features max_value min_features max_features & {:keys [out_type name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "QuantizedReluX" [features max_value min_features max_features] {:out_type out_type :name name }))

(defn Range 
  "Creates a sequence of numbers.

  This operation creates a sequence of numbers that begins at `start` and
  extends by increments of `delta` up to but not including `limit`.

  For example:

  ```
  # 'start' is 3
  # 'limit' is 18
  # 'delta' is 3
  tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
  ```

  Args:
    start: A `Tensor`. Must be one of the following types: `bfloat16`, `float32`, `float64`, `int32`, `int64`.
      0-D (scalar). First entry in the sequence.
    limit: A `Tensor`. Must have the same type as `start`.
      0-D (scalar). Upper limit of sequence, exclusive.
    delta: A `Tensor`. Must have the same type as `start`.
      0-D (scalar). Optional. Default is 1. Number that increments `start`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `start`.
  "
  [ start limit delta name ]
  (py/call-attr quantization "Range"  start limit delta name ))

(defn Real 
  "Returns the real part of a complex number.

  Given a tensor `input` of complex numbers, this operation returns a tensor of
  type `float` that is the real part of each element in `input`. All elements in
  `input` must be complex numbers of the form \\(a + bj\\), where *a* is the real
   part returned by this operation and *b* is the imaginary part.

  For example:

  ```
  # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.real(input) ==> [-2.25, 3.25]
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
    Tout: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  "
  [input & {:keys [Tout name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Real" [input] {:Tout Tout :name name }))

(defn RealDiv 
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
  (py/call-attr quantization "RealDiv"  x y name ))

(defn Reciprocal 
  "Computes the reciprocal of x element-wise.

  I.e., \\(y = 1 / x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Reciprocal"  x name ))

(defn ReciprocalGrad 
  "Computes the gradient for the inverse of `x` wrt its input.

  Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
  is the corresponding input gradient.

  Args:
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    dy: A `Tensor`. Must have the same type as `y`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `y`.
  "
  [ y dy name ]
  (py/call-attr quantization "ReciprocalGrad"  y dy name ))

(defn Relu 
  "Computes rectified linear: `max(features, 0)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`, `qint8`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  "
  [ features name ]
  (py/call-attr quantization "Relu"  features name ))

(defn Relu6 
  "Computes rectified linear 6: `min(max(features, 0), 6)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  "
  [ features name ]
  (py/call-attr quantization "Relu6"  features name ))

(defn Relu6Grad 
  "Computes rectified linear 6 gradients for a Relu6 operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The backpropagated gradients to the corresponding Relu6 operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding Relu6 operation, or
      its output; using either one produces the same result.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  "
  [ gradients features name ]
  (py/call-attr quantization "Relu6Grad"  gradients features name ))

(defn ReluGrad 
  "Computes rectified linear gradients for a Relu operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The backpropagated gradients to the corresponding Relu operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding Relu operation, OR
      the outputs of that operation (both work equivalently).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  "
  [ gradients features name ]
  (py/call-attr quantization "ReluGrad"  gradients features name ))

(defn RequantizationRange 
  "Computes a range that covers the actual values present in a quantized tensor.

  Given a quantized tensor described by `(input, input_min, input_max)`, outputs a
  range that covers the actual values present in that tensor. This op is typically
  used to produce the `requested_output_min` and `requested_output_max` for
  `Requantize`.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    input_min: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    input_max: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_min, output_max).

    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  "
  [ input input_min input_max name ]
  (py/call-attr quantization "RequantizationRange"  input input_min input_max name ))

(defn RequantizationRangePerChannel 
  "Computes requantization range per channel.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    input_min: A `Tensor` of type `float32`.
      The minimum value of the input tensor
    input_max: A `Tensor` of type `float32`.
      The maximum value of the input tensor.
    clip_value_max: A `float`.
      The maximum value of the output that needs to be clipped.
      Example: set this to 6 for Relu6.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_min, output_max).

    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  "
  [ input input_min input_max clip_value_max name ]
  (py/call-attr quantization "RequantizationRangePerChannel"  input input_min input_max clip_value_max name ))

(defn Requantize 
  "Converts the quantized `input` tensor into a lower-precision `output`.

  Converts the quantized `input` tensor into a lower-precision `output`, using the
  output range specified with `requested_output_min` and `requested_output_max`.

  `[input_min, input_max]` are scalar floats that specify the range for the float
  interpretation of the `input` data. For example, if `input_min` is -1.0f and
  `input_max` is 1.0f, and we are dealing with `quint16` quantized data, then a 0
  value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    input_min: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    input_max: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    requested_output_min: A `Tensor` of type `float32`.
      The float value that the minimum quantized output value represents.
    requested_output_max: A `Tensor` of type `float32`.
      The float value that the maximum quantized output value represents.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`.
      The type of the output. Should be a lower bit depth than Tinput.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).

    output: A `Tensor` of type `out_type`.
    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  "
  [ input input_min input_max requested_output_min requested_output_max out_type name ]
  (py/call-attr quantization "Requantize"  input input_min input_max requested_output_min requested_output_max out_type name ))

(defn RequantizePerChannel 
  "Requantizes input with min and max values known per channel.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    input_min: A `Tensor` of type `float32`.
      The minimum value of the input tensor
    input_max: A `Tensor` of type `float32`.
      The maximum value of the input tensor.
    requested_output_min: A `Tensor` of type `float32`.
      The minimum value of the output tensor requested.
    requested_output_max: A `Tensor` of type `float32`.
      The maximum value of the output tensor requested.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
      The quantized type of output tensor that needs to be converted.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).

    output: A `Tensor` of type `out_type`.
    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  "
  [input input_min input_max requested_output_min requested_output_max & {:keys [out_type name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "RequantizePerChannel" [input input_min input_max requested_output_min requested_output_max] {:out_type out_type :name name }))

(defn Rint 
  "Returns element-wise integer closest to x.

  If the result is midway between two representable values,
  the even representable is chosen.
  For example:

  ```
  rint(-1.5) ==> -2.0
  rint(0.5000001) ==> 1.0
  rint([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) ==> [-2., -2., -0., 0., 2., 2., 2.]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Rint"  x name ))

(defn Round 
  "Rounds the values of a tensor to the nearest integer, element-wise.

  Rounds half to even.  Also known as bankers rounding. If you want to round
  according to the current system rounding mode use std::cint.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Round"  x name ))

(defn Rsqrt 
  "Computes reciprocal of square root of x element-wise.

  I.e., \\(y = 1 / \sqrt{x}\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Rsqrt"  x name ))

(defn RsqrtGrad 
  "Computes the gradient for the rsqrt of `x` wrt its input.

  Specifically, `grad = dy * -0.5 * y^3`, where `y = rsqrt(x)`, and `dy`
  is the corresponding input gradient.

  Args:
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    dy: A `Tensor`. Must have the same type as `y`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `y`.
  "
  [ y dy name ]
  (py/call-attr quantization "RsqrtGrad"  y dy name ))

(defn SegmentMax 
  "Computes the maximum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \max_j(data_j)\\) where `max` is over `j` such
  that `segment_ids[j] == i`.

  If the max is empty for a given segment ID `i`, `output[i] = 0`.

  <div style=\"width:70%; margin:auto; margin-bottom:10px; margin-top:20px;\">
  <img style=\"width:100%\" src=\"https://www.tensorflow.org/images/SegmentMax.png\" alt>
  </div>

  For example:

  ```
  c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
  tf.segment_max(c, tf.constant([0, 0, 1]))
  # ==> [[4, 3, 3, 4],
  #      [5, 6, 7, 8]]
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data segment_ids name ]
  (py/call-attr quantization "SegmentMax"  data segment_ids name ))

(defn SegmentMean 
  "Computes the mean along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \frac{\sum_j data_j}{N}\\) where `mean` is
  over `j` such that `segment_ids[j] == i` and `N` is the total number of
  values summed.

  If the mean is empty for a given segment ID `i`, `output[i] = 0`.

  <div style=\"width:70%; margin:auto; margin-bottom:10px; margin-top:20px;\">
  <img style=\"width:100%\" src=\"https://www.tensorflow.org/images/SegmentMean.png\" alt>
  </div>

  For example:

  ```
  c = tf.constant([[1.0,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
  tf.segment_mean(c, tf.constant([0, 0, 1]))
  # ==> [[2.5, 2.5, 2.5, 2.5],
  #      [5, 6, 7, 8]]
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data segment_ids name ]
  (py/call-attr quantization "SegmentMean"  data segment_ids name ))

(defn SegmentMin 
  "Computes the minimum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \min_j(data_j)\\) where `min` is over `j` such
  that `segment_ids[j] == i`.

  If the min is empty for a given segment ID `i`, `output[i] = 0`.

  <div style=\"width:70%; margin:auto; margin-bottom:10px; margin-top:20px;\">
  <img style=\"width:100%\" src=\"https://www.tensorflow.org/images/SegmentMin.png\" alt>
  </div>

  For example:

  ```
  c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
  tf.segment_min(c, tf.constant([0, 0, 1]))
  # ==> [[1, 2, 2, 1],
  #      [5, 6, 7, 8]]
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data segment_ids name ]
  (py/call-attr quantization "SegmentMin"  data segment_ids name ))

(defn SegmentProd 
  "Computes the product along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \prod_j data_j\\) where the product is over `j` such
  that `segment_ids[j] == i`.

  If the product is empty for a given segment ID `i`, `output[i] = 1`.

  <div style=\"width:70%; margin:auto; margin-bottom:10px; margin-top:20px;\">
  <img style=\"width:100%\" src=\"https://www.tensorflow.org/images/SegmentProd.png\" alt>
  </div>

  For example:

  ```
  c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
  tf.segment_prod(c, tf.constant([0, 0, 1]))
  # ==> [[4, 6, 6, 4],
  #      [5, 6, 7, 8]]
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data segment_ids name ]
  (py/call-attr quantization "SegmentProd"  data segment_ids name ))

(defn SegmentSum 
  "Computes the sum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \sum_j data_j\\) where sum is over `j` such
  that `segment_ids[j] == i`.

  If the sum is empty for a given segment ID `i`, `output[i] = 0`.

  <div style=\"width:70%; margin:auto; margin-bottom:10px; margin-top:20px;\">
  <img style=\"width:100%\" src=\"https://www.tensorflow.org/images/SegmentSum.png\" alt>
  </div>

  For example:

  ```
  c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
  tf.segment_sum(c, tf.constant([0, 0, 1]))
  # ==> [[5, 5, 5, 5],
  #      [5, 6, 7, 8]]
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data segment_ids name ]
  (py/call-attr quantization "SegmentSum"  data segment_ids name ))

(defn Select 
  "Selects elements from `x` or `y`, depending on `condition`.

  The `x`, and `y` tensors must all have the same shape, and the
  output will also have that shape.

  The `condition` tensor must be a scalar if `x` and `y` are scalars.
  If `x` and `y` are vectors or higher rank, then `condition` must be either a
  scalar, a vector with size matching the first dimension of `x`, or must have
  the same shape as `x`.

  The `condition` tensor acts as a mask that chooses, based on the value at each
  element, whether the corresponding element / row in the output should be
  taken from `x` (if true) or `y` (if false).

  If `condition` is a vector and `x` and `y` are higher rank matrices, then
  it chooses which row (outer dimension) to copy from `x` and `y`.
  If `condition` has the same shape as `x` and `y`, then it chooses which
  element to copy from `x` and `y`.

  For example:

  ```python
  # 'condition' tensor is [[True,  False]
  #                        [False, True]]
  # 't' is [[1, 2],
  #         [3, 4]]
  # 'e' is [[5, 6],
  #         [7, 8]]
  select(condition, t, e)  # => [[1, 6], [7, 4]]


  # 'condition' tensor is [True, False]
  # 't' is [[1, 2],
  #         [3, 4]]
  # 'e' is [[5, 6],
  #         [7, 8]]
  select(condition, t, e) ==> [[1, 2],
                               [7, 8]]

  ```

  Args:
    condition: A `Tensor` of type `bool`.
    x:  A `Tensor` which may have the same shape as `condition`.
      If `condition` is rank 1, `x` may have higher rank,
      but its first dimension must match the size of `condition`.
    y:  A `Tensor` with the same type and shape as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `t`.
  "
  [ condition x y name ]
  (py/call-attr quantization "Select"  condition x y name ))

(defn SelectV2 
  "TODO: add doc.

  Args:
    condition: A `Tensor` of type `bool`.
    t: A `Tensor`.
    e: A `Tensor`. Must have the same type as `t`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `t`.
  "
  [ condition t e name ]
  (py/call-attr quantization "SelectV2"  condition t e name ))

(defn Selu 
  "Computes scaled exponential linear: `scale * alpha * (exp(features) - 1)`

  if < 0, `scale * features` otherwise.

  To be used together with
  `initializer = tf.variance_scaling_initializer(factor=1.0, mode='FAN_IN')`.
  For correct dropout, use `tf.contrib.nn.alpha_dropout`.

  See [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  "
  [ features name ]
  (py/call-attr quantization "Selu"  features name ))

(defn SeluGrad 
  "Computes gradients for the scaled exponential linear (Selu) operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      The backpropagated gradients to the corresponding Selu operation.
    outputs: A `Tensor`. Must have the same type as `gradients`.
      The outputs of the corresponding Selu operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  "
  [ gradients outputs name ]
  (py/call-attr quantization "SeluGrad"  gradients outputs name ))

(defn Sigmoid 
  "Computes sigmoid of `x` element-wise.

  Specifically, `y = 1 / (1 + exp(-x))`.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Sigmoid"  x name ))

(defn SigmoidGrad 
  "Computes the gradient of the sigmoid of `x` wrt its input.

  Specifically, `grad = dy * y * (1 - y)`, where `y = sigmoid(x)`, and
  `dy` is the corresponding input gradient.

  Args:
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    dy: A `Tensor`. Must have the same type as `y`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `y`.
  "
  [ y dy name ]
  (py/call-attr quantization "SigmoidGrad"  y dy name ))

(defn Sign 
  "Returns an element-wise indication of the sign of a number.

  `y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.

  For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Sign"  x name ))

(defn Sin 
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
  (py/call-attr quantization "Sin"  x name ))

(defn Sinh 
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
  (py/call-attr quantization "Sinh"  x name ))

(defn Softmax 
  "Computes softmax activations.

  For each batch `i` and class `j` we have

      $$softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))$$

  Args:
    logits: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      2-D with shape `[batch_size, num_classes]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `logits`.
  "
  [ logits name ]
  (py/call-attr quantization "Softmax"  logits name ))

(defn SoftmaxCrossEntropyWithLogits 
  "Computes softmax cross entropy cost and gradients to backpropagate.

  Inputs are the logits, not probabilities.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      batch_size x num_classes matrix
    labels: A `Tensor`. Must have the same type as `features`.
      batch_size x num_classes matrix
      The caller must ensure that each batch of labels represents a valid
      probability distribution.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (loss, backprop).

    loss: A `Tensor`. Has the same type as `features`.
    backprop: A `Tensor`. Has the same type as `features`.
  "
  [ features labels name ]
  (py/call-attr quantization "SoftmaxCrossEntropyWithLogits"  features labels name ))

(defn Softplus 
  "Computes softplus: `log(exp(features) + 1)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  "
  [ features name ]
  (py/call-attr quantization "Softplus"  features name ))

(defn SoftplusGrad 
  "Computes softplus gradients for a softplus operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      The backpropagated gradients to the corresponding softplus operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding softplus operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  "
  [ gradients features name ]
  (py/call-attr quantization "SoftplusGrad"  gradients features name ))

(defn Softsign 
  "Computes softsign: `features / (abs(features) + 1)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  "
  [ features name ]
  (py/call-attr quantization "Softsign"  features name ))

(defn SoftsignGrad 
  "Computes softsign gradients for a softsign operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      The backpropagated gradients to the corresponding softsign operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding softsign operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  "
  [ gradients features name ]
  (py/call-attr quantization "SoftsignGrad"  gradients features name ))

(defn SparseMatMul 
  "Multiply matrix \"a\" by matrix \"b\".

  The inputs must be two-dimensional matrices and the inner dimension of \"a\" must
  match the outer dimension of \"b\". Both \"a\" and \"b\" must be `Tensor`s not
  `SparseTensor`s.  This op is optimized for the case where at least one of \"a\" or
  \"b\" is sparse, in the sense that they have a large proportion of zero values.
  The breakeven for using this versus a dense matrix multiply on one platform was
  30% zero values in the sparse matrix.

  The gradient computation of this operation will only take advantage of sparsity
  in the input gradient when that gradient comes from a Relu.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `bfloat16`.
    b: A `Tensor`. Must be one of the following types: `float32`, `bfloat16`.
    transpose_a: An optional `bool`. Defaults to `False`.
    transpose_b: An optional `bool`. Defaults to `False`.
    a_is_sparse: An optional `bool`. Defaults to `False`.
    b_is_sparse: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  "
  [a b & {:keys [transpose_a transpose_b a_is_sparse b_is_sparse name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "SparseMatMul" [a b] {:transpose_a transpose_a :transpose_b transpose_b :a_is_sparse a_is_sparse :b_is_sparse b_is_sparse :name name }))

(defn SparseSegmentMean 
  "Computes the mean along sparse segments of a tensor.

  See `tf.sparse.segment_sum` for usage examples.

  Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
  dimension, selecting a subset of dimension 0, specified by `indices`.

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor` of type `int32`.
      A 1-D tensor. Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data indices segment_ids name ]
  (py/call-attr quantization "SparseSegmentMean"  data indices segment_ids name ))

(defn SparseSegmentMeanGrad 
  "Computes gradients for SparseSegmentMean.

  Returns tensor \"output\" with same shape as grad, except for dimension 0 whose
  value is output_dim0.

  Args:
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      gradient propagated to the SparseSegmentMean op.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      indices passed to the corresponding SparseSegmentMean op.
    segment_ids: A `Tensor` of type `int32`.
      segment_ids passed to the corresponding SparseSegmentMean op.
    output_dim0: A `Tensor` of type `int32`.
      dimension 0 of \"data\" passed to SparseSegmentMean op.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  "
  [ grad indices segment_ids output_dim0 name ]
  (py/call-attr quantization "SparseSegmentMeanGrad"  grad indices segment_ids output_dim0 name ))

(defn SparseSegmentMeanWithNumSegments 
  "Computes the mean along sparse segments of a tensor.

  Like `SparseSegmentMean`, but allows missing ids in `segment_ids`. If an id is
  misisng, the `output` tensor at that position will be zeroed.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor` of type `int32`.
      A 1-D tensor. Values should be sorted and can be repeated.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Should equal the number of distinct segment IDs.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data indices segment_ids num_segments name ]
  (py/call-attr quantization "SparseSegmentMeanWithNumSegments"  data indices segment_ids num_segments name ))

(defn SparseSegmentSqrtN 
  "Computes the sum along sparse segments of a tensor divided by the sqrt of N.

  N is the size of the segment being reduced.

  See `tf.sparse.segment_sum` for usage examples.

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor` of type `int32`.
      A 1-D tensor. Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data indices segment_ids name ]
  (py/call-attr quantization "SparseSegmentSqrtN"  data indices segment_ids name ))

(defn SparseSegmentSqrtNGrad 
  "Computes gradients for SparseSegmentSqrtN.

  Returns tensor \"output\" with same shape as grad, except for dimension 0 whose
  value is output_dim0.

  Args:
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      gradient propagated to the SparseSegmentSqrtN op.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      indices passed to the corresponding SparseSegmentSqrtN op.
    segment_ids: A `Tensor` of type `int32`.
      segment_ids passed to the corresponding SparseSegmentSqrtN op.
    output_dim0: A `Tensor` of type `int32`.
      dimension 0 of \"data\" passed to SparseSegmentSqrtN op.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  "
  [ grad indices segment_ids output_dim0 name ]
  (py/call-attr quantization "SparseSegmentSqrtNGrad"  grad indices segment_ids output_dim0 name ))

(defn SparseSegmentSqrtNWithNumSegments 
  "Computes the sum along sparse segments of a tensor divided by the sqrt of N.

  N is the size of the segment being reduced.

  Like `SparseSegmentSqrtN`, but allows missing ids in `segment_ids`. If an id is
  misisng, the `output` tensor at that position will be zeroed.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor` of type `int32`.
      A 1-D tensor. Values should be sorted and can be repeated.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Should equal the number of distinct segment IDs.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data indices segment_ids num_segments name ]
  (py/call-attr quantization "SparseSegmentSqrtNWithNumSegments"  data indices segment_ids num_segments name ))

(defn SparseSegmentSum 
  "Computes the sum along sparse segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
  dimension, selecting a subset of dimension 0, specified by `indices`.

  For example:

  ```python
  c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])

  # Select two rows, one segment.
  tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
  # => [[0 0 0 0]]

  # Select two rows, two segment.
  tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
  # => [[ 1  2  3  4]
  #     [-1 -2 -3 -4]]

  # Select all rows, two segments.
  tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
  # => [[0 0 0 0]
  #     [5 6 7 8]]

  # Which is equivalent to:
  tf.segment_sum(c, tf.constant([0, 0, 1]))
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor` of type `int32`.
      A 1-D tensor. Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data indices segment_ids name ]
  (py/call-attr quantization "SparseSegmentSum"  data indices segment_ids name ))

(defn SparseSegmentSumWithNumSegments 
  "Computes the sum along sparse segments of a tensor.

  Like `SparseSegmentSum`, but allows missing ids in `segment_ids`. If an id is
  misisng, the `output` tensor at that position will be zeroed.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/sparse#Segmentation)
  for an explanation of segments.

  For example:

  ```python
  c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])

  tf.sparse_segment_sum_with_num_segments(
      c, tf.constant([0, 1]), tf.constant([0, 0]), num_segments=3)
  # => [[0 0 0 0]
  #     [0 0 0 0]
  #     [0 0 0 0]]

  tf.sparse_segment_sum_with_num_segments(c,
                                          tf.constant([0, 1]),
                                          tf.constant([0, 2],
                                          num_segments=4))
  # => [[ 1  2  3  4]
  #     [ 0  0  0  0]
  #     [-1 -2 -3 -4]
  #     [ 0  0  0  0]]
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor` of type `int32`.
      A 1-D tensor. Values should be sorted and can be repeated.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Should equal the number of distinct segment IDs.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data indices segment_ids num_segments name ]
  (py/call-attr quantization "SparseSegmentSumWithNumSegments"  data indices segment_ids num_segments name ))

(defn SparseSoftmaxCrossEntropyWithLogits 
  "Computes softmax cross entropy cost and gradients to backpropagate.

  Unlike `SoftmaxCrossEntropyWithLogits`, this operation does not accept
  a matrix of label probabilities, but rather a single label per row
  of features.  This label is considered to have probability 1.0 for the
  given row.

  Inputs are the logits, not probabilities.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      batch_size x num_classes matrix
    labels: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      batch_size vector with values in [0, num_classes).
      This is the label for the given minibatch entry.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (loss, backprop).

    loss: A `Tensor`. Has the same type as `features`.
    backprop: A `Tensor`. Has the same type as `features`.
  "
  [ features labels name ]
  (py/call-attr quantization "SparseSoftmaxCrossEntropyWithLogits"  features labels name ))

(defn Sqrt 
  "Computes square root of x element-wise.

  I.e., \\(y = \sqrt{x} = x^{1/2}\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Sqrt"  x name ))

(defn SqrtGrad 
  "Computes the gradient for the sqrt of `x` wrt its input.

  Specifically, `grad = dy * 0.5 / y`, where `y = sqrt(x)`, and `dy`
  is the corresponding input gradient.

  Args:
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    dy: A `Tensor`. Must have the same type as `y`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `y`.
  "
  [ y dy name ]
  (py/call-attr quantization "SqrtGrad"  y dy name ))

(defn Square 
  "Computes square of x element-wise.

  I.e., \\(y = x * x = x^2\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "Square"  x name ))

(defn SquaredDifference 
  "Returns (x - y)(x - y) element-wise.

  *NOTE*: `math.squared_difference` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "SquaredDifference"  x y name ))

(defn Sub 
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
  (py/call-attr quantization "Sub"  x y name ))

(defn Sum 
  "Computes the sum of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The tensor to reduce.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce. Must be in the range
      `[-rank(input), rank(input))`.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input axis & {:keys [keep_dims name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "Sum" [input axis] {:keep_dims keep_dims :name name }))

(defn Tan 
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
  (py/call-attr quantization "Tan"  x name ))

(defn Tanh 
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
  "
  [ x name ]
  (py/call-attr quantization "Tanh"  x name ))

(defn TanhGrad 
  "Computes the gradient for the tanh of `x` wrt its input.

  Specifically, `grad = dy * (1 - y*y)`, where `y = tanh(x)`, and `dy`
  is the corresponding input gradient.

  Args:
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    dy: A `Tensor`. Must have the same type as `y`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `y`.
  "
  [ y dy name ]
  (py/call-attr quantization "TanhGrad"  y dy name ))

(defn TopK 
  "Finds values and indices of the `k` largest elements for the last dimension.

  If the input is a vector (rank-1), finds the `k` largest entries in the vector
  and outputs their values and indices as vectors.  Thus `values[j]` is the
  `j`-th largest entry in `input`, and its index is `indices[j]`.

  For matrices (resp. higher rank input), computes the top `k` entries in each
  row (resp. vector along the last dimension).  Thus,

      values.shape = indices.shape = input.shape[:-1] + [k]

  If two elements are equal, the lower-index element appears first.

  If `k` varies dynamically, use `TopKV2` below.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      1-D or higher with last dimension at least `k`.
    k: An `int` that is `>= 0`.
      Number of top elements to look for along the last dimension (along each
      row for matrices).
    sorted: An optional `bool`. Defaults to `True`.
      If true the resulting `k` elements will be sorted by the values in
      descending order.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (values, indices).

    values: A `Tensor`. Has the same type as `input`.
    indices: A `Tensor` of type `int32`.
  "
  [input k & {:keys [sorted name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "TopK" [input k] {:sorted sorted :name name }))

(defn TopKV2 
  "Finds values and indices of the `k` largest elements for the last dimension.

  If the input is a vector (rank-1), finds the `k` largest entries in the vector
  and outputs their values and indices as vectors.  Thus `values[j]` is the
  `j`-th largest entry in `input`, and its index is `indices[j]`.

  For matrices (resp. higher rank input), computes the top `k` entries in each
  row (resp. vector along the last dimension).  Thus,

      values.shape = indices.shape = input.shape[:-1] + [k]

  If two elements are equal, the lower-index element appears first.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      1-D or higher with last dimension at least `k`.
    k: A `Tensor` of type `int32`.
      0-D.  Number of top elements to look for along the last dimension (along each
      row for matrices).
    sorted: An optional `bool`. Defaults to `True`.
      If true the resulting `k` elements will be sorted by the values in
      descending order.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (values, indices).

    values: A `Tensor`. Has the same type as `input`.
    indices: A `Tensor` of type `int32`.
  "
  [input k & {:keys [sorted name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "TopKV2" [input k] {:sorted sorted :name name }))

(defn TruncateDiv 
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
  (py/call-attr quantization "TruncateDiv"  x y name ))

(defn TruncateMod 
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
  (py/call-attr quantization "TruncateMod"  x y name ))

(defn UnsortedSegmentMax 
  "Computes the maximum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  This operator is similar to the unsorted segment sum operator found
  [(here)](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
  Instead of computing the sum over segments, it computes the maximum such that:

  \\(output_i = \max_{j...} data[j...]\\) where max is over tuples `j...` such
  that `segment_ids[j...] == i`.

  If the maximum is empty for a given segment ID `i`, it outputs the smallest
  possible value for the specific numeric type,
  `output[i] = numeric_limits<T>::lowest()`.

  If the given segment ID `i` is negative, then the corresponding value is
  dropped, and will not be included in the result.

  <div style=\"width:70%; margin:auto; margin-bottom:10px; margin-top:20px;\">
  <img style=\"width:100%\" src=\"https://www.tensorflow.org/images/UnsortedSegmentMax.png\" alt>
  </div>

  For example:

  ``` python
  c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
  tf.unsorted_segment_max(c, tf.constant([0, 1, 0]), num_segments=2)
  # ==> [[ 4,  3, 3, 4],
  #       [5,  6, 7, 8]]
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A tensor whose shape is a prefix of `data.shape`.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data segment_ids num_segments name ]
  (py/call-attr quantization "UnsortedSegmentMax"  data segment_ids num_segments name ))

(defn UnsortedSegmentMin 
  "Computes the minimum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  This operator is similar to the unsorted segment sum operator found
  [(here)](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
  Instead of computing the sum over segments, it computes the minimum such that:

  \\(output_i = \min_{j...} data_[j...]\\) where min is over tuples `j...` such
  that `segment_ids[j...] == i`.

  If the minimum is empty for a given segment ID `i`, it outputs the largest
  possible value for the specific numeric type,
  `output[i] = numeric_limits<T>::max()`.

  For example:

  ``` python
  c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
  tf.unsorted_segment_min(c, tf.constant([0, 1, 0]), num_segments=2)
  # ==> [[ 1,  2, 2, 1],
  #       [5,  6, 7, 8]]
  ```

  If the given segment ID `i` is negative, then the corresponding value is
  dropped, and will not be included in the result.

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A tensor whose shape is a prefix of `data.shape`.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data segment_ids num_segments name ]
  (py/call-attr quantization "UnsortedSegmentMin"  data segment_ids num_segments name ))

(defn UnsortedSegmentProd 
  "Computes the product along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  This operator is similar to the unsorted segment sum operator found
  [(here)](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
  Instead of computing the sum over segments, it computes the product of all
  entries belonging to a segment such that:

  \\(output_i = \prod_{j...} data[j...]\\) where the product is over tuples
  `j...` such that `segment_ids[j...] == i`.

  For example:

  ``` python
  c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
  tf.unsorted_segment_prod(c, tf.constant([0, 1, 0]), num_segments=2)
  # ==> [[ 4,  6, 6, 4],
  #       [5,  6, 7, 8]]
  ```

  If there is no entry for a given segment ID `i`, it outputs 1.

  If the given segment ID `i` is negative, then the corresponding value is
  dropped, and will not be included in the result.

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A tensor whose shape is a prefix of `data.shape`.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data segment_ids num_segments name ]
  (py/call-attr quantization "UnsortedSegmentProd"  data segment_ids num_segments name ))

(defn UnsortedSegmentSum 
  "Computes the sum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output[i] = \sum_{j...} data[j...]\\) where the sum is over tuples `j...` such
  that `segment_ids[j...] == i`.  Unlike `SegmentSum`, `segment_ids`
  need not be sorted and need not cover all values in the full
  range of valid values.

  If the sum is empty for a given segment ID `i`, `output[i] = 0`.
  If the given segment ID `i` is negative, the value is dropped and will not be
  added to the sum of the segment.

  `num_segments` should equal the number of distinct segment IDs.

  <div style=\"width:70%; margin:auto; margin-bottom:10px; margin-top:20px;\">
  <img style=\"width:100%\" src=\"https://www.tensorflow.org/images/UnsortedSegmentSum.png\" alt>
  </div>

  ``` python
  c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
  tf.unsorted_segment_sum(c, tf.constant([0, 1, 0]), num_segments=2)
  # ==> [[ 5,  5, 5, 5],
  #       [5,  6, 7, 8]]
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A tensor whose shape is a prefix of `data.shape`.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data segment_ids num_segments name ]
  (py/call-attr quantization "UnsortedSegmentSum"  data segment_ids num_segments name ))

(defn Xdivy 
  "Returns 0 if x == 0, and x / y otherwise, elementwise.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "Xdivy"  x y name ))

(defn Xlogy 
  "Returns 0 if x == 0, and x * log(y) otherwise, elementwise.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "Xlogy"  x y name ))

(defn Zeta 
  "Compute the Hurwitz zeta function \\(\zeta(x, q)\\).

  The Hurwitz zeta function is defined as:


  \\(\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}\\)

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    q: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x q name ]
  (py/call-attr quantization "Zeta"  x q name ))

(defn accumulate-nv2 
  "Returns the element-wise sum of a list of tensors.

  `tf.accumulate_n_v2` performs the same operation as `tf.add_n`, but does not
  wait for all of its inputs to be ready before beginning to sum. This can
  save memory if inputs are ready at different times, since minimum temporary
  storage is proportional to the output size rather than the inputs size.

  Unlike the original `accumulate_n`, `accumulate_n_v2` is differentiable.

  Returns a `Tensor` of same shape and type as the elements of `inputs`.

  Args:
    inputs: A list of at least 1 `Tensor` objects with the same type in: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A list of `Tensor` objects, each with same shape and type.
    shape: A `tf.TensorShape` or list of `ints`.
      Shape of elements of `inputs`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `inputs`.
  "
  [ inputs shape name ]
  (py/call-attr quantization "accumulate_nv2"  inputs shape name ))

(defn accumulate-nv2-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function accumulate_nv2
  "
  [ inputs shape name ctx ]
  (py/call-attr quantization "accumulate_nv2_eager_fallback"  inputs shape name ctx ))

(defn acos 
  "Computes acos of x element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "acos"  x name ))

(defn acos-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function acos
  "
  [ x name ctx ]
  (py/call-attr quantization "acos_eager_fallback"  x name ctx ))

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
  (py/call-attr quantization "acosh"  x name ))

(defn acosh-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function acosh
  "
  [ x name ctx ]
  (py/call-attr quantization "acosh_eager_fallback"  x name ctx ))

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
  (py/call-attr quantization "add"  x y name ))

(defn add-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function add
  "
  [ x y name ctx ]
  (py/call-attr quantization "add_eager_fallback"  x y name ctx ))

(defn add-n 
  "Add all input tensors element wise.

    Inputs must be of same size and shape.

    ```python
    x = [9, 7, 10]
    tf.math.add_n(x) ==> 26
    ```

  Args:
    inputs: A list of at least 1 `Tensor` objects with the same type in: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`, `variant`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `inputs`.
  "
  [ inputs name ]
  (py/call-attr quantization "add_n"  inputs name ))

(defn add-n-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function add_n
  "
  [ inputs name ctx ]
  (py/call-attr quantization "add_n_eager_fallback"  inputs name ctx ))

(defn add-v2 
  "Returns x + y element-wise.

  *NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "add_v2"  x y name ))

(defn add-v2-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function add_v2
  "
  [ x y name ctx ]
  (py/call-attr quantization "add_v2_eager_fallback"  x y name ctx ))

(defn angle 
  "Returns the argument of a complex number.

  Given a tensor `input` of complex numbers, this operation returns a tensor of
  type `float` that is the argument of each element in `input`. All elements in
  `input` must be complex numbers of the form \\(a + bj\\), where *a*
  is the real part and *b* is the imaginary part.

  The argument returned by this operation is of the form \\(atan2(b, a)\\).

  For example:

  ```
  # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.angle(input) ==> [2.0132, 1.056]
  ```

  @compatibility(numpy)
  Equivalent to np.angle.
  @end_compatibility

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
    Tout: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  "
  [input & {:keys [Tout name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "angle" [input] {:Tout Tout :name name }))

(defn angle-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function angle
  "
  [input & {:keys [Tout name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "angle_eager_fallback" [input] {:Tout Tout :name name :ctx ctx }))

(defn approximate-equal 
  "Returns the truth value of abs(x-y) < tolerance element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    tolerance: An optional `float`. Defaults to `1e-05`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [x y & {:keys [tolerance name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "approximate_equal" [x y] {:tolerance tolerance :name name }))

(defn approximate-equal-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function approximate_equal
  "
  [x y & {:keys [tolerance name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "approximate_equal_eager_fallback" [x y] {:tolerance tolerance :name name :ctx ctx }))

(defn arg-max 
  "Returns the index with the largest value across dimensions of a tensor.

  Note that in case of ties the identity of the return value is not guaranteed.

  Usage:
    ```python
    import tensorflow as tf
    a = [1, 10, 26.9, 2.8, 166.32, 62.3]
    b = tf.math.argmax(input = a)
    c = tf.keras.backend.eval(b)
    # c = 4
    # here a[4] = 166.32 which is the largest element of a across axis 0
    ```

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    dimension: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      int32 or int64, must be in the range `[-rank(input), rank(input))`.
      Describes which dimension of the input Tensor to reduce across. For vectors,
      use dimension = 0.
    output_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_type`.
  "
  [input dimension & {:keys [output_type name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "arg_max" [input dimension] {:output_type output_type :name name }))

(defn arg-max-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function arg_max
  "
  [input dimension & {:keys [output_type name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "arg_max_eager_fallback" [input dimension] {:output_type output_type :name name :ctx ctx }))

(defn arg-min 
  "Returns the index with the smallest value across dimensions of a tensor.

  Note that in case of ties the identity of the return value is not guaranteed.

  Usage:
    ```python
    import tensorflow as tf
    a = [1, 10, 26.9, 2.8, 166.32, 62.3]
    b = tf.math.argmin(input = a)
    c = tf.keras.backend.eval(b)
    # c = 0
    # here a[0] = 1 which is the smallest element of a across axis 0
    ```

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    dimension: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      int32 or int64, must be in the range `[-rank(input), rank(input))`.
      Describes which dimension of the input Tensor to reduce across. For vectors,
      use dimension = 0.
    output_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `output_type`.
  "
  [input dimension & {:keys [output_type name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "arg_min" [input dimension] {:output_type output_type :name name }))

(defn arg-min-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function arg_min
  "
  [input dimension & {:keys [output_type name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "arg_min_eager_fallback" [input dimension] {:output_type output_type :name name :ctx ctx }))

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
  (py/call-attr quantization "asin"  x name ))

(defn asin-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function asin
  "
  [ x name ctx ]
  (py/call-attr quantization "asin_eager_fallback"  x name ctx ))

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
  (py/call-attr quantization "asinh"  x name ))

(defn asinh-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function asinh
  "
  [ x name ctx ]
  (py/call-attr quantization "asinh_eager_fallback"  x name ctx ))

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
  (py/call-attr quantization "atan"  x name ))

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
  (py/call-attr quantization "atan2"  y x name ))

(defn atan2-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function atan2
  "
  [ y x name ctx ]
  (py/call-attr quantization "atan2_eager_fallback"  y x name ctx ))

(defn atan-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function atan
  "
  [ x name ctx ]
  (py/call-attr quantization "atan_eager_fallback"  x name ctx ))

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
  (py/call-attr quantization "atanh"  x name ))

(defn atanh-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function atanh
  "
  [ x name ctx ]
  (py/call-attr quantization "atanh_eager_fallback"  x name ctx ))

(defn avg-pool 
  "Performs average pooling on the input.

  Each entry in `output` is the mean of the corresponding size `ksize`
  window in `value`.

  Args:
    value: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the sliding window for each dimension of `value`.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of `value`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
  "
  [value ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "avg_pool" [value ksize strides padding] {:data_format data_format :name name }))

(defn avg-pool3d 
  "Performs 3D average pooling on the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NDHWC\", \"NCDHW\"`. Defaults to `\"NDHWC\"`.
      The data format of the input and output data. With the
      default format \"NDHWC\", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCDHW\", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "avg_pool3d" [input ksize strides padding] {:data_format data_format :name name }))

(defn avg-pool3d-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function avg_pool3d
  "
  [input ksize strides padding & {:keys [data_format name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "avg_pool3d_eager_fallback" [input ksize strides padding] {:data_format data_format :name name :ctx ctx }))

(defn avg-pool3d-grad 
  "Computes gradients of average pooling function.

  Args:
    orig_input_shape: A `Tensor` of type `int32`.
      The original input dimensions.
    grad: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      Output backprop of shape `[batch, depth, rows, cols, channels]`.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NDHWC\", \"NCDHW\"`. Defaults to `\"NDHWC\"`.
      The data format of the input and output data. With the
      default format \"NDHWC\", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCDHW\", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  "
  [orig_input_shape grad ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "avg_pool3d_grad" [orig_input_shape grad ksize strides padding] {:data_format data_format :name name }))

(defn avg-pool3d-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function avg_pool3d_grad
  "
  [orig_input_shape grad ksize strides padding & {:keys [data_format name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "avg_pool3d_grad_eager_fallback" [orig_input_shape grad ksize strides padding] {:data_format data_format :name name :ctx ctx }))

(defn avg-pool-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function avg_pool
  "
  [value ksize strides padding & {:keys [data_format name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "avg_pool_eager_fallback" [value ksize strides padding] {:data_format data_format :name name :ctx ctx }))

(defn avg-pool-grad 
  "Computes gradients of the average pooling function.

  Args:
    orig_input_shape: A `Tensor` of type `int32`.
      1-D.  Shape of the original input to `avg_pool`.
    grad: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t.
      the output of `avg_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the sliding window for each dimension of the input.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the input.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  "
  [orig_input_shape grad ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "avg_pool_grad" [orig_input_shape grad ksize strides padding] {:data_format data_format :name name }))

(defn avg-pool-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function avg_pool_grad
  "
  [orig_input_shape grad ksize strides padding & {:keys [data_format name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "avg_pool_grad_eager_fallback" [orig_input_shape grad ksize strides padding] {:data_format data_format :name name :ctx ctx }))

(defn batch-mat-mul 
  "Multiplies slices of two tensors in batches.

  Multiplies all slices of `Tensor` `x` and `y` (each slice can be
  viewed as an element of a batch), and arranges the individual results
  in a single output tensor of the same batch size. Each of the
  individual slices can optionally be adjointed (to adjoint a matrix
  means to transpose and conjugate it) before multiplication by setting
  the `adj_x` or `adj_y` flag to `True`, which are by default `False`.

  The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
  and `[..., r_y, c_y]`.

  The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:

      r_o = c_x if adj_x else r_x
      c_o = r_y if adj_y else c_y

  It is computed as:

      output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
      2-D or higher with shape `[..., r_x, c_x]`.
    y: A `Tensor`. Must have the same type as `x`.
      2-D or higher with shape `[..., r_y, c_y]`.
    adj_x: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `x`. Defaults to `False`.
    adj_y: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `y`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [x y & {:keys [adj_x adj_y name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "batch_mat_mul" [x y] {:adj_x adj_x :adj_y adj_y :name name }))

(defn batch-mat-mul-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function batch_mat_mul
  "
  [x y & {:keys [adj_x adj_y name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "batch_mat_mul_eager_fallback" [x y] {:adj_x adj_x :adj_y adj_y :name name :ctx ctx }))

(defn batch-mat-mul-v2 
  "Multiplies slices of two tensors in batches.

  Multiplies all slices of `Tensor` `x` and `y` (each slice can be
  viewed as an element of a batch), and arranges the individual results
  in a single output tensor of the same batch size. Each of the
  individual slices can optionally be adjointed (to adjoint a matrix
  means to transpose and conjugate it) before multiplication by setting
  the `adj_x` or `adj_y` flag to `True`, which are by default `False`.

  The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
  and `[..., r_y, c_y]`.

  The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:

      r_o = c_x if adj_x else r_x
      c_o = r_y if adj_y else c_y

  It is computed as:

      output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])

  *NOTE*: `BatchMatMulV2` supports broadcasting in the batch dimensions. More
  about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
      2-D or higher with shape `[..., r_x, c_x]`.
    y: A `Tensor`. Must have the same type as `x`.
      2-D or higher with shape `[..., r_y, c_y]`.
    adj_x: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `x`. Defaults to `False`.
    adj_y: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `y`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [x y & {:keys [adj_x adj_y name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "batch_mat_mul_v2" [x y] {:adj_x adj_x :adj_y adj_y :name name }))

(defn batch-mat-mul-v2-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function batch_mat_mul_v2
  "
  [x y & {:keys [adj_x adj_y name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "batch_mat_mul_v2_eager_fallback" [x y] {:adj_x adj_x :adj_y adj_y :name name :ctx ctx }))

(defn batch-norm-with-global-normalization-grad 
  "Gradients for batch normalization.

  This op is deprecated. See `tf.nn.batch_normalization`.

  Args:
    t: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A 4D input Tensor.
    m: A `Tensor`. Must have the same type as `t`.
      A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    v: A `Tensor`. Must have the same type as `t`.
      A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    gamma: A `Tensor`. Must have the same type as `t`.
      A 1D gamma Tensor with size matching the last dimension of t.
      If \"scale_after_normalization\" is true, this Tensor will be multiplied
      with the normalized Tensor.
    backprop: A `Tensor`. Must have the same type as `t`. 4D backprop Tensor.
    variance_epsilon: A `float`. A small float number to avoid dividing by 0.
    scale_after_normalization: A `bool`.
      A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (dx, dm, dv, db, dg).

    dx: A `Tensor`. Has the same type as `t`.
    dm: A `Tensor`. Has the same type as `t`.
    dv: A `Tensor`. Has the same type as `t`.
    db: A `Tensor`. Has the same type as `t`.
    dg: A `Tensor`. Has the same type as `t`.
  "
  [ t m v gamma backprop variance_epsilon scale_after_normalization name ]
  (py/call-attr quantization "batch_norm_with_global_normalization_grad"  t m v gamma backprop variance_epsilon scale_after_normalization name ))

(defn batch-norm-with-global-normalization-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function batch_norm_with_global_normalization_grad
  "
  [ t m v gamma backprop variance_epsilon scale_after_normalization name ctx ]
  (py/call-attr quantization "batch_norm_with_global_normalization_grad_eager_fallback"  t m v gamma backprop variance_epsilon scale_after_normalization name ctx ))

(defn bessel-i0e 
  "Computes the Bessel i0e function of `x` element-wise.

  Exponentially scaled modified Bessel function of order 0 defined as
  `bessel_i0e(x) = exp(-abs(x)) bessel_i0(x)`.

  This function is faster and numerically stabler than `bessel_i0(x)`.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.

    If `x` is a `SparseTensor`, returns
    `SparseTensor(x.indices, tf.math.bessel_i0e(x.values, ...), x.dense_shape)`"
  [ x name ]
  (py/call-attr quantization "bessel_i0e"  x name ))

(defn bessel-i0e-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function bessel_i0e
  "
  [ x name ctx ]
  (py/call-attr quantization "bessel_i0e_eager_fallback"  x name ctx ))

(defn bessel-i1e 
  "Computes the Bessel i1e function of `x` element-wise.

  Exponentially scaled modified Bessel function of order 0 defined as
  `bessel_i1e(x) = exp(-abs(x)) bessel_i1(x)`.

  This function is faster and numerically stabler than `bessel_i1(x)`.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.

    If `x` is a `SparseTensor`, returns
    `SparseTensor(x.indices, tf.math.bessel_i1e(x.values, ...), x.dense_shape)`"
  [ x name ]
  (py/call-attr quantization "bessel_i1e"  x name ))

(defn bessel-i1e-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function bessel_i1e
  "
  [ x name ctx ]
  (py/call-attr quantization "bessel_i1e_eager_fallback"  x name ctx ))

(defn betainc 
  "Compute the regularized incomplete beta integral \\(I_x(a, b)\\).

  The regularized incomplete beta integral is defined as:


  \\(I_x(a, b) = \frac{B(x; a, b)}{B(a, b)}\\)

  where


  \\(B(x; a, b) = \int_0^x t^{a-1} (1 - t)^{b-1} dt\\)


  is the incomplete beta function and \\(B(a, b)\\) is the *complete*
  beta function.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    b: A `Tensor`. Must have the same type as `a`.
    x: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  "
  [ a b x name ]
  (py/call-attr quantization "betainc"  a b x name ))

(defn betainc-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function betainc
  "
  [ a b x name ctx ]
  (py/call-attr quantization "betainc_eager_fallback"  a b x name ctx ))

(defn bias-add 
  "Adds `bias` to `value`.

  This is a special case of `tf.add` where `bias` is restricted to be 1-D.
  Broadcasting is supported, so `value` may have any number of dimensions.

  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Any number of dimensions.
    bias: A `Tensor`. Must have the same type as `value`.
      1-D with size the last dimension of `value`.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the bias tensor will be added to the last dimension
      of the value tensor.
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
      The tensor will be added to \"in_channels\", the third-to-the-last
          dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
  "
  [value bias & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "bias_add" [value bias] {:data_format data_format :name name }))

(defn bias-add-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function bias_add
  "
  [value bias & {:keys [data_format name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "bias_add_eager_fallback" [value bias] {:data_format data_format :name name :ctx ctx }))

(defn bias-add-grad 
  "The backward operation for \"BiasAdd\" on the \"bias\" tensor.

  It accumulates all the values from out_backprop into the feature dimension.
  For NHWC data format, the feature dimension is the last. For NCHW data format,
  the feature dimension is the third-to-last.

  Args:
    out_backprop: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Any number of dimensions.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the bias tensor will be added to the last dimension
      of the value tensor.
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
      The tensor will be added to \"in_channels\", the third-to-the-last
          dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `out_backprop`.
  "
  [out_backprop & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "bias_add_grad" [out_backprop] {:data_format data_format :name name }))

(defn bias-add-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function bias_add_grad
  "
  [out_backprop & {:keys [data_format name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "bias_add_grad_eager_fallback" [out_backprop] {:data_format data_format :name name :ctx ctx }))

(defn bias-add-v1 
  "Adds `bias` to `value`.

  This is a deprecated version of BiasAdd and will be soon removed.

  This is a special case of `tf.add` where `bias` is restricted to be 1-D.
  Broadcasting is supported, so `value` may have any number of dimensions.

  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Any number of dimensions.
    bias: A `Tensor`. Must have the same type as `value`.
      1-D with size the last dimension of `value`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
  "
  [ value bias name ]
  (py/call-attr quantization "bias_add_v1"  value bias name ))

(defn bias-add-v1-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function bias_add_v1
  "
  [ value bias name ctx ]
  (py/call-attr quantization "bias_add_v1_eager_fallback"  value bias name ctx ))

(defn bincount 
  "Counts the number of occurrences of each value in an integer array.

  Outputs a vector with length `size` and the same dtype as `weights`. If
  `weights` are empty, then index `i` stores the number of times the value `i` is
  counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
  the value in `weights` at each index where the corresponding value in `arr` is
  `i`.

  Values in `arr` outside of the range [0, size) are ignored.

  Args:
    arr: A `Tensor` of type `int32`. int32 `Tensor`.
    size: A `Tensor` of type `int32`. non-negative int32 scalar `Tensor`.
    weights: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      is an int32, int64, float32, or float64 `Tensor` with the same
      shape as `arr`, or a length-0 `Tensor`, in which case it acts as all weights
      equal to 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `weights`.
  "
  [ arr size weights name ]
  (py/call-attr quantization "bincount"  arr size weights name ))

(defn bincount-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function bincount
  "
  [ arr size weights name ctx ]
  (py/call-attr quantization "bincount_eager_fallback"  arr size weights name ctx ))

(defn bucketize 
  "Bucketizes 'input' based on 'boundaries'.

  For example, if the inputs are
      boundaries = [0, 10, 100]
      input = [[-5, 10000]
               [150,   10]
               [5,    100]]

  then the output will be
      output = [[0, 3]
                [3, 2]
                [1, 3]]

  Args:
    input: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      Any shape of Tensor contains with int or float type.
    boundaries: A list of `floats`.
      A sorted list of floats gives the boundary of the buckets.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  "
  [ input boundaries name ]
  (py/call-attr quantization "bucketize"  input boundaries name ))

(defn bucketize-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function bucketize
  "
  [ input boundaries name ctx ]
  (py/call-attr quantization "bucketize_eager_fallback"  input boundaries name ctx ))

(defn cast 
  "Cast x of type SrcT to y of DstT.

  Args:
    x: A `Tensor`.
    DstT: A `tf.DType`.
    Truncate: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `DstT`.
  "
  [x DstT & {:keys [Truncate name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "cast" [x DstT] {:Truncate Truncate :name name }))

(defn cast-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function cast
  "
  [x DstT & {:keys [Truncate name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "cast_eager_fallback" [x DstT] {:Truncate Truncate :name name :ctx ctx }))

(defn ceil 
  "Returns element-wise smallest integer not less than x.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "ceil"  x name ))

(defn ceil-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function ceil
  "
  [ x name ctx ]
  (py/call-attr quantization "ceil_eager_fallback"  x name ctx ))

(defn compare-and-bitpack 
  "Compare values of `input` to `threshold` and pack resulting bits into a `uint8`.

  Each comparison returns a boolean `true` (if `input_value > threshold`)
  or and `false` otherwise.

  This operation is useful for Locality-Sensitive-Hashing (LSH) and other
  algorithms that use hashing approximations of cosine and `L2` distances;
  codes can be generated from an input via:

  ```python
  codebook_size = 50
  codebook_bits = codebook_size * 32
  codebook = tf.get_variable('codebook', [x.shape[-1].value, codebook_bits],
                             dtype=x.dtype,
                             initializer=tf.orthogonal_initializer())
  codes = compare_and_threshold(tf.matmul(x, codebook), threshold=0.)
  codes = tf.bitcast(codes, tf.int32)  # go from uint8 to int32
  # now codes has shape x.shape[:-1] + [codebook_size]
  ```

  **NOTE**: Currently, the innermost dimension of the tensor must be divisible
  by 8.

  Given an `input` shaped `[s0, s1, ..., s_n]`, the output is
  a `uint8` tensor shaped `[s0, s1, ..., s_n / 8]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `bool`, `half`, `float32`, `float64`, `int8`, `int16`, `int32`, `int64`.
      Values to compare against `threshold` and bitpack.
    threshold: A `Tensor`. Must have the same type as `input`.
      Threshold to compare against.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8`.
  "
  [ input threshold name ]
  (py/call-attr quantization "compare_and_bitpack"  input threshold name ))

(defn compare-and-bitpack-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function compare_and_bitpack
  "
  [ input threshold name ctx ]
  (py/call-attr quantization "compare_and_bitpack_eager_fallback"  input threshold name ctx ))

(defn complex-abs 
  "Computes the complex absolute value of a tensor.

  Given a tensor `x` of complex numbers, this operation returns a tensor of type
  `float` or `double` that is the absolute value of each element in `x`. All
  elements in `x` must be complex numbers of the form \\(a + bj\\). The absolute
  value is computed as \\( \sqrt{a^2 + b^2}\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
    Tout: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  "
  [x & {:keys [Tout name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "complex_abs" [x] {:Tout Tout :name name }))

(defn complex-abs-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function complex_abs
  "
  [x & {:keys [Tout name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "complex_abs_eager_fallback" [x] {:Tout Tout :name name :ctx ctx }))

(defn conj 
  "Returns the complex conjugate of a complex number.

  Given a tensor `input` of complex numbers, this operation returns a tensor of
  complex numbers that are the complex conjugate of each element in `input`. The
  complex numbers in `input` must be of the form \\(a + bj\\), where *a* is the
  real part and *b* is the imaginary part.

  The complex conjugate returned by this operation is of the form \\(a - bj\\).

  For example:

  ```
  # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`, `variant`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input name ]
  (py/call-attr quantization "conj"  input name ))

(defn conj-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function conj
  "
  [ input name ctx ]
  (py/call-attr quantization "conj_eager_fallback"  input name ctx ))

(defn conv2d 
  "Computes a 2-D convolution given 4-D `input` and `filter` tensors.

  Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
  and a filter / kernel tensor of shape
  `[filter_height, filter_width, in_channels, out_channels]`, this op
  performs the following:

  1. Flattens the filter to a 2-D matrix with shape
     `[filter_height * filter_width * in_channels, output_channels]`.
  2. Extracts image patches from the input tensor to form a *virtual*
     tensor of shape `[batch, out_height, out_width,
     filter_height * filter_width * in_channels]`.
  3. For each patch, right-multiplies the filter matrix and the image patch
     vector.

  In detail, with the default NHWC format,

      output[b, i, j, k] =
          sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                          filter[di, dj, q, k]

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      A 4-D tensor. The dimension order is interpreted according to the value
      of `data_format`, see below for details.
    filter: A `Tensor`. Must have the same type as `input`.
      A 4-D tensor of shape
      `[filter_height, filter_width, in_channels, out_channels]`
    strides: A list of `ints`.
      1-D tensor of length 4.  The stride of the sliding window for each
      dimension of `input`. The dimension order is determined by the value of
      `data_format`, see below for details.
    padding: A `string` from: `\"SAME\", \"VALID\", \"EXPLICIT\"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    explicit_paddings: An optional list of `ints`. Defaults to `[]`.
      If `padding` is `\"EXPLICIT\"`, the list of explicit padding amounts. For the ith
      dimension, the amount of padding inserted before and after the dimension is
      `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
      `padding` is not `\"EXPLICIT\"`, `explicit_paddings` must be empty.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, channels, height, width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each
      filter element on that dimension. The dimension order is determined by the
      value of `data_format`, see above for details. Dilations in the batch and
      depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input filter strides padding & {:keys [use_cudnn_on_gpu explicit_paddings data_format dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "conv2d" [input filter strides padding] {:use_cudnn_on_gpu use_cudnn_on_gpu :explicit_paddings explicit_paddings :data_format data_format :dilations dilations :name name }))

(defn conv2d-backprop-filter 
  "Computes the gradients of convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
    filter_sizes: A `Tensor` of type `int32`.
      An integer vector representing the tensor shape of `filter`,
      where `filter` is a 4-D
      `[filter_height, filter_width, in_channels, out_channels]` tensor.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution. Must be in the same order as the dimension specified with
      format.
    padding: A `string` from: `\"SAME\", \"VALID\", \"EXPLICIT\"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    explicit_paddings: An optional list of `ints`. Defaults to `[]`.
      If `padding` is `\"EXPLICIT\"`, the list of explicit padding amounts. For the ith
      dimension, the amount of padding inserted before and after the dimension is
      `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
      `padding` is not `\"EXPLICIT\"`, `explicit_paddings` must be empty.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input filter_sizes out_backprop strides padding & {:keys [use_cudnn_on_gpu explicit_paddings data_format dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "conv2d_backprop_filter" [input filter_sizes out_backprop strides padding] {:use_cudnn_on_gpu use_cudnn_on_gpu :explicit_paddings explicit_paddings :data_format data_format :dilations dilations :name name }))

(defn conv2d-backprop-filter-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function conv2d_backprop_filter
  "
  [input filter_sizes out_backprop strides padding & {:keys [use_cudnn_on_gpu explicit_paddings data_format dilations name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "conv2d_backprop_filter_eager_fallback" [input filter_sizes out_backprop strides padding] {:use_cudnn_on_gpu use_cudnn_on_gpu :explicit_paddings explicit_paddings :data_format data_format :dilations dilations :name name :ctx ctx }))

(defn conv2d-backprop-input 
  "Computes the gradients of convolution with respect to the input.

  Args:
    input_sizes: A `Tensor` of type `int32`.
      An integer vector representing the shape of `input`,
      where `input` is a 4-D `[batch, height, width, channels]` tensor.
    filter: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape
      `[filter_height, filter_width, in_channels, out_channels]`.
    out_backprop: A `Tensor`. Must have the same type as `filter`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution. Must be in the same order as the dimension specified with
      format.
    padding: A `string` from: `\"SAME\", \"VALID\", \"EXPLICIT\"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    explicit_paddings: An optional list of `ints`. Defaults to `[]`.
      If `padding` is `\"EXPLICIT\"`, the list of explicit padding amounts. For the ith
      dimension, the amount of padding inserted before and after the dimension is
      `explicit_paddings[2 * i]` and `explicit_paddings[2 * i + 1]`, respectively. If
      `padding` is not `\"EXPLICIT\"`, `explicit_paddings` must be empty.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `filter`.
  "
  [input_sizes filter out_backprop strides padding & {:keys [use_cudnn_on_gpu explicit_paddings data_format dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "conv2d_backprop_input" [input_sizes filter out_backprop strides padding] {:use_cudnn_on_gpu use_cudnn_on_gpu :explicit_paddings explicit_paddings :data_format data_format :dilations dilations :name name }))

(defn conv2d-backprop-input-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function conv2d_backprop_input
  "
  [input_sizes filter out_backprop strides padding & {:keys [use_cudnn_on_gpu explicit_paddings data_format dilations name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "conv2d_backprop_input_eager_fallback" [input_sizes filter out_backprop strides padding] {:use_cudnn_on_gpu use_cudnn_on_gpu :explicit_paddings explicit_paddings :data_format data_format :dilations dilations :name name :ctx ctx }))

(defn conv2d-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function conv2d
  "
  [input filter strides padding & {:keys [use_cudnn_on_gpu explicit_paddings data_format dilations name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "conv2d_eager_fallback" [input filter strides padding] {:use_cudnn_on_gpu use_cudnn_on_gpu :explicit_paddings explicit_paddings :data_format data_format :dilations dilations :name name :ctx ctx }))

(defn conv3d 
  "Computes a 3-D convolution given 5-D `input` and `filter` tensors.

  In signal processing, cross-correlation is a measure of similarity of
  two waveforms as a function of a time-lag applied to one of them. This
  is also known as a sliding dot product or sliding inner-product.

  Our Conv3D implements a form of cross-correlation.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      Shape `[batch, in_depth, in_height, in_width, in_channels]`.
    filter: A `Tensor`. Must have the same type as `input`.
      Shape `[filter_depth, filter_height, filter_width, in_channels,
      out_channels]`. `in_channels` must match between `input` and `filter`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NDHWC\", \"NCDHW\"`. Defaults to `\"NDHWC\"`.
      The data format of the input and output data. With the
      default format \"NDHWC\", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCDHW\", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1, 1]`.
      1-D tensor of length 5.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each
      filter element on that dimension. The dimension order is determined by the
      value of `data_format`, see above for details. Dilations in the batch and
      depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input filter strides padding & {:keys [data_format dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "conv3d" [input filter strides padding] {:data_format data_format :dilations dilations :name name }))

(defn conv3d-backprop-filter 
  "Computes the gradients of 3-D convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      Shape `[batch, depth, rows, cols, in_channels]`.
    filter: A `Tensor`. Must have the same type as `input`.
      Shape `[depth, rows, cols, in_channels, out_channels]`.
      `in_channels` must match between `input` and `filter`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
      out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1, 1]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input filter out_backprop strides padding & {:keys [dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "conv3d_backprop_filter" [input filter out_backprop strides padding] {:dilations dilations :name name }))

(defn conv3d-backprop-filter-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function conv3d_backprop_filter
  "
  [input filter out_backprop strides padding & {:keys [dilations name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "conv3d_backprop_filter_eager_fallback" [input filter out_backprop strides padding] {:dilations dilations :name name :ctx ctx }))

(defn conv3d-backprop-filter-v2 
  "Computes the gradients of 3-D convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      Shape `[batch, depth, rows, cols, in_channels]`.
    filter_sizes: A `Tensor` of type `int32`.
      An integer vector representing the tensor shape of `filter`,
      where `filter` is a 5-D
      `[filter_depth, filter_height, filter_width, in_channels, out_channels]`
      tensor.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
      out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NDHWC\", \"NCDHW\"`. Defaults to `\"NDHWC\"`.
      The data format of the input and output data. With the
      default format \"NDHWC\", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCDHW\", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1, 1]`.
      1-D tensor of length 5.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each
      filter element on that dimension. The dimension order is determined by the
      value of `data_format`, see above for details. Dilations in the batch and
      depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input filter_sizes out_backprop strides padding & {:keys [data_format dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "conv3d_backprop_filter_v2" [input filter_sizes out_backprop strides padding] {:data_format data_format :dilations dilations :name name }))

(defn conv3d-backprop-filter-v2-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function conv3d_backprop_filter_v2
  "
  [input filter_sizes out_backprop strides padding & {:keys [data_format dilations name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "conv3d_backprop_filter_v2_eager_fallback" [input filter_sizes out_backprop strides padding] {:data_format data_format :dilations dilations :name name :ctx ctx }))

(defn conv3d-backprop-input 
  "Computes the gradients of 3-D convolution with respect to the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      Shape `[batch, depth, rows, cols, in_channels]`.
    filter: A `Tensor`. Must have the same type as `input`.
      Shape `[depth, rows, cols, in_channels, out_channels]`.
      `in_channels` must match between `input` and `filter`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
      out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1, 1]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input filter out_backprop strides padding & {:keys [dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "conv3d_backprop_input" [input filter out_backprop strides padding] {:dilations dilations :name name }))

(defn conv3d-backprop-input-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function conv3d_backprop_input
  "
  [input filter out_backprop strides padding & {:keys [dilations name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "conv3d_backprop_input_eager_fallback" [input filter out_backprop strides padding] {:dilations dilations :name name :ctx ctx }))

(defn conv3d-backprop-input-v2 
  "Computes the gradients of 3-D convolution with respect to the input.

  Args:
    input_sizes: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      An integer vector representing the tensor shape of `input`,
      where `input` is a 5-D
      `[batch, depth, rows, cols, in_channels]` tensor.
    filter: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      Shape `[depth, rows, cols, in_channels, out_channels]`.
      `in_channels` must match between `input` and `filter`.
    out_backprop: A `Tensor`. Must have the same type as `filter`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
      out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NDHWC\", \"NCDHW\"`. Defaults to `\"NDHWC\"`.
      The data format of the input and output data. With the
      default format \"NDHWC\", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCDHW\", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1, 1]`.
      1-D tensor of length 5.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each
      filter element on that dimension. The dimension order is determined by the
      value of `data_format`, see above for details. Dilations in the batch and
      depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `filter`.
  "
  [input_sizes filter out_backprop strides padding & {:keys [data_format dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "conv3d_backprop_input_v2" [input_sizes filter out_backprop strides padding] {:data_format data_format :dilations dilations :name name }))

(defn conv3d-backprop-input-v2-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function conv3d_backprop_input_v2
  "
  [input_sizes filter out_backprop strides padding & {:keys [data_format dilations name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "conv3d_backprop_input_v2_eager_fallback" [input_sizes filter out_backprop strides padding] {:data_format data_format :dilations dilations :name name :ctx ctx }))

(defn conv3d-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function conv3d
  "
  [input filter strides padding & {:keys [data_format dilations name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "conv3d_eager_fallback" [input filter strides padding] {:data_format data_format :dilations dilations :name name :ctx ctx }))

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
  (py/call-attr quantization "cos"  x name ))

(defn cos-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function cos
  "
  [ x name ctx ]
  (py/call-attr quantization "cos_eager_fallback"  x name ctx ))

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
  (py/call-attr quantization "cosh"  x name ))

(defn cosh-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function cosh
  "
  [ x name ctx ]
  (py/call-attr quantization "cosh_eager_fallback"  x name ctx ))

(defn cross 
  "Compute the pairwise cross product.

  `a` and `b` must be the same shape; they can either be simple 3-element vectors,
  or any shape where the innermost dimension is 3. In the latter case, each pair
  of corresponding 3-element vectors is cross-multiplied independently.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      A tensor containing 3-element vectors.
    b: A `Tensor`. Must have the same type as `a`.
      Another tensor, of same type and shape as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  "
  [ a b name ]
  (py/call-attr quantization "cross"  a b name ))

(defn cross-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function cross
  "
  [ a b name ctx ]
  (py/call-attr quantization "cross_eager_fallback"  a b name ctx ))

(defn cumprod 
  "Compute the cumulative product of the tensor `x` along `axis`.

  By default, this op performs an inclusive cumprod, which means that the first
  element of the input is identical to the first element of the output:

  ```python
  tf.cumprod([a, b, c])  # => [a, a * b, a * b * c]
  ```

  By setting the `exclusive` kwarg to `True`, an exclusive cumprod is
  performed instead:

  ```python
  tf.cumprod([a, b, c], exclusive=True)  # => [1, a, a * b]
  ```

  By setting the `reverse` kwarg to `True`, the cumprod is performed in the
  opposite direction:

  ```python
  tf.cumprod([a, b, c], reverse=True)  # => [a * b * c, b * c, c]
  ```

  This is more efficient than using separate `tf.reverse` ops.

  The `reverse` and `exclusive` kwargs can also be combined:

  ```python
  tf.cumprod([a, b, c], exclusive=True, reverse=True)  # => [b * c, c, 1]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A `Tensor`. Must be one of the following types: `float32`, `float64`,
      `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
      `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `Tensor` of type `int32` (default: 0). Must be in the range
      `[-rank(x), rank(x))`.
    exclusive: An optional `bool`. Defaults to `False`.
      If `True`, perform exclusive cumprod.
    reverse: An optional `bool`. Defaults to `False`.
      A `bool` (default: False).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [x axis & {:keys [exclusive reverse name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "cumprod" [x axis] {:exclusive exclusive :reverse reverse :name name }))

(defn cumprod-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function cumprod
  "
  [x axis & {:keys [exclusive reverse name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "cumprod_eager_fallback" [x axis] {:exclusive exclusive :reverse reverse :name name :ctx ctx }))

(defn cumsum 
  "Compute the cumulative sum of the tensor `x` along `axis`.

  By default, this op performs an inclusive cumsum, which means that the first
  element of the input is identical to the first element of the output:

  ```python
  tf.cumsum([a, b, c])  # => [a, a + b, a + b + c]
  ```

  By setting the `exclusive` kwarg to `True`, an exclusive cumsum is
  performed instead:

  ```python
  tf.cumsum([a, b, c], exclusive=True)  # => [0, a, a + b]
  ```

  By setting the `reverse` kwarg to `True`, the cumsum is performed in the
  opposite direction:

  ```python
  tf.cumsum([a, b, c], reverse=True)  # => [a + b + c, b + c, c]
  ```

  This is more efficient than using separate `tf.reverse` ops.

  The `reverse` and `exclusive` kwargs can also be combined:

  ```python
  tf.cumsum([a, b, c], exclusive=True, reverse=True)  # => [b + c, c, 0]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A `Tensor`. Must be one of the following types: `float32`, `float64`,
      `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
      `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `Tensor` of type `int32` (default: 0). Must be in the range
      `[-rank(x), rank(x))`.
    exclusive: An optional `bool`. Defaults to `False`.
      If `True`, perform exclusive cumsum.
    reverse: An optional `bool`. Defaults to `False`.
      A `bool` (default: False).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [x axis & {:keys [exclusive reverse name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "cumsum" [x axis] {:exclusive exclusive :reverse reverse :name name }))

(defn cumsum-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function cumsum
  "
  [x axis & {:keys [exclusive reverse name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "cumsum_eager_fallback" [x axis] {:exclusive exclusive :reverse reverse :name name :ctx ctx }))

(defn cumulative-logsumexp 
  "Compute the cumulative product of the tensor `x` along `axis`.

  By default, this op performs an inclusive cumulative log-sum-exp,
  which means that the first
  element of the input is identical to the first element of the output:
  ```python
  tf.math.cumulative_logsumexp([a, b, c])  # => [a, log(exp(a) + exp(b)), log(exp(a) + exp(b) + exp(c))]
  ```

  By setting the `exclusive` kwarg to `True`, an exclusive cumulative log-sum-exp is
  performed instead:
  ```python
  tf.cumulative_logsumexp([a, b, c], exclusive=True)  # => [-inf, a, log(exp(a) * exp(b))]
  ```
  Note that the neutral element of the log-sum-exp operation is `-inf`,
  however, for performance reasons, the minimal value representable by the
  floating point type is used instead.

  By setting the `reverse` kwarg to `True`, the cumulative log-sum-exp is performed in the
  opposite direction.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      A `Tensor`. Must be one of the following types: `float16`, `float32`, `float64`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `Tensor` of type `int32` (default: 0). Must be in the range
      `[-rank(x), rank(x))`.
    exclusive: An optional `bool`. Defaults to `False`.
      If `True`, perform exclusive cumulative log-sum-exp.
    reverse: An optional `bool`. Defaults to `False`.
      A `bool` (default: False).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [x axis & {:keys [exclusive reverse name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "cumulative_logsumexp" [x axis] {:exclusive exclusive :reverse reverse :name name }))

(defn cumulative-logsumexp-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function cumulative_logsumexp
  "
  [x axis & {:keys [exclusive reverse name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "cumulative_logsumexp_eager_fallback" [x axis] {:exclusive exclusive :reverse reverse :name name :ctx ctx }))

(defn data-format-dim-map 
  "Returns the dimension index in the destination data format given the one in

  the source data format.

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A Tensor with each element as a dimension index in source data format.
      Must be in the range [-4, 4).
    src_format: An optional `string`. Defaults to `\"NHWC\"`.
      source data format.
    dst_format: An optional `string`. Defaults to `\"NCHW\"`.
      destination data format.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [x & {:keys [src_format dst_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "data_format_dim_map" [x] {:src_format src_format :dst_format dst_format :name name }))

(defn data-format-dim-map-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function data_format_dim_map
  "
  [x & {:keys [src_format dst_format name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "data_format_dim_map_eager_fallback" [x] {:src_format src_format :dst_format dst_format :name name :ctx ctx }))

(defn data-format-vec-permute 
  "Returns the permuted vector/tensor in the destination data format given the

  one in the source data format.

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Vector of size 4 or Tensor of shape (4, 2) in source data format.
    src_format: An optional `string`. Defaults to `\"NHWC\"`.
      source data format.
    dst_format: An optional `string`. Defaults to `\"NCHW\"`.
      destination data format.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [x & {:keys [src_format dst_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "data_format_vec_permute" [x] {:src_format src_format :dst_format dst_format :name name }))

(defn data-format-vec-permute-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function data_format_vec_permute
  "
  [x & {:keys [src_format dst_format name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "data_format_vec_permute_eager_fallback" [x] {:src_format src_format :dst_format dst_format :name name :ctx ctx }))

(defn deprecated-endpoints 
  "Decorator for marking endpoints deprecated.

  This decorator does not print deprecation messages.
  TODO(annarev): eventually start printing deprecation warnings when
  @deprecation_endpoints decorator is added.

  Args:
    *args: Deprecated endpoint names.

  Returns:
    A function that takes symbol as an argument and adds
    _tf_deprecated_api_names to that symbol.
    _tf_deprecated_api_names would be set to a list of deprecated
    endpoint names for the symbol.
  "
  [  ]
  (py/call-attr quantization "deprecated_endpoints"  ))

(defn depthwise-conv2d-native 
  "Computes a 2-D depthwise convolution given 4-D `input` and `filter` tensors.

  Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
  and a filter / kernel tensor of shape
  `[filter_height, filter_width, in_channels, channel_multiplier]`, containing
  `in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
  a different filter to each input channel (expanding from 1 channel to
  `channel_multiplier` channels for each), then concatenates the results
  together. Thus, the output has `in_channels * channel_multiplier` channels.

  ```
  for k in 0..in_channels-1
    for q in 0..channel_multiplier-1
      output[b, i, j, k * channel_multiplier + q] =
        sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                          filter[di, dj, k, q]
  ```

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    filter: A `Tensor`. Must have the same type as `input`.
    strides: A list of `ints`.
      1-D of length 4.  The stride of the sliding window for each dimension
      of `input`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, channels, height, width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input filter strides padding & {:keys [data_format dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "depthwise_conv2d_native" [input filter strides padding] {:data_format data_format :dilations dilations :name name }))

(defn depthwise-conv2d-native-backprop-filter 
  "Computes the gradients of depthwise convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape based on `data_format`.  For example, if
      `data_format` is 'NHWC' then `input` is a 4-D `[batch, in_height,
      in_width, in_channels]` tensor.
    filter_sizes: A `Tensor` of type `int32`.
      An integer vector representing the tensor shape of `filter`,
      where `filter` is a 4-D
      `[filter_height, filter_width, in_channels, depthwise_multiplier]` tensor.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape  based on `data_format`.
      For example, if `data_format` is 'NHWC' then
      out_backprop shape is `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, channels, height, width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input filter_sizes out_backprop strides padding & {:keys [data_format dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "depthwise_conv2d_native_backprop_filter" [input filter_sizes out_backprop strides padding] {:data_format data_format :dilations dilations :name name }))

(defn depthwise-conv2d-native-backprop-filter-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function depthwise_conv2d_native_backprop_filter
  "
  [input filter_sizes out_backprop strides padding & {:keys [data_format dilations name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "depthwise_conv2d_native_backprop_filter_eager_fallback" [input filter_sizes out_backprop strides padding] {:data_format data_format :dilations dilations :name name :ctx ctx }))

(defn depthwise-conv2d-native-backprop-input 
  "Computes the gradients of depthwise convolution with respect to the input.

  Args:
    input_sizes: A `Tensor` of type `int32`.
      An integer vector representing the shape of `input`, based
      on `data_format`.  For example, if `data_format` is 'NHWC' then
       `input` is a 4-D `[batch, height, width, channels]` tensor.
    filter: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      4-D with shape
      `[filter_height, filter_width, in_channels, depthwise_multiplier]`.
    out_backprop: A `Tensor`. Must have the same type as `filter`.
      4-D with shape  based on `data_format`.
      For example, if `data_format` is 'NHWC' then
      out_backprop shape is `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, height, width, channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, channels, height, width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each filter
      element on that dimension. The dimension order is determined by the value of
      `data_format`, see above for details. Dilations in the batch and depth
      dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `filter`.
  "
  [input_sizes filter out_backprop strides padding & {:keys [data_format dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "depthwise_conv2d_native_backprop_input" [input_sizes filter out_backprop strides padding] {:data_format data_format :dilations dilations :name name }))

(defn depthwise-conv2d-native-backprop-input-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function depthwise_conv2d_native_backprop_input
  "
  [input_sizes filter out_backprop strides padding & {:keys [data_format dilations name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "depthwise_conv2d_native_backprop_input_eager_fallback" [input_sizes filter out_backprop strides padding] {:data_format data_format :dilations dilations :name name :ctx ctx }))

(defn depthwise-conv2d-native-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function depthwise_conv2d_native
  "
  [input filter strides padding & {:keys [data_format dilations name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "depthwise_conv2d_native_eager_fallback" [input filter strides padding] {:data_format data_format :dilations dilations :name name :ctx ctx }))

(defn dequantize 
  "Dequantize the 'input' tensor into a float Tensor.

  [min_range, max_range] are scalar floats that specify the range for
  the 'input' data. The 'mode' attribute controls exactly which calculations are
  used to convert the float values to their quantized equivalents.

  In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:

  ```
  if T == qint8: in[i] += (range(T) + 1)/ 2.0
  out[i] = min_range + (in[i]* (max_range - min_range) / range(T))
  ```
  here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`

  *MIN_COMBINED Mode Example*

  If the input comes from a QuantizedRelu6, the output type is
  quint8 (range of 0-255) but the possible range of QuantizedRelu6 is
  0-6.  The min_range and max_range values are therefore 0.0 and 6.0.
  Dequantize on quint8 will take each value, cast to float, and multiply
  by 6 / 255.
  Note that if quantizedtype is qint8, the operation will additionally add
  each value by 128 prior to casting.

  If the mode is 'MIN_FIRST', then this approach is used:

  ```c++
  num_discrete_values = 1 << (# of bits in T)
  range_adjust = num_discrete_values / (num_discrete_values - 1)
  range = (range_max - range_min) * range_adjust
  range_scale = range / num_discrete_values
  const double offset_input = static_cast<double>(input) - lowest_quantized;
  result = range_min + ((input - numeric_limits<T>::min()) * range_scale)
  ```

  *SCALED mode Example*

  `SCALED` mode matches the quantization approach used in
  `QuantizeAndDequantize{V2|V3}`.

  If the mode is `SCALED`, we do not use the full range of the output type,
  choosing to elide the lowest possible value for symmetry (e.g., output range is
  -127 to 127, not -128 to 127 for signed 8 bit quantization), so that 0.0 maps to
  0.

  We first find the range of values in our tensor. The
  range we use is always centered on 0, so we find m such that
  ```c++
    m = max(abs(input_min), abs(input_max))
  ```

  Our input tensor range is then `[-m, m]`.

  Next, we choose our fixed-point quantization buckets, `[min_fixed, max_fixed]`.
  If T is signed, this is
  ```
    num_bits = sizeof(T) * 8
    [min_fixed, max_fixed] =
        [-(1 << (num_bits - 1) - 1), (1 << (num_bits - 1)) - 1]
  ```

  Otherwise, if T is unsigned, the fixed-point range is
  ```
    [min_fixed, max_fixed] = [0, (1 << num_bits) - 1]
  ```

  From this we compute our scaling factor, s:
  ```c++
    s = (2 * m) / (max_fixed - min_fixed)
  ```

  Now we can dequantize the elements of our tensor:
  ```c++
  result = input * s
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_range: A `Tensor` of type `float32`.
      The minimum scalar value possibly produced for the input.
    max_range: A `Tensor` of type `float32`.
      The maximum scalar value possibly produced for the input.
    mode: An optional `string` from: `\"MIN_COMBINED\", \"MIN_FIRST\", \"SCALED\"`. Defaults to `\"MIN_COMBINED\"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  "
  [input min_range max_range & {:keys [mode name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "dequantize" [input min_range max_range] {:mode mode :name name }))

(defn digamma 
  "Computes Psi, the derivative of Lgamma (the log of the absolute value of

  `Gamma(x)`), element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "digamma"  x name ))

(defn digamma-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function digamma
  "
  [ x name ctx ]
  (py/call-attr quantization "digamma_eager_fallback"  x name ctx ))

(defn dilation2d 
  "Computes the grayscale dilation of 4-D `input` and 3-D `filter` tensors.

  The `input` tensor has shape `[batch, in_height, in_width, depth]` and the
  `filter` tensor has shape `[filter_height, filter_width, depth]`, i.e., each
  input channel is processed independently of the others with its own structuring
  function. The `output` tensor has shape
  `[batch, out_height, out_width, depth]`. The spatial dimensions of the output
  tensor depend on the `padding` algorithm. We currently only support the default
  \"NHWC\" `data_format`.

  In detail, the grayscale morphological 2-D dilation is the max-sum correlation
  (for consistency with `conv2d`, we use unmirrored filters):

      output[b, y, x, c] =
         max_{dy, dx} input[b,
                            strides[1] * y + rates[1] * dy,
                            strides[2] * x + rates[2] * dx,
                            c] +
                      filter[dy, dx, c]

  Max-pooling is a special case when the filter has size equal to the pooling
  kernel size and contains all zeros.

  Note on duality: The dilation of `input` by the `filter` is equal to the
  negation of the erosion of `-input` by the reflected `filter`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      4-D with shape `[batch, in_height, in_width, depth]`.
    filter: A `Tensor`. Must have the same type as `input`.
      3-D with shape `[filter_height, filter_width, depth]`.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the input
      tensor. Must be: `[1, stride_height, stride_width, 1]`.
    rates: A list of `ints` that has length `>= 4`.
      The input stride for atrous morphological dilation. Must be:
      `[1, rate_height, rate_width, 1]`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input filter strides rates padding name ]
  (py/call-attr quantization "dilation2d"  input filter strides rates padding name ))

(defn dilation2d-backprop-filter 
  "Computes the gradient of morphological 2-D dilation with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      4-D with shape `[batch, in_height, in_width, depth]`.
    filter: A `Tensor`. Must have the same type as `input`.
      3-D with shape `[filter_height, filter_width, depth]`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, depth]`.
    strides: A list of `ints` that has length `>= 4`.
      1-D of length 4. The stride of the sliding window for each dimension of
      the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
    rates: A list of `ints` that has length `>= 4`.
      1-D of length 4. The input stride for atrous morphological dilation.
      Must be: `[1, rate_height, rate_width, 1]`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input filter out_backprop strides rates padding name ]
  (py/call-attr quantization "dilation2d_backprop_filter"  input filter out_backprop strides rates padding name ))

(defn dilation2d-backprop-filter-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function dilation2d_backprop_filter
  "
  [ input filter out_backprop strides rates padding name ctx ]
  (py/call-attr quantization "dilation2d_backprop_filter_eager_fallback"  input filter out_backprop strides rates padding name ctx ))

(defn dilation2d-backprop-input 
  "Computes the gradient of morphological 2-D dilation with respect to the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      4-D with shape `[batch, in_height, in_width, depth]`.
    filter: A `Tensor`. Must have the same type as `input`.
      3-D with shape `[filter_height, filter_width, depth]`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, depth]`.
    strides: A list of `ints` that has length `>= 4`.
      1-D of length 4. The stride of the sliding window for each dimension of
      the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
    rates: A list of `ints` that has length `>= 4`.
      1-D of length 4. The input stride for atrous morphological dilation.
      Must be: `[1, rate_height, rate_width, 1]`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input filter out_backprop strides rates padding name ]
  (py/call-attr quantization "dilation2d_backprop_input"  input filter out_backprop strides rates padding name ))

(defn dilation2d-backprop-input-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function dilation2d_backprop_input
  "
  [ input filter out_backprop strides rates padding name ctx ]
  (py/call-attr quantization "dilation2d_backprop_input_eager_fallback"  input filter out_backprop strides rates padding name ctx ))

(defn dilation2d-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function dilation2d
  "
  [ input filter strides rates padding name ctx ]
  (py/call-attr quantization "dilation2d_eager_fallback"  input filter strides rates padding name ctx ))

(defn div 
  "Returns x / y element-wise.

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
  (py/call-attr quantization "div"  x y name ))

(defn div-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function div
  "
  [ x y name ctx ]
  (py/call-attr quantization "div_eager_fallback"  x y name ctx ))

(defn div-no-nan 
  "Returns 0 if the denominator is zero.

  
  *NOTE*: `DivNoNan` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "div_no_nan"  x y name ))

(defn div-no-nan-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function div_no_nan
  "
  [ x y name ctx ]
  (py/call-attr quantization "div_no_nan_eager_fallback"  x y name ctx ))

(defn elu 
  "Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.

  See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
  ](http://arxiv.org/abs/1511.07289)

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  "
  [ features name ]
  (py/call-attr quantization "elu"  features name ))

(defn elu-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function elu
  "
  [ features name ctx ]
  (py/call-attr quantization "elu_eager_fallback"  features name ctx ))

(defn elu-grad 
  "Computes gradients for the exponential linear (Elu) operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      The backpropagated gradients to the corresponding Elu operation.
    outputs: A `Tensor`. Must have the same type as `gradients`.
      The outputs of the corresponding Elu operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  "
  [ gradients outputs name ]
  (py/call-attr quantization "elu_grad"  gradients outputs name ))

(defn elu-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function elu_grad
  "
  [ gradients outputs name ctx ]
  (py/call-attr quantization "elu_grad_eager_fallback"  gradients outputs name ctx ))

(defn equal 
  "Returns the truth value of (x == y) element-wise.

  *NOTE*: `Equal` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  ```python
  x = tf.constant([2, 4])
  y = tf.constant(2)
  tf.math.equal(x, y) ==> array([True, False])

  x = tf.constant([2, 4])
  y = tf.constant([2, 4])
  tf.math.equal(x, y) ==> array([True,  True])
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `quint8`, `qint8`, `qint32`, `string`, `bool`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    incompatible_shape_error: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [x y & {:keys [incompatible_shape_error name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "equal" [x y] {:incompatible_shape_error incompatible_shape_error :name name }))

(defn equal-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function equal
  "
  [x y & {:keys [incompatible_shape_error name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "equal_eager_fallback" [x y] {:incompatible_shape_error incompatible_shape_error :name name :ctx ctx }))

(defn erf 
  "Computes the Gauss error function of `x` element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.

    If `x` is a `SparseTensor`, returns
    `SparseTensor(x.indices, tf.math.erf(x.values, ...), x.dense_shape)`"
  [ x name ]
  (py/call-attr quantization "erf"  x name ))

(defn erf-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function erf
  "
  [ x name ctx ]
  (py/call-attr quantization "erf_eager_fallback"  x name ctx ))

(defn erfc 
  "Computes the complementary error function of `x` element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "erfc"  x name ))

(defn erfc-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function erfc
  "
  [ x name ctx ]
  (py/call-attr quantization "erfc_eager_fallback"  x name ctx ))

(defn euclidean-norm 
  "Computes the euclidean norm of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The tensor to reduce.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce. Must be in the range
      `[-rank(input), rank(input))`.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input axis & {:keys [keep_dims name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "euclidean_norm" [input axis] {:keep_dims keep_dims :name name }))

(defn euclidean-norm-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function euclidean_norm
  "
  [input axis & {:keys [keep_dims name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "euclidean_norm_eager_fallback" [input axis] {:keep_dims keep_dims :name name :ctx ctx }))

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
  (py/call-attr quantization "exp"  x name ))

(defn exp-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function exp
  "
  [ x name ctx ]
  (py/call-attr quantization "exp_eager_fallback"  x name ctx ))

(defn expm1 
  "Computes `exp(x) - 1` element-wise.

    i.e. `exp(x) - 1` or `e^(x) - 1`, where `x` is the input tensor.
    `e` denotes Euler's number and is approximately equal to 2.718281.

    ```python
    x = tf.constant(2.0)
    tf.math.expm1(x) ==> 6.389056

    x = tf.constant([2.0, 8.0])
    tf.math.expm1(x) ==> array([6.389056, 2979.958], dtype=float32)

    x = tf.constant(1 + 1j)
    tf.math.expm1(x) ==> (0.46869393991588515+2.2873552871788423j)
    ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "expm1"  x name ))

(defn expm1-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function expm1
  "
  [ x name ctx ]
  (py/call-attr quantization "expm1_eager_fallback"  x name ctx ))

(defn floor 
  "Returns element-wise largest integer not greater than x.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "floor"  x name ))

(defn floor-div 
  "Returns x // y element-wise.

  *NOTE*: `floor_div` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "floor_div"  x y name ))

(defn floor-div-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function floor_div
  "
  [ x y name ctx ]
  (py/call-attr quantization "floor_div_eager_fallback"  x y name ctx ))

(defn floor-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function floor
  "
  [ x name ctx ]
  (py/call-attr quantization "floor_eager_fallback"  x name ctx ))

(defn floor-mod 
  "Returns element-wise remainder of division. When `x < 0` xor `y < 0` is

  true, this follows Python semantics in that the result here is consistent
  with a flooring divide. E.g. `floor(x / y) * y + mod(x, y) = x`.

  *NOTE*: `math.floormod` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`, `bfloat16`, `half`, `float32`, `float64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "floor_mod"  x y name ))

(defn floor-mod-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function floor_mod
  "
  [ x y name ctx ]
  (py/call-attr quantization "floor_mod_eager_fallback"  x y name ctx ))

(defn fractional-avg-pool 
  "Performs fractional average pooling on the input.

  Fractional average pooling is similar to Fractional max pooling in the pooling
  region generation step. The only difference is that after pooling regions are
  generated, a mean operation is performed instead of a max operation in each
  pooling region.

  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      4-D with shape `[batch, height, width, channels]`.
    pooling_ratio: A list of `floats` that has length `>= 4`.
      Pooling ratio for each dimension of `value`, currently only
      supports row and col dimension and should be >= 1.0. For example, a valid
      pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements
      must be 1.0 because we don't allow pooling on batch and channels
      dimensions. 1.44 and 1.73 are pooling ratio on height and width dimensions
      respectively.
    pseudo_random: An optional `bool`. Defaults to `False`.
      When set to True, generates the pooling sequence in a
      pseudorandom fashion, otherwise, in a random fashion. Check paper [Benjamin
      Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071) for
      difference between pseudorandom and random.
    overlapping: An optional `bool`. Defaults to `False`.
      When set to True, it means when pooling, the values at the boundary
      of adjacent pooling cells are used by both cells. For example:

      `index  0  1  2  3  4`

      `value  20 5  16 3  7`

      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
      The result would be [41/3, 26/3] for fractional avg pooling.
    deterministic: An optional `bool`. Defaults to `False`.
      When set to True, a fixed pooling region will be used when
      iterating over a FractionalAvgPool node in the computation graph. Mainly used
      in unit test to make FractionalAvgPool deterministic.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, row_pooling_sequence, col_pooling_sequence).

    output: A `Tensor`. Has the same type as `value`.
    row_pooling_sequence: A `Tensor` of type `int64`.
    col_pooling_sequence: A `Tensor` of type `int64`.
  "
  [value pooling_ratio & {:keys [pseudo_random overlapping deterministic seed seed2 name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "fractional_avg_pool" [value pooling_ratio] {:pseudo_random pseudo_random :overlapping overlapping :deterministic deterministic :seed seed :seed2 seed2 :name name }))

(defn fractional-avg-pool-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function fractional_avg_pool
  "
  [value pooling_ratio & {:keys [pseudo_random overlapping deterministic seed seed2 name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "fractional_avg_pool_eager_fallback" [value pooling_ratio] {:pseudo_random pseudo_random :overlapping overlapping :deterministic deterministic :seed seed :seed2 seed2 :name name :ctx ctx }))

(defn fractional-avg-pool-grad 
  "Computes gradient of the FractionalAvgPool function.

  Unlike FractionalMaxPoolGrad, we don't need to find arg_max for
  FractionalAvgPoolGrad, we just need to evenly back-propagate each element of
  out_backprop to those indices that form the same pooling cell. Therefore, we
  just need to know the shape of original input tensor, instead of the whole
  tensor.

  Args:
    orig_input_tensor_shape: A `Tensor` of type `int64`.
      Original input tensor shape for `fractional_avg_pool`
    out_backprop: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      4-D with shape `[batch, height, width, channels]`.  Gradients
      w.r.t. the output of `fractional_avg_pool`.
    row_pooling_sequence: A `Tensor` of type `int64`.
      row pooling sequence, form pooling region with
      col_pooling_sequence.
    col_pooling_sequence: A `Tensor` of type `int64`.
      column pooling sequence, form pooling region with
      row_pooling sequence.
    overlapping: An optional `bool`. Defaults to `False`.
      When set to True, it means when pooling, the values at the boundary
      of adjacent pooling cells are used by both cells. For example:

      `index  0  1  2  3  4`

      `value  20 5  16 3  7`

      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
      The result would be [41/3, 26/3] for fractional avg pooling.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `out_backprop`.
  "
  [orig_input_tensor_shape out_backprop row_pooling_sequence col_pooling_sequence & {:keys [overlapping name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "fractional_avg_pool_grad" [orig_input_tensor_shape out_backprop row_pooling_sequence col_pooling_sequence] {:overlapping overlapping :name name }))

(defn fractional-avg-pool-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function fractional_avg_pool_grad
  "
  [orig_input_tensor_shape out_backprop row_pooling_sequence col_pooling_sequence & {:keys [overlapping name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "fractional_avg_pool_grad_eager_fallback" [orig_input_tensor_shape out_backprop row_pooling_sequence col_pooling_sequence] {:overlapping overlapping :name name :ctx ctx }))

(defn fractional-max-pool 
  "Performs fractional max pooling on the input.

  Fractional max pooling is slightly different than regular max pooling.  In
  regular max pooling, you downsize an input set by taking the maximum value of
  smaller N x N subsections of the set (often 2x2), and try to reduce the set by
  a factor of N, where N is an integer.  Fractional max pooling, as you might
  expect from the word \"fractional\", means that the overall reduction ratio N
  does not have to be an integer.

  The sizes of the pooling regions are generated randomly but are fairly uniform.
  For example, let's look at the height dimension, and the constraints on the
  list of rows that will be pool boundaries.

  First we define the following:

  1.  input_row_length : the number of rows from the input set
  2.  output_row_length : which will be smaller than the input
  3.  alpha = input_row_length / output_row_length : our reduction ratio
  4.  K = floor(alpha)
  5.  row_pooling_sequence : this is the result list of pool boundary rows

  Then, row_pooling_sequence should satisfy:

  1.  a[0] = 0 : the first value of the sequence is 0
  2.  a[end] = input_row_length : the last value of the sequence is the size
  3.  K <= (a[i+1] - a[i]) <= K+1 : all intervals are K or K+1 size
  4.  length(row_pooling_sequence) = output_row_length+1

  For more details on fractional max pooling, see this paper:
  [Benjamin Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071)

  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      4-D with shape `[batch, height, width, channels]`.
    pooling_ratio: A list of `floats` that has length `>= 4`.
      Pooling ratio for each dimension of `value`, currently only
      supports row and col dimension and should be >= 1.0. For example, a valid
      pooling ratio looks like [1.0, 1.44, 1.73, 1.0]. The first and last elements
      must be 1.0 because we don't allow pooling on batch and channels
      dimensions. 1.44 and 1.73 are pooling ratio on height and width dimensions
      respectively.
    pseudo_random: An optional `bool`. Defaults to `False`.
      When set to True, generates the pooling sequence in a
      pseudorandom fashion, otherwise, in a random fashion. Check paper [Benjamin
      Graham, Fractional Max-Pooling](http://arxiv.org/abs/1412.6071) for
      difference between pseudorandom and random.
    overlapping: An optional `bool`. Defaults to `False`.
      When set to True, it means when pooling, the values at the boundary
      of adjacent pooling cells are used by both cells. For example:

      `index  0  1  2  3  4`

      `value  20 5  16 3  7`

      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
      The result would be [20, 16] for fractional max pooling.
    deterministic: An optional `bool`. Defaults to `False`.
      When set to True, a fixed pooling region will be used when
      iterating over a FractionalMaxPool node in the computation graph. Mainly used
      in unit test to make FractionalMaxPool deterministic.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, row_pooling_sequence, col_pooling_sequence).

    output: A `Tensor`. Has the same type as `value`.
    row_pooling_sequence: A `Tensor` of type `int64`.
    col_pooling_sequence: A `Tensor` of type `int64`.
  "
  [value pooling_ratio & {:keys [pseudo_random overlapping deterministic seed seed2 name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "fractional_max_pool" [value pooling_ratio] {:pseudo_random pseudo_random :overlapping overlapping :deterministic deterministic :seed seed :seed2 seed2 :name name }))

(defn fractional-max-pool-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function fractional_max_pool
  "
  [value pooling_ratio & {:keys [pseudo_random overlapping deterministic seed seed2 name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "fractional_max_pool_eager_fallback" [value pooling_ratio] {:pseudo_random pseudo_random :overlapping overlapping :deterministic deterministic :seed seed :seed2 seed2 :name name :ctx ctx }))

(defn fractional-max-pool-grad 
  "Computes gradient of the FractionalMaxPool function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      Original input for `fractional_max_pool`
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      Original output for `fractional_max_pool`
    out_backprop: A `Tensor`. Must have the same type as `orig_input`.
      4-D with shape `[batch, height, width, channels]`.  Gradients
      w.r.t. the output of `fractional_max_pool`.
    row_pooling_sequence: A `Tensor` of type `int64`.
      row pooling sequence, form pooling region with
      col_pooling_sequence.
    col_pooling_sequence: A `Tensor` of type `int64`.
      column pooling sequence, form pooling region with
      row_pooling sequence.
    overlapping: An optional `bool`. Defaults to `False`.
      When set to True, it means when pooling, the values at the boundary
      of adjacent pooling cells are used by both cells. For example:

      `index  0  1  2  3  4`

      `value  20 5  16 3  7`

      If the pooling sequence is [0, 2, 4], then 16, at index 2 will be used twice.
      The result would be [20, 16] for fractional max pooling.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  "
  [orig_input orig_output out_backprop row_pooling_sequence col_pooling_sequence & {:keys [overlapping name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "fractional_max_pool_grad" [orig_input orig_output out_backprop row_pooling_sequence col_pooling_sequence] {:overlapping overlapping :name name }))

(defn fractional-max-pool-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function fractional_max_pool_grad
  "
  [orig_input orig_output out_backprop row_pooling_sequence col_pooling_sequence & {:keys [overlapping name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "fractional_max_pool_grad_eager_fallback" [orig_input orig_output out_backprop row_pooling_sequence col_pooling_sequence] {:overlapping overlapping :name name :ctx ctx }))

(defn fused-batch-norm-grad 
  "Gradient for batch normalization.

  Note that the size of 4D Tensors are defined by either \"NHWC\" or \"NCHW\".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    y_backprop: A `Tensor`. Must be one of the following types: `float32`.
      A 4D Tensor for the gradient with respect to y.
    x: A `Tensor`. Must have the same type as `y_backprop`.
      A 4D Tensor for input data.
    scale: A `Tensor`. Must have the same type as `y_backprop`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    reserve_space_1: A `Tensor`. Must have the same type as `y_backprop`.
      When is_training is True, a 1D Tensor for the computed batch
      mean to be reused in gradient computation. When is_training is
      False, a 1D Tensor for the population mean to be reused in both
      1st and 2nd order gradient computation.
    reserve_space_2: A `Tensor`. Must have the same type as `y_backprop`.
      When is_training is True, a 1D Tensor for the computed batch
      variance (inverted variance in the cuDNN case) to be reused in
      gradient computation. When is_training is False, a 1D Tensor
      for the population variance to be reused in both 1st and 2nd
      order gradient computation.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      The data format for y_backprop, x, x_backprop.
      Either \"NHWC\" (default) or \"NCHW\".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (x_backprop, scale_backprop, offset_backprop, reserve_space_3, reserve_space_4).

    x_backprop: A `Tensor`. Has the same type as `y_backprop`.
    scale_backprop: A `Tensor`. Has the same type as `y_backprop`.
    offset_backprop: A `Tensor`. Has the same type as `y_backprop`.
    reserve_space_3: A `Tensor`. Has the same type as `y_backprop`.
    reserve_space_4: A `Tensor`. Has the same type as `y_backprop`.
  "
  [y_backprop x scale reserve_space_1 reserve_space_2 & {:keys [epsilon data_format is_training name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "fused_batch_norm_grad" [y_backprop x scale reserve_space_1 reserve_space_2] {:epsilon epsilon :data_format data_format :is_training is_training :name name }))

(defn fused-batch-norm-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function fused_batch_norm_grad
  "
  [y_backprop x scale reserve_space_1 reserve_space_2 & {:keys [epsilon data_format is_training name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "fused_batch_norm_grad_eager_fallback" [y_backprop x scale reserve_space_1 reserve_space_2] {:epsilon epsilon :data_format data_format :is_training is_training :name name :ctx ctx }))

(defn fused-batch-norm-grad-v2 
  "Gradient for batch normalization.

  Note that the size of 4D Tensors are defined by either \"NHWC\" or \"NCHW\".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    y_backprop: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      A 4D Tensor for the gradient with respect to y.
    x: A `Tensor`. Must have the same type as `y_backprop`.
      A 4D Tensor for input data.
    scale: A `Tensor` of type `float32`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    reserve_space_1: A `Tensor`. Must be one of the following types: `float32`.
      When is_training is True, a 1D Tensor for the computed batch
      mean to be reused in gradient computation. When is_training is
      False, a 1D Tensor for the population mean to be reused in both
      1st and 2nd order gradient computation.
    reserve_space_2: A `Tensor`. Must have the same type as `reserve_space_1`.
      When is_training is True, a 1D Tensor for the computed batch
      variance (inverted variance in the cuDNN case) to be reused in
      gradient computation. When is_training is False, a 1D Tensor
      for the population variance to be reused in both 1st and 2nd
      order gradient computation.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      The data format for y_backprop, x, x_backprop.
      Either \"NHWC\" (default) or \"NCHW\".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (x_backprop, scale_backprop, offset_backprop, reserve_space_3, reserve_space_4).

    x_backprop: A `Tensor`. Has the same type as `y_backprop`.
    scale_backprop: A `Tensor`. Has the same type as `reserve_space_1`.
    offset_backprop: A `Tensor`. Has the same type as `reserve_space_1`.
    reserve_space_3: A `Tensor`. Has the same type as `reserve_space_1`.
    reserve_space_4: A `Tensor`. Has the same type as `reserve_space_1`.
  "
  [y_backprop x scale reserve_space_1 reserve_space_2 & {:keys [epsilon data_format is_training name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "fused_batch_norm_grad_v2" [y_backprop x scale reserve_space_1 reserve_space_2] {:epsilon epsilon :data_format data_format :is_training is_training :name name }))

(defn fused-batch-norm-grad-v2-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function fused_batch_norm_grad_v2
  "
  [y_backprop x scale reserve_space_1 reserve_space_2 & {:keys [epsilon data_format is_training name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "fused_batch_norm_grad_v2_eager_fallback" [y_backprop x scale reserve_space_1 reserve_space_2] {:epsilon epsilon :data_format data_format :is_training is_training :name name :ctx ctx }))

(defn fused-batch-norm-grad-v3 
  "Gradient for batch normalization.

  Note that the size of 4D Tensors are defined by either \"NHWC\" or \"NCHW\".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    y_backprop: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      A 4D Tensor for the gradient with respect to y.
    x: A `Tensor`. Must have the same type as `y_backprop`.
      A 4D Tensor for input data.
    scale: A `Tensor` of type `float32`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    reserve_space_1: A `Tensor`. Must be one of the following types: `float32`.
      When is_training is True, a 1D Tensor for the computed batch
      mean to be reused in gradient computation. When is_training is
      False, a 1D Tensor for the population mean to be reused in both
      1st and 2nd order gradient computation.
    reserve_space_2: A `Tensor`. Must have the same type as `reserve_space_1`.
      When is_training is True, a 1D Tensor for the computed batch
      variance (inverted variance in the cuDNN case) to be reused in
      gradient computation. When is_training is False, a 1D Tensor
      for the population variance to be reused in both 1st and 2nd
      order gradient computation.
    reserve_space_3: A `Tensor`. Must have the same type as `reserve_space_1`.
      When is_training is True, a 1D Tensor for some intermediate results to be reused
      in gradient computation. When is_training is False, a dummy empty Tensor will be
      created.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      The data format for y_backprop, x, x_backprop.
      Either \"NHWC\" (default) or \"NCHW\".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (x_backprop, scale_backprop, offset_backprop, reserve_space_4, reserve_space_5).

    x_backprop: A `Tensor`. Has the same type as `y_backprop`.
    scale_backprop: A `Tensor`. Has the same type as `reserve_space_1`.
    offset_backprop: A `Tensor`. Has the same type as `reserve_space_1`.
    reserve_space_4: A `Tensor`. Has the same type as `reserve_space_1`.
    reserve_space_5: A `Tensor`. Has the same type as `reserve_space_1`.
  "
  [y_backprop x scale reserve_space_1 reserve_space_2 reserve_space_3 & {:keys [epsilon data_format is_training name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "fused_batch_norm_grad_v3" [y_backprop x scale reserve_space_1 reserve_space_2 reserve_space_3] {:epsilon epsilon :data_format data_format :is_training is_training :name name }))

(defn fused-batch-norm-grad-v3-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function fused_batch_norm_grad_v3
  "
  [y_backprop x scale reserve_space_1 reserve_space_2 reserve_space_3 & {:keys [epsilon data_format is_training name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "fused_batch_norm_grad_v3_eager_fallback" [y_backprop x scale reserve_space_1 reserve_space_2 reserve_space_3] {:epsilon epsilon :data_format data_format :is_training is_training :name name :ctx ctx }))

(defn fused-batch-norm-v2 
  "Batch normalization.

  Note that the size of 4D Tensors are defined by either \"NHWC\" or \"NCHW\".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      A 4D Tensor for input data.
    scale: A `Tensor`. Must be one of the following types: `float32`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    offset: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for offset, to shift to the normalized x.
    mean: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for population mean. Used for inference only;
      must be empty for training.
    variance: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for population variance. Used for inference only;
      must be empty for training.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      The data format for x and y. Either \"NHWC\" (default) or \"NCHW\".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, batch_mean, batch_variance, reserve_space_1, reserve_space_2).

    y: A `Tensor`. Has the same type as `x`.
    batch_mean: A `Tensor`. Has the same type as `scale`.
    batch_variance: A `Tensor`. Has the same type as `scale`.
    reserve_space_1: A `Tensor`. Has the same type as `scale`.
    reserve_space_2: A `Tensor`. Has the same type as `scale`.
  "
  [x scale offset mean variance & {:keys [epsilon data_format is_training name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "fused_batch_norm_v2" [x scale offset mean variance] {:epsilon epsilon :data_format data_format :is_training is_training :name name }))

(defn fused-batch-norm-v2-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function fused_batch_norm_v2
  "
  [x scale offset mean variance & {:keys [epsilon data_format is_training name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "fused_batch_norm_v2_eager_fallback" [x scale offset mean variance] {:epsilon epsilon :data_format data_format :is_training is_training :name name :ctx ctx }))

(defn fused-batch-norm-v3 
  "Batch normalization.

  Note that the size of 4D Tensors are defined by either \"NHWC\" or \"NCHW\".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      A 4D Tensor for input data.
    scale: A `Tensor`. Must be one of the following types: `float32`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    offset: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for offset, to shift to the normalized x.
    mean: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for population mean. Used for inference only;
      must be empty for training.
    variance: A `Tensor`. Must have the same type as `scale`.
      A 1D Tensor for population variance. Used for inference only;
      must be empty for training.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      The data format for x and y. Either \"NHWC\" (default) or \"NCHW\".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, batch_mean, batch_variance, reserve_space_1, reserve_space_2, reserve_space_3).

    y: A `Tensor`. Has the same type as `x`.
    batch_mean: A `Tensor`. Has the same type as `scale`.
    batch_variance: A `Tensor`. Has the same type as `scale`.
    reserve_space_1: A `Tensor`. Has the same type as `scale`.
    reserve_space_2: A `Tensor`. Has the same type as `scale`.
    reserve_space_3: A `Tensor`. Has the same type as `scale`.
  "
  [x scale offset mean variance & {:keys [epsilon data_format is_training name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "fused_batch_norm_v3" [x scale offset mean variance] {:epsilon epsilon :data_format data_format :is_training is_training :name name }))

(defn fused-batch-norm-v3-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function fused_batch_norm_v3
  "
  [x scale offset mean variance & {:keys [epsilon data_format is_training name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "fused_batch_norm_v3_eager_fallback" [x scale offset mean variance] {:epsilon epsilon :data_format data_format :is_training is_training :name name :ctx ctx }))

(defn fused-pad-conv2d 
  "Performs a padding as a preprocess during a convolution.

  Similar to FusedResizeAndPadConv2d, this op allows for an optimized
  implementation where the spatial padding transformation stage is fused with the
  im2col lookup, but in this case without the bilinear filtering required for
  resizing. Fusing the padding prevents the need to write out the intermediate
  results as whole tensors, reducing memory pressure, and we can get some latency
  gains by merging the transformation calculations.
  The data_format attribute for Conv2D isn't supported by this op, and 'NHWC'
  order is used instead.
  Internally this op uses a single per-graph scratch buffer, which means that it
  will block if multiple versions are being run in parallel. This is because this
  operator is primarily an optimization to minimize memory usage.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
    paddings: A `Tensor` of type `int32`.
      A two-column matrix specifying the padding sizes. The number of
      rows must be the same as the rank of `input`.
    filter: A `Tensor`. Must have the same type as `input`. 4-D with shape
      `[filter_height, filter_width, in_channels, out_channels]`.
    mode: A `string` from: `\"REFLECT\", \"SYMMETRIC\"`.
    strides: A list of `ints`.
      1-D of length 4.  The stride of the sliding window for each dimension
      of `input`. Must be in the same order as the dimension specified with format.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [ input paddings filter mode strides padding name ]
  (py/call-attr quantization "fused_pad_conv2d"  input paddings filter mode strides padding name ))

(defn fused-pad-conv2d-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function fused_pad_conv2d
  "
  [ input paddings filter mode strides padding name ctx ]
  (py/call-attr quantization "fused_pad_conv2d_eager_fallback"  input paddings filter mode strides padding name ctx ))

(defn fused-resize-and-pad-conv2d 
  "Performs a resize and padding as a preprocess during a convolution.

  It's often possible to do spatial transformations more efficiently as part of
  the packing stage of a convolution, so this op allows for an optimized
  implementation where these stages are fused together. This prevents the need to
  write out the intermediate results as whole tensors, reducing memory pressure,
  and we can get some latency gains by merging the transformation calculations.
  The data_format attribute for Conv2D isn't supported by this op, and defaults to
  'NHWC' order.
  Internally this op uses a single per-graph scratch buffer, which means that it
  will block if multiple versions are being run in parallel. This is because this
  operator is primarily an optimization to minimize memory usage.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
    size: A `Tensor` of type `int32`.
      A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
      new size for the images.
    paddings: A `Tensor` of type `int32`.
      A two-column matrix specifying the padding sizes. The number of
      rows must be the same as the rank of `input`.
    filter: A `Tensor`. Must have the same type as `input`. 4-D with shape
      `[filter_height, filter_width, in_channels, out_channels]`.
    mode: A `string` from: `\"REFLECT\", \"SYMMETRIC\"`.
    strides: A list of `ints`.
      1-D of length 4.  The stride of the sliding window for each dimension
      of `input`. Must be in the same order as the dimension specified with format.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    resize_align_corners: An optional `bool`. Defaults to `False`.
      If true, the centers of the 4 corner pixels of the input and output tensors are
      aligned, preserving the values at the corner pixels. Defaults to false.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input size paddings filter mode strides padding & {:keys [resize_align_corners name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "fused_resize_and_pad_conv2d" [input size paddings filter mode strides padding] {:resize_align_corners resize_align_corners :name name }))

(defn fused-resize-and-pad-conv2d-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function fused_resize_and_pad_conv2d
  "
  [input size paddings filter mode strides padding & {:keys [resize_align_corners name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "fused_resize_and_pad_conv2d_eager_fallback" [input size paddings filter mode strides padding] {:resize_align_corners resize_align_corners :name name :ctx ctx }))

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
  (py/call-attr quantization "greater"  x y name ))

(defn greater-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function greater
  "
  [ x y name ctx ]
  (py/call-attr quantization "greater_eager_fallback"  x y name ctx ))

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
  (py/call-attr quantization "greater_equal"  x y name ))

(defn greater-equal-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function greater_equal
  "
  [ x y name ctx ]
  (py/call-attr quantization "greater_equal_eager_fallback"  x y name ctx ))

(defn igamma 
  "Compute the lower regularized incomplete Gamma function `P(a, x)`.

  The lower regularized incomplete Gamma function is defined as:


  \\(P(a, x) = gamma(a, x) / Gamma(a) = 1 - Q(a, x)\\)

  where

  \\(gamma(a, x) = \\int_{0}^{x} t^{a-1} exp(-t) dt\\)

  is the lower incomplete Gamma function.

  Note, above `Q(a, x)` (`Igammac`) is the upper regularized complete
  Gamma function.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    x: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  "
  [ a x name ]
  (py/call-attr quantization "igamma"  a x name ))

(defn igamma-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function igamma
  "
  [ a x name ctx ]
  (py/call-attr quantization "igamma_eager_fallback"  a x name ctx ))

(defn igamma-grad-a 
  "Computes the gradient of `igamma(a, x)` wrt `a`.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    x: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  "
  [ a x name ]
  (py/call-attr quantization "igamma_grad_a"  a x name ))

(defn igamma-grad-a-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function igamma_grad_a
  "
  [ a x name ctx ]
  (py/call-attr quantization "igamma_grad_a_eager_fallback"  a x name ctx ))

(defn igammac 
  "Compute the upper regularized incomplete Gamma function `Q(a, x)`.

  The upper regularized incomplete Gamma function is defined as:

  \\(Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)\\)

  where

  \\(Gamma(a, x) = int_{x}^{\infty} t^{a-1} exp(-t) dt\\)

  is the upper incomplete Gama function.

  Note, above `P(a, x)` (`Igamma`) is the lower regularized complete
  Gamma function.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    x: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  "
  [ a x name ]
  (py/call-attr quantization "igammac"  a x name ))

(defn igammac-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function igammac
  "
  [ a x name ctx ]
  (py/call-attr quantization "igammac_eager_fallback"  a x name ctx ))

(defn imag 
  "Returns the imaginary part of a complex number.

  Given a tensor `input` of complex numbers, this operation returns a tensor of
  type `float` that is the imaginary part of each element in `input`. All
  elements in `input` must be complex numbers of the form \\(a + bj\\), where *a*
  is the real part and *b* is the imaginary part returned by this operation.

  For example:

  ```
  # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.imag(input) ==> [4.75, 5.75]
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
    Tout: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  "
  [input & {:keys [Tout name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "imag" [input] {:Tout Tout :name name }))

(defn imag-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function imag
  "
  [input & {:keys [Tout name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "imag_eager_fallback" [input] {:Tout Tout :name name :ctx ctx }))

(defn in-top-k 
  "Says whether the targets are in the top `K` predictions.

  This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
  prediction for the target class is among the top `k` predictions among
  all predictions for example `i`. Note that the behavior of `InTopK` differs
  from the `TopK` op in its handling of ties; if multiple classes have the
  same prediction value and straddle the top-`k` boundary, all of those
  classes are considered to be in the top `k`.

  More formally, let

    \\(predictions_i\\) be the predictions for all classes for example `i`,
    \\(targets_i\\) be the target class for example `i`,
    \\(out_i\\) be the output for example `i`,

  $$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$

  Args:
    predictions: A `Tensor` of type `float32`.
      A `batch_size` x `classes` tensor.
    targets: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `batch_size` vector of class ids.
    k: An `int`. Number of top elements to look at for computing precision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ predictions targets k name ]
  (py/call-attr quantization "in_top_k"  predictions targets k name ))

(defn in-top-k-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function in_top_k
  "
  [ predictions targets k name ctx ]
  (py/call-attr quantization "in_top_k_eager_fallback"  predictions targets k name ctx ))

(defn in-top-kv2 
  "Says whether the targets are in the top `K` predictions.

  This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
  prediction for the target class is among the top `k` predictions among
  all predictions for example `i`. Note that the behavior of `InTopK` differs
  from the `TopK` op in its handling of ties; if multiple classes have the
  same prediction value and straddle the top-`k` boundary, all of those
  classes are considered to be in the top `k`.

  More formally, let

    \\(predictions_i\\) be the predictions for all classes for example `i`,
    \\(targets_i\\) be the target class for example `i`,
    \\(out_i\\) be the output for example `i`,

  $$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$

  Args:
    predictions: A `Tensor` of type `float32`.
      A `batch_size` x `classes` tensor.
    targets: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `batch_size` vector of class ids.
    k: A `Tensor`. Must have the same type as `targets`.
      Number of top elements to look at for computing precision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ predictions targets k name ]
  (py/call-attr quantization "in_top_kv2"  predictions targets k name ))

(defn in-top-kv2-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function in_top_kv2
  "
  [ predictions targets k name ctx ]
  (py/call-attr quantization "in_top_kv2_eager_fallback"  predictions targets k name ctx ))

(defn inv 
  "Computes the reciprocal of x element-wise.

  I.e., \\(y = 1 / x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "inv"  x name ))

(defn inv-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function inv
  "
  [ x name ctx ]
  (py/call-attr quantization "inv_eager_fallback"  x name ctx ))

(defn inv-grad 
  "Computes the gradient for the inverse of `x` wrt its input.

  Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
  is the corresponding input gradient.

  Args:
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    dy: A `Tensor`. Must have the same type as `y`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `y`.
  "
  [ y dy name ]
  (py/call-attr quantization "inv_grad"  y dy name ))

(defn inv-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function inv_grad
  "
  [ y dy name ctx ]
  (py/call-attr quantization "inv_grad_eager_fallback"  y dy name ctx ))

(defn is-finite 
  "Returns which elements of x are finite.

  @compatibility(numpy)
  Equivalent to np.isfinite
  @end_compatibility

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ x name ]
  (py/call-attr quantization "is_finite"  x name ))

(defn is-finite-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function is_finite
  "
  [ x name ctx ]
  (py/call-attr quantization "is_finite_eager_fallback"  x name ctx ))

(defn is-inf 
  "Returns which elements of x are Inf.

  @compatibility(numpy)
  Equivalent to np.isinf
  @end_compatibility

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ x name ]
  (py/call-attr quantization "is_inf"  x name ))

(defn is-inf-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function is_inf
  "
  [ x name ctx ]
  (py/call-attr quantization "is_inf_eager_fallback"  x name ctx ))

(defn is-nan 
  "Returns which elements of x are NaN.

  @compatibility(numpy)
  Equivalent to np.isnan
  @end_compatibility

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ x name ]
  (py/call-attr quantization "is_nan"  x name ))

(defn is-nan-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function is_nan
  "
  [ x name ctx ]
  (py/call-attr quantization "is_nan_eager_fallback"  x name ctx ))

(defn l2-loss 
  "L2 Loss.

  Computes half the L2 norm of a tensor without the `sqrt`:

      output = sum(t ** 2) / 2

  Args:
    t: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      Typically 2-D, but may have any dimensions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `t`.
  "
  [ t name ]
  (py/call-attr quantization "l2_loss"  t name ))

(defn l2-loss-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function l2_loss
  "
  [ t name ctx ]
  (py/call-attr quantization "l2_loss_eager_fallback"  t name ctx ))

(defn leaky-relu 
  "Computes rectified linear: `max(features, features * alpha)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    alpha: An optional `float`. Defaults to `0.2`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  "
  [features & {:keys [alpha name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "leaky_relu" [features] {:alpha alpha :name name }))

(defn leaky-relu-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function leaky_relu
  "
  [features & {:keys [alpha name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "leaky_relu_eager_fallback" [features] {:alpha alpha :name name :ctx ctx }))

(defn leaky-relu-grad 
  "Computes rectified linear gradients for a LeakyRelu operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      The backpropagated gradients to the corresponding LeakyRelu operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding LeakyRelu operation,
      OR the outputs of that operation (both work equivalently).
    alpha: An optional `float`. Defaults to `0.2`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  "
  [gradients features & {:keys [alpha name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "leaky_relu_grad" [gradients features] {:alpha alpha :name name }))

(defn leaky-relu-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function leaky_relu_grad
  "
  [gradients features & {:keys [alpha name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "leaky_relu_grad_eager_fallback" [gradients features] {:alpha alpha :name name :ctx ctx }))

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
  (py/call-attr quantization "less"  x y name ))

(defn less-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function less
  "
  [ x y name ctx ]
  (py/call-attr quantization "less_eager_fallback"  x y name ctx ))

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
  (py/call-attr quantization "less_equal"  x y name ))

(defn less-equal-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function less_equal
  "
  [ x y name ctx ]
  (py/call-attr quantization "less_equal_eager_fallback"  x y name ctx ))

(defn lgamma 
  "Computes the log of the absolute value of `Gamma(x)` element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "lgamma"  x name ))

(defn lgamma-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function lgamma
  "
  [ x name ctx ]
  (py/call-attr quantization "lgamma_eager_fallback"  x name ctx ))

(defn lin-space 
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
  (py/call-attr quantization "lin_space"  start stop num name ))

(defn lin-space-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function lin_space
  "
  [ start stop num name ctx ]
  (py/call-attr quantization "lin_space_eager_fallback"  start stop num name ctx ))

(defn log 
  "Computes natural logarithm of x element-wise.

  I.e., \\(y = \log_e x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "log"  x name ))

(defn log1p 
  "Computes natural logarithm of (1 + x) element-wise.

  I.e., \\(y = \log_e (1 + x)\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "log1p"  x name ))

(defn log1p-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function log1p
  "
  [ x name ctx ]
  (py/call-attr quantization "log1p_eager_fallback"  x name ctx ))

(defn log-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function log
  "
  [ x name ctx ]
  (py/call-attr quantization "log_eager_fallback"  x name ctx ))

(defn log-softmax 
  "Computes log softmax activations.

  For each batch `i` and class `j` we have

      logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))

  Args:
    logits: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      2-D with shape `[batch_size, num_classes]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `logits`.
  "
  [ logits name ]
  (py/call-attr quantization "log_softmax"  logits name ))

(defn log-softmax-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function log_softmax
  "
  [ logits name ctx ]
  (py/call-attr quantization "log_softmax_eager_fallback"  logits name ctx ))

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
  (py/call-attr quantization "logical_and"  x y name ))

(defn logical-and-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function logical_and
  "
  [ x y name ctx ]
  (py/call-attr quantization "logical_and_eager_fallback"  x y name ctx ))

(defn logical-not 
  "Returns the truth value of NOT x element-wise.

  Args:
    x: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ x name ]
  (py/call-attr quantization "logical_not"  x name ))

(defn logical-not-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function logical_not
  "
  [ x name ctx ]
  (py/call-attr quantization "logical_not_eager_fallback"  x name ctx ))

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
  (py/call-attr quantization "logical_or"  x y name ))

(defn logical-or-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function logical_or
  "
  [ x y name ctx ]
  (py/call-attr quantization "logical_or_eager_fallback"  x y name ctx ))

(defn lrn 
  "Local Response Normalization.

  The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
  dimension), and each vector is normalized independently.  Within a given vector,
  each component is divided by the weighted, squared sum of inputs within
  `depth_radius`.  In detail,

      sqr_sum[a, b, c, d] =
          sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
      output = input / (bias + alpha * sqr_sum) ** beta

  For details, see [Krizhevsky et al., ImageNet classification with deep
  convolutional neural networks (NIPS 2012)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      4-D.
    depth_radius: An optional `int`. Defaults to `5`.
      0-D.  Half-width of the 1-D normalization window.
    bias: An optional `float`. Defaults to `1`.
      An offset (usually positive to avoid dividing by 0).
    alpha: An optional `float`. Defaults to `1`.
      A scale factor, usually positive.
    beta: An optional `float`. Defaults to `0.5`. An exponent.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input & {:keys [depth_radius bias alpha beta name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "lrn" [input] {:depth_radius depth_radius :bias bias :alpha alpha :beta beta :name name }))

(defn lrn-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function lrn
  "
  [input & {:keys [depth_radius bias alpha beta name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "lrn_eager_fallback" [input] {:depth_radius depth_radius :bias bias :alpha alpha :beta beta :name name :ctx ctx }))

(defn lrn-grad 
  "Gradients for Local Response Normalization.

  Args:
    input_grads: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      4-D with shape `[batch, height, width, channels]`.
    input_image: A `Tensor`. Must have the same type as `input_grads`.
      4-D with shape `[batch, height, width, channels]`.
    output_image: A `Tensor`. Must have the same type as `input_grads`.
      4-D with shape `[batch, height, width, channels]`.
    depth_radius: An optional `int`. Defaults to `5`. A depth radius.
    bias: An optional `float`. Defaults to `1`.
      An offset (usually > 0 to avoid dividing by 0).
    alpha: An optional `float`. Defaults to `1`.
      A scale factor, usually positive.
    beta: An optional `float`. Defaults to `0.5`. An exponent.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input_grads`.
  "
  [input_grads input_image output_image & {:keys [depth_radius bias alpha beta name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "lrn_grad" [input_grads input_image output_image] {:depth_radius depth_radius :bias bias :alpha alpha :beta beta :name name }))

(defn lrn-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function lrn_grad
  "
  [input_grads input_image output_image & {:keys [depth_radius bias alpha beta name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "lrn_grad_eager_fallback" [input_grads input_image output_image] {:depth_radius depth_radius :bias bias :alpha alpha :beta beta :name name :ctx ctx }))

(defn mat-mul 
  "Multiply the matrix \"a\" by the matrix \"b\".

  The inputs must be two-dimensional matrices and the inner dimension of
  \"a\" (after being transposed if transpose_a is true) must match the
  outer dimension of \"b\" (after being transposed if transposed_b is
  true).

  *Note*: The default kernel implementation for MatMul on GPUs uses
  cublas.

  Args:
    a: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    b: A `Tensor`. Must have the same type as `a`.
    transpose_a: An optional `bool`. Defaults to `False`.
      If true, \"a\" is transposed before multiplication.
    transpose_b: An optional `bool`. Defaults to `False`.
      If true, \"b\" is transposed before multiplication.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  "
  [a b & {:keys [transpose_a transpose_b name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "mat_mul" [a b] {:transpose_a transpose_a :transpose_b transpose_b :name name }))

(defn mat-mul-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function mat_mul
  "
  [a b & {:keys [transpose_a transpose_b name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "mat_mul_eager_fallback" [a b] {:transpose_a transpose_a :transpose_b transpose_b :name name :ctx ctx }))

(defn max-pool 
  "Performs max pooling on the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `qint8`.
      4-D input to pool over.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\", \"NCHW_VECT_C\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "max_pool" [input ksize strides padding] {:data_format data_format :name name }))

(defn max-pool3d 
  "Performs 3D max pooling on the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NDHWC\", \"NCDHW\"`. Defaults to `\"NDHWC\"`.
      The data format of the input and output data. With the
      default format \"NDHWC\", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCDHW\", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "max_pool3d" [input ksize strides padding] {:data_format data_format :name name }))

(defn max-pool3d-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function max_pool3d
  "
  [input ksize strides padding & {:keys [data_format name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "max_pool3d_eager_fallback" [input ksize strides padding] {:data_format data_format :name name :ctx ctx }))

(defn max-pool3d-grad 
  "Computes gradients of max pooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      Output backprop of shape `[batch, depth, rows, cols, channels]`.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NDHWC\", \"NCDHW\"`. Defaults to `\"NDHWC\"`.
      The data format of the input and output data. With the
      default format \"NDHWC\", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCDHW\", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  "
  [orig_input orig_output grad ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "max_pool3d_grad" [orig_input orig_output grad ksize strides padding] {:data_format data_format :name name }))

(defn max-pool3d-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function max_pool3d_grad
  "
  [orig_input orig_output grad ksize strides padding & {:keys [data_format name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "max_pool3d_grad_eager_fallback" [orig_input orig_output grad ksize strides padding] {:data_format data_format :name name :ctx ctx }))

(defn max-pool3d-grad-grad 
  "Computes second-order gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must have the same type as `orig_input`.
      Output backprop of shape `[batch, depth, rows, cols, channels]`.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NDHWC\", \"NCDHW\"`. Defaults to `\"NDHWC\"`.
      The data format of the input and output data. With the
      default format \"NDHWC\", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCDHW\", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  "
  [orig_input orig_output grad ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "max_pool3d_grad_grad" [orig_input orig_output grad ksize strides padding] {:data_format data_format :name name }))

(defn max-pool3d-grad-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function max_pool3d_grad_grad
  "
  [orig_input orig_output grad ksize strides padding & {:keys [data_format name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "max_pool3d_grad_grad_eager_fallback" [orig_input orig_output grad ksize strides padding] {:data_format data_format :name name :ctx ctx }))

(defn max-pool-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function max_pool
  "
  [input ksize strides padding & {:keys [data_format name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "max_pool_eager_fallback" [input ksize strides padding] {:data_format data_format :name name :ctx ctx }))

(defn max-pool-grad 
  "Computes gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must have the same type as `orig_input`.
      4-D.  Gradients w.r.t. the output of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  "
  [orig_input orig_output grad ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "max_pool_grad" [orig_input orig_output grad ksize strides padding] {:data_format data_format :name name }))

(defn max-pool-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function max_pool_grad
  "
  [orig_input orig_output grad ksize strides padding & {:keys [data_format name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "max_pool_grad_eager_fallback" [orig_input orig_output grad ksize strides padding] {:data_format data_format :name name :ctx ctx }))

(defn max-pool-grad-grad 
  "Computes second-order gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must have the same type as `orig_input`.
      4-D.  Gradients of gradients w.r.t. the input of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  "
  [orig_input orig_output grad ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "max_pool_grad_grad" [orig_input orig_output grad ksize strides padding] {:data_format data_format :name name }))

(defn max-pool-grad-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function max_pool_grad_grad
  "
  [orig_input orig_output grad ksize strides padding & {:keys [data_format name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "max_pool_grad_grad_eager_fallback" [orig_input orig_output grad ksize strides padding] {:data_format data_format :name name :ctx ctx }))

(defn max-pool-grad-grad-v2 
  "Computes second-order gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must have the same type as `orig_input`.
      4-D.  Gradients of gradients w.r.t. the input of `max_pool`.
    ksize: A `Tensor` of type `int32`.
      The size of the window for each dimension of the input tensor.
    strides: A `Tensor` of type `int32`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  "
  [orig_input orig_output grad ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "max_pool_grad_grad_v2" [orig_input orig_output grad ksize strides padding] {:data_format data_format :name name }))

(defn max-pool-grad-grad-v2-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function max_pool_grad_grad_v2
  "
  [orig_input orig_output grad ksize strides padding & {:keys [data_format name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "max_pool_grad_grad_v2_eager_fallback" [orig_input orig_output grad ksize strides padding] {:data_format data_format :name name :ctx ctx }))

(defn max-pool-grad-grad-with-argmax 
  "Computes second-order gradients of the maxpooling function.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input.
    grad: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the
      input of `max_pool`.
    argmax: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The indices of the maximum values chosen for each output of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    include_batch_in_index: An optional `bool`. Defaults to `False`.
      Whether to include batch dimension in flattened index of `argmax`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input grad argmax ksize strides padding & {:keys [include_batch_in_index name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "max_pool_grad_grad_with_argmax" [input grad argmax ksize strides padding] {:include_batch_in_index include_batch_in_index :name name }))

(defn max-pool-grad-grad-with-argmax-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function max_pool_grad_grad_with_argmax
  "
  [input grad argmax ksize strides padding & {:keys [include_batch_in_index name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "max_pool_grad_grad_with_argmax_eager_fallback" [input grad argmax ksize strides padding] {:include_batch_in_index include_batch_in_index :name name :ctx ctx }))

(defn max-pool-grad-v2 
  "Computes gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must have the same type as `orig_input`.
      4-D.  Gradients w.r.t. the output of `max_pool`.
    ksize: A `Tensor` of type `int32`.
      The size of the window for each dimension of the input tensor.
    strides: A `Tensor` of type `int32`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `orig_input`.
  "
  [orig_input orig_output grad ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "max_pool_grad_v2" [orig_input orig_output grad ksize strides padding] {:data_format data_format :name name }))

(defn max-pool-grad-v2-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function max_pool_grad_v2
  "
  [orig_input orig_output grad ksize strides padding & {:keys [data_format name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "max_pool_grad_v2_eager_fallback" [orig_input orig_output grad ksize strides padding] {:data_format data_format :name name :ctx ctx }))

(defn max-pool-grad-with-argmax 
  "Computes gradients of the maxpooling function.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The original input.
    grad: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the
      output of `max_pool`.
    argmax: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The indices of the maximum values chosen for each output of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    include_batch_in_index: An optional `bool`. Defaults to `False`.
      Whether to include batch dimension in flattened index of `argmax`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input grad argmax ksize strides padding & {:keys [include_batch_in_index name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "max_pool_grad_with_argmax" [input grad argmax ksize strides padding] {:include_batch_in_index include_batch_in_index :name name }))

(defn max-pool-grad-with-argmax-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function max_pool_grad_with_argmax
  "
  [input grad argmax ksize strides padding & {:keys [include_batch_in_index name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "max_pool_grad_with_argmax_eager_fallback" [input grad argmax ksize strides padding] {:include_batch_in_index include_batch_in_index :name name :ctx ctx }))

(defn max-pool-v2 
  "Performs max pooling on the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `qint8`.
      4-D input to pool over.
    ksize: A `Tensor` of type `int32`.
      The size of the window for each dimension of the input tensor.
    strides: A `Tensor` of type `int32`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `\"NHWC\", \"NCHW\", \"NCHW_VECT_C\"`. Defaults to `\"NHWC\"`.
      Specify the data format of the input and output data. With the
      default format \"NHWC\", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be \"NCHW\", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input ksize strides padding & {:keys [data_format name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "max_pool_v2" [input ksize strides padding] {:data_format data_format :name name }))

(defn max-pool-v2-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function max_pool_v2
  "
  [input ksize strides padding & {:keys [data_format name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "max_pool_v2_eager_fallback" [input ksize strides padding] {:data_format data_format :name name :ctx ctx }))

(defn max-pool-with-argmax 
  "Performs max pooling on the input and outputs both max values and indices.

  The indices in `argmax` are flattened, so that a maximum value at position
  `[b, y, x, c]` becomes flattened index:
  `(y * width + x) * channels + c` if `include_batch_in_index` is False;
  `((b * height + y) * width + x) * channels + c` if `include_batch_in_index` is True.

  The indices returned are always in `[0, height) x [0, width)` before flattening,
  even if padding is involved and the mathematically correct answer is outside
  (either negative or too large).  This is a bug, but fixing it is difficult to do
  in a safe backwards compatible way, especially due to flattening.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      4-D with shape `[batch, height, width, channels]`.  Input to pool over.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    Targmax: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
    include_batch_in_index: An optional `bool`. Defaults to `False`.
      Whether to include batch dimension in flattened index of `argmax`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, argmax).

    output: A `Tensor`. Has the same type as `input`.
    argmax: A `Tensor` of type `Targmax`.
  "
  [input ksize strides padding & {:keys [Targmax include_batch_in_index name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "max_pool_with_argmax" [input ksize strides padding] {:Targmax Targmax :include_batch_in_index include_batch_in_index :name name }))

(defn max-pool-with-argmax-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function max_pool_with_argmax
  "
  [input ksize strides padding & {:keys [Targmax include_batch_in_index name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "max_pool_with_argmax_eager_fallback" [input ksize strides padding] {:Targmax Targmax :include_batch_in_index include_batch_in_index :name name :ctx ctx }))

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
  (py/call-attr quantization "maximum"  x y name ))

(defn maximum-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function maximum
  "
  [ x y name ctx ]
  (py/call-attr quantization "maximum_eager_fallback"  x y name ctx ))

(defn mean 
  "Computes the mean of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The tensor to reduce.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce. Must be in the range
      `[-rank(input), rank(input))`.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input axis & {:keys [keep_dims name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "mean" [input axis] {:keep_dims keep_dims :name name }))

(defn mean-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function mean
  "
  [input axis & {:keys [keep_dims name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "mean_eager_fallback" [input axis] {:keep_dims keep_dims :name name :ctx ctx }))

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
  (py/call-attr quantization "minimum"  x y name ))

(defn minimum-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function minimum
  "
  [ x y name ctx ]
  (py/call-attr quantization "minimum_eager_fallback"  x y name ctx ))

(defn mod 
  "Returns element-wise remainder of division. This emulates C semantics in that

  the result here is consistent with a truncating divide. E.g.
  `tf.truncatediv(x, y) * y + truncate_mod(x, y) = x`.

  *NOTE*: `Mod` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`, `half`, `half`, `bfloat16`, `float32`, `float64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "mod"  x y name ))

(defn mod-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function mod
  "
  [ x y name ctx ]
  (py/call-attr quantization "mod_eager_fallback"  x y name ctx ))

(defn mul 
  "Returns x * y element-wise.

  *NOTE*: `Multiply` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "mul"  x y name ))

(defn mul-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function mul
  "
  [ x y name ctx ]
  (py/call-attr quantization "mul_eager_fallback"  x y name ctx ))

(defn mul-no-nan 
  "Returns x * y element-wise. Returns zero if y is zero, even if x if infinite or NaN.

  *NOTE*: `MulNoNan` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "mul_no_nan"  x y name ))

(defn mul-no-nan-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function mul_no_nan
  "
  [ x y name ctx ]
  (py/call-attr quantization "mul_no_nan_eager_fallback"  x y name ctx ))

(defn neg 
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
  (py/call-attr quantization "neg"  x name ))

(defn neg-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function neg
  "
  [ x name ctx ]
  (py/call-attr quantization "neg_eager_fallback"  x name ctx ))

(defn next-after 
  "Returns the next representable value of `x1` in the direction of `x2`, element-wise.

  This operation returns the same result as the C++ std::nextafter function.

  It can also return a subnormal number.

  @compatibility(cpp)
  Equivalent to C++ std::nextafter function.
  @end_compatibility

  Args:
    x1: A `Tensor`. Must be one of the following types: `float64`, `float32`.
    x2: A `Tensor`. Must have the same type as `x1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x1`.
  "
  [ x1 x2 name ]
  (py/call-attr quantization "next_after"  x1 x2 name ))

(defn next-after-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function next_after
  "
  [ x1 x2 name ctx ]
  (py/call-attr quantization "next_after_eager_fallback"  x1 x2 name ctx ))

(defn not-equal 
  "Returns the truth value of (x != y) element-wise.

  *NOTE*: `NotEqual` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `quint8`, `qint8`, `qint32`, `string`, `bool`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    incompatible_shape_error: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [x y & {:keys [incompatible_shape_error name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "not_equal" [x y] {:incompatible_shape_error incompatible_shape_error :name name }))

(defn not-equal-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function not_equal
  "
  [x y & {:keys [incompatible_shape_error name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "not_equal_eager_fallback" [x y] {:incompatible_shape_error incompatible_shape_error :name name :ctx ctx }))

(defn nth-element 
  "Finds values of the `n`-th order statistic for the last dimension.

  If the input is a vector (rank-1), finds the entries which is the nth-smallest
  value in the vector and outputs their values as scalar tensor.

  For matrices (resp. higher rank input), computes the entries which is the
  nth-smallest value in each row (resp. vector along the last dimension). Thus,

      values.shape = input.shape[:-1]

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      1-D or higher with last dimension at least `n+1`.
    n: A `Tensor` of type `int32`.
      0-D. Position of sorted vector to select along the last dimension (along
      each row for matrices). Valid range of n is `[0, input.shape[:-1])`
    reverse: An optional `bool`. Defaults to `False`.
      When set to True, find the nth-largest value in the vector and vice
      versa.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input n & {:keys [reverse name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "nth_element" [input n] {:reverse reverse :name name }))

(defn nth-element-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function nth_element
  "
  [input n & {:keys [reverse name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "nth_element_eager_fallback" [input n] {:reverse reverse :name name :ctx ctx }))

(defn polygamma 
  "Compute the polygamma function \\(\psi^{(n)}(x)\\).

  The polygamma function is defined as:


  \\(\psi^{(a)}(x) = \frac{d^a}{dx^a} \psi(x)\\)

  where \\(\psi(x)\\) is the digamma function.
  The polygamma function is defined only for non-negative integer orders \\a\\.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    x: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  "
  [ a x name ]
  (py/call-attr quantization "polygamma"  a x name ))

(defn polygamma-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function polygamma
  "
  [ a x name ctx ]
  (py/call-attr quantization "polygamma_eager_fallback"  a x name ctx ))

(defn prod 
  "Computes the product of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `axis`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `axis`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      The tensor to reduce.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce. Must be in the range
      `[-rank(input), rank(input))`.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  "
  [input axis & {:keys [keep_dims name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "prod" [input axis] {:keep_dims keep_dims :name name }))

(defn prod-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function prod
  "
  [input axis & {:keys [keep_dims name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "prod_eager_fallback" [input axis] {:keep_dims keep_dims :name name :ctx ctx }))

(defn quantize-down-and-shrink-range 
  "Convert the quantized 'input' tensor into a lower-precision 'output', using the

  actual distribution of the values to maximize the usage of the lower bit depth
  and adjusting the output min and max ranges accordingly.

  [input_min, input_max] are scalar floats that specify the range for the float
  interpretation of the 'input' data. For example, if input_min is -1.0f and
  input_max is 1.0f, and we are dealing with quint16 quantized data, then a 0
  value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.

  This operator tries to squeeze as much precision as possible into an output with
  a lower bit depth by calculating the actual min and max values found in the
  data. For example, maybe that quint16 input has no values lower than 16,384 and
  none higher than 49,152. That means only half the range is actually needed, all
  the float interpretations are between -0.5f and 0.5f, so if we want to compress
  the data into a quint8 output, we can use that range rather than the theoretical
  -1.0f to 1.0f that is suggested by the input min and max.

  In practice, this is most useful for taking output from operations like
  QuantizedMatMul that can produce higher bit-depth outputs than their inputs and
  may have large potential output ranges, but in practice have a distribution of
  input values that only uses a small fraction of the possible range. By feeding
  that output into this operator, we can reduce it from 32 bits down to 8 with
  minimal loss of accuracy.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    input_min: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    input_max: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`.
      The type of the output. Should be a lower bit depth than Tinput.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).

    output: A `Tensor` of type `out_type`.
    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  "
  [ input input_min input_max out_type name ]
  (py/call-attr quantization "quantize_down_and_shrink_range"  input input_min input_max out_type name ))

(defn quantize-down-and-shrink-range-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantize_down_and_shrink_range
  "
  [ input input_min input_max out_type name ctx ]
  (py/call-attr quantization "quantize_down_and_shrink_range_eager_fallback"  input input_min input_max out_type name ctx ))

(defn quantize-v2 
  "Quantize the 'input' tensor of type float to 'output' tensor of type 'T'.

  [min_range, max_range] are scalar floats that specify the range for
  the 'input' data. The 'mode' attribute controls exactly which calculations are
  used to convert the float values to their quantized equivalents.  The
  'round_mode' attribute controls which rounding tie-breaking algorithm is used
  when rounding float values to their quantized equivalents.

  In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:

  ```
  out[i] = (in[i] - min_range) * range(T) / (max_range - min_range)
  if T == qint8: out[i] -= (range(T) + 1) / 2.0
  ```

  here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`

  *MIN_COMBINED Mode Example*

  Assume the input is type float and has a possible range of [0.0, 6.0] and the
  output type is quint8 ([0, 255]). The min_range and max_range values should be
  specified as 0.0 and 6.0. Quantizing from float to quint8 will multiply each
  value of the input by 255/6 and cast to quint8.

  If the output type was qint8 ([-128, 127]), the operation will additionally
  subtract each value by 128 prior to casting, so that the range of values aligns
  with the range of qint8.

  If the mode is 'MIN_FIRST', then this approach is used:

  ```
  num_discrete_values = 1 << (# of bits in T)
  range_adjust = num_discrete_values / (num_discrete_values - 1)
  range = (range_max - range_min) * range_adjust
  range_scale = num_discrete_values / range
  quantized = round(input * range_scale) - round(range_min * range_scale) +
    numeric_limits<T>::min()
  quantized = max(quantized, numeric_limits<T>::min())
  quantized = min(quantized, numeric_limits<T>::max())
  ```

  The biggest difference between this and MIN_COMBINED is that the minimum range
  is rounded first, before it's subtracted from the rounded value. With
  MIN_COMBINED, a small bias is introduced where repeated iterations of quantizing
  and dequantizing will introduce a larger and larger error.

  *SCALED mode Example*

  `SCALED` mode matches the quantization approach used in
  `QuantizeAndDequantize{V2|V3}`.

  If the mode is `SCALED`, we do not use the full range of the output type,
  choosing to elide the lowest possible value for symmetry (e.g., output range is
  -127 to 127, not -128 to 127 for signed 8 bit quantization), so that 0.0 maps to
  0.

  We first find the range of values in our tensor. The
  range we use is always centered on 0, so we find m such that

  ```c++
    m = max(abs(input_min), abs(input_max))
  ```

  Our input tensor range is then `[-m, m]`.

  Next, we choose our fixed-point quantization buckets, `[min_fixed, max_fixed]`.
  If T is signed, this is

  ```
    num_bits = sizeof(T) * 8
    [min_fixed, max_fixed] =
        [-(1 << (num_bits - 1) - 1), (1 << (num_bits - 1)) - 1]
  ```

  Otherwise, if T is unsigned, the fixed-point range is

  ```
    [min_fixed, max_fixed] = [0, (1 << num_bits) - 1]
  ```

  From this we compute our scaling factor, s:

  ```c++
    s = (max_fixed - min_fixed) / (2 * m)
  ```

  Now we can quantize the elements of our tensor:

  ```c++
  result = round(input * s)
  ```

  One thing to watch out for is that the operator may choose to adjust the
  requested minimum and maximum values slightly during the quantization process,
  so you should always use the output ports as the range for further calculations.
  For example, if the requested minimum and maximum values are close to equal,
  they will be separated by a small epsilon value to prevent ill-formed quantized
  buffers from being created. Otherwise, you can end up with buffers where all the
  quantized values map to the same float value, which causes problems for
  operations that have to perform further calculations on them.

  Args:
    input: A `Tensor` of type `float32`.
    min_range: A `Tensor` of type `float32`.
      The minimum scalar value possibly produced for the input.
    max_range: A `Tensor` of type `float32`.
      The maximum scalar value possibly produced for the input.
    T: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`.
    mode: An optional `string` from: `\"MIN_COMBINED\", \"MIN_FIRST\", \"SCALED\"`. Defaults to `\"MIN_COMBINED\"`.
    round_mode: An optional `string` from: `\"HALF_AWAY_FROM_ZERO\", \"HALF_TO_EVEN\"`. Defaults to `\"HALF_AWAY_FROM_ZERO\"`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).

    output: A `Tensor` of type `T`.
    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  "
  [input min_range max_range T & {:keys [mode round_mode name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantize_v2" [input min_range max_range T] {:mode mode :round_mode round_mode :name name }))

(defn quantized-add 
  "Returns x + y element-wise, working on quantized buffers.

  Args:
    x: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    y: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_x: A `Tensor` of type `float32`.
      The float value that the lowest quantized `x` value represents.
    max_x: A `Tensor` of type `float32`.
      The float value that the highest quantized `x` value represents.
    min_y: A `Tensor` of type `float32`.
      The float value that the lowest quantized `y` value represents.
    max_y: A `Tensor` of type `float32`.
      The float value that the highest quantized `y` value represents.
    Toutput: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (z, min_z, max_z).

    z: A `Tensor` of type `Toutput`.
    min_z: A `Tensor` of type `float32`.
    max_z: A `Tensor` of type `float32`.
  "
  [x y min_x max_x min_y max_y & {:keys [Toutput name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_add" [x y min_x max_x min_y max_y] {:Toutput Toutput :name name }))

(defn quantized-add-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_add
  "
  [x y min_x max_x min_y max_y & {:keys [Toutput name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_add_eager_fallback" [x y min_x max_x min_y max_y] {:Toutput Toutput :name name :ctx ctx }))

(defn quantized-avg-pool 
  "Produces the average pool of the input tensor for quantized types.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      4-D with shape `[batch, height, width, channels]`.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    ksize: A list of `ints`.
      The size of the window for each dimension of the input tensor.
      The length must be 4 to match the number of dimensions of the input.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      tensor.  The length must be 4 to match the number of dimensions of the input.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor`. Has the same type as `input`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [ input min_input max_input ksize strides padding name ]
  (py/call-attr quantization "quantized_avg_pool"  input min_input max_input ksize strides padding name ))

(defn quantized-avg-pool-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_avg_pool
  "
  [ input min_input max_input ksize strides padding name ctx ]
  (py/call-attr quantization "quantized_avg_pool_eager_fallback"  input min_input max_input ksize strides padding name ctx ))

(defn quantized-batch-norm-with-global-normalization 
  "Quantized Batch normalization.

  This op is deprecated and will be removed in the future. Prefer
  `tf.nn.batch_normalization`.

  Args:
    t: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A 4D input Tensor.
    t_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized input.
    t_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized input.
    m: A `Tensor`. Must have the same type as `t`.
      A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    m_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized mean.
    m_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized mean.
    v: A `Tensor`. Must have the same type as `t`.
      A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    v_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized variance.
    v_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized variance.
    beta: A `Tensor`. Must have the same type as `t`.
      A 1D beta Tensor with size matching the last dimension of t.
      An offset to be added to the normalized tensor.
    beta_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized offset.
    beta_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized offset.
    gamma: A `Tensor`. Must have the same type as `t`.
      A 1D gamma Tensor with size matching the last dimension of t.
      If \"scale_after_normalization\" is true, this tensor will be multiplied
      with the normalized tensor.
    gamma_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized gamma.
    gamma_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized gamma.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`.
    variance_epsilon: A `float`. A small float number to avoid dividing by 0.
    scale_after_normalization: A `bool`.
      A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (result, result_min, result_max).

    result: A `Tensor` of type `out_type`.
    result_min: A `Tensor` of type `float32`.
    result_max: A `Tensor` of type `float32`.
  "
  [ t t_min t_max m m_min m_max v v_min v_max beta beta_min beta_max gamma gamma_min gamma_max out_type variance_epsilon scale_after_normalization name ]
  (py/call-attr quantization "quantized_batch_norm_with_global_normalization"  t t_min t_max m m_min m_max v v_min v_max beta beta_min beta_max gamma gamma_min gamma_max out_type variance_epsilon scale_after_normalization name ))

(defn quantized-batch-norm-with-global-normalization-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_batch_norm_with_global_normalization
  "
  [ t t_min t_max m m_min m_max v v_min v_max beta beta_min beta_max gamma gamma_min gamma_max out_type variance_epsilon scale_after_normalization name ctx ]
  (py/call-attr quantization "quantized_batch_norm_with_global_normalization_eager_fallback"  t t_min t_max m m_min m_max v v_min v_max beta beta_min beta_max gamma gamma_min gamma_max out_type variance_epsilon scale_after_normalization name ctx ))

(defn quantized-bias-add 
  "Adds Tensor 'bias' to Tensor 'input' for Quantized types.

  Broadcasts the values of bias on dimensions 0..N-2 of 'input'.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A 1D bias Tensor with size matching the last dimension of 'input'.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    min_bias: A `Tensor` of type `float32`.
      The float value that the lowest quantized bias value represents.
    max_bias: A `Tensor` of type `float32`.
      The float value that the highest quantized bias value represents.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_out, max_out).

    output: A `Tensor` of type `out_type`.
    min_out: A `Tensor` of type `float32`.
    max_out: A `Tensor` of type `float32`.
  "
  [ input bias min_input max_input min_bias max_bias out_type name ]
  (py/call-attr quantization "quantized_bias_add"  input bias min_input max_input min_bias max_bias out_type name ))

(defn quantized-bias-add-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_bias_add
  "
  [ input bias min_input max_input min_bias max_bias out_type name ctx ]
  (py/call-attr quantization "quantized_bias_add_eager_fallback"  input bias min_input max_input min_bias max_bias out_type name ctx ))

(defn quantized-concat 
  "Concatenates quantized tensors along one dimension.

  Args:
    concat_dim: A `Tensor` of type `int32`.
      0-D.  The dimension along which to concatenate.  Must be in the
      range [0, rank(values)).
    values: A list of at least 2 `Tensor` objects with the same type.
      The `N` Tensors to concatenate. Their ranks and types must match,
      and their sizes must match in all dimensions except `concat_dim`.
    input_mins: A list with the same length as `values` of `Tensor` objects with type `float32`.
      The minimum scalar values for each of the input tensors.
    input_maxes: A list with the same length as `values` of `Tensor` objects with type `float32`.
      The maximum scalar values for each of the input tensors.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).

    output: A `Tensor`. Has the same type as `values`.
    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  "
  [ concat_dim values input_mins input_maxes name ]
  (py/call-attr quantization "quantized_concat"  concat_dim values input_mins input_maxes name ))

(defn quantized-conv2d 
  "Computes a 2D convolution given quantized 4D input and filter tensors.

  The inputs are quantized tensors where the lowest value represents the real
  number of the associated minimum, and the highest represents the maximum.
  This means that you can only interpret the quantized output in the same way, by
  taking the returned minimum and maximum values into account.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      filter's input_depth dimension must match input's depth dimensions.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    min_filter: A `Tensor` of type `float32`.
      The float value that the lowest quantized filter value represents.
    max_filter: A `Tensor` of type `float32`.
      The float value that the highest quantized filter value represents.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      tensor.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      1-D tensor of length 4.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each
      filter element on that dimension. The dimension order is determined by the
      value of `data_format`, see above for details. Dilations in the batch and
      depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_conv2d" [input filter min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :name name }))

(defn quantized-conv2d-and-relu 
  "TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations padding_list name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_and_relu" [input filter min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name }))

(defn quantized-conv2d-and-relu-and-requantize 
  "TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding & {:keys [out_type dilations padding_list name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_and_relu_and_requantize" [input filter min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name }))

(defn quantized-conv2d-and-relu-and-requantize-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_conv2d_and_relu_and_requantize
  "
  [input filter min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding & {:keys [out_type dilations padding_list name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_and_relu_and_requantize_eager_fallback" [input filter min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name :ctx ctx }))

(defn quantized-conv2d-and-relu-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_conv2d_and_relu
  "
  [input filter min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations padding_list name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_and_relu_eager_fallback" [input filter min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name :ctx ctx }))

(defn quantized-conv2d-and-requantize 
  "TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint8`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding & {:keys [out_type dilations padding_list name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_and_requantize" [input filter min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name }))

(defn quantized-conv2d-and-requantize-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_conv2d_and_requantize
  "
  [input filter min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding & {:keys [out_type dilations padding_list name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_and_requantize_eager_fallback" [input filter min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name :ctx ctx }))

(defn quantized-conv2d-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_conv2d
  "
  [input filter min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_eager_fallback" [input filter min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :name name :ctx ctx }))

(defn quantized-conv2d-per-channel 
  "Computes QuantizedConv2D per channel.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original filter tensor.
    min_input: A `Tensor` of type `float32`.
      The minimum value of the input tensor
    max_input: A `Tensor` of type `float32`.
      The maximum value of the input tensor.
    min_filter: A `Tensor` of type `float32`.
      The minimum value of the filter tensor.
    max_filter: A `Tensor` of type `float32`.
      The maximum value of the filter tensor.
    strides: A list of `ints`. list of stride values.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
      The quantized type of output tensor that needs to be converted.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      list of dilation values.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_per_channel" [input filter min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :name name }))

(defn quantized-conv2d-per-channel-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_conv2d_per_channel
  "
  [input filter min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_per_channel_eager_fallback" [input filter min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :name name :ctx ctx }))

(defn quantized-conv2d-with-bias 
  "TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor` of type `float32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter bias min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations padding_list name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_with_bias" [input filter bias min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name }))

(defn quantized-conv2d-with-bias-and-relu 
  "TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor` of type `float32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter bias min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations padding_list name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_with_bias_and_relu" [input filter bias min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name }))

(defn quantized-conv2d-with-bias-and-relu-and-requantize 
  "TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding & {:keys [out_type dilations padding_list name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_with_bias_and_relu_and_requantize" [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name }))

(defn quantized-conv2d-with-bias-and-relu-and-requantize-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_conv2d_with_bias_and_relu_and_requantize
  "
  [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding & {:keys [out_type dilations padding_list name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_with_bias_and_relu_and_requantize_eager_fallback" [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name :ctx ctx }))

(defn quantized-conv2d-with-bias-and-relu-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_conv2d_with_bias_and_relu
  "
  [input filter bias min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations padding_list name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_with_bias_and_relu_eager_fallback" [input filter bias min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name :ctx ctx }))

(defn quantized-conv2d-with-bias-and-requantize 
  "TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint8`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding & {:keys [out_type dilations padding_list name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_with_bias_and_requantize" [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name }))

(defn quantized-conv2d-with-bias-and-requantize-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_conv2d_with_bias_and_requantize
  "
  [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding & {:keys [out_type dilations padding_list name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_with_bias_and_requantize_eager_fallback" [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name :ctx ctx }))

(defn quantized-conv2d-with-bias-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_conv2d_with_bias
  "
  [input filter bias min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations padding_list name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_with_bias_eager_fallback" [input filter bias min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name :ctx ctx }))

(defn quantized-conv2d-with-bias-signed-sum-and-relu-and-requantize 
  "TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    summand: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_summand: A `Tensor` of type `float32`.
    max_summand: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output summand min_summand max_summand strides padding & {:keys [out_type dilations padding_list name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_with_bias_signed_sum_and_relu_and_requantize" [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output summand min_summand max_summand strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name }))

(defn quantized-conv2d-with-bias-signed-sum-and-relu-and-requantize-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_conv2d_with_bias_signed_sum_and_relu_and_requantize
  "
  [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output summand min_summand max_summand strides padding & {:keys [out_type dilations padding_list name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_with_bias_signed_sum_and_relu_and_requantize_eager_fallback" [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output summand min_summand max_summand strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name :ctx ctx }))

(defn quantized-conv2d-with-bias-sum-and-relu 
  "TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor` of type `float32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    summand: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter bias min_input max_input min_filter max_filter summand strides padding & {:keys [out_type dilations padding_list name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_with_bias_sum_and_relu" [input filter bias min_input max_input min_filter max_filter summand strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name }))

(defn quantized-conv2d-with-bias-sum-and-relu-and-requantize 
  "TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
    min_input: A `Tensor` of type `float32`.
    max_input: A `Tensor` of type `float32`.
    min_filter: A `Tensor` of type `float32`.
    max_filter: A `Tensor` of type `float32`.
    min_freezed_output: A `Tensor` of type `float32`.
    max_freezed_output: A `Tensor` of type `float32`.
    summand: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_summand: A `Tensor` of type `float32`.
    max_summand: A `Tensor` of type `float32`.
    strides: A list of `ints`.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
    padding_list: An optional list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output summand min_summand max_summand strides padding & {:keys [out_type dilations padding_list name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_with_bias_sum_and_relu_and_requantize" [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output summand min_summand max_summand strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name }))

(defn quantized-conv2d-with-bias-sum-and-relu-and-requantize-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_conv2d_with_bias_sum_and_relu_and_requantize
  "
  [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output summand min_summand max_summand strides padding & {:keys [out_type dilations padding_list name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_with_bias_sum_and_relu_and_requantize_eager_fallback" [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output summand min_summand max_summand strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name :ctx ctx }))

(defn quantized-conv2d-with-bias-sum-and-relu-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_conv2d_with_bias_sum_and_relu
  "
  [input filter bias min_input max_input min_filter max_filter summand strides padding & {:keys [out_type dilations padding_list name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_conv2d_with_bias_sum_and_relu_eager_fallback" [input filter bias min_input max_input min_filter max_filter summand strides padding] {:out_type out_type :dilations dilations :padding_list padding_list :name name :ctx ctx }))

(defn quantized-depthwise-conv2d 
  "Computes quantized depthwise Conv2D.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original filter tensor.
    min_input: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    min_filter: A `Tensor` of type `float32`.
      The float value that the minimum quantized filter value represents.
    max_filter: A `Tensor` of type `float32`.
      The float value that the maximum quantized filter value represents.
    strides: A list of `ints`. List of stride values.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
      The type of the output.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      List of dilation values.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_depthwise_conv2d" [input filter min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :name name }))

(defn quantized-depthwise-conv2d-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_depthwise_conv2d
  "
  [input filter min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_depthwise_conv2d_eager_fallback" [input filter min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :name name :ctx ctx }))

(defn quantized-depthwise-conv2d-with-bias 
  "Computes quantized depthwise Conv2D with Bias.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original filter tensor.
    bias: A `Tensor` of type `float32`. The original bias tensor.
    min_input: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    min_filter: A `Tensor` of type `float32`.
      The float value that the minimum quantized filter value represents.
    max_filter: A `Tensor` of type `float32`.
      The float value that the maximum quantized filter value represents.
    strides: A list of `ints`. List of stride values.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
      The type of the output.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      List of dilation values.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter bias min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_depthwise_conv2d_with_bias" [input filter bias min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :name name }))

(defn quantized-depthwise-conv2d-with-bias-and-relu 
  "Computes quantized depthwise Conv2D with Bias and Relu.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original filter tensor.
    bias: A `Tensor` of type `float32`. The original bias tensor.
    min_input: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    min_filter: A `Tensor` of type `float32`.
      The float value that the minimum quantized filter value represents.
    max_filter: A `Tensor` of type `float32`.
      The float value that the maximum quantized filter value represents.
    strides: A list of `ints`. List of stride values.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
      The type of the output.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      List of dilation values.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter bias min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_depthwise_conv2d_with_bias_and_relu" [input filter bias min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :name name }))

(defn quantized-depthwise-conv2d-with-bias-and-relu-and-requantize 
  "Computes quantized depthwise Conv2D with Bias, Relu and Requantize.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original filter tensor.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
      The original bias tensor.
    min_input: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    min_filter: A `Tensor` of type `float32`.
      The float value that the minimum quantized filter value represents.
    max_filter: A `Tensor` of type `float32`.
      The float value that the maximum quantized filter value represents.
    min_freezed_output: A `Tensor` of type `float32`.
      The minimum float value of the output tensor.
    max_freezed_output: A `Tensor` of type `float32`.
      The maximum float value of the output tensor.
    strides: A list of `ints`. List of stride values.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
      The type of the output.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1]`.
      List of dilation values.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding & {:keys [out_type dilations name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_depthwise_conv2d_with_bias_and_relu_and_requantize" [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding] {:out_type out_type :dilations dilations :name name }))

(defn quantized-depthwise-conv2d-with-bias-and-relu-and-requantize-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_depthwise_conv2d_with_bias_and_relu_and_requantize
  "
  [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding & {:keys [out_type dilations name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_depthwise_conv2d_with_bias_and_relu_and_requantize_eager_fallback" [input filter bias min_input max_input min_filter max_filter min_freezed_output max_freezed_output strides padding] {:out_type out_type :dilations dilations :name name :ctx ctx }))

(defn quantized-depthwise-conv2d-with-bias-and-relu-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_depthwise_conv2d_with_bias_and_relu
  "
  [input filter bias min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_depthwise_conv2d_with_bias_and_relu_eager_fallback" [input filter bias min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :name name :ctx ctx }))

(defn quantized-depthwise-conv2d-with-bias-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_depthwise_conv2d_with_bias
  "
  [input filter bias min_input max_input min_filter max_filter strides padding & {:keys [out_type dilations name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_depthwise_conv2d_with_bias_eager_fallback" [input filter bias min_input max_input min_filter max_filter strides padding] {:out_type out_type :dilations dilations :name name :ctx ctx }))

(defn quantized-mat-mul 
  "Perform a quantized matrix multiplication of  `a` by the matrix `b`.

  The inputs must be two-dimensional matrices and the inner dimension of
  `a` (after being transposed if `transpose_a` is non-zero) must match the
  outer dimension of `b` (after being transposed if `transposed_b` is
  non-zero).

  Args:
    a: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      Must be a two-dimensional tensor.
    b: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      Must be a two-dimensional tensor.
    min_a: A `Tensor` of type `float32`.
      The float value that the lowest quantized `a` value represents.
    max_a: A `Tensor` of type `float32`.
      The float value that the highest quantized `a` value represents.
    min_b: A `Tensor` of type `float32`.
      The float value that the lowest quantized `b` value represents.
    max_b: A `Tensor` of type `float32`.
      The float value that the highest quantized `b` value represents.
    Toutput: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    transpose_a: An optional `bool`. Defaults to `False`.
      If true, `a` is transposed before multiplication.
    transpose_b: An optional `bool`. Defaults to `False`.
      If true, `b` is transposed before multiplication.
    Tactivation: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
      The type of output produced by activation function
      following this operation.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (out, min_out, max_out).

    out: A `Tensor` of type `Toutput`.
    min_out: A `Tensor` of type `float32`.
    max_out: A `Tensor` of type `float32`.
  "
  [a b min_a max_a min_b max_b & {:keys [Toutput transpose_a transpose_b Tactivation name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_mat_mul" [a b min_a max_a min_b max_b] {:Toutput Toutput :transpose_a transpose_a :transpose_b transpose_b :Tactivation Tactivation :name name }))

(defn quantized-mat-mul-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_mat_mul
  "
  [a b min_a max_a min_b max_b & {:keys [Toutput transpose_a transpose_b Tactivation name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_mat_mul_eager_fallback" [a b min_a max_a min_b max_b] {:Toutput Toutput :transpose_a transpose_a :transpose_b transpose_b :Tactivation Tactivation :name name :ctx ctx }))

(defn quantized-mat-mul-with-bias 
  "Performs a quantized matrix multiplication of `a` by the matrix `b` with bias
add.

  The inputs must be two-dimensional matrices and 1D bias vector. And the inner
  dimension of `a` (after being transposed if `transpose_a` is non-zero) must
  match the outer dimension of `b` (after being transposed if `transposed_b` is
  non-zero). Then do broadcast add operation with bias values on the matrix
  mulplication result. The bias size must match inner dimension of `b`.

  Args:
    a: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A matrix to be multiplied. Must be a two-dimensional tensor of type `quint8`.
    b: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A matrix to be multiplied and must be a two-dimensional tensor of type `qint8`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
      A 1D bias tensor with size matching inner dimension of `b` (after being
      transposed if `transposed_b` is non-zero).
    min_a: A `Tensor` of type `float32`.
      The float value that the lowest quantized `a` value represents.
    max_a: A `Tensor` of type `float32`.
      The float value that the highest quantized `a` value represents.
    min_b: A `Tensor` of type `float32`.
      The float value that the lowest quantized `b` value represents.
    max_b: A `Tensor` of type `float32`.
      The float value that the highest quantized `b` value represents.
    Toutput: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    transpose_a: An optional `bool`. Defaults to `False`.
      If true, `a` is transposed before multiplication.
    transpose_b: An optional `bool`. Defaults to `False`.
      If true, `b` is transposed before multiplication.
    input_quant_mode: An optional `string` from: `\"MIN_FIRST\", \"SCALED\"`. Defaults to `\"MIN_FIRST\"`.
      Input data quantization mode. Either MIN_FIRST(default) or SCALED.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (out, min_out, max_out).

    out: A `Tensor` of type `Toutput`.
    min_out: A `Tensor` of type `float32`.
    max_out: A `Tensor` of type `float32`.
  "
  [a b bias min_a max_a min_b max_b & {:keys [Toutput transpose_a transpose_b input_quant_mode name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_mat_mul_with_bias" [a b bias min_a max_a min_b max_b] {:Toutput Toutput :transpose_a transpose_a :transpose_b transpose_b :input_quant_mode input_quant_mode :name name }))

(defn quantized-mat-mul-with-bias-and-relu 
  "Perform a quantized matrix multiplication of  `a` by the matrix `b` with bias
add and relu fusion.

  The inputs must be two-dimensional matrices and 1D bias vector. And the inner
  dimension of `a` (after being transposed if `transpose_a` is non-zero) must
  match the outer dimension of `b` (after being transposed if `transposed_b` is
  non-zero). Then do broadcast add operation with bias values on the matrix
  mulplication result. The bias size must match inner dimension of `b`. Then do
  relu activation to get non-negative result.

  Args:
    a: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A matrix to be multiplied. Must be a two-dimensional tensor of type `quint8`.
    b: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A matrix to be multiplied and must be a two-dimensional tensor of type `qint8`.
    bias: A `Tensor` of type `float32`.
      A 1D bias tensor with size matching with inner dimension of `b` (after being
      transposed if `transposed_b` is non-zero).
    min_a: A `Tensor` of type `float32`.
      The float value that the lowest quantized `a` value represents.
    max_a: A `Tensor` of type `float32`.
      The float value that the highest quantized `a` value represents.
    min_b: A `Tensor` of type `float32`.
      The float value that the lowest quantized `b` value represents.
    max_b: A `Tensor` of type `float32`.
      The float value that the highest quantized `b` value represents.
    Toutput: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    transpose_a: An optional `bool`. Defaults to `False`.
      If true, `a` is transposed before multiplication.
    transpose_b: An optional `bool`. Defaults to `False`.
      If true, `b` is transposed before multiplication.
    input_quant_mode: An optional `string` from: `\"MIN_FIRST\", \"SCALED\"`. Defaults to `\"MIN_FIRST\"`.
      Input data quantization mode. Either MIN_FIRST(default) or SCALED.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (out, min_out, max_out).

    out: A `Tensor` of type `Toutput`.
    min_out: A `Tensor` of type `float32`.
    max_out: A `Tensor` of type `float32`.
  "
  [a b bias min_a max_a min_b max_b & {:keys [Toutput transpose_a transpose_b input_quant_mode name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_mat_mul_with_bias_and_relu" [a b bias min_a max_a min_b max_b] {:Toutput Toutput :transpose_a transpose_a :transpose_b transpose_b :input_quant_mode input_quant_mode :name name }))

(defn quantized-mat-mul-with-bias-and-relu-and-requantize 
  "Perform a quantized matrix multiplication of  `a` by the matrix `b` with bias
add and relu and requantize fusion.

  The inputs must be two-dimensional matrices and 1D bias vector. And the inner
  dimension of `a` (after being transposed if `transpose_a` is non-zero) must
  match the outer dimension of `b` (after being transposed if `transposed_b` is
  non-zero). Then do broadcast add operation with bias values on the matrix
  mulplication result. The bias size must match inner dimension of `b`.  Then do
  relu activation to get non-negative result. Then do requantize operation to get
  final uint8 result.

  Args:
    a: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A matrix to be multiplied. Must be a two-dimensional tensor of type `quint8`.
    b: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A matrix to be multiplied and must be a two-dimensional tensor of type `qint8`.
    bias: A `Tensor`. Must be one of the following types: `float32`, `qint32`.
      A 1D bias tensor with size matching with inner dimension of `b` (after being
      transposed if `transposed_b` is non-zero).
    min_a: A `Tensor` of type `float32`.
      The float value that the lowest quantized `a` value represents.
    max_a: A `Tensor` of type `float32`.
      The float value that the highest quantized `a` value represents.
    min_b: A `Tensor` of type `float32`.
      The float value that the lowest quantized `b` value represents.
    max_b: A `Tensor` of type `float32`.
      The float value that the highest quantized `b` value represents.
    min_freezed_output: A `Tensor` of type `float32`.
      The float value that the highest quantized output value after requantize.
    max_freezed_output: A `Tensor` of type `float32`.
    Toutput: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    transpose_a: An optional `bool`. Defaults to `False`.
      If true, `a` is transposed before multiplication.
    transpose_b: An optional `bool`. Defaults to `False`.
      If true, `b` is transposed before multiplication.
    input_quant_mode: An optional `string` from: `\"MIN_FIRST\", \"SCALED\"`. Defaults to `\"MIN_FIRST\"`.
      Input data quantization mode. Either MIN_FIRST(default) or SCALED.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (out, min_out, max_out).

    out: A `Tensor` of type `Toutput`.
    min_out: A `Tensor` of type `float32`.
    max_out: A `Tensor` of type `float32`.
  "
  [a b bias min_a max_a min_b max_b min_freezed_output max_freezed_output & {:keys [Toutput transpose_a transpose_b input_quant_mode name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_mat_mul_with_bias_and_relu_and_requantize" [a b bias min_a max_a min_b max_b min_freezed_output max_freezed_output] {:Toutput Toutput :transpose_a transpose_a :transpose_b transpose_b :input_quant_mode input_quant_mode :name name }))

(defn quantized-mat-mul-with-bias-and-relu-and-requantize-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_mat_mul_with_bias_and_relu_and_requantize
  "
  [a b bias min_a max_a min_b max_b min_freezed_output max_freezed_output & {:keys [Toutput transpose_a transpose_b input_quant_mode name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_mat_mul_with_bias_and_relu_and_requantize_eager_fallback" [a b bias min_a max_a min_b max_b min_freezed_output max_freezed_output] {:Toutput Toutput :transpose_a transpose_a :transpose_b transpose_b :input_quant_mode input_quant_mode :name name :ctx ctx }))

(defn quantized-mat-mul-with-bias-and-relu-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_mat_mul_with_bias_and_relu
  "
  [a b bias min_a max_a min_b max_b & {:keys [Toutput transpose_a transpose_b input_quant_mode name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_mat_mul_with_bias_and_relu_eager_fallback" [a b bias min_a max_a min_b max_b] {:Toutput Toutput :transpose_a transpose_a :transpose_b transpose_b :input_quant_mode input_quant_mode :name name :ctx ctx }))

(defn quantized-mat-mul-with-bias-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_mat_mul_with_bias
  "
  [a b bias min_a max_a min_b max_b & {:keys [Toutput transpose_a transpose_b input_quant_mode name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_mat_mul_with_bias_eager_fallback" [a b bias min_a max_a min_b max_b] {:Toutput Toutput :transpose_a transpose_a :transpose_b transpose_b :input_quant_mode input_quant_mode :name name :ctx ctx }))

(defn quantized-max-pool 
  "Produces the max pool of the input tensor for quantized types.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The 4D (batch x rows x cols x depth) Tensor to MaxReduce over.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    ksize: A list of `ints`.
      The size of the window for each dimension of the input tensor.
      The length must be 4 to match the number of dimensions of the input.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      tensor. The length must be 4 to match the number of dimensions of the input.
    padding: A `string` from: `\"SAME\", \"VALID\"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).

    output: A `Tensor`. Has the same type as `input`.
    min_output: A `Tensor` of type `float32`.
    max_output: A `Tensor` of type `float32`.
  "
  [ input min_input max_input ksize strides padding name ]
  (py/call-attr quantization "quantized_max_pool"  input min_input max_input ksize strides padding name ))

(defn quantized-max-pool-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_max_pool
  "
  [ input min_input max_input ksize strides padding name ctx ]
  (py/call-attr quantization "quantized_max_pool_eager_fallback"  input min_input max_input ksize strides padding name ctx ))

(defn quantized-mul 
  "Returns x * y element-wise, working on quantized buffers.

  Args:
    x: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    y: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_x: A `Tensor` of type `float32`.
      The float value that the lowest quantized `x` value represents.
    max_x: A `Tensor` of type `float32`.
      The float value that the highest quantized `x` value represents.
    min_y: A `Tensor` of type `float32`.
      The float value that the lowest quantized `y` value represents.
    max_y: A `Tensor` of type `float32`.
      The float value that the highest quantized `y` value represents.
    Toutput: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.qint32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (z, min_z, max_z).

    z: A `Tensor` of type `Toutput`.
    min_z: A `Tensor` of type `float32`.
    max_z: A `Tensor` of type `float32`.
  "
  [x y min_x max_x min_y max_y & {:keys [Toutput name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_mul" [x y min_x max_x min_y max_y] {:Toutput Toutput :name name }))

(defn quantized-mul-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_mul
  "
  [x y min_x max_x min_y max_y & {:keys [Toutput name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_mul_eager_fallback" [x y min_x max_x min_y max_y] {:Toutput Toutput :name name :ctx ctx }))

(defn quantized-relu 
  "Computes Quantized Rectified Linear: `max(features, 0)`

  Args:
    features: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_features: A `Tensor` of type `float32`.
      The float value that the lowest quantized value represents.
    max_features: A `Tensor` of type `float32`.
      The float value that the highest quantized value represents.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (activations, min_activations, max_activations).

    activations: A `Tensor` of type `out_type`.
    min_activations: A `Tensor` of type `float32`.
    max_activations: A `Tensor` of type `float32`.
  "
  [features min_features max_features & {:keys [out_type name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_relu" [features min_features max_features] {:out_type out_type :name name }))

(defn quantized-relu6 
  "Computes Quantized Rectified Linear 6: `min(max(features, 0), 6)`

  Args:
    features: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    min_features: A `Tensor` of type `float32`.
      The float value that the lowest quantized value represents.
    max_features: A `Tensor` of type `float32`.
      The float value that the highest quantized value represents.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (activations, min_activations, max_activations).

    activations: A `Tensor` of type `out_type`.
    min_activations: A `Tensor` of type `float32`.
    max_activations: A `Tensor` of type `float32`.
  "
  [features min_features max_features & {:keys [out_type name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_relu6" [features min_features max_features] {:out_type out_type :name name }))

(defn quantized-relu6-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_relu6
  "
  [features min_features max_features & {:keys [out_type name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_relu6_eager_fallback" [features min_features max_features] {:out_type out_type :name name :ctx ctx }))

(defn quantized-relu-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_relu
  "
  [features min_features max_features & {:keys [out_type name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_relu_eager_fallback" [features min_features max_features] {:out_type out_type :name name :ctx ctx }))

(defn quantized-relu-x 
  "Computes Quantized Rectified Linear X: `min(max(features, 0), max_value)`

  Args:
    features: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    max_value: A `Tensor` of type `float32`.
    min_features: A `Tensor` of type `float32`.
      The float value that the lowest quantized value represents.
    max_features: A `Tensor` of type `float32`.
      The float value that the highest quantized value represents.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (activations, min_activations, max_activations).

    activations: A `Tensor` of type `out_type`.
    min_activations: A `Tensor` of type `float32`.
    max_activations: A `Tensor` of type `float32`.
  "
  [features max_value min_features max_features & {:keys [out_type name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "quantized_relu_x" [features max_value min_features max_features] {:out_type out_type :name name }))

(defn quantized-relu-x-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_relu_x
  "
  [features max_value min_features max_features & {:keys [out_type name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "quantized_relu_x_eager_fallback" [features max_value min_features max_features] {:out_type out_type :name name :ctx ctx }))

(defn real 
  "Returns the real part of a complex number.

  Given a tensor `input` of complex numbers, this operation returns a tensor of
  type `float` that is the real part of each element in `input`. All elements in
  `input` must be complex numbers of the form \\(a + bj\\), where *a* is the real
   part returned by this operation and *b* is the imaginary part.

  For example:

  ```
  # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.real(input) ==> [-2.25, 3.25]
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
    Tout: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  "
  [input & {:keys [Tout name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "real" [input] {:Tout Tout :name name }))

(defn real-div 
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
  (py/call-attr quantization "real_div"  x y name ))

(defn real-div-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function real_div
  "
  [ x y name ctx ]
  (py/call-attr quantization "real_div_eager_fallback"  x y name ctx ))

(defn real-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function real
  "
  [input & {:keys [Tout name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "real_eager_fallback" [input] {:Tout Tout :name name :ctx ctx }))

(defn reciprocal 
  "Computes the reciprocal of x element-wise.

  I.e., \\(y = 1 / x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "reciprocal"  x name ))

(defn reciprocal-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function reciprocal
  "
  [ x name ctx ]
  (py/call-attr quantization "reciprocal_eager_fallback"  x name ctx ))

(defn reciprocal-grad 
  "Computes the gradient for the inverse of `x` wrt its input.

  Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
  is the corresponding input gradient.

  Args:
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    dy: A `Tensor`. Must have the same type as `y`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `y`.
  "
  [ y dy name ]
  (py/call-attr quantization "reciprocal_grad"  y dy name ))

(defn reciprocal-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function reciprocal_grad
  "
  [ y dy name ctx ]
  (py/call-attr quantization "reciprocal_grad_eager_fallback"  y dy name ctx ))

(defn relu 
  "Computes rectified linear: `max(features, 0)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`, `qint8`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  "
  [ features name ]
  (py/call-attr quantization "relu"  features name ))

(defn relu6 
  "Computes rectified linear 6: `min(max(features, 0), 6)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  "
  [ features name ]
  (py/call-attr quantization "relu6"  features name ))

(defn relu6-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function relu6
  "
  [ features name ctx ]
  (py/call-attr quantization "relu6_eager_fallback"  features name ctx ))

(defn relu6-grad 
  "Computes rectified linear 6 gradients for a Relu6 operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The backpropagated gradients to the corresponding Relu6 operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding Relu6 operation, or
      its output; using either one produces the same result.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  "
  [ gradients features name ]
  (py/call-attr quantization "relu6_grad"  gradients features name ))

(defn relu6-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function relu6_grad
  "
  [ gradients features name ctx ]
  (py/call-attr quantization "relu6_grad_eager_fallback"  gradients features name ctx ))

(defn relu-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function relu
  "
  [ features name ctx ]
  (py/call-attr quantization "relu_eager_fallback"  features name ctx ))

(defn relu-grad 
  "Computes rectified linear gradients for a Relu operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      The backpropagated gradients to the corresponding Relu operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding Relu operation, OR
      the outputs of that operation (both work equivalently).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  "
  [ gradients features name ]
  (py/call-attr quantization "relu_grad"  gradients features name ))

(defn relu-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function relu_grad
  "
  [ gradients features name ctx ]
  (py/call-attr quantization "relu_grad_eager_fallback"  gradients features name ctx ))

(defn requantization-range 
  "Computes a range that covers the actual values present in a quantized tensor.

  Given a quantized tensor described by `(input, input_min, input_max)`, outputs a
  range that covers the actual values present in that tensor. This op is typically
  used to produce the `requested_output_min` and `requested_output_max` for
  `Requantize`.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    input_min: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    input_max: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_min, output_max).

    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  "
  [ input input_min input_max name ]
  (py/call-attr quantization "requantization_range"  input input_min input_max name ))

(defn requantization-range-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function requantization_range
  "
  [ input input_min input_max name ctx ]
  (py/call-attr quantization "requantization_range_eager_fallback"  input input_min input_max name ctx ))

(defn requantization-range-per-channel 
  "Computes requantization range per channel.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    input_min: A `Tensor` of type `float32`.
      The minimum value of the input tensor
    input_max: A `Tensor` of type `float32`.
      The maximum value of the input tensor.
    clip_value_max: A `float`.
      The maximum value of the output that needs to be clipped.
      Example: set this to 6 for Relu6.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_min, output_max).

    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  "
  [ input input_min input_max clip_value_max name ]
  (py/call-attr quantization "requantization_range_per_channel"  input input_min input_max clip_value_max name ))

(defn requantization-range-per-channel-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function requantization_range_per_channel
  "
  [ input input_min input_max clip_value_max name ctx ]
  (py/call-attr quantization "requantization_range_per_channel_eager_fallback"  input input_min input_max clip_value_max name ctx ))

(defn requantize 
  "Converts the quantized `input` tensor into a lower-precision `output`.

  Converts the quantized `input` tensor into a lower-precision `output`, using the
  output range specified with `requested_output_min` and `requested_output_max`.

  `[input_min, input_max]` are scalar floats that specify the range for the float
  interpretation of the `input` data. For example, if `input_min` is -1.0f and
  `input_max` is 1.0f, and we are dealing with `quint16` quantized data, then a 0
  value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
    input_min: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    input_max: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    requested_output_min: A `Tensor` of type `float32`.
      The float value that the minimum quantized output value represents.
    requested_output_max: A `Tensor` of type `float32`.
      The float value that the maximum quantized output value represents.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`.
      The type of the output. Should be a lower bit depth than Tinput.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).

    output: A `Tensor` of type `out_type`.
    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  "
  [ input input_min input_max requested_output_min requested_output_max out_type name ]
  (py/call-attr quantization "requantize"  input input_min input_max requested_output_min requested_output_max out_type name ))

(defn requantize-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function requantize
  "
  [ input input_min input_max requested_output_min requested_output_max out_type name ctx ]
  (py/call-attr quantization "requantize_eager_fallback"  input input_min input_max requested_output_min requested_output_max out_type name ctx ))

(defn requantize-per-channel 
  "Requantizes input with min and max values known per channel.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      The original input tensor.
    input_min: A `Tensor` of type `float32`.
      The minimum value of the input tensor
    input_max: A `Tensor` of type `float32`.
      The maximum value of the input tensor.
    requested_output_min: A `Tensor` of type `float32`.
      The minimum value of the output tensor requested.
    requested_output_max: A `Tensor` of type `float32`.
      The maximum value of the output tensor requested.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint32, tf.qint16, tf.quint16`. Defaults to `tf.quint8`.
      The quantized type of output tensor that needs to be converted.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).

    output: A `Tensor` of type `out_type`.
    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  "
  [input input_min input_max requested_output_min requested_output_max & {:keys [out_type name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "requantize_per_channel" [input input_min input_max requested_output_min requested_output_max] {:out_type out_type :name name }))

(defn requantize-per-channel-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function requantize_per_channel
  "
  [input input_min input_max requested_output_min requested_output_max & {:keys [out_type name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "requantize_per_channel_eager_fallback" [input input_min input_max requested_output_min requested_output_max] {:out_type out_type :name name :ctx ctx }))

(defn rint 
  "Returns element-wise integer closest to x.

  If the result is midway between two representable values,
  the even representable is chosen.
  For example:

  ```
  rint(-1.5) ==> -2.0
  rint(0.5000001) ==> 1.0
  rint([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) ==> [-2., -2., -0., 0., 2., 2., 2.]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "rint"  x name ))

(defn rint-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function rint
  "
  [ x name ctx ]
  (py/call-attr quantization "rint_eager_fallback"  x name ctx ))

(defn round 
  "Rounds the values of a tensor to the nearest integer, element-wise.

  Rounds half to even.  Also known as bankers rounding. If you want to round
  according to the current system rounding mode use std::cint.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "round"  x name ))

(defn round-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function round
  "
  [ x name ctx ]
  (py/call-attr quantization "round_eager_fallback"  x name ctx ))

(defn rsqrt 
  "Computes reciprocal of square root of x element-wise.

  I.e., \\(y = 1 / \sqrt{x}\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "rsqrt"  x name ))

(defn rsqrt-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function rsqrt
  "
  [ x name ctx ]
  (py/call-attr quantization "rsqrt_eager_fallback"  x name ctx ))

(defn rsqrt-grad 
  "Computes the gradient for the rsqrt of `x` wrt its input.

  Specifically, `grad = dy * -0.5 * y^3`, where `y = rsqrt(x)`, and `dy`
  is the corresponding input gradient.

  Args:
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    dy: A `Tensor`. Must have the same type as `y`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `y`.
  "
  [ y dy name ]
  (py/call-attr quantization "rsqrt_grad"  y dy name ))

(defn rsqrt-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function rsqrt_grad
  "
  [ y dy name ctx ]
  (py/call-attr quantization "rsqrt_grad_eager_fallback"  y dy name ctx ))

(defn segment-max 
  "Computes the maximum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \max_j(data_j)\\) where `max` is over `j` such
  that `segment_ids[j] == i`.

  If the max is empty for a given segment ID `i`, `output[i] = 0`.

  <div style=\"width:70%; margin:auto; margin-bottom:10px; margin-top:20px;\">
  <img style=\"width:100%\" src=\"https://www.tensorflow.org/images/SegmentMax.png\" alt>
  </div>

  For example:

  ```
  c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
  tf.segment_max(c, tf.constant([0, 0, 1]))
  # ==> [[4, 3, 3, 4],
  #      [5, 6, 7, 8]]
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data segment_ids name ]
  (py/call-attr quantization "segment_max"  data segment_ids name ))

(defn segment-max-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function segment_max
  "
  [ data segment_ids name ctx ]
  (py/call-attr quantization "segment_max_eager_fallback"  data segment_ids name ctx ))

(defn segment-mean 
  "Computes the mean along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \frac{\sum_j data_j}{N}\\) where `mean` is
  over `j` such that `segment_ids[j] == i` and `N` is the total number of
  values summed.

  If the mean is empty for a given segment ID `i`, `output[i] = 0`.

  <div style=\"width:70%; margin:auto; margin-bottom:10px; margin-top:20px;\">
  <img style=\"width:100%\" src=\"https://www.tensorflow.org/images/SegmentMean.png\" alt>
  </div>

  For example:

  ```
  c = tf.constant([[1.0,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
  tf.segment_mean(c, tf.constant([0, 0, 1]))
  # ==> [[2.5, 2.5, 2.5, 2.5],
  #      [5, 6, 7, 8]]
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data segment_ids name ]
  (py/call-attr quantization "segment_mean"  data segment_ids name ))

(defn segment-mean-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function segment_mean
  "
  [ data segment_ids name ctx ]
  (py/call-attr quantization "segment_mean_eager_fallback"  data segment_ids name ctx ))

(defn segment-min 
  "Computes the minimum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \min_j(data_j)\\) where `min` is over `j` such
  that `segment_ids[j] == i`.

  If the min is empty for a given segment ID `i`, `output[i] = 0`.

  <div style=\"width:70%; margin:auto; margin-bottom:10px; margin-top:20px;\">
  <img style=\"width:100%\" src=\"https://www.tensorflow.org/images/SegmentMin.png\" alt>
  </div>

  For example:

  ```
  c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
  tf.segment_min(c, tf.constant([0, 0, 1]))
  # ==> [[1, 2, 2, 1],
  #      [5, 6, 7, 8]]
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data segment_ids name ]
  (py/call-attr quantization "segment_min"  data segment_ids name ))

(defn segment-min-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function segment_min
  "
  [ data segment_ids name ctx ]
  (py/call-attr quantization "segment_min_eager_fallback"  data segment_ids name ctx ))

(defn segment-prod 
  "Computes the product along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \prod_j data_j\\) where the product is over `j` such
  that `segment_ids[j] == i`.

  If the product is empty for a given segment ID `i`, `output[i] = 1`.

  <div style=\"width:70%; margin:auto; margin-bottom:10px; margin-top:20px;\">
  <img style=\"width:100%\" src=\"https://www.tensorflow.org/images/SegmentProd.png\" alt>
  </div>

  For example:

  ```
  c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
  tf.segment_prod(c, tf.constant([0, 0, 1]))
  # ==> [[4, 6, 6, 4],
  #      [5, 6, 7, 8]]
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data segment_ids name ]
  (py/call-attr quantization "segment_prod"  data segment_ids name ))

(defn segment-prod-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function segment_prod
  "
  [ data segment_ids name ctx ]
  (py/call-attr quantization "segment_prod_eager_fallback"  data segment_ids name ctx ))

(defn segment-sum 
  "Computes the sum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \sum_j data_j\\) where sum is over `j` such
  that `segment_ids[j] == i`.

  If the sum is empty for a given segment ID `i`, `output[i] = 0`.

  <div style=\"width:70%; margin:auto; margin-bottom:10px; margin-top:20px;\">
  <img style=\"width:100%\" src=\"https://www.tensorflow.org/images/SegmentSum.png\" alt>
  </div>

  For example:

  ```
  c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
  tf.segment_sum(c, tf.constant([0, 0, 1]))
  # ==> [[5, 5, 5, 5],
  #      [5, 6, 7, 8]]
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data segment_ids name ]
  (py/call-attr quantization "segment_sum"  data segment_ids name ))

(defn segment-sum-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function segment_sum
  "
  [ data segment_ids name ctx ]
  (py/call-attr quantization "segment_sum_eager_fallback"  data segment_ids name ctx ))

(defn select 
  "Selects elements from `x` or `y`, depending on `condition`.

  The `x`, and `y` tensors must all have the same shape, and the
  output will also have that shape.

  The `condition` tensor must be a scalar if `x` and `y` are scalars.
  If `x` and `y` are vectors or higher rank, then `condition` must be either a
  scalar, a vector with size matching the first dimension of `x`, or must have
  the same shape as `x`.

  The `condition` tensor acts as a mask that chooses, based on the value at each
  element, whether the corresponding element / row in the output should be
  taken from `x` (if true) or `y` (if false).

  If `condition` is a vector and `x` and `y` are higher rank matrices, then
  it chooses which row (outer dimension) to copy from `x` and `y`.
  If `condition` has the same shape as `x` and `y`, then it chooses which
  element to copy from `x` and `y`.

  For example:

  ```python
  # 'condition' tensor is [[True,  False]
  #                        [False, True]]
  # 't' is [[1, 2],
  #         [3, 4]]
  # 'e' is [[5, 6],
  #         [7, 8]]
  select(condition, t, e)  # => [[1, 6], [7, 4]]


  # 'condition' tensor is [True, False]
  # 't' is [[1, 2],
  #         [3, 4]]
  # 'e' is [[5, 6],
  #         [7, 8]]
  select(condition, t, e) ==> [[1, 2],
                               [7, 8]]

  ```

  Args:
    condition: A `Tensor` of type `bool`.
    x:  A `Tensor` which may have the same shape as `condition`.
      If `condition` is rank 1, `x` may have higher rank,
      but its first dimension must match the size of `condition`.
    y:  A `Tensor` with the same type and shape as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `t`.
  "
  [ condition x y name ]
  (py/call-attr quantization "select"  condition x y name ))

(defn select-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function select
  "
  [ condition x y name ctx ]
  (py/call-attr quantization "select_eager_fallback"  condition x y name ctx ))

(defn select-v2 
  "TODO: add doc.

  Args:
    condition: A `Tensor` of type `bool`.
    t: A `Tensor`.
    e: A `Tensor`. Must have the same type as `t`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `t`.
  "
  [ condition t e name ]
  (py/call-attr quantization "select_v2"  condition t e name ))

(defn select-v2-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function select_v2
  "
  [ condition t e name ctx ]
  (py/call-attr quantization "select_v2_eager_fallback"  condition t e name ctx ))

(defn selu 
  "Computes scaled exponential linear: `scale * alpha * (exp(features) - 1)`

  if < 0, `scale * features` otherwise.

  To be used together with
  `initializer = tf.variance_scaling_initializer(factor=1.0, mode='FAN_IN')`.
  For correct dropout, use `tf.contrib.nn.alpha_dropout`.

  See [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  "
  [ features name ]
  (py/call-attr quantization "selu"  features name ))

(defn selu-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function selu
  "
  [ features name ctx ]
  (py/call-attr quantization "selu_eager_fallback"  features name ctx ))

(defn selu-grad 
  "Computes gradients for the scaled exponential linear (Selu) operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      The backpropagated gradients to the corresponding Selu operation.
    outputs: A `Tensor`. Must have the same type as `gradients`.
      The outputs of the corresponding Selu operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  "
  [ gradients outputs name ]
  (py/call-attr quantization "selu_grad"  gradients outputs name ))

(defn selu-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function selu_grad
  "
  [ gradients outputs name ctx ]
  (py/call-attr quantization "selu_grad_eager_fallback"  gradients outputs name ctx ))

(defn sigmoid 
  "Computes sigmoid of `x` element-wise.

  Specifically, `y = 1 / (1 + exp(-x))`.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr quantization "sigmoid"  x name ))

(defn sigmoid-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sigmoid
  "
  [ x name ctx ]
  (py/call-attr quantization "sigmoid_eager_fallback"  x name ctx ))

(defn sigmoid-grad 
  "Computes the gradient of the sigmoid of `x` wrt its input.

  Specifically, `grad = dy * y * (1 - y)`, where `y = sigmoid(x)`, and
  `dy` is the corresponding input gradient.

  Args:
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    dy: A `Tensor`. Must have the same type as `y`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `y`.
  "
  [ y dy name ]
  (py/call-attr quantization "sigmoid_grad"  y dy name ))

(defn sigmoid-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sigmoid_grad
  "
  [ y dy name ctx ]
  (py/call-attr quantization "sigmoid_grad_eager_fallback"  y dy name ctx ))

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
  (py/call-attr quantization "sign"  x name ))

(defn sign-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sign
  "
  [ x name ctx ]
  (py/call-attr quantization "sign_eager_fallback"  x name ctx ))

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
  (py/call-attr quantization "sin"  x name ))

(defn sin-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sin
  "
  [ x name ctx ]
  (py/call-attr quantization "sin_eager_fallback"  x name ctx ))

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
  (py/call-attr quantization "sinh"  x name ))

(defn sinh-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sinh
  "
  [ x name ctx ]
  (py/call-attr quantization "sinh_eager_fallback"  x name ctx ))

(defn softmax 
  "Computes softmax activations.

  For each batch `i` and class `j` we have

      $$softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))$$

  Args:
    logits: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      2-D with shape `[batch_size, num_classes]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `logits`.
  "
  [ logits name ]
  (py/call-attr quantization "softmax"  logits name ))

(defn softmax-cross-entropy-with-logits 
  "Computes softmax cross entropy cost and gradients to backpropagate.

  Inputs are the logits, not probabilities.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      batch_size x num_classes matrix
    labels: A `Tensor`. Must have the same type as `features`.
      batch_size x num_classes matrix
      The caller must ensure that each batch of labels represents a valid
      probability distribution.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (loss, backprop).

    loss: A `Tensor`. Has the same type as `features`.
    backprop: A `Tensor`. Has the same type as `features`.
  "
  [ features labels name ]
  (py/call-attr quantization "softmax_cross_entropy_with_logits"  features labels name ))

(defn softmax-cross-entropy-with-logits-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function softmax_cross_entropy_with_logits
  "
  [ features labels name ctx ]
  (py/call-attr quantization "softmax_cross_entropy_with_logits_eager_fallback"  features labels name ctx ))

(defn softmax-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function softmax
  "
  [ logits name ctx ]
  (py/call-attr quantization "softmax_eager_fallback"  logits name ctx ))

(defn softplus 
  "Computes softplus: `log(exp(features) + 1)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  "
  [ features name ]
  (py/call-attr quantization "softplus"  features name ))

(defn softplus-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function softplus
  "
  [ features name ctx ]
  (py/call-attr quantization "softplus_eager_fallback"  features name ctx ))

(defn softplus-grad 
  "Computes softplus gradients for a softplus operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      The backpropagated gradients to the corresponding softplus operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding softplus operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  "
  [ gradients features name ]
  (py/call-attr quantization "softplus_grad"  gradients features name ))

(defn softplus-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function softplus_grad
  "
  [ gradients features name ctx ]
  (py/call-attr quantization "softplus_grad_eager_fallback"  gradients features name ctx ))

(defn softsign 
  "Computes softsign: `features / (abs(features) + 1)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  "
  [ features name ]
  (py/call-attr quantization "softsign"  features name ))

(defn softsign-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function softsign
  "
  [ features name ctx ]
  (py/call-attr quantization "softsign_eager_fallback"  features name ctx ))

(defn softsign-grad 
  "Computes softsign gradients for a softsign operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      The backpropagated gradients to the corresponding softsign operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding softsign operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
  "
  [ gradients features name ]
  (py/call-attr quantization "softsign_grad"  gradients features name ))

(defn softsign-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function softsign_grad
  "
  [ gradients features name ctx ]
  (py/call-attr quantization "softsign_grad_eager_fallback"  gradients features name ctx ))

(defn sparse-mat-mul 
  "Multiply matrix \"a\" by matrix \"b\".

  The inputs must be two-dimensional matrices and the inner dimension of \"a\" must
  match the outer dimension of \"b\". Both \"a\" and \"b\" must be `Tensor`s not
  `SparseTensor`s.  This op is optimized for the case where at least one of \"a\" or
  \"b\" is sparse, in the sense that they have a large proportion of zero values.
  The breakeven for using this versus a dense matrix multiply on one platform was
  30% zero values in the sparse matrix.

  The gradient computation of this operation will only take advantage of sparsity
  in the input gradient when that gradient comes from a Relu.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `bfloat16`.
    b: A `Tensor`. Must be one of the following types: `float32`, `bfloat16`.
    transpose_a: An optional `bool`. Defaults to `False`.
    transpose_b: An optional `bool`. Defaults to `False`.
    a_is_sparse: An optional `bool`. Defaults to `False`.
    b_is_sparse: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  "
  [a b & {:keys [transpose_a transpose_b a_is_sparse b_is_sparse name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "sparse_mat_mul" [a b] {:transpose_a transpose_a :transpose_b transpose_b :a_is_sparse a_is_sparse :b_is_sparse b_is_sparse :name name }))

(defn sparse-mat-mul-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sparse_mat_mul
  "
  [a b & {:keys [transpose_a transpose_b a_is_sparse b_is_sparse name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "sparse_mat_mul_eager_fallback" [a b] {:transpose_a transpose_a :transpose_b transpose_b :a_is_sparse a_is_sparse :b_is_sparse b_is_sparse :name name :ctx ctx }))

(defn sparse-segment-mean 
  "Computes the mean along sparse segments of a tensor.

  See `tf.sparse.segment_sum` for usage examples.

  Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
  dimension, selecting a subset of dimension 0, specified by `indices`.

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor` of type `int32`.
      A 1-D tensor. Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data indices segment_ids name ]
  (py/call-attr quantization "sparse_segment_mean"  data indices segment_ids name ))

(defn sparse-segment-mean-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sparse_segment_mean
  "
  [ data indices segment_ids name ctx ]
  (py/call-attr quantization "sparse_segment_mean_eager_fallback"  data indices segment_ids name ctx ))

(defn sparse-segment-mean-grad 
  "Computes gradients for SparseSegmentMean.

  Returns tensor \"output\" with same shape as grad, except for dimension 0 whose
  value is output_dim0.

  Args:
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      gradient propagated to the SparseSegmentMean op.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      indices passed to the corresponding SparseSegmentMean op.
    segment_ids: A `Tensor` of type `int32`.
      segment_ids passed to the corresponding SparseSegmentMean op.
    output_dim0: A `Tensor` of type `int32`.
      dimension 0 of \"data\" passed to SparseSegmentMean op.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  "
  [ grad indices segment_ids output_dim0 name ]
  (py/call-attr quantization "sparse_segment_mean_grad"  grad indices segment_ids output_dim0 name ))

(defn sparse-segment-mean-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sparse_segment_mean_grad
  "
  [ grad indices segment_ids output_dim0 name ctx ]
  (py/call-attr quantization "sparse_segment_mean_grad_eager_fallback"  grad indices segment_ids output_dim0 name ctx ))

(defn sparse-segment-mean-with-num-segments 
  "Computes the mean along sparse segments of a tensor.

  Like `SparseSegmentMean`, but allows missing ids in `segment_ids`. If an id is
  misisng, the `output` tensor at that position will be zeroed.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor` of type `int32`.
      A 1-D tensor. Values should be sorted and can be repeated.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Should equal the number of distinct segment IDs.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data indices segment_ids num_segments name ]
  (py/call-attr quantization "sparse_segment_mean_with_num_segments"  data indices segment_ids num_segments name ))

(defn sparse-segment-mean-with-num-segments-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sparse_segment_mean_with_num_segments
  "
  [ data indices segment_ids num_segments name ctx ]
  (py/call-attr quantization "sparse_segment_mean_with_num_segments_eager_fallback"  data indices segment_ids num_segments name ctx ))

(defn sparse-segment-sqrt-n 
  "Computes the sum along sparse segments of a tensor divided by the sqrt of N.

  N is the size of the segment being reduced.

  See `tf.sparse.segment_sum` for usage examples.

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor` of type `int32`.
      A 1-D tensor. Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data indices segment_ids name ]
  (py/call-attr quantization "sparse_segment_sqrt_n"  data indices segment_ids name ))

(defn sparse-segment-sqrt-n-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sparse_segment_sqrt_n
  "
  [ data indices segment_ids name ctx ]
  (py/call-attr quantization "sparse_segment_sqrt_n_eager_fallback"  data indices segment_ids name ctx ))

(defn sparse-segment-sqrt-n-grad 
  "Computes gradients for SparseSegmentSqrtN.

  Returns tensor \"output\" with same shape as grad, except for dimension 0 whose
  value is output_dim0.

  Args:
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      gradient propagated to the SparseSegmentSqrtN op.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      indices passed to the corresponding SparseSegmentSqrtN op.
    segment_ids: A `Tensor` of type `int32`.
      segment_ids passed to the corresponding SparseSegmentSqrtN op.
    output_dim0: A `Tensor` of type `int32`.
      dimension 0 of \"data\" passed to SparseSegmentSqrtN op.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  "
  [ grad indices segment_ids output_dim0 name ]
  (py/call-attr quantization "sparse_segment_sqrt_n_grad"  grad indices segment_ids output_dim0 name ))

(defn sparse-segment-sqrt-n-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sparse_segment_sqrt_n_grad
  "
  [ grad indices segment_ids output_dim0 name ctx ]
  (py/call-attr quantization "sparse_segment_sqrt_n_grad_eager_fallback"  grad indices segment_ids output_dim0 name ctx ))

(defn sparse-segment-sqrt-n-with-num-segments 
  "Computes the sum along sparse segments of a tensor divided by the sqrt of N.

  N is the size of the segment being reduced.

  Like `SparseSegmentSqrtN`, but allows missing ids in `segment_ids`. If an id is
  misisng, the `output` tensor at that position will be zeroed.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor` of type `int32`.
      A 1-D tensor. Values should be sorted and can be repeated.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Should equal the number of distinct segment IDs.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data indices segment_ids num_segments name ]
  (py/call-attr quantization "sparse_segment_sqrt_n_with_num_segments"  data indices segment_ids num_segments name ))

(defn sparse-segment-sqrt-n-with-num-segments-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sparse_segment_sqrt_n_with_num_segments
  "
  [ data indices segment_ids num_segments name ctx ]
  (py/call-attr quantization "sparse_segment_sqrt_n_with_num_segments_eager_fallback"  data indices segment_ids num_segments name ctx ))

(defn sparse-segment-sum 
  "Computes the sum along sparse segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
  dimension, selecting a subset of dimension 0, specified by `indices`.

  For example:

  ```python
  c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])

  # Select two rows, one segment.
  tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
  # => [[0 0 0 0]]

  # Select two rows, two segment.
  tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
  # => [[ 1  2  3  4]
  #     [-1 -2 -3 -4]]

  # Select all rows, two segments.
  tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
  # => [[0 0 0 0]
  #     [5 6 7 8]]

  # Which is equivalent to:
  tf.segment_sum(c, tf.constant([0, 0, 1]))
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor` of type `int32`.
      A 1-D tensor. Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data indices segment_ids name ]
  (py/call-attr quantization "sparse_segment_sum"  data indices segment_ids name ))

(defn sparse-segment-sum-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sparse_segment_sum
  "
  [ data indices segment_ids name ctx ]
  (py/call-attr quantization "sparse_segment_sum_eager_fallback"  data indices segment_ids name ctx ))

(defn sparse-segment-sum-with-num-segments 
  "Computes the sum along sparse segments of a tensor.

  Like `SparseSegmentSum`, but allows missing ids in `segment_ids`. If an id is
  misisng, the `output` tensor at that position will be zeroed.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/sparse#Segmentation)
  for an explanation of segments.

  For example:

  ```python
  c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])

  tf.sparse_segment_sum_with_num_segments(
      c, tf.constant([0, 1]), tf.constant([0, 0]), num_segments=3)
  # => [[0 0 0 0]
  #     [0 0 0 0]
  #     [0 0 0 0]]

  tf.sparse_segment_sum_with_num_segments(c,
                                          tf.constant([0, 1]),
                                          tf.constant([0, 2],
                                          num_segments=4))
  # => [[ 1  2  3  4]
  #     [ 0  0  0  0]
  #     [-1 -2 -3 -4]
  #     [ 0  0  0  0]]
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor` of type `int32`.
      A 1-D tensor. Values should be sorted and can be repeated.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Should equal the number of distinct segment IDs.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data indices segment_ids num_segments name ]
  (py/call-attr quantization "sparse_segment_sum_with_num_segments"  data indices segment_ids num_segments name ))

(defn sparse-segment-sum-with-num-segments-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sparse_segment_sum_with_num_segments
  "
  [ data indices segment_ids num_segments name ctx ]
  (py/call-attr quantization "sparse_segment_sum_with_num_segments_eager_fallback"  data indices segment_ids num_segments name ctx ))

(defn sparse-softmax-cross-entropy-with-logits 
  "Computes softmax cross entropy cost and gradients to backpropagate.

  Unlike `SoftmaxCrossEntropyWithLogits`, this operation does not accept
  a matrix of label probabilities, but rather a single label per row
  of features.  This label is considered to have probability 1.0 for the
  given row.

  Inputs are the logits, not probabilities.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      batch_size x num_classes matrix
    labels: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      batch_size vector with values in [0, num_classes).
      This is the label for the given minibatch entry.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (loss, backprop).

    loss: A `Tensor`. Has the same type as `features`.
    backprop: A `Tensor`. Has the same type as `features`.
  "
  [ features labels name ]
  (py/call-attr quantization "sparse_softmax_cross_entropy_with_logits"  features labels name ))

(defn sparse-softmax-cross-entropy-with-logits-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sparse_softmax_cross_entropy_with_logits
  "
  [ features labels name ctx ]
  (py/call-attr quantization "sparse_softmax_cross_entropy_with_logits_eager_fallback"  features labels name ctx ))

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
  (py/call-attr quantization "sqrt"  x name ))

(defn sqrt-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sqrt
  "
  [ x name ctx ]
  (py/call-attr quantization "sqrt_eager_fallback"  x name ctx ))

(defn sqrt-grad 
  "Computes the gradient for the sqrt of `x` wrt its input.

  Specifically, `grad = dy * 0.5 / y`, where `y = sqrt(x)`, and `dy`
  is the corresponding input gradient.

  Args:
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    dy: A `Tensor`. Must have the same type as `y`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `y`.
  "
  [ y dy name ]
  (py/call-attr quantization "sqrt_grad"  y dy name ))

(defn sqrt-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sqrt_grad
  "
  [ y dy name ctx ]
  (py/call-attr quantization "sqrt_grad_eager_fallback"  y dy name ctx ))

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
  (py/call-attr quantization "square"  x name ))

(defn square-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function square
  "
  [ x name ctx ]
  (py/call-attr quantization "square_eager_fallback"  x name ctx ))

(defn squared-difference 
  "Returns (x - y)(x - y) element-wise.

  *NOTE*: `math.squared_difference` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "squared_difference"  x y name ))

(defn squared-difference-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function squared_difference
  "
  [ x y name ctx ]
  (py/call-attr quantization "squared_difference_eager_fallback"  x y name ctx ))

(defn sub 
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
  (py/call-attr quantization "sub"  x y name ))

(defn sub-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sub
  "
  [ x y name ctx ]
  (py/call-attr quantization "sub_eager_fallback"  x y name ctx ))

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
  (py/call-attr quantization "tan"  x name ))

(defn tan-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function tan
  "
  [ x name ctx ]
  (py/call-attr quantization "tan_eager_fallback"  x name ctx ))

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
  (py/call-attr quantization "tanh"  x name ))

(defn tanh-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function tanh
  "
  [ x name ctx ]
  (py/call-attr quantization "tanh_eager_fallback"  x name ctx ))

(defn tanh-grad 
  "Computes the gradient for the tanh of `x` wrt its input.

  Specifically, `grad = dy * (1 - y*y)`, where `y = tanh(x)`, and `dy`
  is the corresponding input gradient.

  Args:
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    dy: A `Tensor`. Must have the same type as `y`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `y`.
  "
  [ y dy name ]
  (py/call-attr quantization "tanh_grad"  y dy name ))

(defn tanh-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function tanh_grad
  "
  [ y dy name ctx ]
  (py/call-attr quantization "tanh_grad_eager_fallback"  y dy name ctx ))

(defn top-k 
  "Finds values and indices of the `k` largest elements for the last dimension.

  If the input is a vector (rank-1), finds the `k` largest entries in the vector
  and outputs their values and indices as vectors.  Thus `values[j]` is the
  `j`-th largest entry in `input`, and its index is `indices[j]`.

  For matrices (resp. higher rank input), computes the top `k` entries in each
  row (resp. vector along the last dimension).  Thus,

      values.shape = indices.shape = input.shape[:-1] + [k]

  If two elements are equal, the lower-index element appears first.

  If `k` varies dynamically, use `TopKV2` below.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      1-D or higher with last dimension at least `k`.
    k: An `int` that is `>= 0`.
      Number of top elements to look for along the last dimension (along each
      row for matrices).
    sorted: An optional `bool`. Defaults to `True`.
      If true the resulting `k` elements will be sorted by the values in
      descending order.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (values, indices).

    values: A `Tensor`. Has the same type as `input`.
    indices: A `Tensor` of type `int32`.
  "
  [input k & {:keys [sorted name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "top_k" [input k] {:sorted sorted :name name }))

(defn top-k-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function top_k
  "
  [input k & {:keys [sorted name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "top_k_eager_fallback" [input k] {:sorted sorted :name name :ctx ctx }))

(defn top-kv2 
  "Finds values and indices of the `k` largest elements for the last dimension.

  If the input is a vector (rank-1), finds the `k` largest entries in the vector
  and outputs their values and indices as vectors.  Thus `values[j]` is the
  `j`-th largest entry in `input`, and its index is `indices[j]`.

  For matrices (resp. higher rank input), computes the top `k` entries in each
  row (resp. vector along the last dimension).  Thus,

      values.shape = indices.shape = input.shape[:-1] + [k]

  If two elements are equal, the lower-index element appears first.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      1-D or higher with last dimension at least `k`.
    k: A `Tensor` of type `int32`.
      0-D.  Number of top elements to look for along the last dimension (along each
      row for matrices).
    sorted: An optional `bool`. Defaults to `True`.
      If true the resulting `k` elements will be sorted by the values in
      descending order.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (values, indices).

    values: A `Tensor`. Has the same type as `input`.
    indices: A `Tensor` of type `int32`.
  "
  [input k & {:keys [sorted name]
                       :or {name None}} ]
    (py/call-attr-kw quantization "top_kv2" [input k] {:sorted sorted :name name }))

(defn top-kv2-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function top_kv2
  "
  [input k & {:keys [sorted name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw quantization "top_kv2_eager_fallback" [input k] {:sorted sorted :name name :ctx ctx }))

(defn truncate-div 
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
  (py/call-attr quantization "truncate_div"  x y name ))

(defn truncate-div-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function truncate_div
  "
  [ x y name ctx ]
  (py/call-attr quantization "truncate_div_eager_fallback"  x y name ctx ))

(defn truncate-mod 
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
  (py/call-attr quantization "truncate_mod"  x y name ))

(defn truncate-mod-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function truncate_mod
  "
  [ x y name ctx ]
  (py/call-attr quantization "truncate_mod_eager_fallback"  x y name ctx ))

(defn unsorted-segment-max 
  "Computes the maximum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  This operator is similar to the unsorted segment sum operator found
  [(here)](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
  Instead of computing the sum over segments, it computes the maximum such that:

  \\(output_i = \max_{j...} data[j...]\\) where max is over tuples `j...` such
  that `segment_ids[j...] == i`.

  If the maximum is empty for a given segment ID `i`, it outputs the smallest
  possible value for the specific numeric type,
  `output[i] = numeric_limits<T>::lowest()`.

  If the given segment ID `i` is negative, then the corresponding value is
  dropped, and will not be included in the result.

  <div style=\"width:70%; margin:auto; margin-bottom:10px; margin-top:20px;\">
  <img style=\"width:100%\" src=\"https://www.tensorflow.org/images/UnsortedSegmentMax.png\" alt>
  </div>

  For example:

  ``` python
  c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
  tf.unsorted_segment_max(c, tf.constant([0, 1, 0]), num_segments=2)
  # ==> [[ 4,  3, 3, 4],
  #       [5,  6, 7, 8]]
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A tensor whose shape is a prefix of `data.shape`.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data segment_ids num_segments name ]
  (py/call-attr quantization "unsorted_segment_max"  data segment_ids num_segments name ))

(defn unsorted-segment-max-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function unsorted_segment_max
  "
  [ data segment_ids num_segments name ctx ]
  (py/call-attr quantization "unsorted_segment_max_eager_fallback"  data segment_ids num_segments name ctx ))

(defn unsorted-segment-min 
  "Computes the minimum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  This operator is similar to the unsorted segment sum operator found
  [(here)](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
  Instead of computing the sum over segments, it computes the minimum such that:

  \\(output_i = \min_{j...} data_[j...]\\) where min is over tuples `j...` such
  that `segment_ids[j...] == i`.

  If the minimum is empty for a given segment ID `i`, it outputs the largest
  possible value for the specific numeric type,
  `output[i] = numeric_limits<T>::max()`.

  For example:

  ``` python
  c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
  tf.unsorted_segment_min(c, tf.constant([0, 1, 0]), num_segments=2)
  # ==> [[ 1,  2, 2, 1],
  #       [5,  6, 7, 8]]
  ```

  If the given segment ID `i` is negative, then the corresponding value is
  dropped, and will not be included in the result.

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A tensor whose shape is a prefix of `data.shape`.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data segment_ids num_segments name ]
  (py/call-attr quantization "unsorted_segment_min"  data segment_ids num_segments name ))

(defn unsorted-segment-min-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function unsorted_segment_min
  "
  [ data segment_ids num_segments name ctx ]
  (py/call-attr quantization "unsorted_segment_min_eager_fallback"  data segment_ids num_segments name ctx ))

(defn unsorted-segment-prod 
  "Computes the product along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  This operator is similar to the unsorted segment sum operator found
  [(here)](../../../api_docs/python/math_ops.md#UnsortedSegmentSum).
  Instead of computing the sum over segments, it computes the product of all
  entries belonging to a segment such that:

  \\(output_i = \prod_{j...} data[j...]\\) where the product is over tuples
  `j...` such that `segment_ids[j...] == i`.

  For example:

  ``` python
  c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
  tf.unsorted_segment_prod(c, tf.constant([0, 1, 0]), num_segments=2)
  # ==> [[ 4,  6, 6, 4],
  #       [5,  6, 7, 8]]
  ```

  If there is no entry for a given segment ID `i`, it outputs 1.

  If the given segment ID `i` is negative, then the corresponding value is
  dropped, and will not be included in the result.

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A tensor whose shape is a prefix of `data.shape`.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data segment_ids num_segments name ]
  (py/call-attr quantization "unsorted_segment_prod"  data segment_ids num_segments name ))

(defn unsorted-segment-prod-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function unsorted_segment_prod
  "
  [ data segment_ids num_segments name ctx ]
  (py/call-attr quantization "unsorted_segment_prod_eager_fallback"  data segment_ids num_segments name ctx ))

(defn unsorted-segment-sum 
  "Computes the sum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output[i] = \sum_{j...} data[j...]\\) where the sum is over tuples `j...` such
  that `segment_ids[j...] == i`.  Unlike `SegmentSum`, `segment_ids`
  need not be sorted and need not cover all values in the full
  range of valid values.

  If the sum is empty for a given segment ID `i`, `output[i] = 0`.
  If the given segment ID `i` is negative, the value is dropped and will not be
  added to the sum of the segment.

  `num_segments` should equal the number of distinct segment IDs.

  <div style=\"width:70%; margin:auto; margin-bottom:10px; margin-top:20px;\">
  <img style=\"width:100%\" src=\"https://www.tensorflow.org/images/UnsortedSegmentSum.png\" alt>
  </div>

  ``` python
  c = tf.constant([[1,2,3,4], [5,6,7,8], [4,3,2,1]])
  tf.unsorted_segment_sum(c, tf.constant([0, 1, 0]), num_segments=2)
  # ==> [[ 5,  5, 5, 5],
  #       [5,  6, 7, 8]]
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A tensor whose shape is a prefix of `data.shape`.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  "
  [ data segment_ids num_segments name ]
  (py/call-attr quantization "unsorted_segment_sum"  data segment_ids num_segments name ))

(defn unsorted-segment-sum-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function unsorted_segment_sum
  "
  [ data segment_ids num_segments name ctx ]
  (py/call-attr quantization "unsorted_segment_sum_eager_fallback"  data segment_ids num_segments name ctx ))

(defn xdivy 
  "Returns 0 if x == 0, and x / y otherwise, elementwise.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "xdivy"  x y name ))

(defn xdivy-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function xdivy
  "
  [ x y name ctx ]
  (py/call-attr quantization "xdivy_eager_fallback"  x y name ctx ))

(defn xlogy 
  "Returns 0 if x == 0, and x * log(y) otherwise, elementwise.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x y name ]
  (py/call-attr quantization "xlogy"  x y name ))

(defn xlogy-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function xlogy
  "
  [ x y name ctx ]
  (py/call-attr quantization "xlogy_eager_fallback"  x y name ctx ))

(defn zeta 
  "Compute the Hurwitz zeta function \\(\zeta(x, q)\\).

  The Hurwitz zeta function is defined as:


  \\(\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}\\)

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    q: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x q name ]
  (py/call-attr quantization "zeta"  x q name ))

(defn zeta-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function zeta
  "
  [ x q name ctx ]
  (py/call-attr quantization "zeta_eager_fallback"  x q name ctx ))
