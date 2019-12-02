(ns tensorflow.contrib.quantization.python.math-ops
  "Quantized Math Operations."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce math-ops (import-module "tensorflow.contrib.quantization.python.math_ops"))

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
  (py/call-attr math-ops "Abs"  x name ))

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
  (py/call-attr math-ops "AccumulateNV2"  inputs shape name ))

(defn Acos 
  "Computes acos of x element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr math-ops "Acos"  x name ))

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
  (py/call-attr math-ops "Acosh"  x name ))

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
  (py/call-attr math-ops "Add"  x y name ))

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
  (py/call-attr math-ops "AddN"  inputs name ))

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
  (py/call-attr math-ops "AddV2"  x y name ))

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
    (py/call-attr-kw math-ops "All" [input axis] {:keep_dims keep_dims :name name }))

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
    (py/call-attr-kw math-ops "Angle" [input] {:Tout Tout :name name }))

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
    (py/call-attr-kw math-ops "Any" [input axis] {:keep_dims keep_dims :name name }))

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
    (py/call-attr-kw math-ops "ApproximateEqual" [x y] {:tolerance tolerance :name name }))

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
    (py/call-attr-kw math-ops "ArgMax" [input dimension] {:output_type output_type :name name }))

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
    (py/call-attr-kw math-ops "ArgMin" [input dimension] {:output_type output_type :name name }))

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
  (py/call-attr math-ops "Asin"  x name ))

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
  (py/call-attr math-ops "Asinh"  x name ))

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
  (py/call-attr math-ops "Atan"  x name ))

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
  (py/call-attr math-ops "Atan2"  y x name ))

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
  (py/call-attr math-ops "Atanh"  x name ))

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
    (py/call-attr-kw math-ops "BatchMatMul" [x y] {:adj_x adj_x :adj_y adj_y :name name }))

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
    (py/call-attr-kw math-ops "BatchMatMulV2" [x y] {:adj_x adj_x :adj_y adj_y :name name }))

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
  (py/call-attr math-ops "BesselI0e"  x name ))

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
  (py/call-attr math-ops "BesselI1e"  x name ))

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
  (py/call-attr math-ops "Betainc"  a b x name ))

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
  (py/call-attr math-ops "Bincount"  arr size weights name ))

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
  (py/call-attr math-ops "Bucketize"  input boundaries name ))

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
    (py/call-attr-kw math-ops "Cast" [x DstT] {:Truncate Truncate :name name }))

(defn Ceil 
  "Returns element-wise smallest integer not less than x.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr math-ops "Ceil"  x name ))

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
  (py/call-attr math-ops "ClipByValue"  t clip_value_min clip_value_max name ))

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
  (py/call-attr math-ops "CompareAndBitpack"  input threshold name ))

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
    (py/call-attr-kw math-ops "Complex" [real imag] {:Tout Tout :name name }))

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
    (py/call-attr-kw math-ops "ComplexAbs" [x] {:Tout Tout :name name }))

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
  (py/call-attr math-ops "Conj"  input name ))

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
  (py/call-attr math-ops "Cos"  x name ))

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
  (py/call-attr math-ops "Cosh"  x name ))

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
  (py/call-attr math-ops "Cross"  a b name ))

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
    (py/call-attr-kw math-ops "Cumprod" [x axis] {:exclusive exclusive :reverse reverse :name name }))

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
    (py/call-attr-kw math-ops "Cumsum" [x axis] {:exclusive exclusive :reverse reverse :name name }))

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
    (py/call-attr-kw math-ops "CumulativeLogsumexp" [x axis] {:exclusive exclusive :reverse reverse :name name }))

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
  (py/call-attr math-ops "Digamma"  x name ))

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
  (py/call-attr math-ops "Div"  x y name ))

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
  (py/call-attr math-ops "DivNoNan"  x y name ))

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
    (py/call-attr-kw math-ops "Equal" [x y] {:incompatible_shape_error incompatible_shape_error :name name }))

(defn Erf 
  "Computes the Gauss error function of `x` element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr math-ops "Erf"  x name ))

(defn Erfc 
  "Computes the complementary error function of `x` element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr math-ops "Erfc"  x name ))

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
    (py/call-attr-kw math-ops "EuclideanNorm" [input axis] {:keep_dims keep_dims :name name }))

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
  (py/call-attr math-ops "Exp"  x name ))

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
  (py/call-attr math-ops "Expm1"  x name ))

(defn Floor 
  "Returns element-wise largest integer not greater than x.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr math-ops "Floor"  x name ))

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
  (py/call-attr math-ops "FloorDiv"  x y name ))

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
  (py/call-attr math-ops "FloorMod"  x y name ))

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
  (py/call-attr math-ops "Greater"  x y name ))

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
  (py/call-attr math-ops "GreaterEqual"  x y name ))

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
    (py/call-attr-kw math-ops "HistogramFixedWidth" [values value_range nbins] {:dtype dtype :name name }))

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
  (py/call-attr math-ops "Igamma"  a x name ))

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
  (py/call-attr math-ops "IgammaGradA"  a x name ))

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
  (py/call-attr math-ops "Igammac"  a x name ))

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
    (py/call-attr-kw math-ops "Imag" [input] {:Tout Tout :name name }))

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
  (py/call-attr math-ops "Inv"  x name ))

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
  (py/call-attr math-ops "InvGrad"  y dy name ))

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
  (py/call-attr math-ops "IsFinite"  x name ))

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
  (py/call-attr math-ops "IsInf"  x name ))

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
  (py/call-attr math-ops "IsNan"  x name ))

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
  (py/call-attr math-ops "Less"  x y name ))

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
  (py/call-attr math-ops "LessEqual"  x y name ))

(defn Lgamma 
  "Computes the log of the absolute value of `Gamma(x)` element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr math-ops "Lgamma"  x name ))

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
  (py/call-attr math-ops "LinSpace"  start stop num name ))

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
  (py/call-attr math-ops "Log"  x name ))

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
  (py/call-attr math-ops "Log1p"  x name ))

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
  (py/call-attr math-ops "LogicalAnd"  x y name ))

(defn LogicalNot 
  "Returns the truth value of NOT x element-wise.

  Args:
    x: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ x name ]
  (py/call-attr math-ops "LogicalNot"  x name ))

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
  (py/call-attr math-ops "LogicalOr"  x y name ))

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
    (py/call-attr-kw math-ops "MatMul" [a b] {:transpose_a transpose_a :transpose_b transpose_b :name name }))

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
    (py/call-attr-kw math-ops "Max" [input axis] {:keep_dims keep_dims :name name }))

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
  (py/call-attr math-ops "Maximum"  x y name ))

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
    (py/call-attr-kw math-ops "Mean" [input axis] {:keep_dims keep_dims :name name }))

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
    (py/call-attr-kw math-ops "Min" [input axis] {:keep_dims keep_dims :name name }))

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
  (py/call-attr math-ops "Minimum"  x y name ))

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
  (py/call-attr math-ops "Mod"  x y name ))

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
  (py/call-attr math-ops "Mul"  x y name ))

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
  (py/call-attr math-ops "MulNoNan"  x y name ))

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
  (py/call-attr math-ops "Neg"  x name ))

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
  (py/call-attr math-ops "NextAfter"  x1 x2 name ))

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
    (py/call-attr-kw math-ops "NotEqual" [x y] {:incompatible_shape_error incompatible_shape_error :name name }))

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
  (py/call-attr math-ops "Polygamma"  a x name ))

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
  (py/call-attr math-ops "Pow"  x y name ))

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
    (py/call-attr-kw math-ops "Prod" [input axis] {:keep_dims keep_dims :name name }))

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
  (py/call-attr math-ops "QuantizeDownAndShrinkRange"  input input_min input_max out_type name ))

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
    (py/call-attr-kw math-ops "QuantizedAdd" [x y min_x max_x min_y max_y] {:Toutput Toutput :name name }))

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
    (py/call-attr-kw math-ops "QuantizedMatMul" [a b min_a max_a min_b max_b] {:Toutput Toutput :transpose_a transpose_a :transpose_b transpose_b :Tactivation Tactivation :name name }))

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
    (py/call-attr-kw math-ops "QuantizedMul" [x y min_x max_x min_y max_y] {:Toutput Toutput :name name }))

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
  (py/call-attr math-ops "Range"  start limit delta name ))

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
    (py/call-attr-kw math-ops "Real" [input] {:Tout Tout :name name }))

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
  (py/call-attr math-ops "RealDiv"  x y name ))

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
  (py/call-attr math-ops "Reciprocal"  x name ))

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
  (py/call-attr math-ops "ReciprocalGrad"  y dy name ))

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
  (py/call-attr math-ops "RequantizationRange"  input input_min input_max name ))

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
  (py/call-attr math-ops "RequantizationRangePerChannel"  input input_min input_max clip_value_max name ))

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
  (py/call-attr math-ops "Requantize"  input input_min input_max requested_output_min requested_output_max out_type name ))

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
    (py/call-attr-kw math-ops "RequantizePerChannel" [input input_min input_max requested_output_min requested_output_max] {:out_type out_type :name name }))

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
  (py/call-attr math-ops "Rint"  x name ))

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
  (py/call-attr math-ops "Round"  x name ))

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
  (py/call-attr math-ops "Rsqrt"  x name ))

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
  (py/call-attr math-ops "RsqrtGrad"  y dy name ))

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
  (py/call-attr math-ops "SegmentMax"  data segment_ids name ))

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
  (py/call-attr math-ops "SegmentMean"  data segment_ids name ))

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
  (py/call-attr math-ops "SegmentMin"  data segment_ids name ))

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
  (py/call-attr math-ops "SegmentProd"  data segment_ids name ))

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
  (py/call-attr math-ops "SegmentSum"  data segment_ids name ))

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
  (py/call-attr math-ops "Select"  condition x y name ))

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
  (py/call-attr math-ops "SelectV2"  condition t e name ))

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
  (py/call-attr math-ops "Sigmoid"  x name ))

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
  (py/call-attr math-ops "SigmoidGrad"  y dy name ))

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
  (py/call-attr math-ops "Sign"  x name ))

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
  (py/call-attr math-ops "Sin"  x name ))

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
  (py/call-attr math-ops "Sinh"  x name ))

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
    (py/call-attr-kw math-ops "SparseMatMul" [a b] {:transpose_a transpose_a :transpose_b transpose_b :a_is_sparse a_is_sparse :b_is_sparse b_is_sparse :name name }))

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
  (py/call-attr math-ops "SparseSegmentMean"  data indices segment_ids name ))

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
  (py/call-attr math-ops "SparseSegmentMeanGrad"  grad indices segment_ids output_dim0 name ))

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
  (py/call-attr math-ops "SparseSegmentMeanWithNumSegments"  data indices segment_ids num_segments name ))

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
  (py/call-attr math-ops "SparseSegmentSqrtN"  data indices segment_ids name ))

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
  (py/call-attr math-ops "SparseSegmentSqrtNGrad"  grad indices segment_ids output_dim0 name ))

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
  (py/call-attr math-ops "SparseSegmentSqrtNWithNumSegments"  data indices segment_ids num_segments name ))

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
  (py/call-attr math-ops "SparseSegmentSum"  data indices segment_ids name ))

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
  (py/call-attr math-ops "SparseSegmentSumWithNumSegments"  data indices segment_ids num_segments name ))

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
  (py/call-attr math-ops "Sqrt"  x name ))

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
  (py/call-attr math-ops "SqrtGrad"  y dy name ))

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
  (py/call-attr math-ops "Square"  x name ))

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
  (py/call-attr math-ops "SquaredDifference"  x y name ))

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
  (py/call-attr math-ops "Sub"  x y name ))

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
    (py/call-attr-kw math-ops "Sum" [input axis] {:keep_dims keep_dims :name name }))

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
  (py/call-attr math-ops "Tan"  x name ))

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
  (py/call-attr math-ops "Tanh"  x name ))

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
  (py/call-attr math-ops "TanhGrad"  y dy name ))

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
  (py/call-attr math-ops "TruncateDiv"  x y name ))

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
  (py/call-attr math-ops "TruncateMod"  x y name ))

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
  (py/call-attr math-ops "UnsortedSegmentMax"  data segment_ids num_segments name ))

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
  (py/call-attr math-ops "UnsortedSegmentMin"  data segment_ids num_segments name ))

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
  (py/call-attr math-ops "UnsortedSegmentProd"  data segment_ids num_segments name ))

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
  (py/call-attr math-ops "UnsortedSegmentSum"  data segment_ids num_segments name ))

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
  (py/call-attr math-ops "Xdivy"  x y name ))

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
  (py/call-attr math-ops "Xlogy"  x y name ))

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
  (py/call-attr math-ops "Zeta"  x q name ))

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
  (py/call-attr math-ops "accumulate_nv2"  inputs shape name ))

(defn accumulate-nv2-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function accumulate_nv2
  "
  [ inputs shape name ctx ]
  (py/call-attr math-ops "accumulate_nv2_eager_fallback"  inputs shape name ctx ))

(defn acos 
  "Computes acos of x element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr math-ops "acos"  x name ))

(defn acos-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function acos
  "
  [ x name ctx ]
  (py/call-attr math-ops "acos_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "acosh"  x name ))

(defn acosh-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function acosh
  "
  [ x name ctx ]
  (py/call-attr math-ops "acosh_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "add"  x y name ))

(defn add-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function add
  "
  [ x y name ctx ]
  (py/call-attr math-ops "add_eager_fallback"  x y name ctx ))

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
  (py/call-attr math-ops "add_n"  inputs name ))

(defn add-n-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function add_n
  "
  [ inputs name ctx ]
  (py/call-attr math-ops "add_n_eager_fallback"  inputs name ctx ))

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
  (py/call-attr math-ops "add_v2"  x y name ))

(defn add-v2-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function add_v2
  "
  [ x y name ctx ]
  (py/call-attr math-ops "add_v2_eager_fallback"  x y name ctx ))

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
    (py/call-attr-kw math-ops "angle" [input] {:Tout Tout :name name }))

(defn angle-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function angle
  "
  [input & {:keys [Tout name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "angle_eager_fallback" [input] {:Tout Tout :name name :ctx ctx }))

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
    (py/call-attr-kw math-ops "approximate_equal" [x y] {:tolerance tolerance :name name }))

(defn approximate-equal-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function approximate_equal
  "
  [x y & {:keys [tolerance name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "approximate_equal_eager_fallback" [x y] {:tolerance tolerance :name name :ctx ctx }))

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
    (py/call-attr-kw math-ops "arg_max" [input dimension] {:output_type output_type :name name }))

(defn arg-max-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function arg_max
  "
  [input dimension & {:keys [output_type name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "arg_max_eager_fallback" [input dimension] {:output_type output_type :name name :ctx ctx }))

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
    (py/call-attr-kw math-ops "arg_min" [input dimension] {:output_type output_type :name name }))

(defn arg-min-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function arg_min
  "
  [input dimension & {:keys [output_type name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "arg_min_eager_fallback" [input dimension] {:output_type output_type :name name :ctx ctx }))

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
  (py/call-attr math-ops "asin"  x name ))

(defn asin-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function asin
  "
  [ x name ctx ]
  (py/call-attr math-ops "asin_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "asinh"  x name ))

(defn asinh-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function asinh
  "
  [ x name ctx ]
  (py/call-attr math-ops "asinh_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "atan"  x name ))

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
  (py/call-attr math-ops "atan2"  y x name ))

(defn atan2-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function atan2
  "
  [ y x name ctx ]
  (py/call-attr math-ops "atan2_eager_fallback"  y x name ctx ))

(defn atan-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function atan
  "
  [ x name ctx ]
  (py/call-attr math-ops "atan_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "atanh"  x name ))

(defn atanh-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function atanh
  "
  [ x name ctx ]
  (py/call-attr math-ops "atanh_eager_fallback"  x name ctx ))

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
    (py/call-attr-kw math-ops "batch_mat_mul" [x y] {:adj_x adj_x :adj_y adj_y :name name }))

(defn batch-mat-mul-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function batch_mat_mul
  "
  [x y & {:keys [adj_x adj_y name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "batch_mat_mul_eager_fallback" [x y] {:adj_x adj_x :adj_y adj_y :name name :ctx ctx }))

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
    (py/call-attr-kw math-ops "batch_mat_mul_v2" [x y] {:adj_x adj_x :adj_y adj_y :name name }))

(defn batch-mat-mul-v2-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function batch_mat_mul_v2
  "
  [x y & {:keys [adj_x adj_y name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "batch_mat_mul_v2_eager_fallback" [x y] {:adj_x adj_x :adj_y adj_y :name name :ctx ctx }))

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
  (py/call-attr math-ops "bessel_i0e"  x name ))

(defn bessel-i0e-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function bessel_i0e
  "
  [ x name ctx ]
  (py/call-attr math-ops "bessel_i0e_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "bessel_i1e"  x name ))

(defn bessel-i1e-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function bessel_i1e
  "
  [ x name ctx ]
  (py/call-attr math-ops "bessel_i1e_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "betainc"  a b x name ))

(defn betainc-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function betainc
  "
  [ a b x name ctx ]
  (py/call-attr math-ops "betainc_eager_fallback"  a b x name ctx ))

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
  (py/call-attr math-ops "bincount"  arr size weights name ))

(defn bincount-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function bincount
  "
  [ arr size weights name ctx ]
  (py/call-attr math-ops "bincount_eager_fallback"  arr size weights name ctx ))

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
  (py/call-attr math-ops "bucketize"  input boundaries name ))

(defn bucketize-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function bucketize
  "
  [ input boundaries name ctx ]
  (py/call-attr math-ops "bucketize_eager_fallback"  input boundaries name ctx ))

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
    (py/call-attr-kw math-ops "cast" [x DstT] {:Truncate Truncate :name name }))

(defn cast-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function cast
  "
  [x DstT & {:keys [Truncate name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "cast_eager_fallback" [x DstT] {:Truncate Truncate :name name :ctx ctx }))

(defn ceil 
  "Returns element-wise smallest integer not less than x.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr math-ops "ceil"  x name ))

(defn ceil-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function ceil
  "
  [ x name ctx ]
  (py/call-attr math-ops "ceil_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "compare_and_bitpack"  input threshold name ))

(defn compare-and-bitpack-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function compare_and_bitpack
  "
  [ input threshold name ctx ]
  (py/call-attr math-ops "compare_and_bitpack_eager_fallback"  input threshold name ctx ))

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
    (py/call-attr-kw math-ops "complex_abs" [x] {:Tout Tout :name name }))

(defn complex-abs-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function complex_abs
  "
  [x & {:keys [Tout name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "complex_abs_eager_fallback" [x] {:Tout Tout :name name :ctx ctx }))

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
  (py/call-attr math-ops "conj"  input name ))

(defn conj-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function conj
  "
  [ input name ctx ]
  (py/call-attr math-ops "conj_eager_fallback"  input name ctx ))

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
  (py/call-attr math-ops "cos"  x name ))

(defn cos-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function cos
  "
  [ x name ctx ]
  (py/call-attr math-ops "cos_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "cosh"  x name ))

(defn cosh-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function cosh
  "
  [ x name ctx ]
  (py/call-attr math-ops "cosh_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "cross"  a b name ))

(defn cross-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function cross
  "
  [ a b name ctx ]
  (py/call-attr math-ops "cross_eager_fallback"  a b name ctx ))

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
    (py/call-attr-kw math-ops "cumprod" [x axis] {:exclusive exclusive :reverse reverse :name name }))

(defn cumprod-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function cumprod
  "
  [x axis & {:keys [exclusive reverse name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "cumprod_eager_fallback" [x axis] {:exclusive exclusive :reverse reverse :name name :ctx ctx }))

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
    (py/call-attr-kw math-ops "cumsum" [x axis] {:exclusive exclusive :reverse reverse :name name }))

(defn cumsum-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function cumsum
  "
  [x axis & {:keys [exclusive reverse name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "cumsum_eager_fallback" [x axis] {:exclusive exclusive :reverse reverse :name name :ctx ctx }))

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
    (py/call-attr-kw math-ops "cumulative_logsumexp" [x axis] {:exclusive exclusive :reverse reverse :name name }))

(defn cumulative-logsumexp-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function cumulative_logsumexp
  "
  [x axis & {:keys [exclusive reverse name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "cumulative_logsumexp_eager_fallback" [x axis] {:exclusive exclusive :reverse reverse :name name :ctx ctx }))

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
  (py/call-attr math-ops "deprecated_endpoints"  ))

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
  (py/call-attr math-ops "digamma"  x name ))

(defn digamma-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function digamma
  "
  [ x name ctx ]
  (py/call-attr math-ops "digamma_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "div"  x y name ))

(defn div-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function div
  "
  [ x y name ctx ]
  (py/call-attr math-ops "div_eager_fallback"  x y name ctx ))

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
  (py/call-attr math-ops "div_no_nan"  x y name ))

(defn div-no-nan-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function div_no_nan
  "
  [ x y name ctx ]
  (py/call-attr math-ops "div_no_nan_eager_fallback"  x y name ctx ))

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
    (py/call-attr-kw math-ops "equal" [x y] {:incompatible_shape_error incompatible_shape_error :name name }))

(defn equal-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function equal
  "
  [x y & {:keys [incompatible_shape_error name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "equal_eager_fallback" [x y] {:incompatible_shape_error incompatible_shape_error :name name :ctx ctx }))

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
  (py/call-attr math-ops "erf"  x name ))

(defn erf-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function erf
  "
  [ x name ctx ]
  (py/call-attr math-ops "erf_eager_fallback"  x name ctx ))

(defn erfc 
  "Computes the complementary error function of `x` element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr math-ops "erfc"  x name ))

(defn erfc-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function erfc
  "
  [ x name ctx ]
  (py/call-attr math-ops "erfc_eager_fallback"  x name ctx ))

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
    (py/call-attr-kw math-ops "euclidean_norm" [input axis] {:keep_dims keep_dims :name name }))

(defn euclidean-norm-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function euclidean_norm
  "
  [input axis & {:keys [keep_dims name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "euclidean_norm_eager_fallback" [input axis] {:keep_dims keep_dims :name name :ctx ctx }))

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
  (py/call-attr math-ops "exp"  x name ))

(defn exp-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function exp
  "
  [ x name ctx ]
  (py/call-attr math-ops "exp_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "expm1"  x name ))

(defn expm1-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function expm1
  "
  [ x name ctx ]
  (py/call-attr math-ops "expm1_eager_fallback"  x name ctx ))

(defn floor 
  "Returns element-wise largest integer not greater than x.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr math-ops "floor"  x name ))

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
  (py/call-attr math-ops "floor_div"  x y name ))

(defn floor-div-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function floor_div
  "
  [ x y name ctx ]
  (py/call-attr math-ops "floor_div_eager_fallback"  x y name ctx ))

(defn floor-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function floor
  "
  [ x name ctx ]
  (py/call-attr math-ops "floor_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "floor_mod"  x y name ))

(defn floor-mod-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function floor_mod
  "
  [ x y name ctx ]
  (py/call-attr math-ops "floor_mod_eager_fallback"  x y name ctx ))

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
  (py/call-attr math-ops "greater"  x y name ))

(defn greater-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function greater
  "
  [ x y name ctx ]
  (py/call-attr math-ops "greater_eager_fallback"  x y name ctx ))

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
  (py/call-attr math-ops "greater_equal"  x y name ))

(defn greater-equal-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function greater_equal
  "
  [ x y name ctx ]
  (py/call-attr math-ops "greater_equal_eager_fallback"  x y name ctx ))

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
  (py/call-attr math-ops "igamma"  a x name ))

(defn igamma-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function igamma
  "
  [ a x name ctx ]
  (py/call-attr math-ops "igamma_eager_fallback"  a x name ctx ))

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
  (py/call-attr math-ops "igamma_grad_a"  a x name ))

(defn igamma-grad-a-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function igamma_grad_a
  "
  [ a x name ctx ]
  (py/call-attr math-ops "igamma_grad_a_eager_fallback"  a x name ctx ))

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
  (py/call-attr math-ops "igammac"  a x name ))

(defn igammac-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function igammac
  "
  [ a x name ctx ]
  (py/call-attr math-ops "igammac_eager_fallback"  a x name ctx ))

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
    (py/call-attr-kw math-ops "imag" [input] {:Tout Tout :name name }))

(defn imag-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function imag
  "
  [input & {:keys [Tout name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "imag_eager_fallback" [input] {:Tout Tout :name name :ctx ctx }))

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
  (py/call-attr math-ops "inv"  x name ))

(defn inv-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function inv
  "
  [ x name ctx ]
  (py/call-attr math-ops "inv_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "inv_grad"  y dy name ))

(defn inv-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function inv_grad
  "
  [ y dy name ctx ]
  (py/call-attr math-ops "inv_grad_eager_fallback"  y dy name ctx ))

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
  (py/call-attr math-ops "is_finite"  x name ))

(defn is-finite-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function is_finite
  "
  [ x name ctx ]
  (py/call-attr math-ops "is_finite_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "is_inf"  x name ))

(defn is-inf-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function is_inf
  "
  [ x name ctx ]
  (py/call-attr math-ops "is_inf_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "is_nan"  x name ))

(defn is-nan-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function is_nan
  "
  [ x name ctx ]
  (py/call-attr math-ops "is_nan_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "less"  x y name ))

(defn less-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function less
  "
  [ x y name ctx ]
  (py/call-attr math-ops "less_eager_fallback"  x y name ctx ))

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
  (py/call-attr math-ops "less_equal"  x y name ))

(defn less-equal-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function less_equal
  "
  [ x y name ctx ]
  (py/call-attr math-ops "less_equal_eager_fallback"  x y name ctx ))

(defn lgamma 
  "Computes the log of the absolute value of `Gamma(x)` element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  "
  [ x name ]
  (py/call-attr math-ops "lgamma"  x name ))

(defn lgamma-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function lgamma
  "
  [ x name ctx ]
  (py/call-attr math-ops "lgamma_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "lin_space"  start stop num name ))

(defn lin-space-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function lin_space
  "
  [ start stop num name ctx ]
  (py/call-attr math-ops "lin_space_eager_fallback"  start stop num name ctx ))

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
  (py/call-attr math-ops "log"  x name ))

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
  (py/call-attr math-ops "log1p"  x name ))

(defn log1p-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function log1p
  "
  [ x name ctx ]
  (py/call-attr math-ops "log1p_eager_fallback"  x name ctx ))

(defn log-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function log
  "
  [ x name ctx ]
  (py/call-attr math-ops "log_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "logical_and"  x y name ))

(defn logical-and-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function logical_and
  "
  [ x y name ctx ]
  (py/call-attr math-ops "logical_and_eager_fallback"  x y name ctx ))

(defn logical-not 
  "Returns the truth value of NOT x element-wise.

  Args:
    x: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  "
  [ x name ]
  (py/call-attr math-ops "logical_not"  x name ))

(defn logical-not-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function logical_not
  "
  [ x name ctx ]
  (py/call-attr math-ops "logical_not_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "logical_or"  x y name ))

(defn logical-or-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function logical_or
  "
  [ x y name ctx ]
  (py/call-attr math-ops "logical_or_eager_fallback"  x y name ctx ))

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
    (py/call-attr-kw math-ops "mat_mul" [a b] {:transpose_a transpose_a :transpose_b transpose_b :name name }))

(defn mat-mul-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function mat_mul
  "
  [a b & {:keys [transpose_a transpose_b name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "mat_mul_eager_fallback" [a b] {:transpose_a transpose_a :transpose_b transpose_b :name name :ctx ctx }))

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
  (py/call-attr math-ops "maximum"  x y name ))

(defn maximum-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function maximum
  "
  [ x y name ctx ]
  (py/call-attr math-ops "maximum_eager_fallback"  x y name ctx ))

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
    (py/call-attr-kw math-ops "mean" [input axis] {:keep_dims keep_dims :name name }))

(defn mean-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function mean
  "
  [input axis & {:keys [keep_dims name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "mean_eager_fallback" [input axis] {:keep_dims keep_dims :name name :ctx ctx }))

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
  (py/call-attr math-ops "minimum"  x y name ))

(defn minimum-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function minimum
  "
  [ x y name ctx ]
  (py/call-attr math-ops "minimum_eager_fallback"  x y name ctx ))

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
  (py/call-attr math-ops "mod"  x y name ))

(defn mod-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function mod
  "
  [ x y name ctx ]
  (py/call-attr math-ops "mod_eager_fallback"  x y name ctx ))

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
  (py/call-attr math-ops "mul"  x y name ))

(defn mul-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function mul
  "
  [ x y name ctx ]
  (py/call-attr math-ops "mul_eager_fallback"  x y name ctx ))

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
  (py/call-attr math-ops "mul_no_nan"  x y name ))

(defn mul-no-nan-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function mul_no_nan
  "
  [ x y name ctx ]
  (py/call-attr math-ops "mul_no_nan_eager_fallback"  x y name ctx ))

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
  (py/call-attr math-ops "neg"  x name ))

(defn neg-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function neg
  "
  [ x name ctx ]
  (py/call-attr math-ops "neg_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "next_after"  x1 x2 name ))

(defn next-after-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function next_after
  "
  [ x1 x2 name ctx ]
  (py/call-attr math-ops "next_after_eager_fallback"  x1 x2 name ctx ))

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
    (py/call-attr-kw math-ops "not_equal" [x y] {:incompatible_shape_error incompatible_shape_error :name name }))

(defn not-equal-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function not_equal
  "
  [x y & {:keys [incompatible_shape_error name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "not_equal_eager_fallback" [x y] {:incompatible_shape_error incompatible_shape_error :name name :ctx ctx }))

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
  (py/call-attr math-ops "polygamma"  a x name ))

(defn polygamma-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function polygamma
  "
  [ a x name ctx ]
  (py/call-attr math-ops "polygamma_eager_fallback"  a x name ctx ))

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
    (py/call-attr-kw math-ops "prod" [input axis] {:keep_dims keep_dims :name name }))

(defn prod-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function prod
  "
  [input axis & {:keys [keep_dims name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "prod_eager_fallback" [input axis] {:keep_dims keep_dims :name name :ctx ctx }))

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
  (py/call-attr math-ops "quantize_down_and_shrink_range"  input input_min input_max out_type name ))

(defn quantize-down-and-shrink-range-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantize_down_and_shrink_range
  "
  [ input input_min input_max out_type name ctx ]
  (py/call-attr math-ops "quantize_down_and_shrink_range_eager_fallback"  input input_min input_max out_type name ctx ))

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
    (py/call-attr-kw math-ops "quantized_add" [x y min_x max_x min_y max_y] {:Toutput Toutput :name name }))

(defn quantized-add-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_add
  "
  [x y min_x max_x min_y max_y & {:keys [Toutput name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "quantized_add_eager_fallback" [x y min_x max_x min_y max_y] {:Toutput Toutput :name name :ctx ctx }))

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
    (py/call-attr-kw math-ops "quantized_mat_mul" [a b min_a max_a min_b max_b] {:Toutput Toutput :transpose_a transpose_a :transpose_b transpose_b :Tactivation Tactivation :name name }))

(defn quantized-mat-mul-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_mat_mul
  "
  [a b min_a max_a min_b max_b & {:keys [Toutput transpose_a transpose_b Tactivation name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "quantized_mat_mul_eager_fallback" [a b min_a max_a min_b max_b] {:Toutput Toutput :transpose_a transpose_a :transpose_b transpose_b :Tactivation Tactivation :name name :ctx ctx }))

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
    (py/call-attr-kw math-ops "quantized_mul" [x y min_x max_x min_y max_y] {:Toutput Toutput :name name }))

(defn quantized-mul-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function quantized_mul
  "
  [x y min_x max_x min_y max_y & {:keys [Toutput name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "quantized_mul_eager_fallback" [x y min_x max_x min_y max_y] {:Toutput Toutput :name name :ctx ctx }))

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
    (py/call-attr-kw math-ops "real" [input] {:Tout Tout :name name }))

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
  (py/call-attr math-ops "real_div"  x y name ))

(defn real-div-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function real_div
  "
  [ x y name ctx ]
  (py/call-attr math-ops "real_div_eager_fallback"  x y name ctx ))

(defn real-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function real
  "
  [input & {:keys [Tout name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "real_eager_fallback" [input] {:Tout Tout :name name :ctx ctx }))

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
  (py/call-attr math-ops "reciprocal"  x name ))

(defn reciprocal-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function reciprocal
  "
  [ x name ctx ]
  (py/call-attr math-ops "reciprocal_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "reciprocal_grad"  y dy name ))

(defn reciprocal-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function reciprocal_grad
  "
  [ y dy name ctx ]
  (py/call-attr math-ops "reciprocal_grad_eager_fallback"  y dy name ctx ))

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
  (py/call-attr math-ops "requantization_range"  input input_min input_max name ))

(defn requantization-range-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function requantization_range
  "
  [ input input_min input_max name ctx ]
  (py/call-attr math-ops "requantization_range_eager_fallback"  input input_min input_max name ctx ))

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
  (py/call-attr math-ops "requantization_range_per_channel"  input input_min input_max clip_value_max name ))

(defn requantization-range-per-channel-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function requantization_range_per_channel
  "
  [ input input_min input_max clip_value_max name ctx ]
  (py/call-attr math-ops "requantization_range_per_channel_eager_fallback"  input input_min input_max clip_value_max name ctx ))

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
  (py/call-attr math-ops "requantize"  input input_min input_max requested_output_min requested_output_max out_type name ))

(defn requantize-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function requantize
  "
  [ input input_min input_max requested_output_min requested_output_max out_type name ctx ]
  (py/call-attr math-ops "requantize_eager_fallback"  input input_min input_max requested_output_min requested_output_max out_type name ctx ))

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
    (py/call-attr-kw math-ops "requantize_per_channel" [input input_min input_max requested_output_min requested_output_max] {:out_type out_type :name name }))

(defn requantize-per-channel-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function requantize_per_channel
  "
  [input input_min input_max requested_output_min requested_output_max & {:keys [out_type name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "requantize_per_channel_eager_fallback" [input input_min input_max requested_output_min requested_output_max] {:out_type out_type :name name :ctx ctx }))

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
  (py/call-attr math-ops "rint"  x name ))

(defn rint-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function rint
  "
  [ x name ctx ]
  (py/call-attr math-ops "rint_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "round"  x name ))

(defn round-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function round
  "
  [ x name ctx ]
  (py/call-attr math-ops "round_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "rsqrt"  x name ))

(defn rsqrt-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function rsqrt
  "
  [ x name ctx ]
  (py/call-attr math-ops "rsqrt_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "rsqrt_grad"  y dy name ))

(defn rsqrt-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function rsqrt_grad
  "
  [ y dy name ctx ]
  (py/call-attr math-ops "rsqrt_grad_eager_fallback"  y dy name ctx ))

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
  (py/call-attr math-ops "segment_max"  data segment_ids name ))

(defn segment-max-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function segment_max
  "
  [ data segment_ids name ctx ]
  (py/call-attr math-ops "segment_max_eager_fallback"  data segment_ids name ctx ))

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
  (py/call-attr math-ops "segment_mean"  data segment_ids name ))

(defn segment-mean-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function segment_mean
  "
  [ data segment_ids name ctx ]
  (py/call-attr math-ops "segment_mean_eager_fallback"  data segment_ids name ctx ))

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
  (py/call-attr math-ops "segment_min"  data segment_ids name ))

(defn segment-min-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function segment_min
  "
  [ data segment_ids name ctx ]
  (py/call-attr math-ops "segment_min_eager_fallback"  data segment_ids name ctx ))

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
  (py/call-attr math-ops "segment_prod"  data segment_ids name ))

(defn segment-prod-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function segment_prod
  "
  [ data segment_ids name ctx ]
  (py/call-attr math-ops "segment_prod_eager_fallback"  data segment_ids name ctx ))

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
  (py/call-attr math-ops "segment_sum"  data segment_ids name ))

(defn segment-sum-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function segment_sum
  "
  [ data segment_ids name ctx ]
  (py/call-attr math-ops "segment_sum_eager_fallback"  data segment_ids name ctx ))

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
  (py/call-attr math-ops "select"  condition x y name ))

(defn select-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function select
  "
  [ condition x y name ctx ]
  (py/call-attr math-ops "select_eager_fallback"  condition x y name ctx ))

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
  (py/call-attr math-ops "select_v2"  condition t e name ))

(defn select-v2-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function select_v2
  "
  [ condition t e name ctx ]
  (py/call-attr math-ops "select_v2_eager_fallback"  condition t e name ctx ))

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
  (py/call-attr math-ops "sigmoid"  x name ))

(defn sigmoid-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sigmoid
  "
  [ x name ctx ]
  (py/call-attr math-ops "sigmoid_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "sigmoid_grad"  y dy name ))

(defn sigmoid-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sigmoid_grad
  "
  [ y dy name ctx ]
  (py/call-attr math-ops "sigmoid_grad_eager_fallback"  y dy name ctx ))

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
  (py/call-attr math-ops "sign"  x name ))

(defn sign-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sign
  "
  [ x name ctx ]
  (py/call-attr math-ops "sign_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "sin"  x name ))

(defn sin-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sin
  "
  [ x name ctx ]
  (py/call-attr math-ops "sin_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "sinh"  x name ))

(defn sinh-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sinh
  "
  [ x name ctx ]
  (py/call-attr math-ops "sinh_eager_fallback"  x name ctx ))

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
    (py/call-attr-kw math-ops "sparse_mat_mul" [a b] {:transpose_a transpose_a :transpose_b transpose_b :a_is_sparse a_is_sparse :b_is_sparse b_is_sparse :name name }))

(defn sparse-mat-mul-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sparse_mat_mul
  "
  [a b & {:keys [transpose_a transpose_b a_is_sparse b_is_sparse name ctx]
                       :or {name None ctx None}} ]
    (py/call-attr-kw math-ops "sparse_mat_mul_eager_fallback" [a b] {:transpose_a transpose_a :transpose_b transpose_b :a_is_sparse a_is_sparse :b_is_sparse b_is_sparse :name name :ctx ctx }))

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
  (py/call-attr math-ops "sparse_segment_mean"  data indices segment_ids name ))

(defn sparse-segment-mean-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sparse_segment_mean
  "
  [ data indices segment_ids name ctx ]
  (py/call-attr math-ops "sparse_segment_mean_eager_fallback"  data indices segment_ids name ctx ))

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
  (py/call-attr math-ops "sparse_segment_mean_grad"  grad indices segment_ids output_dim0 name ))

(defn sparse-segment-mean-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sparse_segment_mean_grad
  "
  [ grad indices segment_ids output_dim0 name ctx ]
  (py/call-attr math-ops "sparse_segment_mean_grad_eager_fallback"  grad indices segment_ids output_dim0 name ctx ))

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
  (py/call-attr math-ops "sparse_segment_mean_with_num_segments"  data indices segment_ids num_segments name ))

(defn sparse-segment-mean-with-num-segments-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sparse_segment_mean_with_num_segments
  "
  [ data indices segment_ids num_segments name ctx ]
  (py/call-attr math-ops "sparse_segment_mean_with_num_segments_eager_fallback"  data indices segment_ids num_segments name ctx ))

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
  (py/call-attr math-ops "sparse_segment_sqrt_n"  data indices segment_ids name ))

(defn sparse-segment-sqrt-n-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sparse_segment_sqrt_n
  "
  [ data indices segment_ids name ctx ]
  (py/call-attr math-ops "sparse_segment_sqrt_n_eager_fallback"  data indices segment_ids name ctx ))

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
  (py/call-attr math-ops "sparse_segment_sqrt_n_grad"  grad indices segment_ids output_dim0 name ))

(defn sparse-segment-sqrt-n-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sparse_segment_sqrt_n_grad
  "
  [ grad indices segment_ids output_dim0 name ctx ]
  (py/call-attr math-ops "sparse_segment_sqrt_n_grad_eager_fallback"  grad indices segment_ids output_dim0 name ctx ))

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
  (py/call-attr math-ops "sparse_segment_sqrt_n_with_num_segments"  data indices segment_ids num_segments name ))

(defn sparse-segment-sqrt-n-with-num-segments-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sparse_segment_sqrt_n_with_num_segments
  "
  [ data indices segment_ids num_segments name ctx ]
  (py/call-attr math-ops "sparse_segment_sqrt_n_with_num_segments_eager_fallback"  data indices segment_ids num_segments name ctx ))

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
  (py/call-attr math-ops "sparse_segment_sum"  data indices segment_ids name ))

(defn sparse-segment-sum-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sparse_segment_sum
  "
  [ data indices segment_ids name ctx ]
  (py/call-attr math-ops "sparse_segment_sum_eager_fallback"  data indices segment_ids name ctx ))

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
  (py/call-attr math-ops "sparse_segment_sum_with_num_segments"  data indices segment_ids num_segments name ))

(defn sparse-segment-sum-with-num-segments-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sparse_segment_sum_with_num_segments
  "
  [ data indices segment_ids num_segments name ctx ]
  (py/call-attr math-ops "sparse_segment_sum_with_num_segments_eager_fallback"  data indices segment_ids num_segments name ctx ))

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
  (py/call-attr math-ops "sqrt"  x name ))

(defn sqrt-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sqrt
  "
  [ x name ctx ]
  (py/call-attr math-ops "sqrt_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "sqrt_grad"  y dy name ))

(defn sqrt-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sqrt_grad
  "
  [ y dy name ctx ]
  (py/call-attr math-ops "sqrt_grad_eager_fallback"  y dy name ctx ))

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
  (py/call-attr math-ops "square"  x name ))

(defn square-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function square
  "
  [ x name ctx ]
  (py/call-attr math-ops "square_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "squared_difference"  x y name ))

(defn squared-difference-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function squared_difference
  "
  [ x y name ctx ]
  (py/call-attr math-ops "squared_difference_eager_fallback"  x y name ctx ))

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
  (py/call-attr math-ops "sub"  x y name ))

(defn sub-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function sub
  "
  [ x y name ctx ]
  (py/call-attr math-ops "sub_eager_fallback"  x y name ctx ))

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
  (py/call-attr math-ops "tan"  x name ))

(defn tan-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function tan
  "
  [ x name ctx ]
  (py/call-attr math-ops "tan_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "tanh"  x name ))

(defn tanh-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function tanh
  "
  [ x name ctx ]
  (py/call-attr math-ops "tanh_eager_fallback"  x name ctx ))

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
  (py/call-attr math-ops "tanh_grad"  y dy name ))

(defn tanh-grad-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function tanh_grad
  "
  [ y dy name ctx ]
  (py/call-attr math-ops "tanh_grad_eager_fallback"  y dy name ctx ))

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
  (py/call-attr math-ops "truncate_div"  x y name ))

(defn truncate-div-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function truncate_div
  "
  [ x y name ctx ]
  (py/call-attr math-ops "truncate_div_eager_fallback"  x y name ctx ))

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
  (py/call-attr math-ops "truncate_mod"  x y name ))

(defn truncate-mod-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function truncate_mod
  "
  [ x y name ctx ]
  (py/call-attr math-ops "truncate_mod_eager_fallback"  x y name ctx ))

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
  (py/call-attr math-ops "unsorted_segment_max"  data segment_ids num_segments name ))

(defn unsorted-segment-max-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function unsorted_segment_max
  "
  [ data segment_ids num_segments name ctx ]
  (py/call-attr math-ops "unsorted_segment_max_eager_fallback"  data segment_ids num_segments name ctx ))

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
  (py/call-attr math-ops "unsorted_segment_min"  data segment_ids num_segments name ))

(defn unsorted-segment-min-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function unsorted_segment_min
  "
  [ data segment_ids num_segments name ctx ]
  (py/call-attr math-ops "unsorted_segment_min_eager_fallback"  data segment_ids num_segments name ctx ))

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
  (py/call-attr math-ops "unsorted_segment_prod"  data segment_ids num_segments name ))

(defn unsorted-segment-prod-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function unsorted_segment_prod
  "
  [ data segment_ids num_segments name ctx ]
  (py/call-attr math-ops "unsorted_segment_prod_eager_fallback"  data segment_ids num_segments name ctx ))

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
  (py/call-attr math-ops "unsorted_segment_sum"  data segment_ids num_segments name ))

(defn unsorted-segment-sum-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function unsorted_segment_sum
  "
  [ data segment_ids num_segments name ctx ]
  (py/call-attr math-ops "unsorted_segment_sum_eager_fallback"  data segment_ids num_segments name ctx ))

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
  (py/call-attr math-ops "xdivy"  x y name ))

(defn xdivy-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function xdivy
  "
  [ x y name ctx ]
  (py/call-attr math-ops "xdivy_eager_fallback"  x y name ctx ))

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
  (py/call-attr math-ops "xlogy"  x y name ))

(defn xlogy-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function xlogy
  "
  [ x y name ctx ]
  (py/call-attr math-ops "xlogy_eager_fallback"  x y name ctx ))

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
  (py/call-attr math-ops "zeta"  x q name ))

(defn zeta-eager-fallback 
  "This is the slowpath function for Eager mode.
  This is for function zeta
  "
  [ x q name ctx ]
  (py/call-attr math-ops "zeta_eager_fallback"  x q name ctx ))
