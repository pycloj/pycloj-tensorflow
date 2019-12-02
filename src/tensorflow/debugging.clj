(ns tensorflow.debugging
  "Public API for tf.debugging namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce debugging (import-module "tensorflow.debugging"))

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
  (py/call-attr debugging "Assert"  condition data summarize name ))

(defn assert-all-finite 
  "Assert that the tensor does not contain any NaN's or Inf's.

  Args:
    t: Tensor to check.
    msg: Message to log on failure.
    name: A name for this operation (optional).
    x: Alias for t.
    message: Alias for msg.

  Returns:
    Same tensor as `t`.
  "
  [ t msg name x message ]
  (py/call-attr debugging "assert_all_finite"  t msg name x message ))

(defn assert-equal 
  "
    Assert the condition `x == y` holds element-wise.

    This condition holds if for every pair of (possibly broadcast) elements
    `x[i]`, `y[i]`, we have `x[i] == y[i]`.
    If both `x` and `y` are empty, this is trivially satisfied.

    When running in graph mode, you should add a dependency on this operation
    to ensure that it runs. Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.compat.v1.assert_equal(x, y)]):
      output = tf.reduce_sum(x)
    ```

    Args:
      x:  Numeric `Tensor`.
      y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
      data:  The tensors to print out if the condition is False.  Defaults to
        error message and first few entries of `x`, `y`.
      summarize: Print this many entries of each tensor.
      message: A string to prefix to the default message.
      name: A name for this operation (optional).  Defaults to \"assert_equal\".

    Returns:
      Op that raises `InvalidArgumentError` if `x == y` is False.
      @compatibility(eager)
        returns None
      @end_compatibility

    Raises:
      InvalidArgumentError: if the check can be performed immediately and
        `x == y` is False. The check can be performed immediately during 
        eager execution or if `x` and `y` are statically known.
    "
  [ x y data summarize message name ]
  (py/call-attr debugging "assert_equal"  x y data summarize message name ))

(defn assert-greater 
  "
    Assert the condition `x > y` holds element-wise.

    This condition holds if for every pair of (possibly broadcast) elements
    `x[i]`, `y[i]`, we have `x[i] > y[i]`.
    If both `x` and `y` are empty, this is trivially satisfied.

    When running in graph mode, you should add a dependency on this operation
    to ensure that it runs. Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.compat.v1.assert_greater(x, y)]):
      output = tf.reduce_sum(x)
    ```

    Args:
      x:  Numeric `Tensor`.
      y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
      data:  The tensors to print out if the condition is False.  Defaults to
        error message and first few entries of `x`, `y`.
      summarize: Print this many entries of each tensor.
      message: A string to prefix to the default message.
      name: A name for this operation (optional).  Defaults to \"assert_greater\".

    Returns:
      Op that raises `InvalidArgumentError` if `x > y` is False.
      @compatibility(eager)
        returns None
      @end_compatibility

    Raises:
      InvalidArgumentError: if the check can be performed immediately and
        `x > y` is False. The check can be performed immediately during 
        eager execution or if `x` and `y` are statically known.
    "
  [ x y data summarize message name ]
  (py/call-attr debugging "assert_greater"  x y data summarize message name ))

(defn assert-greater-equal 
  "
    Assert the condition `x >= y` holds element-wise.

    This condition holds if for every pair of (possibly broadcast) elements
    `x[i]`, `y[i]`, we have `x[i] >= y[i]`.
    If both `x` and `y` are empty, this is trivially satisfied.

    When running in graph mode, you should add a dependency on this operation
    to ensure that it runs. Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.compat.v1.assert_greater_equal(x, y)]):
      output = tf.reduce_sum(x)
    ```

    Args:
      x:  Numeric `Tensor`.
      y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
      data:  The tensors to print out if the condition is False.  Defaults to
        error message and first few entries of `x`, `y`.
      summarize: Print this many entries of each tensor.
      message: A string to prefix to the default message.
      name: A name for this operation (optional).  Defaults to \"assert_greater_equal\".

    Returns:
      Op that raises `InvalidArgumentError` if `x >= y` is False.
      @compatibility(eager)
        returns None
      @end_compatibility

    Raises:
      InvalidArgumentError: if the check can be performed immediately and
        `x >= y` is False. The check can be performed immediately during 
        eager execution or if `x` and `y` are statically known.
    "
  [ x y data summarize message name ]
  (py/call-attr debugging "assert_greater_equal"  x y data summarize message name ))

(defn assert-integer 
  "Assert that `x` is of integer dtype.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.compat.v1.assert_integer(x)]):
    output = tf.reduce_sum(x)
  ```

  Args:
    x: `Tensor` whose basetype is integer and is not quantized.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to \"assert_integer\".

  Raises:
    TypeError:  If `x.dtype` is anything other than non-quantized integer.

  Returns:
    A `no_op` that does nothing.  Type can be determined statically.
  "
  [ x message name ]
  (py/call-attr debugging "assert_integer"  x message name ))

(defn assert-less 
  "
    Assert the condition `x < y` holds element-wise.

    This condition holds if for every pair of (possibly broadcast) elements
    `x[i]`, `y[i]`, we have `x[i] < y[i]`.
    If both `x` and `y` are empty, this is trivially satisfied.

    When running in graph mode, you should add a dependency on this operation
    to ensure that it runs. Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.compat.v1.assert_less(x, y)]):
      output = tf.reduce_sum(x)
    ```

    Args:
      x:  Numeric `Tensor`.
      y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
      data:  The tensors to print out if the condition is False.  Defaults to
        error message and first few entries of `x`, `y`.
      summarize: Print this many entries of each tensor.
      message: A string to prefix to the default message.
      name: A name for this operation (optional).  Defaults to \"assert_less\".

    Returns:
      Op that raises `InvalidArgumentError` if `x < y` is False.
      @compatibility(eager)
        returns None
      @end_compatibility

    Raises:
      InvalidArgumentError: if the check can be performed immediately and
        `x < y` is False. The check can be performed immediately during 
        eager execution or if `x` and `y` are statically known.
    "
  [ x y data summarize message name ]
  (py/call-attr debugging "assert_less"  x y data summarize message name ))

(defn assert-less-equal 
  "
    Assert the condition `x <= y` holds element-wise.

    This condition holds if for every pair of (possibly broadcast) elements
    `x[i]`, `y[i]`, we have `x[i] <= y[i]`.
    If both `x` and `y` are empty, this is trivially satisfied.

    When running in graph mode, you should add a dependency on this operation
    to ensure that it runs. Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.compat.v1.assert_less_equal(x, y)]):
      output = tf.reduce_sum(x)
    ```

    Args:
      x:  Numeric `Tensor`.
      y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
      data:  The tensors to print out if the condition is False.  Defaults to
        error message and first few entries of `x`, `y`.
      summarize: Print this many entries of each tensor.
      message: A string to prefix to the default message.
      name: A name for this operation (optional).  Defaults to \"assert_less_equal\".

    Returns:
      Op that raises `InvalidArgumentError` if `x <= y` is False.
      @compatibility(eager)
        returns None
      @end_compatibility

    Raises:
      InvalidArgumentError: if the check can be performed immediately and
        `x <= y` is False. The check can be performed immediately during 
        eager execution or if `x` and `y` are statically known.
    "
  [ x y data summarize message name ]
  (py/call-attr debugging "assert_less_equal"  x y data summarize message name ))

(defn assert-near 
  "Assert the condition `x` and `y` are close element-wise.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.compat.v1.assert_near(x, y)]):
    output = tf.reduce_sum(x)
  ```

  This condition holds if for every pair of (possibly broadcast) elements
  `x[i]`, `y[i]`, we have

  ```tf.abs(x[i] - y[i]) <= atol + rtol * tf.abs(y[i])```.

  If both `x` and `y` are empty, this is trivially satisfied.

  The default `atol` and `rtol` is `10 * eps`, where `eps` is the smallest
  representable positive number such that `1 + eps != 1`.  This is about
  `1.2e-6` in `32bit`, `2.22e-15` in `64bit`, and `0.00977` in `16bit`.
  See `numpy.finfo`.

  Args:
    x:  Float or complex `Tensor`.
    y:  Float or complex `Tensor`, same `dtype` as, and broadcastable to, `x`.
    rtol:  `Tensor`.  Same `dtype` as, and broadcastable to, `x`.
      The relative tolerance.  Default is `10 * eps`.
    atol:  `Tensor`.  Same `dtype` as, and broadcastable to, `x`.
      The absolute tolerance.  Default is `10 * eps`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`, `y`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to \"assert_near\".

  Returns:
    Op that raises `InvalidArgumentError` if `x` and `y` are not close enough.

  @compatibility(numpy)
  Similar to `numpy.assert_allclose`, except tolerance depends on data type.
  This is due to the fact that `TensorFlow` is often used with `32bit`, `64bit`,
  and even `16bit` data.
  @end_compatibility
  "
  [ x y rtol atol data summarize message name ]
  (py/call-attr debugging "assert_near"  x y rtol atol data summarize message name ))

(defn assert-negative 
  "
    Assert the condition `x < 0` holds element-wise.

    When running in graph mode, you should add a dependency on this operation
    to ensure that it runs. Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.debugging.assert_negative(x, y)]):
      output = tf.reduce_sum(x)
    ```

    Negative means, for every element `x[i]` of `x`, we have `x[i] < 0`.
    If `x` is empty this is trivially satisfied.

    Args:
      x:  Numeric `Tensor`.
      data:  The tensors to print out if the condition is False.  Defaults to
        error message and first few entries of `x`.
      summarize: Print this many entries of each tensor.
      message: A string to prefix to the default message.
      name: A name for this operation (optional).  Defaults to \"assert_negative\".

    Returns:
      Op that raises `InvalidArgumentError` if `x < 0` is False.
      @compatibility(eager)
        returns None
      @end_compatibility

    Raises:
      InvalidArgumentError: if the check can be performed immediately and
        `x < 0` is False. The check can be performed immediately during 
        eager execution or if `x` is statically known.
    "
  [ x data summarize message name ]
  (py/call-attr debugging "assert_negative"  x data summarize message name ))

(defn assert-non-negative 
  "
    Assert the condition `x >= 0` holds element-wise.

    When running in graph mode, you should add a dependency on this operation
    to ensure that it runs. Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.debugging.assert_non_negative(x, y)]):
      output = tf.reduce_sum(x)
    ```

    Non-negative means, for every element `x[i]` of `x`, we have `x[i] >= 0`.
    If `x` is empty this is trivially satisfied.

    Args:
      x:  Numeric `Tensor`.
      data:  The tensors to print out if the condition is False.  Defaults to
        error message and first few entries of `x`.
      summarize: Print this many entries of each tensor.
      message: A string to prefix to the default message.
      name: A name for this operation (optional).  Defaults to \"assert_non_negative\".

    Returns:
      Op that raises `InvalidArgumentError` if `x >= 0` is False.
      @compatibility(eager)
        returns None
      @end_compatibility

    Raises:
      InvalidArgumentError: if the check can be performed immediately and
        `x >= 0` is False. The check can be performed immediately during 
        eager execution or if `x` is statically known.
    "
  [ x data summarize message name ]
  (py/call-attr debugging "assert_non_negative"  x data summarize message name ))

(defn assert-non-positive 
  "
    Assert the condition `x <= 0` holds element-wise.

    When running in graph mode, you should add a dependency on this operation
    to ensure that it runs. Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.debugging.assert_non_positive(x, y)]):
      output = tf.reduce_sum(x)
    ```

    Non-positive means, for every element `x[i]` of `x`, we have `x[i] <= 0`.
    If `x` is empty this is trivially satisfied.

    Args:
      x:  Numeric `Tensor`.
      data:  The tensors to print out if the condition is False.  Defaults to
        error message and first few entries of `x`.
      summarize: Print this many entries of each tensor.
      message: A string to prefix to the default message.
      name: A name for this operation (optional).  Defaults to \"assert_non_positive\".

    Returns:
      Op that raises `InvalidArgumentError` if `x <= 0` is False.
      @compatibility(eager)
        returns None
      @end_compatibility

    Raises:
      InvalidArgumentError: if the check can be performed immediately and
        `x <= 0` is False. The check can be performed immediately during 
        eager execution or if `x` is statically known.
    "
  [ x data summarize message name ]
  (py/call-attr debugging "assert_non_positive"  x data summarize message name ))

(defn assert-none-equal 
  "
    Assert the condition `x != y` holds element-wise.

    This condition holds if for every pair of (possibly broadcast) elements
    `x[i]`, `y[i]`, we have `x[i] != y[i]`.
    If both `x` and `y` are empty, this is trivially satisfied.

    When running in graph mode, you should add a dependency on this operation
    to ensure that it runs. Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.compat.v1.assert_none_equal(x, y)]):
      output = tf.reduce_sum(x)
    ```

    Args:
      x:  Numeric `Tensor`.
      y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.
      data:  The tensors to print out if the condition is False.  Defaults to
        error message and first few entries of `x`, `y`.
      summarize: Print this many entries of each tensor.
      message: A string to prefix to the default message.
      name: A name for this operation (optional).  Defaults to \"assert_none_equal\".

    Returns:
      Op that raises `InvalidArgumentError` if `x != y` is False.
      @compatibility(eager)
        returns None
      @end_compatibility

    Raises:
      InvalidArgumentError: if the check can be performed immediately and
        `x != y` is False. The check can be performed immediately during 
        eager execution or if `x` and `y` are statically known.
    "
  [ x y data summarize message name ]
  (py/call-attr debugging "assert_none_equal"  x y data summarize message name ))

(defn assert-positive 
  "
    Assert the condition `x > 0` holds element-wise.

    When running in graph mode, you should add a dependency on this operation
    to ensure that it runs. Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.debugging.assert_positive(x, y)]):
      output = tf.reduce_sum(x)
    ```

    Positive means, for every element `x[i]` of `x`, we have `x[i] > 0`.
    If `x` is empty this is trivially satisfied.

    Args:
      x:  Numeric `Tensor`.
      data:  The tensors to print out if the condition is False.  Defaults to
        error message and first few entries of `x`.
      summarize: Print this many entries of each tensor.
      message: A string to prefix to the default message.
      name: A name for this operation (optional).  Defaults to \"assert_positive\".

    Returns:
      Op that raises `InvalidArgumentError` if `x > 0` is False.
      @compatibility(eager)
        returns None
      @end_compatibility

    Raises:
      InvalidArgumentError: if the check can be performed immediately and
        `x > 0` is False. The check can be performed immediately during 
        eager execution or if `x` is statically known.
    "
  [ x data summarize message name ]
  (py/call-attr debugging "assert_positive"  x data summarize message name ))

(defn assert-proper-iterable 
  "Static assert that values is a \"proper\" iterable.

  `Ops` that expect iterables of `Tensor` can call this to validate input.
  Useful since `Tensor`, `ndarray`, byte/text type are all iterables themselves.

  Args:
    values:  Object to be checked.

  Raises:
    TypeError:  If `values` is not iterable or is one of
      `Tensor`, `SparseTensor`, `np.array`, `tf.compat.bytes_or_text_types`.
  "
  [ values ]
  (py/call-attr debugging "assert_proper_iterable"  values ))

(defn assert-rank 
  "Assert `x` has rank equal to `rank`.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.compat.v1.assert_rank(x, 2)]):
    output = tf.reduce_sum(x)
  ```

  Args:
    x:  Numeric `Tensor`.
    rank:  Scalar integer `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and the shape of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to \"assert_rank\".

  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank.
    If static checks determine `x` has correct rank, a `no_op` is returned.

  Raises:
    ValueError:  If static checks determine `x` has wrong rank.
  "
  [ x rank data summarize message name ]
  (py/call-attr debugging "assert_rank"  x rank data summarize message name ))

(defn assert-rank-at-least 
  "Assert `x` has rank equal to `rank` or higher.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.compat.v1.assert_rank_at_least(x, 2)]):
    output = tf.reduce_sum(x)
  ```

  Args:
    x:  Numeric `Tensor`.
    rank:  Scalar `Tensor`.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).
      Defaults to \"assert_rank_at_least\".

  Returns:
    Op raising `InvalidArgumentError` unless `x` has specified rank or higher.
    If static checks determine `x` has correct rank, a `no_op` is returned.

  Raises:
    ValueError:  If static checks determine `x` has wrong rank.
  "
  [ x rank data summarize message name ]
  (py/call-attr debugging "assert_rank_at_least"  x rank data summarize message name ))

(defn assert-rank-in 
  "Assert `x` has rank in `ranks`.

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.compat.v1.assert_rank_in(x, (2, 4))]):
    output = tf.reduce_sum(x)
  ```

  Args:
    x:  Numeric `Tensor`.
    ranks:  Iterable of scalar `Tensor` objects.
    data:  The tensors to print out if the condition is False.  Defaults to
      error message and first few entries of `x`.
    summarize: Print this many entries of each tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).
      Defaults to \"assert_rank_in\".

  Returns:
    Op raising `InvalidArgumentError` unless rank of `x` is in `ranks`.
    If static checks determine `x` has matching rank, a `no_op` is returned.

  Raises:
    ValueError:  If static checks determine `x` has mismatched rank.
  "
  [ x ranks data summarize message name ]
  (py/call-attr debugging "assert_rank_in"  x ranks data summarize message name ))

(defn assert-same-float-dtype 
  "Validate and return float type based on `tensors` and `dtype`.

  For ops such as matrix multiplication, inputs and weights must be of the
  same float type. This function validates that all `tensors` are the same type,
  validates that type is `dtype` (if supplied), and returns the type. Type must
  be a floating point type. If neither `tensors` nor `dtype` is supplied,
  the function will return `dtypes.float32`.

  Args:
    tensors: Tensors of input values. Can include `None` elements, which will be
        ignored.
    dtype: Expected type.

  Returns:
    Validated type.

  Raises:
    ValueError: if neither `tensors` nor `dtype` is supplied, or result is not
        float, or the common type of the inputs is not a floating point type.
  "
  [ tensors dtype ]
  (py/call-attr debugging "assert_same_float_dtype"  tensors dtype ))

(defn assert-scalar 
  "Asserts that the given `tensor` is a scalar (i.e. zero-dimensional).

  This function raises `ValueError` unless it can be certain that the given
  `tensor` is a scalar. `ValueError` is also raised if the shape of `tensor` is
  unknown.

  Args:
    tensor: A `Tensor`.
    name:  A name for this operation. Defaults to \"assert_scalar\"
    message: A string to prefix to the default message.

  Returns:
    The input tensor (potentially converted to a `Tensor`).

  Raises:
    ValueError: If the tensor is not scalar (rank 0), or if its shape is
      unknown.
  "
  [ tensor name message ]
  (py/call-attr debugging "assert_scalar"  tensor name message ))

(defn assert-shapes 
  "Assert tensor shapes and dimension size relationships between tensors.

  This Op checks that a collection of tensors shape relationships
  satisfies given constraints.

  Example:

  ```python
  tf.assert_shapes({
    (x, ('N', 'Q')),
    (y, ('N', 'D')),
    (param, ('Q',)),
    (scalar, ())
  })
  ```

  Example of adding a dependency to an operation:

  ```python
  with tf.control_dependencies([tf.assert_shapes(shapes)]):
    output = tf.matmul(x, y, transpose_a=True)
  ```

  If `x`, `y`, `param` or `scalar` does not have a shape that satisfies
  all specified constraints, `message`, as well as the first `summarize` entries
  of the first encountered violating tensor are printed, and
  `InvalidArgumentError` is raised.

  Size entries in the specified shapes are checked against other entries by
  their __hash__, except:
    - a size entry is interpreted as an explicit size if it can be parsed as an
      integer primitive.
    - a size entry is interpreted as *any* size if it is None or '.'.

  If the first entry of a shape is `...` (type `Ellipsis`) or '*' that indicates
  a variable number of outer dimensions of unspecified size, i.e. the constraint
  applies to the inner-most dimensions only.

  Scalar tensors and specified shapes of length zero (excluding the 'inner-most'
  prefix) are both treated as having a single dimension of size one.

  Args:
    shapes: dictionary with (`Tensor` to shape) items. A shape must be an
      iterable.
    data: The tensors to print out if the condition is False.  Defaults to error
      message and first few entries of the violating tensor.
    summarize: Print this many entries of the tensor.
    message: A string to prefix to the default message.
    name: A name for this operation (optional).  Defaults to \"assert_shapes\".

  Returns:
    Op raising `InvalidArgumentError` unless all shape constraints are
    satisfied.
    If static checks determine all constraints are satisfied, a `no_op` is
    returned.

  Raises:
    ValueError:  If static checks determine any shape constraint is violated.
  "
  [ shapes data summarize message name ]
  (py/call-attr debugging "assert_shapes"  shapes data summarize message name ))

(defn assert-type 
  "Statically asserts that the given `Tensor` is of the specified type.

  Args:
    tensor: A `Tensor`.
    tf_type: A tensorflow type (`dtypes.float32`, `tf.int64`, `dtypes.bool`,
      etc).
    message: A string to prefix to the default message.
    name:  A name to give this `Op`.  Defaults to \"assert_type\"

  Raises:
    TypeError: If the tensors data type doesn't match `tf_type`.

  Returns:
    A `no_op` that does nothing.  Type can be determined statically.
  "
  [ tensor tf_type message name ]
  (py/call-attr debugging "assert_type"  tensor tf_type message name ))

(defn check-numerics 
  "Checks a tensor for NaN and Inf values.

  When run, reports an `InvalidArgument` error if `tensor` has any values
  that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is.

  Args:
    tensor: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    message: A `string`. Prefix of the error message.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  "
  [ tensor message name ]
  (py/call-attr debugging "check_numerics"  tensor message name ))

(defn get-log-device-placement 
  "Get if device placements are logged.

  Returns:
    If device placements are logged.
  "
  [  ]
  (py/call-attr debugging "get_log_device_placement"  ))

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
  (py/call-attr debugging "is_finite"  x name ))

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
  (py/call-attr debugging "is_inf"  x name ))

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
  (py/call-attr debugging "is_nan"  x name ))

(defn is-non-decreasing 
  "Returns `True` if `x` is non-decreasing.

  Elements of `x` are compared in row-major order.  The tensor `[x[0],...]`
  is non-decreasing if for every adjacent pair we have `x[i] <= x[i+1]`.
  If `x` has less than two elements, it is trivially non-decreasing.

  See also:  `is_strictly_increasing`

  Args:
    x: Numeric `Tensor`.
    name: A name for this operation (optional).  Defaults to \"is_non_decreasing\"

  Returns:
    Boolean `Tensor`, equal to `True` iff `x` is non-decreasing.

  Raises:
    TypeError: if `x` is not a numeric tensor.
  "
  [ x name ]
  (py/call-attr debugging "is_non_decreasing"  x name ))

(defn is-numeric-tensor 
  "Returns `True` if the elements of `tensor` are numbers.

  Specifically, returns `True` if the dtype of `tensor` is one of the following:

  * `tf.float32`
  * `tf.float64`
  * `tf.int8`
  * `tf.int16`
  * `tf.int32`
  * `tf.int64`
  * `tf.uint8`
  * `tf.qint8`
  * `tf.qint32`
  * `tf.quint8`
  * `tf.complex64`

  Returns `False` if `tensor` is of a non-numeric type or if `tensor` is not
  a `tf.Tensor` object.
  "
  [ tensor ]
  (py/call-attr debugging "is_numeric_tensor"  tensor ))

(defn is-strictly-increasing 
  "Returns `True` if `x` is strictly increasing.

  Elements of `x` are compared in row-major order.  The tensor `[x[0],...]`
  is strictly increasing if for every adjacent pair we have `x[i] < x[i+1]`.
  If `x` has less than two elements, it is trivially strictly increasing.

  See also:  `is_non_decreasing`

  Args:
    x: Numeric `Tensor`.
    name: A name for this operation (optional).
      Defaults to \"is_strictly_increasing\"

  Returns:
    Boolean `Tensor`, equal to `True` iff `x` is strictly increasing.

  Raises:
    TypeError: if `x` is not a numeric tensor.
  "
  [ x name ]
  (py/call-attr debugging "is_strictly_increasing"  x name ))

(defn set-log-device-placement 
  "Set if device placements should be logged.

  Args:
    enabled: Whether to enabled device placement logging.
  "
  [ enabled ]
  (py/call-attr debugging "set_log_device_placement"  enabled ))
