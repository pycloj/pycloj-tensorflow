(ns tensorflow.contrib.labeled-tensor.python.ops.core
  "Core classes and core ops for LabeledTensor.

Core ops are ops which will eventually be called by LabeledTensor methods,
and ops which a core op depends upon.
For example, `add` is a core op because we'll eventually support the `+`
operator.
Non-core ops should go in `ops.py`.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce core (import-module "tensorflow.contrib.labeled_tensor.python.ops.core"))

(defn abs-function 
  "LabeledTensor version of `tf.abs`.

    See `tf.abs` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.abs` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "abs_function"  labeled_tensor name ))

(defn acos 
  "LabeledTensor version of `tf.acos`.

    See `tf.acos` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.acos` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "acos"  labeled_tensor name ))

(defn add 
  "LabeledTensor version of `tf.add` with label based alignment.

    See `tf.add` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.add` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "add"  labeled_tensor_0 labeled_tensor_1 name ))

(defn align 
  "Align the axes of two tensors so they may be broadcast to each other.

  Axes are ordered by the current axis order scope, if present, or by the left-
  most consistent ordering. An exception is raised if it is impossible to align
  the tensors without a transpose (align never copies the input data).

  Example usage:

    >>> a = lt.LabeledTensor(tf.ones((2, 4)), ['x', 'z'])
    >>> b = lt.LabeledTensor(tf.ones((3, 4)), ['y', 'z'])
    >>> a2, b2, axes = lt.align(a, b)
    >>> a2
    <LabeledTensor 'lt_align_1/lt_align_1/0:...' shape=(2, 1, 4) dtype=float32
     axes=[('x', Dimension(2)),
           ('y', Dimension(1)),
           ('z', Dimension(4))]>
    >>> b2
    <LabeledTensor 'lt_align_1/lt_align_1/1:...' shape=(1, 3, 4) dtype=float32
     axes=[('x', Dimension(1)),
           ('y', Dimension(3)),
           ('z', Dimension(4))]>
    >>> axes
    Axes([('x', Dimension(2)),
          ('y', Dimension(3)),
          ('z', Dimension(4))])

  Args:
    labeled_tensor_0: An input tensor.
    labeled_tensor_1: An input tensor.
    name: Optional op name.

  Returns:
    The aligned tensors and the axes the resulting tensor would have if the two
    aligned tensors were broadcast to each other. The aligned tensors have the
    same rank but not necessarily the same shape, with axes in the same order.

  Raises:
    ValueError: If axes with the same name on the inputs are not equal.
    AxisOrderError: If there is no way to reshape the input tensors into the
      output without a transpose.
  "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "align"  labeled_tensor_0 labeled_tensor_1 name ))

(defn as-axis 
  "Convert an AxisLike object into an Axis.

  Args:
    axis_data: Axis object or tuple (axis_name, axis_value) describing an axis.

  Returns:
    Axis object. This may be the original object if axis_data is an Axis.
  "
  [ axis_data ]
  (py/call-attr core "as_axis"  axis_data ))

(defn asin 
  "LabeledTensor version of `tf.asin`.

    See `tf.asin` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.asin` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "asin"  labeled_tensor name ))

(defn atan 
  "LabeledTensor version of `tf.atan`.

    See `tf.atan` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.atan` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "atan"  labeled_tensor name ))

(defn axis-order-scope 
  "Set axis order for the result of broadcasting operations within a scope.

  This allows you to ensure that tensors resulting from arithmetic have a
  predictable axis order.

  Example usage:

    with lt.axis_order_scope(['x', 'y', 'z']):
      # result is guaranteed to have the correct axis order
      result = w + b

  You can nest scopes, in which case only the inner-most scope applies, e.g.,

    with lt.axis_order(['x', 'y', 'z']):
      with lt.axis_order():
        result = w + b  # uses the default (left-most) axis ordering

  Args:
    axis_order: optional list of strings providing axis names. By default,
      creates a scope without axis order.

  Yields:
    The provided axis_order or `None`.
  "
  [ axis_order ]
  (py/call-attr core "axis_order_scope"  axis_order ))

(defn ceil 
  "LabeledTensor version of `tf.ceil`.

    See `tf.ceil` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.ceil` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "ceil"  labeled_tensor name ))

(defn check-axis-order 
  "Verify that the given tensor has a consistent axis order.

  Args:
    labeled_tensor: The input tensor. All axes on this tensor must appear in
      axis_order.
    axis_order: Optional desired axis order, as a list of names. If not
      provided, defaults to the current axis_order_scope (if set).

  Raises:
    AxisOrderError: If the axis_order is unavailable, inconsistent or does not
      include all existing axes.
  "
  [ labeled_tensor axis_order ]
  (py/call-attr core "check_axis_order"  labeled_tensor axis_order ))

(defn concat-axes 
  "Concatenate a list of Axes.

  Args:
    axes: A collection of Axis objects.

  Returns:
    The concatenation of the axes.
    If all axes have labels, the result has the concatenation of the labels.
    Else, the result has no labels, and its size is the sum of the sizes
    of the axes.

  Raises:
    ValueError: If `others` is not a collection of Axes or if it is empty.
  "
  [ axes ]
  (py/call-attr core "concat_axes"  axes ))

(defn convert-to-labeled-tensor 
  "Converts the given `value` to a `LabeledTensor`.

  This function accepts `LabeledTensor` objects, 0-dimensional `Tensor` objects
  and numpy arrays, and Python scalars. Higher dimensional unlabeled tensors
  must use the `LabeledTensor` constructor explicitly.

  Args:
    value: Object to convert.
    dtype: Optional element type for the returned tensor. If missing, the type
      is inferred from the type of value.
    name: Optional name to use if a new Tensor is created.

  Returns:
    `value` converted into a `LabeledTensor` object.

  Raises:
    ValueError: If the output would have rank>0 but the input was not already a
      `LabeledTensor`.
  "
  [ value dtype name ]
  (py/call-attr core "convert_to_labeled_tensor"  value dtype name ))

(defn cos 
  "LabeledTensor version of `tf.cos`.

    See `tf.cos` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.cos` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "cos"  labeled_tensor name ))

(defn define-binary-op 
  "Define a binary operation that broadcasts labeled tensors.

  Args:
    op_name: string name of the TensorFlow op.
    elementwise_function: function to call to evaluate the op on tf.Tensor
      objects. This function must accept three arguments: two tf.Tensor objects,
        and an optional `name`.

  Returns:
    Function defining the given op that acts on LabeledTensors.
  "
  [ op_name elementwise_function ]
  (py/call-attr core "define_binary_op"  op_name elementwise_function ))

(defn define-unary-op 
  "Define a unary operation for labeled tensors.

  Args:
    op_name: string name of the TensorFlow op.
    elementwise_function: function to call to evaluate the op on a single
      tf.Tensor object. This function must accept two arguments: a tf.Tensor
        object, and an optional `name`.

  Returns:
    Function defining the given op that acts on LabeledTensors.
  "
  [ op_name elementwise_function ]
  (py/call-attr core "define_unary_op"  op_name elementwise_function ))

(defn digamma 
  "LabeledTensor version of `tf.digamma`.

    See `tf.digamma` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.digamma` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "digamma"  labeled_tensor name ))

(defn div 
  "LabeledTensor version of `tf.div` with label based alignment.

    See `tf.div` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.div` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "div"  labeled_tensor_0 labeled_tensor_1 name ))

(defn equal 
  "LabeledTensor version of `tf.equal` with label based alignment.

    See `tf.equal` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.equal` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "equal"  labeled_tensor_0 labeled_tensor_1 name ))

(defn erf 
  "LabeledTensor version of `tf.erf`.

    See `tf.erf` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.erf` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "erf"  labeled_tensor name ))

(defn erfc 
  "LabeledTensor version of `tf.erfc`.

    See `tf.erfc` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.erfc` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "erfc"  labeled_tensor name ))

(defn exp 
  "LabeledTensor version of `tf.exp`.

    See `tf.exp` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.exp` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "exp"  labeled_tensor name ))

(defn expand-dims 
  "Insert dimensions of size 1.

  See tf.expand_dims.

  Args:
    labeled_tensor: The input tensor.
    axes: The desired axis names as strings or tuples of (name, label), where
      `label` is the coordinate name for the new dimension `name`. These must
      include the existing axis names, and the existing names must appear in the
      same order in this list as they do in the input tensor.
    name: Optional op name.

  Returns:
    A tensor with an axis for each axis in axes.
    New axes are created with size 1 and do not have labeled coordinates.

  Raises:
    AxisOrderError: If axis names don't appear in the same order in axes
      and the labeled tensor.
  "
  [ labeled_tensor axes name ]
  (py/call-attr core "expand_dims"  labeled_tensor axes name ))

(defn floor 
  "LabeledTensor version of `tf.floor`.

    See `tf.floor` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.floor` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "floor"  labeled_tensor name ))

(defn get-axis-order 
  "Get the axis_order set by any containing axis_order_scope.

  Returns:
    List of strings giving an order to use for axis names, or None, if no axis
    order is set.
  "
  [  ]
  (py/call-attr core "get_axis_order"  ))

(defn greater 
  "LabeledTensor version of `tf.greater` with label based alignment.

    See `tf.greater` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.greater` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "greater"  labeled_tensor_0 labeled_tensor_1 name ))

(defn greater-equal 
  "LabeledTensor version of `tf.greater_equal` with label based alignment.

    See `tf.greater_equal` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.greater_equal` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "greater_equal"  labeled_tensor_0 labeled_tensor_1 name ))

(defn identity 
  "The identity op.

  See tf.identity.

  Args:
    labeled_tensor: The input tensor.
    name: Optional op name.

  Returns:
    The tensor.
  "
  [ labeled_tensor name ]
  (py/call-attr core "identity"  labeled_tensor name ))

(defn igamma 
  "LabeledTensor version of `tf.igamma` with label based alignment.

    See `tf.igamma` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.igamma` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "igamma"  labeled_tensor_0 labeled_tensor_1 name ))

(defn igammac 
  "LabeledTensor version of `tf.igammac` with label based alignment.

    See `tf.igammac` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.igammac` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "igammac"  labeled_tensor_0 labeled_tensor_1 name ))

(defn impose-axis-order 
  "Impose desired axis order on a labeled tensor.

  Args:
    labeled_tensor: The input tensor.
    axis_order: Optional desired axis order, as a list of names. If not
      provided, defaults to the current axis_order_scope (if set).
    name: Optional op name.

  Returns:
    Labeled tensor with possibly transposed axes.

  Raises:
    AxisOrderError: If no axis_order is provided or axis_order does not contain
      all axes on the input tensor.
  "
  [ labeled_tensor axis_order name ]
  (py/call-attr core "impose_axis_order"  labeled_tensor axis_order name ))

(defn less 
  "LabeledTensor version of `tf.less` with label based alignment.

    See `tf.less` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.less` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "less"  labeled_tensor_0 labeled_tensor_1 name ))

(defn less-equal 
  "LabeledTensor version of `tf.less_equal` with label based alignment.

    See `tf.less_equal` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.less_equal` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "less_equal"  labeled_tensor_0 labeled_tensor_1 name ))

(defn lgamma 
  "LabeledTensor version of `tf.lgamma`.

    See `tf.lgamma` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.lgamma` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "lgamma"  labeled_tensor name ))

(defn log 
  "LabeledTensor version of `tf.log`.

    See `tf.log` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.log` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "log"  labeled_tensor name ))

(defn logical-and 
  "LabeledTensor version of `tf.logical_and` with label based alignment.

    See `tf.logical_and` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.logical_and` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "logical_and"  labeled_tensor_0 labeled_tensor_1 name ))

(defn logical-not 
  "LabeledTensor version of `tf.logical_not`.

    See `tf.logical_not` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.logical_not` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "logical_not"  labeled_tensor name ))

(defn logical-or 
  "LabeledTensor version of `tf.logical_or` with label based alignment.

    See `tf.logical_or` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.logical_or` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "logical_or"  labeled_tensor_0 labeled_tensor_1 name ))

(defn logical-xor 
  "LabeledTensor version of `tf.logical_xor` with label based alignment.

    See `tf.logical_xor` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.logical_xor` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "logical_xor"  labeled_tensor_0 labeled_tensor_1 name ))

(defn maximum 
  "LabeledTensor version of `tf.maximum` with label based alignment.

    See `tf.maximum` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.maximum` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "maximum"  labeled_tensor_0 labeled_tensor_1 name ))

(defn minimum 
  "LabeledTensor version of `tf.minimum` with label based alignment.

    See `tf.minimum` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.minimum` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "minimum"  labeled_tensor_0 labeled_tensor_1 name ))

(defn mod 
  "LabeledTensor version of `tf.mod` with label based alignment.

    See `tf.mod` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.mod` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "mod"  labeled_tensor_0 labeled_tensor_1 name ))

(defn mul 
  "LabeledTensor version of `tf.mul` with label based alignment.

    See `tf.mul` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.mul` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "mul"  labeled_tensor_0 labeled_tensor_1 name ))

(defn neg 
  "LabeledTensor version of `tf.neg`.

    See `tf.neg` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.neg` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "neg"  labeled_tensor name ))

(defn not-equal 
  "LabeledTensor version of `tf.not_equal` with label based alignment.

    See `tf.not_equal` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.not_equal` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "not_equal"  labeled_tensor_0 labeled_tensor_1 name ))

(defn polygamma 
  "LabeledTensor version of `tf.polygamma` with label based alignment.

    See `tf.polygamma` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.polygamma` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "polygamma"  labeled_tensor_0 labeled_tensor_1 name ))

(defn pow-function 
  "LabeledTensor version of `tf.pow` with label based alignment.

    See `tf.pow` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.pow` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "pow_function"  labeled_tensor_0 labeled_tensor_1 name ))

(defn reciprocal 
  "LabeledTensor version of `tf.reciprocal`.

    See `tf.reciprocal` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.reciprocal` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "reciprocal"  labeled_tensor name ))

(defn round-function 
  "LabeledTensor version of `tf.round`.

    See `tf.round` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.round` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "round_function"  labeled_tensor name ))

(defn rsqrt 
  "LabeledTensor version of `tf.rsqrt`.

    See `tf.rsqrt` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.rsqrt` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "rsqrt"  labeled_tensor name ))

(defn sigmoid 
  "LabeledTensor version of `tf.sigmoid`.

    See `tf.sigmoid` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.sigmoid` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "sigmoid"  labeled_tensor name ))

(defn sign 
  "LabeledTensor version of `tf.sign`.

    See `tf.sign` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.sign` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "sign"  labeled_tensor name ))

(defn sin 
  "LabeledTensor version of `tf.sin`.

    See `tf.sin` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.sin` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "sin"  labeled_tensor name ))

(defn slice-function 
  "Slice out a subset of the tensor.

  This is an analog of tf.slice.
  For example:
  >>> tensor = tf.reshape(tf.range(0, 6), [3, 2])
  >>> labeled_tensor = lt.LabeledTensor(tensor, ['a', ('b', ['foo', 'bar'])])
  >>> lt.slice(labeled_tensor, {'a': slice(0, 2), 'b': 1})
  <LabeledTensor 'lt_slice:...' shape=(2,) dtype=int32
   axes=[('a', Dimension(2))]>

  Args:
    labeled_tensor: The input tensor.
    selection: A dictionary of type str -> Union(int, slice of int) mapping axis
      names to sub-selections.
    name: Optional op name.

  Returns:
    The slice as a `LabeledTensor`.
  "
  [ labeled_tensor selection name ]
  (py/call-attr core "slice_function"  labeled_tensor selection name ))

(defn sqrt 
  "LabeledTensor version of `tf.sqrt`.

    See `tf.sqrt` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.sqrt` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "sqrt"  labeled_tensor name ))

(defn square 
  "LabeledTensor version of `tf.square`.

    See `tf.square` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.square` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "square"  labeled_tensor name ))

(defn squared-difference 
  "LabeledTensor version of `tf.squared_difference` with label based alignment.

    See `tf.squared_difference` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.squared_difference` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "squared_difference"  labeled_tensor_0 labeled_tensor_1 name ))

(defn sub 
  "LabeledTensor version of `tf.sub` with label based alignment.

    See `tf.sub` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.sub` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "sub"  labeled_tensor_0 labeled_tensor_1 name ))

(defn tan 
  "LabeledTensor version of `tf.tan`.

    See `tf.tan` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.tan` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "tan"  labeled_tensor name ))

(defn tanh 
  "LabeledTensor version of `tf.tanh`.

    See `tf.tanh` for full details.

    Args:
      labeled_tensor: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.tanh` elementwise.
    "
  [ labeled_tensor name ]
  (py/call-attr core "tanh"  labeled_tensor name ))

(defn transpose 
  "Permute a tensor's axes.

  See tf.transpose.

  Args:
    labeled_tensor: The input tensor.
    axis_order: Optional desired axis order, as a list of names. By default, the
      order of axes is reversed.
    name: Optional op name.

  Returns:
    The permuted tensor.

  Raises:
    ValueError: If axis_order isn't a permutation of the existing axes.
  "
  [ labeled_tensor axis_order name ]
  (py/call-attr core "transpose"  labeled_tensor axis_order name ))

(defn zeta 
  "LabeledTensor version of `tf.zeta` with label based alignment.

    See `tf.zeta` for full details.

    Args:
      labeled_tensor_0: Input tensor.
      labeled_tensor_1: Input tensor.
      name: Optional op name.

    Returns:
      A LabeledTensor with result of applying `tf.zeta` elementwise.
    "
  [ labeled_tensor_0 labeled_tensor_1 name ]
  (py/call-attr core "zeta"  labeled_tensor_0 labeled_tensor_1 name ))
