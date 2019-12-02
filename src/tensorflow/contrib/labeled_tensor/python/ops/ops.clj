(ns tensorflow.contrib.labeled-tensor.python.ops.ops
  "Non-core ops for LabeledTensor."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce ops (import-module "tensorflow.contrib.labeled_tensor.python.ops.ops"))

(defn batch 
  "Rebatch a tensor.

  See tf.batch.

  Args:
    labeled_tensors: The input tensors.
    batch_size: The output batch size.
    num_threads: See tf.batch.
    capacity: See tf.batch.
    enqueue_many: If true, the input tensors must contain a 'batch' axis as
      their first axis.
      If false, the input tensors must not contain a 'batch' axis.
      See tf.batch.
    allow_smaller_final_batch: See tf.batch.
    name: Optional op name.

  Returns:
    The rebatched tensors.
    If enqueue_many is false, the output tensors will have a new 'batch' axis
      as their first axis.

  Raises:
    ValueError: If enqueue_many is True and the first axis of the tensors
      isn't \"batch\".
  "
  [labeled_tensors batch_size & {:keys [num_threads capacity enqueue_many allow_smaller_final_batch name]
                       :or {name None}} ]
    (py/call-attr-kw ops "batch" [labeled_tensors batch_size] {:num_threads num_threads :capacity capacity :enqueue_many enqueue_many :allow_smaller_final_batch allow_smaller_final_batch :name name }))

(defn boolean-mask 
  "Apply a boolean mask to a labeled tensor.

  Unlike `tf.boolean_mask`, this currently only works on 1-dimensional masks.
  The mask is applied to the first axis of `labeled_tensor`. Labels on the first
  axis are removed, because True indices in `mask` may not be known dynamically.

  Args:
    labeled_tensor: The input tensor.
    mask: The type of the returned tensor.
    name: Optional op name.

  Returns:
    The masked labeled tensor.

  Raises:
    ValueError: if the first axis of the mask
  "
  [ labeled_tensor mask name ]
  (py/call-attr ops "boolean_mask"  labeled_tensor mask name ))

(defn cast 
  "Casts a labeled tensor to a new type.

  Args:
    labeled_tensor: The input tensor.
    dtype: The type of the returned tensor.
    name: Optional op name.

  Returns:
    A labeled tensor with the new dtype.
  "
  [ labeled_tensor dtype name ]
  (py/call-attr ops "cast"  labeled_tensor dtype name ))

(defn concat 
  "Concatenate tensors along a dimension.

  See tf.concat.

  Args:
    labeled_tensors: A list of input LabeledTensors.
    axis_name: The name of the axis along which to concatenate.
    name: Optional op name.

  Returns:
    The concatenated tensor.
    The coordinate labels for the concatenation dimension are also concatenated,
    if they are available for every tensor.

  Raises:
    ValueError: If fewer than one tensor inputs is provided, if the tensors
      have incompatible axes, or if `axis_name` isn't the name of an axis.
  "
  [ labeled_tensors axis_name name ]
  (py/call-attr ops "concat"  labeled_tensors axis_name name ))

(defn constant 
  "Creates a constant tensor.

  If `axes` includes any strings, shape is inferred from `value`. Otherwise,
  the sizes of the given `axes` are used to set `shape` for `tf.constant`.

  See tf.constant for more details.

  Args:
    value: The input tensor.
    dtype: The type of the returned tensor.
    axes: Optional Axes, list of strings or list of objects coercible to Axis
      objects. By default, axes are assumed to be an empty list (i.e., `value`
      is treated as a scalar).
    name: Optional op name.

  Returns:
    The tensor with elements set to zero.
  "
  [ value dtype axes name ]
  (py/call-attr ops "constant"  value dtype axes name ))

(defn define-reduce-op 
  "Define a reduction op for labeled tensors.

  Args:
    op_name: string name of the TensorFlow op.
    reduce_fn: function to call to evaluate the op on a tf.Tensor.

  Returns:
    Function defining the given reduction op that acts on a LabeledTensor.
  "
  [ op_name reduce_fn ]
  (py/call-attr ops "define_reduce_op"  op_name reduce_fn ))

(defn foldl 
  "Left fold on the list of tensors unpacked from labeled_tensor.

  See tf.foldl.

  Args:
    fn: The function to apply to each unpacked LabeledTensor.
      It should have type (LabeledTensor, LabeledTensor) -> LabeledTensor.
      Its arguments are (accumulated_value, next_value).
    labeled_tensor: The input tensor.
    initial_value: The initial value of the accumulator.
    name: Optional op name.

  Returns:
    The accumulated value.
  "
  [ fn labeled_tensor initial_value name ]
  (py/call-attr ops "foldl"  fn labeled_tensor initial_value name ))

(defn map-fn 
  "Map on the list of tensors unpacked from labeled_tensor.

  See tf.map_fn.

  Args:
    fn: The function to apply to each unpacked LabeledTensor.
      It should have type LabeledTensor -> LabeledTensor.
    labeled_tensor: The input tensor.
    name: Optional op name.

  Returns:
    A tensor that packs the results of applying fn to the list of tensors
    unpacked from labeled_tensor.
  "
  [ fn labeled_tensor name ]
  (py/call-attr ops "map_fn"  fn labeled_tensor name ))

(defn matmul 
  "Matrix multiply two tensors with rank 1 or 2.

  If both tensors have rank 2, a matrix-matrix product is performed.
  If one tensor has rank 1 and the other has rank 2, then a matrix-vector
  product is performed.
  If both tensors have rank 1, then a vector dot-product is performed.
  (This behavior matches that of `numpy.dot`.)

  Both tensors must share exactly one dimension in common, which is the
  dimension the operation is summed along. The inputs will be automatically
  transposed if necessary as part of the matmul op.

  We intend to eventually support `matmul` on higher rank input, and also
  eventually support summing over any number shared dimensions (via an `axis`
  argument), but neither of these features has been implemented yet.

  Args:
    a: First LabeledTensor.
    b: Second LabeledTensor.
    name: Optional op name.

  Returns:
    LabeledTensor with the result of matrix multiplication. Axes are ordered by
    the current axis_order_scope, if set, or in or order of appearance on the
    inputs.

  Raises:
    NotImplementedError: If inputs have rank >2 or share multiple axes.
    ValueError: If the inputs have rank 0 or do not share any axes.
  "
  [ a b name ]
  (py/call-attr ops "matmul"  a b name ))

(defn ones-like 
  "Creates an identical tensor with all elements set to one.

  Args:
    labeled_tensor: The input tensor.
    dtype: The type of the returned tensor.
    name: Optional op name.

  Returns:
    The tensor with elements set to one.
  "
  [ labeled_tensor dtype name ]
  (py/call-attr ops "ones_like"  labeled_tensor dtype name ))

(defn pack 
  "Pack tensors along a new axis.

  See tf.pack.

  Args:
    labeled_tensors: The input tensors, which must have identical axes.
    new_axis: The name of the new axis, or a tuple containing the name
      and coordinate labels.
    axis_position: Optional integer position at which to insert the new axis.
    name: Optional op name.

  Returns:
    The packed tensors as a single LabeledTensor, with `new_axis` in the given
    `axis_position`.

  Raises:
    ValueError: If fewer than one input tensors is provided, or if the tensors
      don't have identical axes.
  "
  [labeled_tensors new_axis & {:keys [axis_position name]
                       :or {name None}} ]
    (py/call-attr-kw ops "pack" [labeled_tensors new_axis] {:axis_position axis_position :name name }))

(defn pad 
  "Pads a tensor.

  See tf.pad.

  Args:
    labeled_tensor: The input tensor.
    paddings: A mapping where the keys are axis names and the values are
      tuples where the first element is the padding to insert at the beginning
      of the axis and the second is the padding to insert at the end of the
      axis.
    mode: One of \"CONSTANT\", \"REFLECT\", or \"SYMMETRIC\".
    name: Optional op name.

  Returns:
    A tensor with the indicated axes padded, optionally with those axes extended
    with the provided labels.

  Raises:
    ValueError: If the padded axes are not axes in the input tensor.
  "
  [labeled_tensor paddings & {:keys [mode name]
                       :or {name None}} ]
    (py/call-attr-kw ops "pad" [labeled_tensor paddings] {:mode mode :name name }))

(defn random-crop 
  "Randomly crops a tensor to a given size.

  See tf.random_crop.

  Args:
    labeled_tensor: The input tensor.
    shape_map: A dictionary mapping axis names to the size of the random crop
      for that dimension.
    seed: An optional random seed.
    name: An optional op name.

  Returns:
    A tensor of the same rank as `labeled_tensor`, cropped randomly in the
    selected dimensions.

  Raises:
    ValueError: If the shape map contains an axis name not in the input tensor.
  "
  [ labeled_tensor shape_map seed name ]
  (py/call-attr ops "random_crop"  labeled_tensor shape_map seed name ))

(defn reduce-all 
  "Computes the given reduction across the given axes of a LabeledTensor.

    See `tf.reduce_all` for full details.

    Args:
      labeled_tensor: The input tensor.
      axes: A set of axes or None.
        If None, all axes will be reduced.
        Axes must all be strings, in which case those dimensions will be
        removed, or pairs of (name, None) or (name, label), in which case those
        dimensions will be kept.
      name: Optional op name.

    Returns:
      The reduced LabeledTensor.

    Raises:
      ValueError: if any of the axes to reduce over are not found on
        `labeled_tensor`.
    "
  [ labeled_tensor axes name ]
  (py/call-attr ops "reduce_all"  labeled_tensor axes name ))

(defn reduce-any 
  "Computes the given reduction across the given axes of a LabeledTensor.

    See `tf.reduce_any` for full details.

    Args:
      labeled_tensor: The input tensor.
      axes: A set of axes or None.
        If None, all axes will be reduced.
        Axes must all be strings, in which case those dimensions will be
        removed, or pairs of (name, None) or (name, label), in which case those
        dimensions will be kept.
      name: Optional op name.

    Returns:
      The reduced LabeledTensor.

    Raises:
      ValueError: if any of the axes to reduce over are not found on
        `labeled_tensor`.
    "
  [ labeled_tensor axes name ]
  (py/call-attr ops "reduce_any"  labeled_tensor axes name ))

(defn reduce-logsumexp 
  "Computes the given reduction across the given axes of a LabeledTensor.

    See `tf.reduce_logsumexp` for full details.

    Args:
      labeled_tensor: The input tensor.
      axes: A set of axes or None.
        If None, all axes will be reduced.
        Axes must all be strings, in which case those dimensions will be
        removed, or pairs of (name, None) or (name, label), in which case those
        dimensions will be kept.
      name: Optional op name.

    Returns:
      The reduced LabeledTensor.

    Raises:
      ValueError: if any of the axes to reduce over are not found on
        `labeled_tensor`.
    "
  [ labeled_tensor axes name ]
  (py/call-attr ops "reduce_logsumexp"  labeled_tensor axes name ))

(defn reduce-max 
  "Computes the given reduction across the given axes of a LabeledTensor.

    See `tf.reduce_max` for full details.

    Args:
      labeled_tensor: The input tensor.
      axes: A set of axes or None.
        If None, all axes will be reduced.
        Axes must all be strings, in which case those dimensions will be
        removed, or pairs of (name, None) or (name, label), in which case those
        dimensions will be kept.
      name: Optional op name.

    Returns:
      The reduced LabeledTensor.

    Raises:
      ValueError: if any of the axes to reduce over are not found on
        `labeled_tensor`.
    "
  [ labeled_tensor axes name ]
  (py/call-attr ops "reduce_max"  labeled_tensor axes name ))

(defn reduce-mean 
  "Computes the given reduction across the given axes of a LabeledTensor.

    See `tf.reduce_mean` for full details.

    Args:
      labeled_tensor: The input tensor.
      axes: A set of axes or None.
        If None, all axes will be reduced.
        Axes must all be strings, in which case those dimensions will be
        removed, or pairs of (name, None) or (name, label), in which case those
        dimensions will be kept.
      name: Optional op name.

    Returns:
      The reduced LabeledTensor.

    Raises:
      ValueError: if any of the axes to reduce over are not found on
        `labeled_tensor`.
    "
  [ labeled_tensor axes name ]
  (py/call-attr ops "reduce_mean"  labeled_tensor axes name ))

(defn reduce-min 
  "Computes the given reduction across the given axes of a LabeledTensor.

    See `tf.reduce_min` for full details.

    Args:
      labeled_tensor: The input tensor.
      axes: A set of axes or None.
        If None, all axes will be reduced.
        Axes must all be strings, in which case those dimensions will be
        removed, or pairs of (name, None) or (name, label), in which case those
        dimensions will be kept.
      name: Optional op name.

    Returns:
      The reduced LabeledTensor.

    Raises:
      ValueError: if any of the axes to reduce over are not found on
        `labeled_tensor`.
    "
  [ labeled_tensor axes name ]
  (py/call-attr ops "reduce_min"  labeled_tensor axes name ))

(defn reduce-prod 
  "Computes the given reduction across the given axes of a LabeledTensor.

    See `tf.reduce_prod` for full details.

    Args:
      labeled_tensor: The input tensor.
      axes: A set of axes or None.
        If None, all axes will be reduced.
        Axes must all be strings, in which case those dimensions will be
        removed, or pairs of (name, None) or (name, label), in which case those
        dimensions will be kept.
      name: Optional op name.

    Returns:
      The reduced LabeledTensor.

    Raises:
      ValueError: if any of the axes to reduce over are not found on
        `labeled_tensor`.
    "
  [ labeled_tensor axes name ]
  (py/call-attr ops "reduce_prod"  labeled_tensor axes name ))

(defn reduce-sum 
  "Computes the given reduction across the given axes of a LabeledTensor.

    See `tf.reduce_sum` for full details.

    Args:
      labeled_tensor: The input tensor.
      axes: A set of axes or None.
        If None, all axes will be reduced.
        Axes must all be strings, in which case those dimensions will be
        removed, or pairs of (name, None) or (name, label), in which case those
        dimensions will be kept.
      name: Optional op name.

    Returns:
      The reduced LabeledTensor.

    Raises:
      ValueError: if any of the axes to reduce over are not found on
        `labeled_tensor`.
    "
  [ labeled_tensor axes name ]
  (py/call-attr ops "reduce_sum"  labeled_tensor axes name ))

(defn rename-axis 
  "Rename an axis of LabeledTensor.

  Args:
    labeled_tensor: The input tensor.
    existing_name: Name for an existing axis on the input.
    new_name: Desired replacement name.
    name: Optional op name.

  Returns:
    LabeledTensor with renamed axis.

  Raises:
    ValueError: If `existing_name` is not an axis on the input.
  "
  [ labeled_tensor existing_name new_name name ]
  (py/call-attr ops "rename_axis"  labeled_tensor existing_name new_name name ))

(defn reshape 
  "Reshape specific axes of a LabeledTensor.

  Non-indicated axes remain in their original locations.

  Args:
    labeled_tensor: The input tensor.
    existing_axes: List of axis names found on the input tensor. These must
      appear sequentially in the list of axis names on the input. In other
      words, they must be a valid slice of `list(labeled_tensor.axes.keys())`.
    new_axes: List of strings, tuples of (axis_name, axis_value) or Axis objects
      providing new axes with which to replace `existing_axes` in the reshaped
      result. At most one element of `new_axes` may be a string, indicating an
      axis with unknown size.
    name: Optional op name.

  Returns:
    The reshaped LabeledTensor.

  Raises:
    ValueError: If `existing_axes` are not all axes on the input, or if more
     than one of `new_axes` has unknown size.
    AxisOrderError: If `existing_axes` are not a slice of axis names on the
      input.
  "
  [ labeled_tensor existing_axes new_axes name ]
  (py/call-attr ops "reshape"  labeled_tensor existing_axes new_axes name ))

(defn select 
  "Slice out a subset of the tensor.

  Args:
    labeled_tensor: The input tensor.
    selection: A dictionary mapping an axis name to a scalar, slice or list of
      values to select. Currently supports two types of selections:
        (a) Any number of scalar and/or slice selections.
        (b) Exactly one list selection, without any scalars or slices.
    name: Optional op name.

  Returns:
    The selection as a `LabeledTensor`.

  Raises:
    ValueError: If the tensor doesn't have an axis in the selection or if
      that axis lacks labels.
    KeyError: If any labels in a selection are not found in the original axis.
    NotImplementedError: If you attempt to combine a list selection with
      scalar selection or another list selection.
  "
  [ labeled_tensor selection name ]
  (py/call-attr ops "select"  labeled_tensor selection name ))

(defn shuffle-batch 
  "Rebatch a tensor, with shuffling.

  See tf.batch.

  Args:
    labeled_tensors: The input tensors.
    batch_size: The output batch size.
    num_threads: See tf.batch.
    capacity: See tf.batch.
    enqueue_many: If true, the input tensors must contain a 'batch' axis as
      their first axis.
      If false, the input tensors must not contain a 'batch' axis.
      See tf.batch.
    min_after_dequeue: Minimum number of elements in the queue after a dequeue,
      used to ensure mixing.
    seed: Optional random seed.
    allow_smaller_final_batch: See tf.batch.
    name: Optional op name.

  Returns:
    The rebatched tensors.
    If enqueue_many is false, the output tensors will have a new 'batch' axis
      as their first axis.

  Raises:
    ValueError: If enqueue_many is True and the first axis of the tensors
      isn't \"batch\".
  "
  [labeled_tensors batch_size & {:keys [num_threads capacity enqueue_many min_after_dequeue seed allow_smaller_final_batch name]
                       :or {seed None name None}} ]
    (py/call-attr-kw ops "shuffle_batch" [labeled_tensors batch_size] {:num_threads num_threads :capacity capacity :enqueue_many enqueue_many :min_after_dequeue min_after_dequeue :seed seed :allow_smaller_final_batch allow_smaller_final_batch :name name }))

(defn squeeze 
  "Remove size-1 dimensions.

  See tf.squeeze.

  Args:
    labeled_tensor: The input tensor.
    axis_names: The names of the dimensions to remove, or None to remove
      all size-1 dimensions.
    name: Optional op name.

  Returns:
    A tensor with the specified dimensions removed.

  Raises:
    ValueError: If the named axes are not in the tensor, or if they are
      not size-1.
  "
  [ labeled_tensor axis_names name ]
  (py/call-attr ops "squeeze"  labeled_tensor axis_names name ))

(defn tile 
  "Constructs a tensor by tiling a given tensor.

  Only axes without tick-labels can be tiled. (Otherwise, axis labels on tiled
  tensors would no longer be unique.)

  See lt.tile.

  Args:
    labeled_tensor: The input tensor.
    multiples: A mapping where the keys are axis names and the values are the
      integer number of times to tile along that axis. Only axes with a multiple
      different than 1 need be included.
    name: Optional op name.

  Returns:
    A tensor with the indicated axes tiled.

  Raises:
    ValueError: If the tiled axes are not axes in the input tensor, or if any
      axes in multiples have tick labels.
  "
  [ labeled_tensor multiples name ]
  (py/call-attr ops "tile"  labeled_tensor multiples name ))

(defn unpack 
  "Unpack the tensor.

  See tf.unpack.

  Args:
    labeled_tensor: The input tensor.
    axis_name: Optional name of axis to unpack. By default, the first axis is
      used.
    name: Optional op name.

  Returns:
    The list of unpacked LabeledTensors.

  Raises:
    ValueError: If `axis_name` is not an axis on the input.
  "
  [ labeled_tensor axis_name name ]
  (py/call-attr ops "unpack"  labeled_tensor axis_name name ))

(defn verify-tensor-all-finite 
  "Asserts a tensor doesn't contain NaNs or Infs.

  See tf.verify_tensor_all_finite.

  Args:
    labeled_tensor: The input tensor.
    message: Message to log on failure.
    name: Optional op name.

  Returns:
    The input tensor.
  "
  [ labeled_tensor message name ]
  (py/call-attr ops "verify_tensor_all_finite"  labeled_tensor message name ))

(defn where 
  "Return elements from x or y depending on condition.

  See `tf.where` for more details. This function currently only implements the
  three argument version of where.

  Args:
    condition: LabeledTensor of type `bool`.
    x: LabeledTensor for values where condition is true.
    y: LabeledTensor for values where condition is false.
    name: Optional op name.

  Returns:
    The labeled tensor with values according to condition.

  Raises:
    ValueError: if `x` and `y` have different axes, or if the axes of `x` do not
      start with the axes of `condition`.
  "
  [ condition x y name ]
  (py/call-attr ops "where"  condition x y name ))

(defn zeros-like 
  "Creates an identical tensor with all elements set to zero.

  Args:
    labeled_tensor: The input tensor.
    dtype: The type of the returned tensor.
    name: Optional op name.

  Returns:
    The tensor with elements set to zero.
  "
  [ labeled_tensor dtype name ]
  (py/call-attr ops "zeros_like"  labeled_tensor dtype name ))
