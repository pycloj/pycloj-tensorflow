(ns tensorflow.-api.v1.compat.v2.ragged
  "Ragged Tensors.

This package defines ops for manipulating ragged tensors (`tf.RaggedTensor`),
which are tensors with non-uniform shapes.  In particular, each `RaggedTensor`
has one or more *ragged dimensions*, which are dimensions whose slices may have
different lengths.  For example, the inner (column) dimension of
`rt=[[3, 1, 4, 1], [], [5, 9, 2], [6], []]` is ragged, since the column slices
(`rt[0, :]`, ..., `rt[4, :]`) have different lengths.  For a more detailed
description of ragged tensors, see the `tf.RaggedTensor` class documentation
and the [Ragged Tensor Guide](/guide/ragged_tensors).


### Additional ops that support `RaggedTensor`

Arguments that accept `RaggedTensor`s are marked in **bold**.

* `tf.batch_gather`(**params**, **indices**, name=`None`)
* `tf.bitwise.bitwise_and`(**x**, **y**, name=`None`)
* `tf.bitwise.bitwise_or`(**x**, **y**, name=`None`)
* `tf.bitwise.bitwise_xor`(**x**, **y**, name=`None`)
* `tf.bitwise.invert`(**x**, name=`None`)
* `tf.bitwise.left_shift`(**x**, **y**, name=`None`)
* `tf.bitwise.right_shift`(**x**, **y**, name=`None`)
* `tf.cast`(**x**, dtype, name=`None`)
* `tf.clip_by_value`(**t**, clip_value_min, clip_value_max, name=`None`)
* `tf.concat`(**values**, axis, name=`'concat'`)
* `tf.debugging.check_numerics`(**tensor**, message, name=`None`)
* `tf.dtypes.complex`(**real**, **imag**, name=`None`)
* `tf.dtypes.saturate_cast`(**value**, dtype, name=`None`)
* `tf.dynamic_partition`(**data**, **partitions**, num_partitions, name=`None`)
* `tf.expand_dims`(**input**, axis=`None`, name=`None`, dim=`None`)
* `tf.gather_nd`(**params**, **indices**, name=`None`, batch_dims=`0`)
* `tf.gather`(**params**, **indices**, validate_indices=`None`, name=`None`, axis=`None`, batch_dims=`0`)
* `tf.identity`(**input**, name=`None`)
* `tf.io.decode_base64`(**input**, name=`None`)
* `tf.io.decode_compressed`(**bytes**, compression_type=`''`, name=`None`)
* `tf.io.encode_base64`(**input**, pad=`False`, name=`None`)
* `tf.math.abs`(**x**, name=`None`)
* `tf.math.acos`(**x**, name=`None`)
* `tf.math.acosh`(**x**, name=`None`)
* `tf.math.add_n`(**inputs**, name=`None`)
* `tf.math.add`(**x**, **y**, name=`None`)
* `tf.math.angle`(**input**, name=`None`)
* `tf.math.asin`(**x**, name=`None`)
* `tf.math.asinh`(**x**, name=`None`)
* `tf.math.atan2`(**y**, **x**, name=`None`)
* `tf.math.atan`(**x**, name=`None`)
* `tf.math.atanh`(**x**, name=`None`)
* `tf.math.ceil`(**x**, name=`None`)
* `tf.math.conj`(**x**, name=`None`)
* `tf.math.cos`(**x**, name=`None`)
* `tf.math.cosh`(**x**, name=`None`)
* `tf.math.digamma`(**x**, name=`None`)
* `tf.math.divide_no_nan`(**x**, **y**, name=`None`)
* `tf.math.divide`(**x**, **y**, name=`None`)
* `tf.math.equal`(**x**, **y**, name=`None`)
* `tf.math.erf`(**x**, name=`None`)
* `tf.math.erfc`(**x**, name=`None`)
* `tf.math.exp`(**x**, name=`None`)
* `tf.math.expm1`(**x**, name=`None`)
* `tf.math.floor`(**x**, name=`None`)
* `tf.math.floordiv`(**x**, **y**, name=`None`)
* `tf.math.floormod`(**x**, **y**, name=`None`)
* `tf.math.greater_equal`(**x**, **y**, name=`None`)
* `tf.math.greater`(**x**, **y**, name=`None`)
* `tf.math.imag`(**input**, name=`None`)
* `tf.math.is_finite`(**x**, name=`None`)
* `tf.math.is_inf`(**x**, name=`None`)
* `tf.math.is_nan`(**x**, name=`None`)
* `tf.math.less_equal`(**x**, **y**, name=`None`)
* `tf.math.less`(**x**, **y**, name=`None`)
* `tf.math.lgamma`(**x**, name=`None`)
* `tf.math.log1p`(**x**, name=`None`)
* `tf.math.log_sigmoid`(**x**, name=`None`)
* `tf.math.log`(**x**, name=`None`)
* `tf.math.logical_and`(**x**, **y**, name=`None`)
* `tf.math.logical_not`(**x**, name=`None`)
* `tf.math.logical_or`(**x**, **y**, name=`None`)
* `tf.math.logical_xor`(**x**, **y**, name=`'LogicalXor'`)
* `tf.math.maximum`(**x**, **y**, name=`None`)
* `tf.math.minimum`(**x**, **y**, name=`None`)
* `tf.math.multiply`(**x**, **y**, name=`None`)
* `tf.math.negative`(**x**, name=`None`)
* `tf.math.not_equal`(**x**, **y**, name=`None`)
* `tf.math.pow`(**x**, **y**, name=`None`)
* `tf.math.real`(**input**, name=`None`)
* `tf.math.reciprocal`(**x**, name=`None`)
* `tf.math.reduce_any`(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* `tf.math.reduce_max`(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* `tf.math.reduce_mean`(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* `tf.math.reduce_min`(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* `tf.math.reduce_prod`(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* `tf.math.reduce_sum`(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* `tf.math.rint`(**x**, name=`None`)
* `tf.math.round`(**x**, name=`None`)
* `tf.math.rsqrt`(**x**, name=`None`)
* `tf.math.sign`(**x**, name=`None`)
* `tf.math.sin`(**x**, name=`None`)
* `tf.math.sinh`(**x**, name=`None`)
* `tf.math.sqrt`(**x**, name=`None`)
* `tf.math.square`(**x**, name=`None`)
* `tf.math.squared_difference`(**x**, **y**, name=`None`)
* `tf.math.subtract`(**x**, **y**, name=`None`)
* `tf.math.tan`(**x**, name=`None`)
* `tf.math.truediv`(**x**, **y**, name=`None`)
* `tf.math.unsorted_segment_max`(**data**, **segment_ids**, num_segments, name=`None`)
* `tf.math.unsorted_segment_mean`(**data**, **segment_ids**, num_segments, name=`None`)
* `tf.math.unsorted_segment_min`(**data**, **segment_ids**, num_segments, name=`None`)
* `tf.math.unsorted_segment_prod`(**data**, **segment_ids**, num_segments, name=`None`)
* `tf.math.unsorted_segment_sqrt_n`(**data**, **segment_ids**, num_segments, name=`None`)
* `tf.math.unsorted_segment_sum`(**data**, **segment_ids**, num_segments, name=`None`)
* `tf.one_hot`(**indices**, depth, on_value=`None`, off_value=`None`, axis=`None`, dtype=`None`, name=`None`)
* `tf.ones_like`(**tensor**, dtype=`None`, name=`None`, optimize=`True`)
* `tf.rank`(**input**, name=`None`)
* `tf.realdiv`(**x**, **y**, name=`None`)
* `tf.reduce_all`(**input_tensor**, axis=`None`, keepdims=`False`, name=`None`)
* `tf.size`(**input**, name=`None`, out_type=`tf.int32`)
* `tf.squeeze`(**input**, axis=`None`, name=`None`, squeeze_dims=`None`)
* `tf.stack`(**values**, axis=`0`, name=`'stack'`)
* `tf.strings.as_string`(**input**, precision=`-1`, scientific=`False`, shortest=`False`, width=`-1`, fill=`''`, name=`None`)
* `tf.strings.join`(**inputs**, separator=`''`, name=`None`)
* `tf.strings.length`(**input**, name=`None`, unit=`'BYTE'`)
* `tf.strings.reduce_join`(**inputs**, axis=`None`, keepdims=`False`, separator=`''`, name=`None`)
* `tf.strings.regex_full_match`(**input**, pattern, name=`None`)
* `tf.strings.regex_replace`(**input**, pattern, rewrite, replace_global=`True`, name=`None`)
* `tf.strings.strip`(**input**, name=`None`)
* `tf.strings.substr`(**input**, pos, len, name=`None`, unit=`'BYTE'`)
* `tf.strings.to_hash_bucket_fast`(**input**, num_buckets, name=`None`)
* `tf.strings.to_hash_bucket_strong`(**input**, num_buckets, key, name=`None`)
* `tf.strings.to_hash_bucket`(**input**, num_buckets, name=`None`)
* `tf.strings.to_hash_bucket`(**input**, num_buckets, name=`None`)
* `tf.strings.to_number`(**input**, out_type=`tf.float32`, name=`None`)
* `tf.strings.unicode_script`(**input**, name=`None`)
* `tf.tile`(**input**, multiples, name=`None`)
* `tf.truncatediv`(**x**, **y**, name=`None`)
* `tf.truncatemod`(**x**, **y**, name=`None`)
* `tf.where`(**condition**, **x**=`None`, **y**=`None`, name=`None`)
* `tf.zeros_like`(**tensor**, dtype=`None`, name=`None`, optimize=`True`)n
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce ragged (import-module "tensorflow._api.v1.compat.v2.ragged"))

(defn boolean-mask 
  "Applies a boolean mask to `data` without flattening the mask dimensions.

  Returns a potentially ragged tensor that is formed by retaining the elements
  in `data` where the corresponding value in `mask` is `True`.

  * `output[a1...aA, i, b1...bB] = data[a1...aA, j, b1...bB]`

     Where `j` is the `i`th `True` entry of `mask[a1...aA]`.

  Note that `output` preserves the mask dimensions `a1...aA`; this differs
  from `tf.boolean_mask`, which flattens those dimensions.

  Args:
    data: A potentially ragged tensor.
    mask: A potentially ragged boolean tensor.  `mask`'s shape must be a prefix
      of `data`'s shape.  `rank(mask)` must be known statically.
    name: A name prefix for the returned tensor (optional).

  Returns:
    A potentially ragged tensor that is formed by retaining the elements in
    `data` where the corresponding value in `mask` is `True`.

    * `rank(output) = rank(data)`.
    * `output.ragged_rank = max(data.ragged_rank, rank(mask) - 1)`.

  Raises:
    ValueError: if `rank(mask)` is not known statically; or if `mask.shape` is
      not a prefix of `data.shape`.

  #### Examples:
    ```python
    >>> # Aliases for True & False so data and mask line up.
    >>> T, F = (True, False)

    >>> tf.ragged.boolean_mask(  # Mask a 2D Tensor.
    ...     data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ...     mask=[[T, F, T], [F, F, F], [T, F, F]]).tolist()
    [[1, 3], [], [7]]

    >>> tf.ragged.boolean_mask(  # Mask a 2D RaggedTensor.
    ...     tf.ragged.constant([[1, 2, 3], [4], [5, 6]]),
    ...     tf.ragged.constant([[F, F, T], [F], [T, T]])).tolist()
    [[3], [], [5, 6]]

    >>> tf.ragged.boolean_mask(  # Mask rows of a 2D RaggedTensor.
    ...     tf.ragged.constant([[1, 2, 3], [4], [5, 6]]),
    ...     tf.ragged.constant([True, False, True])).tolist()
    [[1, 2, 3], [5, 6]]
    ```
  "
  [ data mask name ]
  (py/call-attr ragged "boolean_mask"  data mask name ))
(defn constant 
  "Constructs a constant RaggedTensor from a nested Python list.

  Example:

  ```python
  >>> ragged.constant([[1, 2], [3], [4, 5, 6]]).eval()
  RaggedTensorValue(values=[1, 2, 3, 4, 5, 6], splits=[0, 2, 3, 6])
  ```

  All scalar values in `pylist` must have the same nesting depth `K`, and the
  returned `RaggedTensor` will have rank `K`.  If `pylist` contains no scalar
  values, then `K` is one greater than the maximum depth of empty lists in
  `pylist`.  All scalar values in `pylist` must be compatible with `dtype`.

  Args:
    pylist: A nested `list`, `tuple` or `np.ndarray`.  Any nested element that
      is not a `list`, `tuple` or `np.ndarray` must be a scalar value
      compatible with `dtype`.
    dtype: The type of elements for the returned `RaggedTensor`.  If not
      specified, then a default is chosen based on the scalar values in
      `pylist`.
    ragged_rank: An integer specifying the ragged rank of the returned
      `RaggedTensor`.  Must be nonnegative and less than `K`. Defaults to
      `max(0, K - 1)` if `inner_shape` is not specified.  Defaults to `max(0, K
      - 1 - len(inner_shape))` if `inner_shape` is specified.
    inner_shape: A tuple of integers specifying the shape for individual inner
      values in the returned `RaggedTensor`.  Defaults to `()` if `ragged_rank`
      is not specified.  If `ragged_rank` is specified, then a default is chosen
      based on the contents of `pylist`.
    name: A name prefix for the returned tensor (optional).
    row_splits_dtype: data type for the constructed `RaggedTensor`'s row_splits.
      One of `tf.int32` or `tf.int64`.

  Returns:
    A potentially ragged tensor with rank `K` and the specified `ragged_rank`,
    containing the values from `pylist`.

  Raises:
    ValueError: If the scalar values in `pylist` have inconsistent nesting
      depth; or if ragged_rank or inner_shape are incompatible with `pylist`.
  "
  [pylist dtype ragged_rank inner_shape name  & {:keys [row_splits_dtype]} ]
    (py/call-attr-kw ragged "constant" [pylist dtype ragged_rank inner_shape name] {:row_splits_dtype row_splits_dtype }))

(defn map-flat-values 
  "Applies `op` to the values of one or more RaggedTensors.

  Replaces any `RaggedTensor` in `args` or `kwargs` with its `flat_values`
  tensor, and then calls `op`.  Returns a `RaggedTensor` that is constructed
  from the input `RaggedTensor`s' `nested_row_splits` and the value returned by
  the `op`.

  If the input arguments contain multiple `RaggedTensor`s, then they must have
  identical `nested_row_splits`.

  Examples:

  ```python
  >>> rt = ragged.constant([[1, 2, 3], [], [4, 5], [6]])
  >>> ragged.map_flat_values(tf.ones_like, rt).eval().tolist()
  [[1, 1, 1], [], [1, 1], [1]]
  >>> ragged.map_flat_values(tf.multiply, rt, rt).eval().tolist()
  [[1, 4, 9], [], [16, 25], [36]]
  >>> ragged.map_flat_values(tf.add, rt, 5).eval().tolist()
  [[6, 7, 8], [], [9, 10], [11]]
  ```

  Args:
    op: The operation that should be applied to the RaggedTensor `flat_values`.
      `op` is typically an element-wise operation (such as math_ops.add), but
      any operation that preserves the size of the outermost dimension can be
      used.  I.e., `shape[0]` of the value returned by `op` must match
      `shape[0]` of the `RaggedTensor`s' `flat_values` tensors.
    *args: Arguments for `op`.
    **kwargs: Keyword arguments for `op`.

  Returns:
    A `RaggedTensor` whose `ragged_rank` matches the `ragged_rank` of all
    input `RaggedTensor`s.
  Raises:
    ValueError: If args contains no `RaggedTensors`, or if the `nested_splits`
      of the input `RaggedTensor`s are not identical.
  "
  [ op ]
  (py/call-attr ragged "map_flat_values"  op ))

(defn range 
  "Returns a `RaggedTensor` containing the specified sequences of numbers.

  Each row of the returned `RaggedTensor` contains a single sequence:

  ```python
  ragged.range(starts, limits, deltas)[i] ==
      tf.range(starts[i], limits[i], deltas[i])
  ```

  If `start[i] < limits[i] and deltas[i] > 0`, then `output[i]` will be an
  empty list.  Similarly, if `start[i] > limits[i] and deltas[i] < 0`, then
  `output[i]` will be an empty list.  This behavior is consistent with the
  Python `range` function, but differs from the `tf.range` op, which returns
  an error for these cases.

  Examples:

  ```python
  >>> ragged.range([3, 5, 2]).eval().tolist()
  [[0, 1, 2], [0, 1, 2, 3, 4], [0, 1]]
  >>> ragged.range([0, 5, 8], [3, 3, 12]).eval().tolist()
  [[0, 1, 2], [], [8, 9, 10, 11]]
  >>> ragged.range([0, 5, 8], [3, 3, 12], 2).eval().tolist()
  [[0, 2], [], [8, 10]]
  ```

  The input tensors `starts`, `limits`, and `deltas` may be scalars or vectors.
  The vector inputs must all have the same size.  Scalar inputs are broadcast
  to match the size of the vector inputs.

  Args:
    starts: Vector or scalar `Tensor`.  Specifies the first entry for each range
      if `limits` is not `None`; otherwise, specifies the range limits, and the
      first entries default to `0`.
    limits: Vector or scalar `Tensor`.  Specifies the exclusive upper limits for
      each range.
    deltas: Vector or scalar `Tensor`.  Specifies the increment for each range.
      Defaults to `1`.
    dtype: The type of the elements of the resulting tensor.  If not specified,
      then a value is chosen based on the other args.
    name: A name for the operation.
    row_splits_dtype: `dtype` for the returned `RaggedTensor`'s `row_splits`
      tensor.  One of `tf.int32` or `tf.int64`.

  Returns:
    A `RaggedTensor` of type `dtype` with `ragged_rank=1`.
  "
  [starts limits & {:keys [deltas dtype name row_splits_dtype]
                       :or {dtype None name None}} ]
    (py/call-attr-kw ragged "range" [starts limits] {:deltas deltas :dtype dtype :name name :row_splits_dtype row_splits_dtype }))

(defn row-splits-to-segment-ids 
  "Generates the segmentation corresponding to a RaggedTensor `row_splits`.

  Returns an integer vector `segment_ids`, where `segment_ids[i] == j` if
  `splits[j] <= i < splits[j+1]`.  Example:

  ```python
  >>> ragged.row_splits_to_segment_ids([0, 3, 3, 5, 6, 9]).eval()
  [ 0 0 0 2 2 3 4 4 4 ]
  ```

  Args:
    splits: A sorted 1-D integer Tensor.  `splits[0]` must be zero.
    name: A name prefix for the returned tensor (optional).
    out_type: The dtype for the return value.  Defaults to `splits.dtype`,
      or `tf.int64` if `splits` does not have a dtype.

  Returns:
    A sorted 1-D integer Tensor, with `shape=[splits[-1]]`

  Raises:
    ValueError: If `splits` is invalid.
  "
  [ splits name out_type ]
  (py/call-attr ragged "row_splits_to_segment_ids"  splits name out_type ))

(defn segment-ids-to-row-splits 
  "Generates the RaggedTensor `row_splits` corresponding to a segmentation.

  Returns an integer vector `splits`, where `splits[0] = 0` and
  `splits[i] = splits[i-1] + count(segment_ids==i)`.  Example:

  ```python
  >>> ragged.segment_ids_to_row_splits([0, 0, 0, 2, 2, 3, 4, 4, 4]).eval()
  [ 0 3 3 5 6 9 ]
  ```

  Args:
    segment_ids: A 1-D integer Tensor.
    num_segments: A scalar integer indicating the number of segments.  Defaults
      to `max(segment_ids) + 1` (or zero if `segment_ids` is empty).
    out_type: The dtype for the return value.  Defaults to `segment_ids.dtype`,
      or `tf.int64` if `segment_ids` does not have a dtype.
    name: A name prefix for the returned tensor (optional).

  Returns:
    A sorted 1-D integer Tensor, with `shape=[num_segments + 1]`.
  "
  [ segment_ids num_segments out_type name ]
  (py/call-attr ragged "segment_ids_to_row_splits"  segment_ids num_segments out_type name ))

(defn stack 
  "Stacks a list of rank-`R` tensors into one rank-`(R+1)` `RaggedTensor`.

  Given a list of tensors or ragged tensors with the same rank `R`
  (`R >= axis`), returns a rank-`R+1` `RaggedTensor` `result` such that
  `result[i0...iaxis]` is `[value[i0...iaxis] for value in values]`.

  #### Example:
    ```python
    >>> t1 = tf.ragged.constant([[1, 2], [3, 4, 5]])
    >>> t2 = tf.ragged.constant([[6], [7, 8, 9]])
    >>> tf.ragged.stack([t1, t2], axis=0)
    [[[1, 2], [3, 4, 5]], [[6], [7, 9, 0]]]
    >>> tf.ragged.stack([t1, t2], axis=1)
    [[[1, 2], [6]], [[3, 4, 5], [7, 8, 9]]]
    ```

  Args:
    values: A list of `tf.Tensor` or `tf.RaggedTensor`.  May not be empty. All
      `values` must have the same rank and the same dtype; but unlike
      `tf.stack`, they can have arbitrary dimension sizes.
    axis: A python integer, indicating the dimension along which to stack.
      (Note: Unlike `tf.stack`, the `axis` parameter must be statically known.)
      Negative values are supported only if the rank of at least one
      `values` value is statically known.
    name: A name prefix for the returned tensor (optional).

  Returns:
    A `RaggedTensor` with rank `R+1`.
    `result.ragged_rank=1+max(axis, max(rt.ragged_rank for rt in values]))`.

  Raises:
    ValueError: If `values` is empty, if `axis` is out of bounds or if
      the input tensors have different ranks.
  "
  [values & {:keys [axis name]
                       :or {name None}} ]
    (py/call-attr-kw ragged "stack" [values] {:axis axis :name name }))

(defn stack-dynamic-partitions 
  "Stacks dynamic partitions of a Tensor or RaggedTensor.

  Returns a RaggedTensor `output` with `num_partitions` rows, where the row
  `output[i]` is formed by stacking all slices `data[j1...jN]` such that
  `partitions[j1...jN] = i`.  Slices of `data` are stacked in row-major
  order.

  If `num_partitions` is an `int` (not a `Tensor`), then this is equivalent to
  `tf.ragged.stack(tf.dynamic_partition(data, partitions, num_partitions))`.

  ####Example:
    ```python
    >>> data           = ['a', 'b', 'c', 'd', 'e']
    >>> partitions     = [  3,   0,   2,   2,   3]
    >>> num_partitions = 5
    >>> tf.ragged.stack_dynamic_partitions(data, partitions, num_partitions)
    <RaggedTensor [['b'], [], ['c', 'd'], ['a', 'e'], []]>
    ```

  Args:
    data: A `Tensor` or `RaggedTensor` containing the values to stack.
    partitions: An `int32` or `int64` `Tensor` or `RaggedTensor` specifying the
      partition that each slice of `data` should be added to.
      `partitions.shape` must be a prefix of `data.shape`.  Values must be
      greater than or equal to zero, and less than `num_partitions`.
      `partitions` is not required to be sorted.
    num_partitions: An `int32` or `int64` scalar specifying the number of
      partitions to output.  This determines the number of rows in `output`.
    name: A name prefix for the returned tensor (optional).

  Returns:
    A `RaggedTensor` containing the stacked partitions.  The returned tensor
    has the same dtype as `data`, and its shape is
    `[num_partitions, (D)] + data.shape[partitions.rank:]`, where `(D)` is a
    ragged dimension whose length is the number of data slices stacked for
    each `partition`.
  "
  [ data partitions num_partitions name ]
  (py/call-attr ragged "stack_dynamic_partitions"  data partitions num_partitions name ))
