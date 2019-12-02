(ns tensorflow.-api.v1.compat.v2.RaggedTensor
  "Represents a ragged tensor.

  A `RaggedTensor` is a tensor with one or more *ragged dimensions*, which are
  dimensions whose slices may have different lengths.  For example, the inner
  (column) dimension of `rt=[[3, 1, 4, 1], [], [5, 9, 2], [6], []]` is ragged,
  since the column slices (`rt[0, :]`, ..., `rt[4, :]`) have different lengths.
  Dimensions whose slices all have the same length are called *uniform
  dimensions*.  The outermost dimension of a `RaggedTensor` is always uniform,
  since it consists of a single slice (and so there is no possibility for
  differing slice lengths).

  The total number of dimensions in a `RaggedTensor` is called its *rank*,
  and the number of ragged dimensions in a `RaggedTensor` is called its
  *ragged-rank*.  A `RaggedTensor`'s ragged-rank is fixed at graph creation
  time: it can't depend on the runtime values of `Tensor`s, and can't vary
  dynamically for different session runs.

  ### Potentially Ragged Tensors

  Many ops support both `Tensor`s and `RaggedTensor`s.  The term \"potentially
  ragged tensor\" may be used to refer to a tensor that might be either a
  `Tensor` or a `RaggedTensor`.  The ragged-rank of a `Tensor` is zero.

  ### Documenting RaggedTensor Shapes

  When documenting the shape of a RaggedTensor, ragged dimensions can be
  indicated by enclosing them in parentheses.  For example, the shape of
  a 3-D `RaggedTensor` that stores the fixed-size word embedding for each
  word in a sentence, for each sentence in a batch, could be written as
  `[num_sentences, (num_words), embedding_size]`.  The parentheses around
  `(num_words)` indicate that dimension is ragged, and that the length
  of each element list in that dimension may vary for each item.

  ### Component Tensors

  Internally, a `RaggedTensor` consists of a concatenated list of values that
  are partitioned into variable-length rows.  In particular, each `RaggedTensor`
  consists of:

    * A `values` tensor, which concatenates the variable-length rows into a
      flattened list.  For example, the `values` tensor for
      `[[3, 1, 4, 1], [], [5, 9, 2], [6], []]` is `[3, 1, 4, 1, 5, 9, 2, 6]`.

    * A `row_splits` vector, which indicates how those flattened values are
      divided into rows.  In particular, the values for row `rt[i]` are stored
      in the slice `rt.values[rt.row_splits[i]:rt.row_splits[i+1]]`.

  Example:

  ```python
  >>> print(tf.RaggedTensor.from_row_splits(
  ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
  ...     row_splits=[0, 4, 4, 7, 8, 8]))
  <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
  ```

  ### Alternative Row-Partitioning Schemes

  In addition to `row_splits`, ragged tensors provide support for four other
  row-partitioning schemes:

    * `row_lengths`: a vector with shape `[nrows]`, which specifies the length
      of each row.

    * `value_rowids` and `nrows`: `value_rowids` is a vector with shape
      `[nvals]`, corresponding one-to-one with `values`, which specifies
      each value's row index.  In particular, the row `rt[row]` consists of the
      values `rt.values[j]` where `value_rowids[j]==row`.  `nrows` is an
      integer scalar that specifies the number of rows in the
      `RaggedTensor`. (`nrows` is used to indicate trailing empty rows.)

    * `row_starts`: a vector with shape `[nrows]`, which specifies the start
      offset of each row.  Equivalent to `row_splits[:-1]`.

    * `row_limits`: a vector with shape `[nrows]`, which specifies the stop
      offset of each row.  Equivalent to `row_splits[1:]`.

  Example: The following ragged tensors are equivalent, and all represent the
  nested list `[[3, 1, 4, 1], [], [5, 9, 2], [6], []]`.

  ```python
  >>> values = [3, 1, 4, 1, 5, 9, 2, 6]
  >>> rt1 = RaggedTensor.from_row_splits(values, row_splits=[0, 4, 4, 7, 8, 8])
  >>> rt2 = RaggedTensor.from_row_lengths(values, row_lengths=[4, 0, 3, 1, 0])
  >>> rt3 = RaggedTensor.from_value_rowids(
  ...     values, value_rowids=[0, 0, 0, 0, 2, 2, 2, 3], nrows=5)
  >>> rt4 = RaggedTensor.from_row_starts(values, row_starts=[0, 4, 4, 7, 8])
  >>> rt5 = RaggedTensor.from_row_limits(values, row_limits=[4, 4, 7, 8, 8])
  ```

  ### Multiple Ragged Dimensions

  `RaggedTensor`s with multiple ragged dimensions can be defined by using
  a nested `RaggedTensor` for the `values` tensor.  Each nested `RaggedTensor`
  adds a single ragged dimension.

  ```python
  >>> inner_rt = RaggedTensor.from_row_splits(  # =rt1 from above
  ...     values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
  >>> outer_rt = RaggedTensor.from_row_splits(
  ...     values=inner_rt, row_splits=[0, 3, 3, 5])
  >>> print outer_rt.to_list()
  [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]]
  >>> print outer_rt.ragged_rank
  2
  ```

  The factory function `RaggedTensor.from_nested_row_splits` may be used to
  construct a `RaggedTensor` with multiple ragged dimensions directly, by
  providing a list of `row_splits` tensors:

  ```python
  >>> RaggedTensor.from_nested_row_splits(
  ...     flat_values=[3, 1, 4, 1, 5, 9, 2, 6],
  ...     nested_row_splits=([0, 3, 3, 5], [0, 4, 4, 7, 8, 8])).to_list()
  [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]]
  ```

  ### Uniform Inner Dimensions

  `RaggedTensor`s with uniform inner dimensions can be defined
  by using a multidimensional `Tensor` for `values`.

  ```python
  >>> rt = RaggedTensor.from_row_splits(values=tf.ones([5, 3]),
  ..                                    row_splits=[0, 2, 5])
  >>> print rt.to_list()
  [[[1, 1, 1], [1, 1, 1]],
   [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
   >>> print rt.shape
   (2, ?, 3)
  ```

  ### RaggedTensor Shape Restrictions

  The shape of a RaggedTensor is currently restricted to have the following
  form:

    * A single uniform dimension
    * Followed by one or more ragged dimensions
    * Followed by zero or more uniform dimensions.

  This restriction follows from the fact that each nested `RaggedTensor`
  replaces the uniform outermost dimension of its `values` with a uniform
  dimension followed by a ragged dimension.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v2 (import-module "tensorflow._api.v1.compat.v2"))
(defn RaggedTensor 
  "Represents a ragged tensor.

  A `RaggedTensor` is a tensor with one or more *ragged dimensions*, which are
  dimensions whose slices may have different lengths.  For example, the inner
  (column) dimension of `rt=[[3, 1, 4, 1], [], [5, 9, 2], [6], []]` is ragged,
  since the column slices (`rt[0, :]`, ..., `rt[4, :]`) have different lengths.
  Dimensions whose slices all have the same length are called *uniform
  dimensions*.  The outermost dimension of a `RaggedTensor` is always uniform,
  since it consists of a single slice (and so there is no possibility for
  differing slice lengths).

  The total number of dimensions in a `RaggedTensor` is called its *rank*,
  and the number of ragged dimensions in a `RaggedTensor` is called its
  *ragged-rank*.  A `RaggedTensor`'s ragged-rank is fixed at graph creation
  time: it can't depend on the runtime values of `Tensor`s, and can't vary
  dynamically for different session runs.

  ### Potentially Ragged Tensors

  Many ops support both `Tensor`s and `RaggedTensor`s.  The term \"potentially
  ragged tensor\" may be used to refer to a tensor that might be either a
  `Tensor` or a `RaggedTensor`.  The ragged-rank of a `Tensor` is zero.

  ### Documenting RaggedTensor Shapes

  When documenting the shape of a RaggedTensor, ragged dimensions can be
  indicated by enclosing them in parentheses.  For example, the shape of
  a 3-D `RaggedTensor` that stores the fixed-size word embedding for each
  word in a sentence, for each sentence in a batch, could be written as
  `[num_sentences, (num_words), embedding_size]`.  The parentheses around
  `(num_words)` indicate that dimension is ragged, and that the length
  of each element list in that dimension may vary for each item.

  ### Component Tensors

  Internally, a `RaggedTensor` consists of a concatenated list of values that
  are partitioned into variable-length rows.  In particular, each `RaggedTensor`
  consists of:

    * A `values` tensor, which concatenates the variable-length rows into a
      flattened list.  For example, the `values` tensor for
      `[[3, 1, 4, 1], [], [5, 9, 2], [6], []]` is `[3, 1, 4, 1, 5, 9, 2, 6]`.

    * A `row_splits` vector, which indicates how those flattened values are
      divided into rows.  In particular, the values for row `rt[i]` are stored
      in the slice `rt.values[rt.row_splits[i]:rt.row_splits[i+1]]`.

  Example:

  ```python
  >>> print(tf.RaggedTensor.from_row_splits(
  ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
  ...     row_splits=[0, 4, 4, 7, 8, 8]))
  <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
  ```

  ### Alternative Row-Partitioning Schemes

  In addition to `row_splits`, ragged tensors provide support for four other
  row-partitioning schemes:

    * `row_lengths`: a vector with shape `[nrows]`, which specifies the length
      of each row.

    * `value_rowids` and `nrows`: `value_rowids` is a vector with shape
      `[nvals]`, corresponding one-to-one with `values`, which specifies
      each value's row index.  In particular, the row `rt[row]` consists of the
      values `rt.values[j]` where `value_rowids[j]==row`.  `nrows` is an
      integer scalar that specifies the number of rows in the
      `RaggedTensor`. (`nrows` is used to indicate trailing empty rows.)

    * `row_starts`: a vector with shape `[nrows]`, which specifies the start
      offset of each row.  Equivalent to `row_splits[:-1]`.

    * `row_limits`: a vector with shape `[nrows]`, which specifies the stop
      offset of each row.  Equivalent to `row_splits[1:]`.

  Example: The following ragged tensors are equivalent, and all represent the
  nested list `[[3, 1, 4, 1], [], [5, 9, 2], [6], []]`.

  ```python
  >>> values = [3, 1, 4, 1, 5, 9, 2, 6]
  >>> rt1 = RaggedTensor.from_row_splits(values, row_splits=[0, 4, 4, 7, 8, 8])
  >>> rt2 = RaggedTensor.from_row_lengths(values, row_lengths=[4, 0, 3, 1, 0])
  >>> rt3 = RaggedTensor.from_value_rowids(
  ...     values, value_rowids=[0, 0, 0, 0, 2, 2, 2, 3], nrows=5)
  >>> rt4 = RaggedTensor.from_row_starts(values, row_starts=[0, 4, 4, 7, 8])
  >>> rt5 = RaggedTensor.from_row_limits(values, row_limits=[4, 4, 7, 8, 8])
  ```

  ### Multiple Ragged Dimensions

  `RaggedTensor`s with multiple ragged dimensions can be defined by using
  a nested `RaggedTensor` for the `values` tensor.  Each nested `RaggedTensor`
  adds a single ragged dimension.

  ```python
  >>> inner_rt = RaggedTensor.from_row_splits(  # =rt1 from above
  ...     values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
  >>> outer_rt = RaggedTensor.from_row_splits(
  ...     values=inner_rt, row_splits=[0, 3, 3, 5])
  >>> print outer_rt.to_list()
  [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]]
  >>> print outer_rt.ragged_rank
  2
  ```

  The factory function `RaggedTensor.from_nested_row_splits` may be used to
  construct a `RaggedTensor` with multiple ragged dimensions directly, by
  providing a list of `row_splits` tensors:

  ```python
  >>> RaggedTensor.from_nested_row_splits(
  ...     flat_values=[3, 1, 4, 1, 5, 9, 2, 6],
  ...     nested_row_splits=([0, 3, 3, 5], [0, 4, 4, 7, 8, 8])).to_list()
  [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]]
  ```

  ### Uniform Inner Dimensions

  `RaggedTensor`s with uniform inner dimensions can be defined
  by using a multidimensional `Tensor` for `values`.

  ```python
  >>> rt = RaggedTensor.from_row_splits(values=tf.ones([5, 3]),
  ..                                    row_splits=[0, 2, 5])
  >>> print rt.to_list()
  [[[1, 1, 1], [1, 1, 1]],
   [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
   >>> print rt.shape
   (2, ?, 3)
  ```

  ### RaggedTensor Shape Restrictions

  The shape of a RaggedTensor is currently restricted to have the following
  form:

    * A single uniform dimension
    * Followed by one or more ragged dimensions
    * Followed by zero or more uniform dimensions.

  This restriction follows from the fact that each nested `RaggedTensor`
  replaces the uniform outermost dimension of its `values` with a uniform
  dimension followed by a ragged dimension.
  "
  [values row_splits cached_row_lengths cached_value_rowids cached_nrows  & {:keys [internal]} ]
    (py/call-attr-kw v2 "RaggedTensor" [values row_splits cached_row_lengths cached_value_rowids cached_nrows] {:internal internal }))

(defn bounding-shape 
  "Returns the tight bounding box shape for this `RaggedTensor`.

    Args:
      axis: An integer scalar or vector indicating which axes to return the
        bounding box for.  If not specified, then the full bounding box is
        returned.
      name: A name prefix for the returned tensor (optional).
      out_type: `dtype` for the returned tensor.  Defaults to
        `self.row_splits.dtype`.

    Returns:
      An integer `Tensor` (`dtype=self.row_splits.dtype`).  If `axis` is not
      specified, then `output` is a vector with
      `output.shape=[self.shape.ndims]`.  If `axis` is a scalar, then the
      `output` is a scalar.  If `axis` is a vector, then `output` is a vector,
      where `output[i]` is the bounding size for dimension `axis[i]`.

    #### Example:
      ```python
      >>> rt = ragged.constant([[1, 2, 3, 4], [5], [], [6, 7, 8, 9], [10]])
      >>> rt.bounding_shape()
      [5, 4]
      ```
    "
  [ self axis name out_type ]
  (py/call-attr self "bounding_shape"  self axis name out_type ))

(defn consumers 
  ""
  [ self  ]
  (py/call-attr self "consumers"  self  ))

(defn dtype 
  "The `DType` of values in this tensor."
  [ self ]
    (py/call-attr self "dtype"))

(defn flat-values 
  "The innermost `values` tensor for this ragged tensor.

    Concretely, if `rt.values` is a `Tensor`, then `rt.flat_values` is
    `rt.values`; otherwise, `rt.flat_values` is `rt.values.flat_values`.

    Conceptually, `flat_values` is the tensor formed by flattening the
    outermost dimension and all of the ragged dimensions into a single
    dimension.

    `rt.flat_values.shape = [nvals] + rt.shape[rt.ragged_rank + 1:]`
    (where `nvals` is the number of items in the flattened dimensions).

    Returns:
      A `Tensor`.

    #### Example:

      ```python
      >>> rt = ragged.constant([[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]])
      >>> print rt.flat_values()
      tf.Tensor([3, 1, 4, 1, 5, 9, 2, 6])
      ```
    "
  [ self ]
    (py/call-attr self "flat_values"))

(defn nested-row-lengths 
  "Returns a tuple containing the row_lengths for all ragged dimensions.

    `rt.nested_row_lengths()` is a tuple containing the `row_lengths` tensors
    for all ragged dimensions in `rt`, ordered from outermost to innermost.

    Args:
      name: A name prefix for the returned tensors (optional).

    Returns:
      A `tuple` of 1-D integer `Tensors`.  The length of the tuple is equal to
      `self.ragged_rank`.
    "
  [ self name ]
  (py/call-attr self "nested_row_lengths"  self name ))

(defn nested-row-splits 
  "A tuple containing the row_splits for all ragged dimensions.

    `rt.nested_row_splits` is a tuple containing the `row_splits` tensors for
    all ragged dimensions in `rt`, ordered from outermost to innermost.  In
    particular, `rt.nested_row_splits = (rt.row_splits,) + value_splits` where:

        * `value_splits = ()` if `rt.values` is a `Tensor`.
        * `value_splits = rt.values.nested_row_splits` otherwise.

    Returns:
      A `tuple` of 1-D integer `Tensor`s.

    #### Example:

      ```python
      >>> rt = ragged.constant([[[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]]])
      >>> for i, splits in enumerate(rt.nested_row_splits()):
      ...   print('Splits for dimension %d: %s' % (i+1, splits))
      Splits for dimension 1: [0, 1]
      Splits for dimension 2: [0, 3, 3, 5]
      Splits for dimension 3: [0, 4, 4, 7, 8, 8]
      ```

    "
  [ self ]
    (py/call-attr self "nested_row_splits"))

(defn nested-value-rowids 
  "Returns a tuple containing the value_rowids for all ragged dimensions.

    `rt.nested_value_rowids` is a tuple containing the `value_rowids` tensors
    for
    all ragged dimensions in `rt`, ordered from outermost to innermost.  In
    particular, `rt.nested_value_rowids = (rt.value_rowids(),) + value_ids`
    where:

        * `value_ids = ()` if `rt.values` is a `Tensor`.
        * `value_ids = rt.values.nested_value_rowids` otherwise.

    Args:
      name: A name prefix for the returned tensors (optional).

    Returns:
      A `tuple` of 1-D integer `Tensor`s.

    #### Example:

      ```python
      >>> rt = ragged.constant([[[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]]])
      >>> for i, ids in enumerate(rt.nested_value_rowids()):
      ...   print('row ids for dimension %d: %s' % (i+1, ids))
      row ids for dimension 1: [0]
      row ids for dimension 2: [0, 0, 0, 2, 2]
      row ids for dimension 3: [0, 0, 0, 0, 2, 2, 2, 3]
      ```

    "
  [ self name ]
  (py/call-attr self "nested_value_rowids"  self name ))

(defn nrows 
  "Returns the number of rows in this ragged tensor.

    I.e., the size of the outermost dimension of the tensor.

    Args:
      out_type: `dtype` for the returned tensor.  Defaults to
        `self.row_splits.dtype`.
      name: A name prefix for the returned tensor (optional).

    Returns:
      A scalar `Tensor` with dtype `out_type`.

    #### Example:
      ```python
      >>> rt = ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
      >>> rt.nrows()  # rt has 5 rows.
      5
      ```
    "
  [ self out_type name ]
  (py/call-attr self "nrows"  self out_type name ))

(defn ragged-rank 
  "The number of ragged dimensions in this ragged tensor.

    Returns:
      A Python `int` indicating the number of ragged dimensions in this ragged
      tensor.  The outermost dimension is not considered ragged.
    "
  [ self ]
    (py/call-attr self "ragged_rank"))

(defn row-lengths 
  "Returns the lengths of the rows in this ragged tensor.

    `rt.row_lengths()[i]` indicates the number of values in the
    `i`th row of `rt`.

    Args:
      axis: An integer constant indicating the axis whose row lengths should be
        returned.
      name: A name prefix for the returned tensor (optional).

    Returns:
      A potentially ragged integer Tensor with shape `self.shape[:axis]`.

    Raises:
      ValueError: If `axis` is out of bounds.

    #### Example:
      ```python
      >>> rt = ragged.constant([[[3, 1, 4], [1]], [], [[5, 9], [2]], [[6]], []])
      >>> rt.row_lengths(rt)  # lengths of rows in rt
      tf.Tensor([2, 0, 2, 1, 0])
      >>> rt.row_lengths(axis=2)  # lengths of axis=2 rows.
      <tf.RaggedTensor [[3, 1], [], [2, 1], [1], []]>
      ```
    "
  [self  & {:keys [axis name]
                       :or {name None}} ]
    (py/call-attr-kw self "row_lengths" [] {:axis axis :name name }))

(defn row-limits 
  "Returns the limit indices for rows in this ragged tensor.

    These indices specify where the values for each row end in
    `self.values`.  `rt.row_limits(self)` is equal to `rt.row_splits[:-1]`.

    Args:
      name: A name prefix for the returned tensor (optional).

    Returns:
      A 1-D integer Tensor with shape `[nrows]`.
      The returned tensor is nonnegative, and is sorted in ascending order.

    #### Example:
      ```python
      >>> rt = ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
      >>> rt.values
      tf.Tensor([3, 1, 4, 1, 5, 9, 2, 6])
      >>> rt.row_limits()  # indices of row limits in rt.values
      tf.Tensor([4, 4, 7, 8, 8])
      ```
    "
  [ self name ]
  (py/call-attr self "row_limits"  self name ))

(defn row-splits 
  "The row-split indices for this ragged tensor's `values`.

    `rt.row_splits` specifies where the values for each row begin and end in
    `rt.values`.  In particular, the values for row `rt[i]` are stored in
    the slice `rt.values[rt.row_splits[i]:rt.row_splits[i+1]]`.

    Returns:
      A 1-D integer `Tensor` with shape `[self.nrows+1]`.
      The returned tensor is non-empty, and is sorted in ascending order.
      `self.row_splits[0]` is zero, and `self.row_splits[-1]` is equal to
      `self.values.shape[0]`.

    #### Example:
      ```python
      >>> rt = ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
      >>> print rt.row_splits  # indices of row splits in rt.values
      tf.Tensor([0, 4, 4, 7, 8, 8])
      ```
    "
  [ self ]
    (py/call-attr self "row_splits"))

(defn row-starts 
  "Returns the start indices for rows in this ragged tensor.

    These indices specify where the values for each row begin in
    `self.values`.  `rt.row_starts()` is equal to `rt.row_splits[:-1]`.

    Args:
      name: A name prefix for the returned tensor (optional).

    Returns:
      A 1-D integer Tensor with shape `[nrows]`.
      The returned tensor is nonnegative, and is sorted in ascending order.

    #### Example:
      ```python
      >>> rt = ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
      >>> rt.values
      tf.Tensor([3, 1, 4, 1, 5, 9, 2, 6])
      >>> rt.row_starts()  # indices of row starts in rt.values
      tf.Tensor([0, 4, 4, 7, 8])
      ```
    "
  [ self name ]
  (py/call-attr self "row_starts"  self name ))

(defn shape 
  "The statically known shape of this ragged tensor.

    Returns:
      A `TensorShape` containing the statically known shape of this ragged
      tensor.  Ragged dimensions have a size of `None`.

    Examples:

      ```python
      >>> ragged.constant([[0], [1, 2]]).shape
      TensorShape([Dimension(2), Dimension(None)])

      >>> ragged.constant([[[0, 1]], [[1, 2], [3, 4]]], ragged_rank=1).shape
      TensorShape([Dimension(2), Dimension(None), Dimension(2)
      ```
    "
  [ self ]
    (py/call-attr self "shape"))

(defn to-list 
  "Returns a nested Python `list` with the values for this `RaggedTensor`.

    Requires that `rt` was constructed in eager execution mode.

    Returns:
      A nested Python `list`.
    "
  [ self  ]
  (py/call-attr self "to_list"  self  ))

(defn to-sparse 
  "Converts this `RaggedTensor` into a `tf.SparseTensor`.

    Example:

    ```python
    >>> rt = ragged.constant([[1, 2, 3], [4], [], [5, 6]])
    >>> rt.to_sparse().eval()
    SparseTensorValue(indices=[[0, 0], [0, 1], [0, 2], [1, 0], [3, 0], [3, 1]],
                      values=[1, 2, 3, 4, 5, 6],
                      dense_shape=[4, 3])
    ```

    Args:
      name: A name prefix for the returned tensors (optional).

    Returns:
      A SparseTensor with the same values as `self`.
    "
  [ self name ]
  (py/call-attr self "to_sparse"  self name ))

(defn to-tensor 
  "Converts this `RaggedTensor` into a `tf.Tensor`.

    Example:

    ```python
    >>> rt = ragged.constant([[9, 8, 7], [], [6, 5], [4]])
    >>> print rt.to_tensor()
    [[9 8 7]
     [0 0 0]
     [6 5 0]
     [4 0 0]]
    ```

    Args:
      default_value: Value to set for indices not specified in `self`. Defaults
        to zero.  `default_value` must be broadcastable to
        `self.shape[self.ragged_rank + 1:]`.
      name: A name prefix for the returned tensors (optional).

    Returns:
      A `Tensor` with shape `ragged.bounding_shape(self)` and the
      values specified by the non-empty values in `self`.  Empty values are
      assigned `default_value`.
    "
  [ self default_value name ]
  (py/call-attr self "to_tensor"  self default_value name ))

(defn value-rowids 
  "Returns the row indices for the `values` in this ragged tensor.

    `rt.value_rowids()` corresponds one-to-one with the outermost dimension of
    `rt.values`, and specifies the row containing each value.  In particular,
    the row `rt[row]` consists of the values `rt.values[j]` where
    `rt.value_rowids()[j] == row`.

    Args:
      name: A name prefix for the returned tensor (optional).

    Returns:
      A 1-D integer `Tensor` with shape `self.values.shape[:1]`.
      The returned tensor is nonnegative, and is sorted in ascending order.

    #### Example:
      ```python
      >>> rt = ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
      >>> rt.values
      tf.Tensor([3, 1, 4, 1, 5, 9, 2, 6])
      >>> rt.value_rowids()
      tf.Tensor([0, 0, 0, 0, 2, 2, 2, 3])  # corresponds 1:1 with rt.values
      ```
    "
  [ self name ]
  (py/call-attr self "value_rowids"  self name ))

(defn values 
  "The concatenated rows for this ragged tensor.

    `rt.values` is a potentially ragged tensor formed by flattening the two
    outermost dimensions of `rt` into a single dimension.

    `rt.values.shape = [nvals] + rt.shape[2:]` (where `nvals` is the
    number of items in the outer two dimensions of `rt`).

    `rt.ragged_rank = self.ragged_rank - 1`

    Returns:
      A potentially ragged tensor.

    #### Example:
      ```python
      >>> rt = ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
      >>> print rt.values
      tf.Tensor([3, 1, 4, 1, 5, 9, 2, 6])
      ```
    "
  [ self ]
    (py/call-attr self "values"))

(defn with-flat-values 
  "Returns a copy of `self` with `flat_values` replaced by `new_value`.

    Preserves cached row-partitioning tensors such as `self.cached_nrows` and
    `self.cached_value_rowids` if they have values.

    Args:
      new_values: Potentially ragged tensor that should replace
      `self.flat_values`.  Must have `rank > 0`, and must have the same
      number of rows as `self.flat_values`.

    Returns:
      A `RaggedTensor`.
      `result.rank = self.ragged_rank + new_values.rank`.
      `result.ragged_rank = self.ragged_rank + new_values.ragged_rank`.
    "
  [ self new_values ]
  (py/call-attr self "with_flat_values"  self new_values ))

(defn with-row-splits-dtype 
  "Returns a copy of this RaggedTensor with the given `row_splits` dtype.

    For RaggedTensors with multiple ragged dimensions, the `row_splits` for all
    nested `RaggedTensor` objects are cast to the given dtype.

    Args:
      dtype: The dtype for `row_splits`.  One of `tf.int32` or `tf.int64`.

    Returns:
      A copy of this RaggedTensor, with the `row_splits` cast to the given
      type.
    "
  [ self dtype ]
  (py/call-attr self "with_row_splits_dtype"  self dtype ))

(defn with-values 
  "Returns a copy of `self` with `values` replaced by `new_value`.

    Preserves cached row-partitioning tensors such as `self.cached_nrows` and
    `self.cached_value_rowids` if they have values.

    Args:
      new_values: Potentially ragged tensor to use as the `values` for the
        returned `RaggedTensor`.  Must have `rank > 0`, and must have the same
        number of rows as `self.values`.

    Returns:
      A `RaggedTensor`.  `result.rank = 1 + new_values.rank`.
      `result.ragged_rank = 1 + new_values.ragged_rank`
    "
  [ self new_values ]
  (py/call-attr self "with_values"  self new_values ))
