(ns tensorflow.-api.v1.compat.v1.sets
  "Tensorflow set operations.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce sets (import-module "tensorflow._api.v1.compat.v1.sets"))
(defn difference 
  "Compute set difference of elements in last dimension of `a` and `b`.

  All but the last dimension of `a` and `b` must match.

  Example:

  ```python
    import tensorflow as tf
    import collections

    # Represent the following array of sets as a sparse tensor:
    # a = np.array([[{1, 2}, {3}], [{4}, {5, 6}]])
    a = collections.OrderedDict([
        ((0, 0, 0), 1),
        ((0, 0, 1), 2),
        ((0, 1, 0), 3),
        ((1, 0, 0), 4),
        ((1, 1, 0), 5),
        ((1, 1, 1), 6),
    ])
    a = tf.SparseTensor(list(a.keys()), list(a.values()), dense_shape=[2, 2, 2])

    # np.array([[{1, 3}, {2}], [{4, 5}, {5, 6, 7, 8}]])
    b = collections.OrderedDict([
        ((0, 0, 0), 1),
        ((0, 0, 1), 3),
        ((0, 1, 0), 2),
        ((1, 0, 0), 4),
        ((1, 0, 1), 5),
        ((1, 1, 0), 5),
        ((1, 1, 1), 6),
        ((1, 1, 2), 7),
        ((1, 1, 3), 8),
    ])
    b = tf.SparseTensor(list(b.keys()), list(b.values()), dense_shape=[2, 2, 4])

    # `set_difference` is applied to each aligned pair of sets.
    tf.sets.difference(a, b)

    # The result will be equivalent to either of:
    #
    # np.array([[{2}, {3}], [{}, {}]])
    #
    # collections.OrderedDict([
    #     ((0, 0, 0), 2),
    #     ((0, 1, 0), 3),
    # ])
  ```

  Args:
    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
        must be sorted in row-major order.
    b: `Tensor` or `SparseTensor` of the same type as `a`. If sparse, indices
        must be sorted in row-major order.
    aminusb: Whether to subtract `b` from `a`, vs vice versa.
    validate_indices: Whether to validate the order and range of sparse indices
       in `a` and `b`.

  Returns:
    A `SparseTensor` whose shape is the same rank as `a` and `b`, and all but
    the last dimension the same. Elements along the last dimension contain the
    differences.

  Raises:
    TypeError: If inputs are invalid types, or if `a` and `b` have
        different types.
    ValueError: If `a` is sparse and `b` is dense.
    errors_impl.InvalidArgumentError: If the shapes of `a` and `b` do not
        match in any dimension other than the last dimension.
  "
  [a b  & {:keys [aminusb validate_indices]} ]
    (py/call-attr-kw sets "difference" [a b] {:aminusb aminusb :validate_indices validate_indices }))
(defn intersection 
  "Compute set intersection of elements in last dimension of `a` and `b`.

  All but the last dimension of `a` and `b` must match.

  Example:

  ```python
    import tensorflow as tf
    import collections

    # Represent the following array of sets as a sparse tensor:
    # a = np.array([[{1, 2}, {3}], [{4}, {5, 6}]])
    a = collections.OrderedDict([
        ((0, 0, 0), 1),
        ((0, 0, 1), 2),
        ((0, 1, 0), 3),
        ((1, 0, 0), 4),
        ((1, 1, 0), 5),
        ((1, 1, 1), 6),
    ])
    a = tf.SparseTensor(list(a.keys()), list(a.values()), dense_shape=[2,2,2])

    # b = np.array([[{1}, {}], [{4}, {5, 6, 7, 8}]])
    b = collections.OrderedDict([
        ((0, 0, 0), 1),
        ((1, 0, 0), 4),
        ((1, 1, 0), 5),
        ((1, 1, 1), 6),
        ((1, 1, 2), 7),
        ((1, 1, 3), 8),
    ])
    b = tf.SparseTensor(list(b.keys()), list(b.values()), dense_shape=[2, 2, 4])

    # `tf.sets.intersection` is applied to each aligned pair of sets.
    tf.sets.intersection(a, b)

    # The result will be equivalent to either of:
    #
    # np.array([[{1}, {}], [{4}, {5, 6}]])
    #
    # collections.OrderedDict([
    #     ((0, 0, 0), 1),
    #     ((1, 0, 0), 4),
    #     ((1, 1, 0), 5),
    #     ((1, 1, 1), 6),
    # ])
  ```

  Args:
    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
        must be sorted in row-major order.
    b: `Tensor` or `SparseTensor` of the same type as `a`. If sparse, indices
        must be sorted in row-major order.
    validate_indices: Whether to validate the order and range of sparse indices
       in `a` and `b`.

  Returns:
    A `SparseTensor` whose shape is the same rank as `a` and `b`, and all but
    the last dimension the same. Elements along the last dimension contain the
    intersections.
  "
  [a b  & {:keys [validate_indices]} ]
    (py/call-attr-kw sets "intersection" [a b] {:validate_indices validate_indices }))
(defn set-difference 
  "Compute set difference of elements in last dimension of `a` and `b`.

  All but the last dimension of `a` and `b` must match.

  Example:

  ```python
    import tensorflow as tf
    import collections

    # Represent the following array of sets as a sparse tensor:
    # a = np.array([[{1, 2}, {3}], [{4}, {5, 6}]])
    a = collections.OrderedDict([
        ((0, 0, 0), 1),
        ((0, 0, 1), 2),
        ((0, 1, 0), 3),
        ((1, 0, 0), 4),
        ((1, 1, 0), 5),
        ((1, 1, 1), 6),
    ])
    a = tf.SparseTensor(list(a.keys()), list(a.values()), dense_shape=[2, 2, 2])

    # np.array([[{1, 3}, {2}], [{4, 5}, {5, 6, 7, 8}]])
    b = collections.OrderedDict([
        ((0, 0, 0), 1),
        ((0, 0, 1), 3),
        ((0, 1, 0), 2),
        ((1, 0, 0), 4),
        ((1, 0, 1), 5),
        ((1, 1, 0), 5),
        ((1, 1, 1), 6),
        ((1, 1, 2), 7),
        ((1, 1, 3), 8),
    ])
    b = tf.SparseTensor(list(b.keys()), list(b.values()), dense_shape=[2, 2, 4])

    # `set_difference` is applied to each aligned pair of sets.
    tf.sets.difference(a, b)

    # The result will be equivalent to either of:
    #
    # np.array([[{2}, {3}], [{}, {}]])
    #
    # collections.OrderedDict([
    #     ((0, 0, 0), 2),
    #     ((0, 1, 0), 3),
    # ])
  ```

  Args:
    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
        must be sorted in row-major order.
    b: `Tensor` or `SparseTensor` of the same type as `a`. If sparse, indices
        must be sorted in row-major order.
    aminusb: Whether to subtract `b` from `a`, vs vice versa.
    validate_indices: Whether to validate the order and range of sparse indices
       in `a` and `b`.

  Returns:
    A `SparseTensor` whose shape is the same rank as `a` and `b`, and all but
    the last dimension the same. Elements along the last dimension contain the
    differences.

  Raises:
    TypeError: If inputs are invalid types, or if `a` and `b` have
        different types.
    ValueError: If `a` is sparse and `b` is dense.
    errors_impl.InvalidArgumentError: If the shapes of `a` and `b` do not
        match in any dimension other than the last dimension.
  "
  [a b  & {:keys [aminusb validate_indices]} ]
    (py/call-attr-kw sets "set_difference" [a b] {:aminusb aminusb :validate_indices validate_indices }))
(defn set-intersection 
  "Compute set intersection of elements in last dimension of `a` and `b`.

  All but the last dimension of `a` and `b` must match.

  Example:

  ```python
    import tensorflow as tf
    import collections

    # Represent the following array of sets as a sparse tensor:
    # a = np.array([[{1, 2}, {3}], [{4}, {5, 6}]])
    a = collections.OrderedDict([
        ((0, 0, 0), 1),
        ((0, 0, 1), 2),
        ((0, 1, 0), 3),
        ((1, 0, 0), 4),
        ((1, 1, 0), 5),
        ((1, 1, 1), 6),
    ])
    a = tf.SparseTensor(list(a.keys()), list(a.values()), dense_shape=[2,2,2])

    # b = np.array([[{1}, {}], [{4}, {5, 6, 7, 8}]])
    b = collections.OrderedDict([
        ((0, 0, 0), 1),
        ((1, 0, 0), 4),
        ((1, 1, 0), 5),
        ((1, 1, 1), 6),
        ((1, 1, 2), 7),
        ((1, 1, 3), 8),
    ])
    b = tf.SparseTensor(list(b.keys()), list(b.values()), dense_shape=[2, 2, 4])

    # `tf.sets.intersection` is applied to each aligned pair of sets.
    tf.sets.intersection(a, b)

    # The result will be equivalent to either of:
    #
    # np.array([[{1}, {}], [{4}, {5, 6}]])
    #
    # collections.OrderedDict([
    #     ((0, 0, 0), 1),
    #     ((1, 0, 0), 4),
    #     ((1, 1, 0), 5),
    #     ((1, 1, 1), 6),
    # ])
  ```

  Args:
    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
        must be sorted in row-major order.
    b: `Tensor` or `SparseTensor` of the same type as `a`. If sparse, indices
        must be sorted in row-major order.
    validate_indices: Whether to validate the order and range of sparse indices
       in `a` and `b`.

  Returns:
    A `SparseTensor` whose shape is the same rank as `a` and `b`, and all but
    the last dimension the same. Elements along the last dimension contain the
    intersections.
  "
  [a b  & {:keys [validate_indices]} ]
    (py/call-attr-kw sets "set_intersection" [a b] {:validate_indices validate_indices }))
(defn set-size 
  "Compute number of unique elements along last dimension of `a`.

  Args:
    a: `SparseTensor`, with indices sorted in row-major order.
    validate_indices: Whether to validate the order and range of sparse indices
       in `a`.

  Returns:
    `int32` `Tensor` of set sizes. For `a` ranked `n`, this is a `Tensor` with
    rank `n-1`, and the same 1st `n-1` dimensions as `a`. Each value is the
    number of unique elements in the corresponding `[0...n-1]` dimension of `a`.

  Raises:
    TypeError: If `a` is an invalid types.
  "
  [a  & {:keys [validate_indices]} ]
    (py/call-attr-kw sets "set_size" [a] {:validate_indices validate_indices }))
(defn set-union 
  "Compute set union of elements in last dimension of `a` and `b`.

  All but the last dimension of `a` and `b` must match.

  Example:

  ```python
    import tensorflow as tf
    import collections

    # [[{1, 2}, {3}], [{4}, {5, 6}]]
    a = collections.OrderedDict([
        ((0, 0, 0), 1),
        ((0, 0, 1), 2),
        ((0, 1, 0), 3),
        ((1, 0, 0), 4),
        ((1, 1, 0), 5),
        ((1, 1, 1), 6),
    ])
    a = tf.SparseTensor(list(a.keys()), list(a.values()), dense_shape=[2, 2, 2])

    # [[{1, 3}, {2}], [{4, 5}, {5, 6, 7, 8}]]
    b = collections.OrderedDict([
        ((0, 0, 0), 1),
        ((0, 0, 1), 3),
        ((0, 1, 0), 2),
        ((1, 0, 0), 4),
        ((1, 0, 1), 5),
        ((1, 1, 0), 5),
        ((1, 1, 1), 6),
        ((1, 1, 2), 7),
        ((1, 1, 3), 8),
    ])
    b = tf.SparseTensor(list(b.keys()), list(b.values()), dense_shape=[2, 2, 4])

    # `set_union` is applied to each aligned pair of sets.
    tf.sets.union(a, b)

    # The result will be a equivalent to either of:
    #
    # np.array([[{1, 2, 3}, {2, 3}], [{4, 5}, {5, 6, 7, 8}]])
    #
    # collections.OrderedDict([
    #     ((0, 0, 0), 1),
    #     ((0, 0, 1), 2),
    #     ((0, 0, 2), 3),
    #     ((0, 1, 0), 2),
    #     ((0, 1, 1), 3),
    #     ((1, 0, 0), 4),
    #     ((1, 0, 1), 5),
    #     ((1, 1, 0), 5),
    #     ((1, 1, 1), 6),
    #     ((1, 1, 2), 7),
    #     ((1, 1, 3), 8),
    # ])
  ```

  Args:
    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
        must be sorted in row-major order.
    b: `Tensor` or `SparseTensor` of the same type as `a`. If sparse, indices
        must be sorted in row-major order.
    validate_indices: Whether to validate the order and range of sparse indices
       in `a` and `b`.

  Returns:
    A `SparseTensor` whose shape is the same rank as `a` and `b`, and all but
    the last dimension the same. Elements along the last dimension contain the
    unions.
  "
  [a b  & {:keys [validate_indices]} ]
    (py/call-attr-kw sets "set_union" [a b] {:validate_indices validate_indices }))
(defn size 
  "Compute number of unique elements along last dimension of `a`.

  Args:
    a: `SparseTensor`, with indices sorted in row-major order.
    validate_indices: Whether to validate the order and range of sparse indices
       in `a`.

  Returns:
    `int32` `Tensor` of set sizes. For `a` ranked `n`, this is a `Tensor` with
    rank `n-1`, and the same 1st `n-1` dimensions as `a`. Each value is the
    number of unique elements in the corresponding `[0...n-1]` dimension of `a`.

  Raises:
    TypeError: If `a` is an invalid types.
  "
  [a  & {:keys [validate_indices]} ]
    (py/call-attr-kw sets "size" [a] {:validate_indices validate_indices }))
(defn union 
  "Compute set union of elements in last dimension of `a` and `b`.

  All but the last dimension of `a` and `b` must match.

  Example:

  ```python
    import tensorflow as tf
    import collections

    # [[{1, 2}, {3}], [{4}, {5, 6}]]
    a = collections.OrderedDict([
        ((0, 0, 0), 1),
        ((0, 0, 1), 2),
        ((0, 1, 0), 3),
        ((1, 0, 0), 4),
        ((1, 1, 0), 5),
        ((1, 1, 1), 6),
    ])
    a = tf.SparseTensor(list(a.keys()), list(a.values()), dense_shape=[2, 2, 2])

    # [[{1, 3}, {2}], [{4, 5}, {5, 6, 7, 8}]]
    b = collections.OrderedDict([
        ((0, 0, 0), 1),
        ((0, 0, 1), 3),
        ((0, 1, 0), 2),
        ((1, 0, 0), 4),
        ((1, 0, 1), 5),
        ((1, 1, 0), 5),
        ((1, 1, 1), 6),
        ((1, 1, 2), 7),
        ((1, 1, 3), 8),
    ])
    b = tf.SparseTensor(list(b.keys()), list(b.values()), dense_shape=[2, 2, 4])

    # `set_union` is applied to each aligned pair of sets.
    tf.sets.union(a, b)

    # The result will be a equivalent to either of:
    #
    # np.array([[{1, 2, 3}, {2, 3}], [{4, 5}, {5, 6, 7, 8}]])
    #
    # collections.OrderedDict([
    #     ((0, 0, 0), 1),
    #     ((0, 0, 1), 2),
    #     ((0, 0, 2), 3),
    #     ((0, 1, 0), 2),
    #     ((0, 1, 1), 3),
    #     ((1, 0, 0), 4),
    #     ((1, 0, 1), 5),
    #     ((1, 1, 0), 5),
    #     ((1, 1, 1), 6),
    #     ((1, 1, 2), 7),
    #     ((1, 1, 3), 8),
    # ])
  ```

  Args:
    a: `Tensor` or `SparseTensor` of the same type as `b`. If sparse, indices
        must be sorted in row-major order.
    b: `Tensor` or `SparseTensor` of the same type as `a`. If sparse, indices
        must be sorted in row-major order.
    validate_indices: Whether to validate the order and range of sparse indices
       in `a` and `b`.

  Returns:
    A `SparseTensor` whose shape is the same rank as `a` and `b`, and all but
    the last dimension the same. Elements along the last dimension contain the
    unions.
  "
  [a b  & {:keys [validate_indices]} ]
    (py/call-attr-kw sets "union" [a b] {:validate_indices validate_indices }))
