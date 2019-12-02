(ns tensorflow.sparse.SparseTensor
  "Represents a sparse tensor.

  TensorFlow represents a sparse tensor as three separate dense tensors:
  `indices`, `values`, and `dense_shape`.  In Python, the three tensors are
  collected into a `SparseTensor` class for ease of use.  If you have separate
  `indices`, `values`, and `dense_shape` tensors, wrap them in a `SparseTensor`
  object before passing to the ops below.

  Concretely, the sparse tensor `SparseTensor(indices, values, dense_shape)`
  comprises the following components, where `N` and `ndims` are the number
  of values and number of dimensions in the `SparseTensor`, respectively:

  * `indices`: A 2-D int64 tensor of dense_shape `[N, ndims]`, which specifies
    the indices of the elements in the sparse tensor that contain nonzero
    values (elements are zero-indexed). For example, `indices=[[1,3], [2,4]]`
    specifies that the elements with indexes of [1,3] and [2,4] have
    nonzero values.

  * `values`: A 1-D tensor of any type and dense_shape `[N]`, which supplies the
    values for each element in `indices`. For example, given
    `indices=[[1,3], [2,4]]`, the parameter `values=[18, 3.6]` specifies
    that element [1,3] of the sparse tensor has a value of 18, and element
    [2,4] of the tensor has a value of 3.6.

  * `dense_shape`: A 1-D int64 tensor of dense_shape `[ndims]`, which specifies
    the dense_shape of the sparse tensor. Takes a list indicating the number of
    elements in each dimension. For example, `dense_shape=[3,6]` specifies a
    two-dimensional 3x6 tensor, `dense_shape=[2,3,4]` specifies a
    three-dimensional 2x3x4 tensor, and `dense_shape=[9]` specifies a
    one-dimensional tensor with 9 elements.

  The corresponding dense tensor satisfies:

  ```python
  dense.shape = dense_shape
  dense[tuple(indices[i])] = values[i]
  ```

  By convention, `indices` should be sorted in row-major order (or equivalently
  lexicographic order on the tuples `indices[i]`). This is not enforced when
  `SparseTensor` objects are constructed, but most ops assume correct ordering.
  If the ordering of sparse tensor `st` is wrong, a fixed version can be
  obtained by calling `tf.sparse.reorder(st)`.

  Example: The sparse tensor

  ```python
  SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
  ```

  represents the dense tensor

  ```python
  [[1, 0, 0, 0]
   [0, 0, 2, 0]
   [0, 0, 0, 0]]
  ```
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce sparse (import-module "tensorflow.sparse"))

(defn SparseTensor 
  "Represents a sparse tensor.

  TensorFlow represents a sparse tensor as three separate dense tensors:
  `indices`, `values`, and `dense_shape`.  In Python, the three tensors are
  collected into a `SparseTensor` class for ease of use.  If you have separate
  `indices`, `values`, and `dense_shape` tensors, wrap them in a `SparseTensor`
  object before passing to the ops below.

  Concretely, the sparse tensor `SparseTensor(indices, values, dense_shape)`
  comprises the following components, where `N` and `ndims` are the number
  of values and number of dimensions in the `SparseTensor`, respectively:

  * `indices`: A 2-D int64 tensor of dense_shape `[N, ndims]`, which specifies
    the indices of the elements in the sparse tensor that contain nonzero
    values (elements are zero-indexed). For example, `indices=[[1,3], [2,4]]`
    specifies that the elements with indexes of [1,3] and [2,4] have
    nonzero values.

  * `values`: A 1-D tensor of any type and dense_shape `[N]`, which supplies the
    values for each element in `indices`. For example, given
    `indices=[[1,3], [2,4]]`, the parameter `values=[18, 3.6]` specifies
    that element [1,3] of the sparse tensor has a value of 18, and element
    [2,4] of the tensor has a value of 3.6.

  * `dense_shape`: A 1-D int64 tensor of dense_shape `[ndims]`, which specifies
    the dense_shape of the sparse tensor. Takes a list indicating the number of
    elements in each dimension. For example, `dense_shape=[3,6]` specifies a
    two-dimensional 3x6 tensor, `dense_shape=[2,3,4]` specifies a
    three-dimensional 2x3x4 tensor, and `dense_shape=[9]` specifies a
    one-dimensional tensor with 9 elements.

  The corresponding dense tensor satisfies:

  ```python
  dense.shape = dense_shape
  dense[tuple(indices[i])] = values[i]
  ```

  By convention, `indices` should be sorted in row-major order (or equivalently
  lexicographic order on the tuples `indices[i]`). This is not enforced when
  `SparseTensor` objects are constructed, but most ops assume correct ordering.
  If the ordering of sparse tensor `st` is wrong, a fixed version can be
  obtained by calling `tf.sparse.reorder(st)`.

  Example: The sparse tensor

  ```python
  SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
  ```

  represents the dense tensor

  ```python
  [[1, 0, 0, 0]
   [0, 0, 2, 0]
   [0, 0, 0, 0]]
  ```
  "
  [ indices values dense_shape ]
  (py/call-attr sparse "SparseTensor"  indices values dense_shape ))

(defn consumers 
  ""
  [ self  ]
  (py/call-attr self "consumers"  self  ))

(defn dense-shape 
  "A 1-D Tensor of int64 representing the shape of the dense tensor."
  [ self ]
    (py/call-attr self "dense_shape"))

(defn dtype 
  "The `DType` of elements in this tensor."
  [ self ]
    (py/call-attr self "dtype"))

(defn eval 
  "Evaluates this sparse tensor in a `Session`.

    Calling this method will execute all preceding operations that
    produce the inputs needed for the operation that produces this
    tensor.

    *N.B.* Before invoking `SparseTensor.eval()`, its graph must have been
    launched in a session, and either a default session must be
    available, or `session` must be specified explicitly.

    Args:
      feed_dict: A dictionary that maps `Tensor` objects to feed values. See
        `tf.Session.run` for a description of the valid feed values.
      session: (Optional.) The `Session` to be used to evaluate this sparse
        tensor. If none, the default session will be used.

    Returns:
      A `SparseTensorValue` object.
    "
  [ self feed_dict session ]
  (py/call-attr self "eval"  self feed_dict session ))

(defn get-shape 
  "Get the `TensorShape` representing the shape of the dense tensor.

    Returns:
      A `TensorShape` object.
    "
  [ self  ]
  (py/call-attr self "get_shape"  self  ))

(defn graph 
  "The `Graph` that contains the index, value, and dense_shape tensors."
  [ self ]
    (py/call-attr self "graph"))

(defn indices 
  "The indices of non-zero values in the represented dense tensor.

    Returns:
      A 2-D Tensor of int64 with dense_shape `[N, ndims]`, where `N` is the
        number of non-zero values in the tensor, and `ndims` is the rank.
    "
  [ self ]
    (py/call-attr self "indices"))

(defn op 
  "The `Operation` that produces `values` as an output."
  [ self ]
    (py/call-attr self "op"))

(defn shape 
  "Get the `TensorShape` representing the shape of the dense tensor.

    Returns:
      A `TensorShape` object.
    "
  [ self ]
    (py/call-attr self "shape"))

(defn values 
  "The non-zero values in the represented dense tensor.

    Returns:
      A 1-D Tensor of any data type.
    "
  [ self ]
    (py/call-attr self "values"))
