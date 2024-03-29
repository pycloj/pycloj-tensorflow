(ns tensorflow.-api.v1.compat.v2.linalg.LinearOperatorDiag
  "`LinearOperator` acting like a [batch] square diagonal matrix.

  This operator acts like a [batch] diagonal matrix `A` with shape
  `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  `LinearOperatorDiag` is initialized with a (batch) vector.

  ```python
  # Create a 2 x 2 diagonal linear operator.
  diag = [1., -1.]
  operator = LinearOperatorDiag(diag)

  operator.to_dense()
  ==> [[1.,  0.]
       [0., -1.]]

  operator.shape
  ==> [2, 2]

  operator.log_abs_determinant()
  ==> scalar Tensor

  x = ... Shape [2, 4] Tensor
  operator.matmul(x)
  ==> Shape [2, 4] Tensor

  # Create a [2, 3] batch of 4 x 4 linear operators.
  diag = tf.random.normal(shape=[2, 3, 4])
  operator = LinearOperatorDiag(diag)

  # Create a shape [2, 1, 4, 2] vector.  Note that this shape is compatible
  # since the batch dimensions, [2, 1], are broadcast to
  # operator.batch_shape = [2, 3].
  y = tf.random.normal(shape=[2, 1, 4, 2])
  x = operator.solve(y)
  ==> operator.matmul(x) = y
  ```

  #### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
  x.shape =   [C1,...,Cc] + [N, R],
  and [C1,...,Cc] broadcasts with [B1,...,Bb] to [D1,...,Dd]
  ```

  #### Performance

  Suppose `operator` is a `LinearOperatorDiag` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then

  * `operator.matmul(x)` involves `N * R` multiplications.
  * `operator.solve(x)` involves `N` divisions and `N * R` multiplications.
  * `operator.determinant()` involves a size `N` `reduce_prod`.

  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning:

  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce linalg (import-module "tensorflow._api.v1.compat.v2.linalg"))
(defn LinearOperatorDiag 
  "`LinearOperator` acting like a [batch] square diagonal matrix.

  This operator acts like a [batch] diagonal matrix `A` with shape
  `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  `LinearOperatorDiag` is initialized with a (batch) vector.

  ```python
  # Create a 2 x 2 diagonal linear operator.
  diag = [1., -1.]
  operator = LinearOperatorDiag(diag)

  operator.to_dense()
  ==> [[1.,  0.]
       [0., -1.]]

  operator.shape
  ==> [2, 2]

  operator.log_abs_determinant()
  ==> scalar Tensor

  x = ... Shape [2, 4] Tensor
  operator.matmul(x)
  ==> Shape [2, 4] Tensor

  # Create a [2, 3] batch of 4 x 4 linear operators.
  diag = tf.random.normal(shape=[2, 3, 4])
  operator = LinearOperatorDiag(diag)

  # Create a shape [2, 1, 4, 2] vector.  Note that this shape is compatible
  # since the batch dimensions, [2, 1], are broadcast to
  # operator.batch_shape = [2, 3].
  y = tf.random.normal(shape=[2, 1, 4, 2])
  x = operator.solve(y)
  ==> operator.matmul(x) = y
  ```

  #### Shape compatibility

  This operator acts on [batch] matrix with compatible shape.
  `x` is a batch matrix with compatible shape for `matmul` and `solve` if

  ```
  operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
  x.shape =   [C1,...,Cc] + [N, R],
  and [C1,...,Cc] broadcasts with [B1,...,Bb] to [D1,...,Dd]
  ```

  #### Performance

  Suppose `operator` is a `LinearOperatorDiag` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then

  * `operator.matmul(x)` involves `N * R` multiplications.
  * `operator.solve(x)` involves `N` divisions and `N * R` multiplications.
  * `operator.determinant()` involves a size `N` `reduce_prod`.

  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning:

  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  "
  [diag is_non_singular is_self_adjoint is_positive_definite is_square  & {:keys [name]} ]
    (py/call-attr-kw linalg "LinearOperatorDiag" [diag is_non_singular is_self_adjoint is_positive_definite is_square] {:name name }))

(defn H 
  "Returns the adjoint of the current `LinearOperator`.

    Given `A` representing this `LinearOperator`, return `A*`.
    Note that calling `self.adjoint()` and `self.H` are equivalent.

    Args:
      name:  A name for this `Op`.

    Returns:
      `LinearOperator` which represents the adjoint of this `LinearOperator`.
    "
  [ self ]
    (py/call-attr self "H"))
(defn add-to-tensor 
  "Add matrix represented by this operator to `x`.  Equivalent to `A + x`.

    Args:
      x:  `Tensor` with same `dtype` and shape broadcastable to `self.shape`.
      name:  A name to give this `Op`.

    Returns:
      A `Tensor` with broadcast shape and same `dtype` as `self`.
    "
  [self x  & {:keys [name]} ]
    (py/call-attr-kw self "add_to_tensor" [x] {:name name }))
(defn adjoint 
  "Returns the adjoint of the current `LinearOperator`.

    Given `A` representing this `LinearOperator`, return `A*`.
    Note that calling `self.adjoint()` and `self.H` are equivalent.

    Args:
      name:  A name for this `Op`.

    Returns:
      `LinearOperator` which represents the adjoint of this `LinearOperator`.
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "adjoint" [] {:name name }))
(defn assert-non-singular 
  "Returns an `Op` that asserts this operator is non singular.

    This operator is considered non-singular if

    ```
    ConditionNumber < max{100, range_dimension, domain_dimension} * eps,
    eps := np.finfo(self.dtype.as_numpy_dtype).eps
    ```

    Args:
      name:  A string name to prepend to created ops.

    Returns:
      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if
        the operator is singular.
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "assert_non_singular" [] {:name name }))
(defn assert-positive-definite 
  "Returns an `Op` that asserts this operator is positive definite.

    Here, positive definite means that the quadratic form `x^H A x` has positive
    real part for all nonzero `x`.  Note that we do not require the operator to
    be self-adjoint to be positive definite.

    Args:
      name:  A name to give this `Op`.

    Returns:
      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if
        the operator is not positive definite.
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "assert_positive_definite" [] {:name name }))
(defn assert-self-adjoint 
  "Returns an `Op` that asserts this operator is self-adjoint.

    Here we check that this operator is *exactly* equal to its hermitian
    transpose.

    Args:
      name:  A string name to prepend to created ops.

    Returns:
      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if
        the operator is not self-adjoint.
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "assert_self_adjoint" [] {:name name }))

(defn batch-shape 
  "`TensorShape` of batch dimensions of this `LinearOperator`.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns
    `TensorShape([B1,...,Bb])`, equivalent to `A.shape[:-2]`

    Returns:
      `TensorShape`, statically determined, may be undefined.
    "
  [ self ]
    (py/call-attr self "batch_shape"))
(defn batch-shape-tensor 
  "Shape of batch dimensions of this operator, determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
    `[B1,...,Bb]`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "batch_shape_tensor" [] {:name name }))
(defn cholesky 
  "Returns a Cholesky factor as a `LinearOperator`.

    Given `A` representing this `LinearOperator`, if `A` is positive definite
    self-adjoint, return `L`, where `A = L L^T`, i.e. the cholesky
    decomposition.

    Args:
      name:  A name for this `Op`.

    Returns:
      `LinearOperator` which represents the lower triangular matrix
      in the Cholesky decomposition.

    Raises:
      ValueError: When the `LinearOperator` is not hinted to be positive
        definite and self adjoint.
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "cholesky" [] {:name name }))
(defn determinant 
  "Determinant for every batch member.

    Args:
      name:  A name for this `Op`.

    Returns:
      `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.

    Raises:
      NotImplementedError:  If `self.is_square` is `False`.
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "determinant" [] {:name name }))

(defn diag 
  ""
  [ self ]
    (py/call-attr self "diag"))
(defn diag-part 
  "Efficiently get the [batch] diagonal part of this operator.

    If this operator has shape `[B1,...,Bb, M, N]`, this returns a
    `Tensor` `diagonal`, of shape `[B1,...,Bb, min(M, N)]`, where
    `diagonal[b1,...,bb, i] = self.to_dense()[b1,...,bb, i, i]`.

    ```
    my_operator = LinearOperatorDiag([1., 2.])

    # Efficiently get the diagonal
    my_operator.diag_part()
    ==> [1., 2.]

    # Equivalent, but inefficient method
    tf.linalg.diag_part(my_operator.to_dense())
    ==> [1., 2.]
    ```

    Args:
      name:  A name for this `Op`.

    Returns:
      diag_part:  A `Tensor` of same `dtype` as self.
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "diag_part" [] {:name name }))

(defn domain-dimension 
  "Dimension (in the sense of vector spaces) of the domain of this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

    Returns:
      `Dimension` object.
    "
  [ self ]
    (py/call-attr self "domain_dimension"))
(defn domain-dimension-tensor 
  "Dimension (in the sense of vector spaces) of the domain of this operator.

    Determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "domain_dimension_tensor" [] {:name name }))

(defn dtype 
  "The `DType` of `Tensor`s handled by this `LinearOperator`."
  [ self ]
    (py/call-attr self "dtype"))

(defn graph-parents 
  "List of graph dependencies of this `LinearOperator`."
  [ self ]
    (py/call-attr self "graph_parents"))
(defn inverse 
  "Returns the Inverse of this `LinearOperator`.

    Given `A` representing this `LinearOperator`, return a `LinearOperator`
    representing `A^-1`.

    Args:
      name: A name scope to use for ops added by this method.

    Returns:
      `LinearOperator` representing inverse of this matrix.

    Raises:
      ValueError: When the `LinearOperator` is not hinted to be `non_singular`.
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "inverse" [] {:name name }))

(defn is-non-singular 
  ""
  [ self ]
    (py/call-attr self "is_non_singular"))

(defn is-positive-definite 
  ""
  [ self ]
    (py/call-attr self "is_positive_definite"))

(defn is-self-adjoint 
  ""
  [ self ]
    (py/call-attr self "is_self_adjoint"))

(defn is-square 
  "Return `True/False` depending on if this operator is square."
  [ self ]
    (py/call-attr self "is_square"))
(defn log-abs-determinant 
  "Log absolute value of determinant for every batch member.

    Args:
      name:  A name for this `Op`.

    Returns:
      `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.

    Raises:
      NotImplementedError:  If `self.is_square` is `False`.
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "log_abs_determinant" [] {:name name }))
(defn matmul 
  "Transform [batch] matrix `x` with left multiplication:  `x --> Ax`.

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    X = ... # shape [..., N, R], batch matrix, R > 0.

    Y = operator.matmul(X)
    Y.shape
    ==> [..., M, R]

    Y[..., :, r] = sum_j A[..., :, j] X[j, r]
    ```

    Args:
      x: `LinearOperator` or `Tensor` with compatible shape and same `dtype` as
        `self`. See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, left multiply by the adjoint: `A^H x`.
      adjoint_arg:  Python `bool`.  If `True`, compute `A x^H` where `x^H` is
        the hermitian transpose (transposition and complex conjugation).
      name:  A name for this `Op`.

    Returns:
      A `LinearOperator` or `Tensor` with shape `[..., M, R]` and same `dtype`
        as `self`.
    "
  [self x  & {:keys [adjoint adjoint_arg name]} ]
    (py/call-attr-kw self "matmul" [x] {:adjoint adjoint :adjoint_arg adjoint_arg :name name }))
(defn matvec 
  "Transform [batch] vector `x` with left multiplication:  `x --> Ax`.

    ```python
    # Make an operator acting like batch matric A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)

    X = ... # shape [..., N], batch vector

    Y = operator.matvec(X)
    Y.shape
    ==> [..., M]

    Y[..., :] = sum_j A[..., :, j] X[..., j]
    ```

    Args:
      x: `Tensor` with compatible shape and same `dtype` as `self`.
        `x` is treated as a [batch] vector meaning for every set of leading
        dimensions, the last dimension defines a vector.
        See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, left multiply by the adjoint: `A^H x`.
      name:  A name for this `Op`.

    Returns:
      A `Tensor` with shape `[..., M]` and same `dtype` as `self`.
    "
  [self x  & {:keys [adjoint name]} ]
    (py/call-attr-kw self "matvec" [x] {:adjoint adjoint :name name }))

(defn name 
  "Name prepended to all ops created by this `LinearOperator`."
  [ self ]
    (py/call-attr self "name"))

(defn name-scope 
  "Returns a `tf.name_scope` instance for this class."
  [ self ]
    (py/call-attr self "name_scope"))

(defn range-dimension 
  "Dimension (in the sense of vector spaces) of the range of this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

    Returns:
      `Dimension` object.
    "
  [ self ]
    (py/call-attr self "range_dimension"))
(defn range-dimension-tensor 
  "Dimension (in the sense of vector spaces) of the range of this operator.

    Determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "range_dimension_tensor" [] {:name name }))

(defn shape 
  "`TensorShape` of this `LinearOperator`.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns
    `TensorShape([B1,...,Bb, M, N])`, equivalent to `A.shape`.

    Returns:
      `TensorShape`, statically determined, may be undefined.
    "
  [ self ]
    (py/call-attr self "shape"))
(defn shape-tensor 
  "Shape of this `LinearOperator`, determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
    `[B1,...,Bb, M, N]`, equivalent to `tf.shape(A)`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "shape_tensor" [] {:name name }))
(defn solve 
  "Solve (exact or approx) `R` (batch) systems of equations: `A X = rhs`.

    The returned `Tensor` will be close to an exact solution if `A` is well
    conditioned. Otherwise closeness will vary. See class docstring for details.

    Examples:

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    # Solve R > 0 linear systems for every member of the batch.
    RHS = ... # shape [..., M, R]

    X = operator.solve(RHS)
    # X[..., :, r] is the solution to the r'th linear system
    # sum_j A[..., :, j] X[..., j, r] = RHS[..., :, r]

    operator.matmul(X)
    ==> RHS
    ```

    Args:
      rhs: `Tensor` with same `dtype` as this operator and compatible shape.
        `rhs` is treated like a [batch] matrix meaning for every set of leading
        dimensions, the last two dimensions defines a matrix.
        See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, solve the system involving the adjoint
        of this `LinearOperator`:  `A^H X = rhs`.
      adjoint_arg:  Python `bool`.  If `True`, solve `A X = rhs^H` where `rhs^H`
        is the hermitian transpose (transposition and complex conjugation).
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` with shape `[...,N, R]` and same `dtype` as `rhs`.

    Raises:
      NotImplementedError:  If `self.is_non_singular` or `is_square` is False.
    "
  [self rhs  & {:keys [adjoint adjoint_arg name]} ]
    (py/call-attr-kw self "solve" [rhs] {:adjoint adjoint :adjoint_arg adjoint_arg :name name }))
(defn solvevec 
  "Solve single equation with best effort: `A X = rhs`.

    The returned `Tensor` will be close to an exact solution if `A` is well
    conditioned. Otherwise closeness will vary. See class docstring for details.

    Examples:

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    # Solve one linear system for every member of the batch.
    RHS = ... # shape [..., M]

    X = operator.solvevec(RHS)
    # X is the solution to the linear system
    # sum_j A[..., :, j] X[..., j] = RHS[..., :]

    operator.matvec(X)
    ==> RHS
    ```

    Args:
      rhs: `Tensor` with same `dtype` as this operator.
        `rhs` is treated like a [batch] vector meaning for every set of leading
        dimensions, the last dimension defines a vector.  See class docstring
        for definition of compatibility regarding batch dimensions.
      adjoint: Python `bool`.  If `True`, solve the system involving the adjoint
        of this `LinearOperator`:  `A^H X = rhs`.
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` with shape `[...,N]` and same `dtype` as `rhs`.

    Raises:
      NotImplementedError:  If `self.is_non_singular` or `is_square` is False.
    "
  [self rhs  & {:keys [adjoint name]} ]
    (py/call-attr-kw self "solvevec" [rhs] {:adjoint adjoint :name name }))

(defn submodules 
  "Sequence of all sub-modules.

    Submodules are modules which are properties of this module, or found as
    properties of modules which are properties of this module (and so on).

    ```
    a = tf.Module()
    b = tf.Module()
    c = tf.Module()
    a.b = b
    b.c = c
    assert list(a.submodules) == [b, c]
    assert list(b.submodules) == [c]
    assert list(c.submodules) == []
    ```

    Returns:
      A sequence of all submodules.
    "
  [ self ]
    (py/call-attr self "submodules"))

(defn tensor-rank 
  "Rank (in the sense of tensors) of matrix corresponding to this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

    Args:
      name:  A name for this `Op`.

    Returns:
      Python integer, or None if the tensor rank is undefined.
    "
  [ self ]
    (py/call-attr self "tensor_rank"))
(defn tensor-rank-tensor 
  "Rank (in the sense of tensors) of matrix corresponding to this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`, determined at runtime.
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "tensor_rank_tensor" [] {:name name }))
(defn to-dense 
  "Return a dense (batch) matrix representing this operator."
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "to_dense" [] {:name name }))
(defn trace 
  "Trace of the linear operator, equal to sum of `self.diag_part()`.

    If the operator is square, this is also the sum of the eigenvalues.

    Args:
      name:  A name for this `Op`.

    Returns:
      Shape `[B1,...,Bb]` `Tensor` of same `dtype` as `self`.
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "trace" [] {:name name }))

(defn trainable-variables 
  "Sequence of variables owned by this module and it's submodules.

    Note: this method uses reflection to find variables on the current instance
    and submodules. For performance reasons you may wish to cache the result
    of calling this method if you don't expect the return value to change.

    Returns:
      A sequence of variables for the current module (sorted by attribute
      name) followed by variables from all submodules recursively (breadth
      first).
    "
  [ self ]
    (py/call-attr self "trainable_variables"))

(defn variables 
  "Sequence of variables owned by this module and it's submodules.

    Note: this method uses reflection to find variables on the current instance
    and submodules. For performance reasons you may wish to cache the result
    of calling this method if you don't expect the return value to change.

    Returns:
      A sequence of variables for the current module (sorted by attribute
      name) followed by variables from all submodules recursively (breadth
      first).
    "
  [ self ]
    (py/call-attr self "variables"))
