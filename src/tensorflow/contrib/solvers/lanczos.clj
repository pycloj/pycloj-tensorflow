(ns tensorflow.contrib.solvers.python.ops.lanczos
  "Lanczos algorithms."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce lanczos (import-module "tensorflow.contrib.solvers.python.ops.lanczos"))
(defn bidiag-matmul 
  "Multiplies a matrix by a bidiagonal matrix.

  alpha and beta are length k vectors representing the diagonal and first lower
  subdiagonal of (K+1) x K matrix B.
  If adjoint_b is False, computes A * B as follows:

    A * B =  A[:, :-1] * diag(alpha) + A[:, 1:] * diag(beta)

  If adjoint_b is True, computes A * B[:-1, :]' as follows

    A * B[:-1, :]' =
      A * diag(alpha) + [zeros(m,1), A[:, :-1] * diag(beta[:-1])]

  Args:
    matrix: A rank-2 `Tensor` representing matrix A.
    alpha: A rank-1 `Tensor` representing the diagonal of B.
    beta: A rank-1 `Tensor` representing the lower subdiagonal diagonal of B.
    adjoint_b: `bool` determining what to compute.
    name: A name scope for the operation.

  Returns:
    If `adjoint_b` is False the `A * B` is returned.
    If `adjoint_b` is True the `A * B'` is returned.
  "
  [matrix alpha beta  & {:keys [adjoint_b name]} ]
    (py/call-attr-kw lanczos "bidiag_matmul" [matrix alpha beta] {:adjoint_b adjoint_b :name name }))

(defn lanczos-bidiag 
  "Computes a Lanczos bidiagonalization for a linear operator.

  Computes matrices `U` of shape `[m, k+1]`, `V` of shape `[n, k]` and lower
  bidiagonal matrix `B` of shape `[k+1, k]`, that satisfy the equations
  `A * V = U * B` and `A' * U[:, :-1] = V * B[:-1, :]'`.

  The columns of `U` are orthonormal and form a basis for the Krylov subspace
  `K(A*A', U[:,0])`.

  The columns of `V` are orthonormal and form a basis for the Krylov subspace
  `K(A'*A, A' U[:,0])`.

  Args:
    operator: An object representing a linear operator with attributes:
      - shape: Either a list of integers or a 1-D `Tensor` of type `int32` of
        length 2. `shape[0]` is the dimension on the domain of the operator,
        `shape[1]` is the dimension of the co-domain of the operator. On other
        words, if operator represents an M x N matrix A, `shape` must contain
        `[M, N]`.
      - dtype: The datatype of input to and output from `apply` and
        `apply_adjoint`.
      - apply: Callable object taking a vector `x` as input and returning a
        vector with the result of applying the operator to `x`, i.e. if
       `operator` represents matrix `A`, `apply` should return `A * x`.
      - apply_adjoint: Callable object taking a vector `x` as input and
        returning a vector with the result of applying the adjoint operator
        to `x`, i.e. if `operator` represents matrix `A`, `apply_adjoint` should
        return `conj(transpose(A)) * x`.
    k: An integer or a scalar Tensor of type `int32`. Determines the maximum
      number of steps to run. If an invariant subspace is found, the algorithm
      may terminate before `k` steps have been run.
    orthogonalize: If `True`, perform full orthogonalization. If `False` no
      orthogonalization is performed.
    starting_vector: If not null, must be a `Tensor` of shape `[n]`.
    name: A name scope for the operation.

  Returns:
    output: A namedtuple representing a Lanczos bidiagonalization of
      `operator` with attributes:
      u: A rank-2 `Tensor` of type `operator.dtype` and shape
        `[operator.shape[0], k_actual+1]`, where `k_actual` is the number of
        steps run.
      v: A rank-2 `Tensor` of type `operator.dtype` and shape
        `[operator.shape[1], k_actual]`, where `k_actual` is the number of steps
        run.
      alpha: A rank-1 `Tensor` of type `operator.dtype` and shape `[k]`.
      beta: A rank-1 `Tensor` of type `operator.dtype` and shape `[k]`.
  "
  [operator k & {:keys [orthogonalize starting_vector name]
                       :or {starting_vector None}} ]
    (py/call-attr-kw lanczos "lanczos_bidiag" [operator k] {:orthogonalize orthogonalize :starting_vector starting_vector :name name }))
