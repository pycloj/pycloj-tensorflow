(ns tensorflow.contrib.solvers.python.ops.least-squares
  "Solvers for linear least-squares."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce least-squares (import-module "tensorflow.contrib.solvers.python.ops.least_squares"))
(defn cgls 
  "Conjugate gradient least squares solver.

  Solves a linear least squares problem \\(||A x - rhs||_2\\) for a single
  right-hand side, using an iterative, matrix-free algorithm where the action of
  the matrix A is represented by `operator`. The CGLS algorithm implicitly
  applies the symmetric conjugate gradient algorithm to the normal equations
  \\(A^* A x = A^* rhs\\). The iteration terminates when either
  the number of iterations exceeds `max_iter` or when the norm of the conjugate
  residual (residual of the normal equations) have been reduced to `tol` times
  its initial initial value, i.e.
  \\(||A^* (rhs - A x_k)|| <= tol ||A^* rhs||\\).

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

    rhs: A rank-1 `Tensor` of shape `[M]` containing the right-hand size vector.
    tol: A float scalar convergence tolerance.
    max_iter: An integer giving the maximum number of iterations.
    name: A name scope for the operation.


  Returns:
    output: A namedtuple representing the final state with fields:
      - i: A scalar `int32` `Tensor`. Number of iterations executed.
      - x: A rank-1 `Tensor` of shape `[N]` containing the computed solution.
      - r: A rank-1 `Tensor` of shape `[M]` containing the residual vector.
      - p: A rank-1 `Tensor` of shape `[N]`. The next descent direction.
      - gamma: \\(||A^* r||_2^2\\)
  "
  [operator rhs  & {:keys [tol max_iter name]} ]
    (py/call-attr-kw least-squares "cgls" [operator rhs] {:tol tol :max_iter max_iter :name name }))
