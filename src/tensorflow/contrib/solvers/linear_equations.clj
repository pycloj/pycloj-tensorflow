(ns tensorflow.contrib.solvers.python.ops.linear-equations
  "Solvers for linear equations."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce linear-equations (import-module "tensorflow.contrib.solvers.python.ops.linear_equations"))
(defn conjugate-gradient 
  "Conjugate gradient solver.

  Solves a linear system of equations `A*x = rhs` for selfadjoint, positive
  definite matrix `A` and right-hand side vector `rhs`, using an iterative,
  matrix-free algorithm where the action of the matrix A is represented by
  `operator`. The iteration terminates when either the number of iterations
  exceeds `max_iter` or when the residual norm has been reduced to `tol`
  times its initial value, i.e. \\(||rhs - A x_k|| <= tol ||rhs||\\).

  Args:
    operator: An object representing a linear operator with attributes:
      - shape: Either a list of integers or a 1-D `Tensor` of type `int32` of
        length 2. `shape[0]` is the dimension on the domain of the operator,
        `shape[1]` is the dimension of the co-domain of the operator. On other
        words, if operator represents an N x N matrix A, `shape` must contain
        `[N, N]`.
      - dtype: The datatype of input to and output from `apply`.
      - apply: Callable object taking a vector `x` as input and returning a
        vector with the result of applying the operator to `x`, i.e. if
       `operator` represents matrix `A`, `apply` should return `A * x`.
    rhs: A rank-1 `Tensor` of shape `[N]` containing the right-hand size vector.
    preconditioner: An object representing a linear operator, see `operator`
      for detail. The preconditioner should approximate the inverse of `A`.
      An efficient preconditioner could dramatically improve the rate of
      convergence. If `preconditioner` represents matrix `M`(`M` approximates
      `A^{-1}`), the algorithm uses `preconditioner.apply(x)` to estimate
      `A^{-1}x`. For this to be useful, the cost of applying `M` should be
      much lower than computing `A^{-1}` directly.
    x: A rank-1 `Tensor` of shape `[N]` containing the initial guess for the
      solution.
    tol: A float scalar convergence tolerance.
    max_iter: An integer giving the maximum number of iterations.
    name: A name scope for the operation.

  Returns:
    output: A namedtuple representing the final state with fields:
      - i: A scalar `int32` `Tensor`. Number of iterations executed.
      - x: A rank-1 `Tensor` of shape `[N]` containing the computed solution.
      - r: A rank-1 `Tensor` of shape `[M]` containing the residual vector.
      - p: A rank-1 `Tensor` of shape `[N]`. `A`-conjugate basis vector.
      - gamma: \\(r \dot M \dot r\\), equivalent to  \\(||r||_2^2\\) when
        `preconditioner=None`.
  "
  [operator rhs preconditioner x  & {:keys [tol max_iter name]} ]
    (py/call-attr-kw linear-equations "conjugate_gradient" [operator rhs preconditioner x] {:tol tol :max_iter max_iter :name name }))
