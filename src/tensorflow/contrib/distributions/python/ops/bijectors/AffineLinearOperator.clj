(ns tensorflow.contrib.distributions.python.ops.bijectors.AffineLinearOperator
  "Compute `Y = g(X; shift, scale) = scale @ X + shift`.

  `shift` is a numeric `Tensor` and `scale` is a `LinearOperator`.

  If `X` is a scalar then the forward transformation is: `scale * X + shift`
  where `*` denotes the scalar product.

  Note: we don't always simply transpose `X` (but write it this way for
  brevity). Actually the input `X` undergoes the following transformation
  before being premultiplied by `scale`:

  1. If there are no sample dims, we call `X = tf.expand_dims(X, 0)`, i.e.,
     `new_sample_shape = [1]`. Otherwise do nothing.
  2. The sample shape is flattened to have one dimension, i.e.,
     `new_sample_shape = [n]` where `n = tf.reduce_prod(old_sample_shape)`.
  3. The sample dim is cyclically rotated left by 1, i.e.,
     `new_shape = [B1,...,Bb, k, n]` where `n` is as above, `k` is the
     event_shape, and `B1,...,Bb` are the batch shapes for each of `b` batch
     dimensions.

  (For more details see `shape.make_batch_of_event_sample_matrices`.)

  The result of the above transformation is that `X` can be regarded as a batch
  of matrices where each column is a draw from the distribution. After
  premultiplying by `scale`, we take the inverse of this procedure. The input
  `Y` also undergoes the same transformation before/after premultiplying by
  `inv(scale)`.

  Example Use:

  ```python
  linalg = tf.linalg

  x = [1., 2, 3]

  shift = [-1., 0., 1]
  diag = [1., 2, 3]
  scale = linalg.LinearOperatorDiag(diag)
  affine = AffineLinearOperator(shift, scale)
  # In this case, `forward` is equivalent to:
  # y = scale @ x + shift
  y = affine.forward(x)  # [0., 4, 10]

  shift = [2., 3, 1]
  tril = [[1., 0, 0],
          [2, 1, 0],
          [3, 2, 1]]
  scale = linalg.LinearOperatorLowerTriangular(tril)
  affine = AffineLinearOperator(shift, scale)
  # In this case, `forward` is equivalent to:
  # np.squeeze(np.matmul(tril, np.expand_dims(x, -1)), -1) + shift
  y = affine.forward(x)  # [3., 7, 11]
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
(defonce bijectors (import-module "tensorflow.contrib.distributions.python.ops.bijectors"))
(defn AffineLinearOperator 
  "Compute `Y = g(X; shift, scale) = scale @ X + shift`.

  `shift` is a numeric `Tensor` and `scale` is a `LinearOperator`.

  If `X` is a scalar then the forward transformation is: `scale * X + shift`
  where `*` denotes the scalar product.

  Note: we don't always simply transpose `X` (but write it this way for
  brevity). Actually the input `X` undergoes the following transformation
  before being premultiplied by `scale`:

  1. If there are no sample dims, we call `X = tf.expand_dims(X, 0)`, i.e.,
     `new_sample_shape = [1]`. Otherwise do nothing.
  2. The sample shape is flattened to have one dimension, i.e.,
     `new_sample_shape = [n]` where `n = tf.reduce_prod(old_sample_shape)`.
  3. The sample dim is cyclically rotated left by 1, i.e.,
     `new_shape = [B1,...,Bb, k, n]` where `n` is as above, `k` is the
     event_shape, and `B1,...,Bb` are the batch shapes for each of `b` batch
     dimensions.

  (For more details see `shape.make_batch_of_event_sample_matrices`.)

  The result of the above transformation is that `X` can be regarded as a batch
  of matrices where each column is a draw from the distribution. After
  premultiplying by `scale`, we take the inverse of this procedure. The input
  `Y` also undergoes the same transformation before/after premultiplying by
  `inv(scale)`.

  Example Use:

  ```python
  linalg = tf.linalg

  x = [1., 2, 3]

  shift = [-1., 0., 1]
  diag = [1., 2, 3]
  scale = linalg.LinearOperatorDiag(diag)
  affine = AffineLinearOperator(shift, scale)
  # In this case, `forward` is equivalent to:
  # y = scale @ x + shift
  y = affine.forward(x)  # [0., 4, 10]

  shift = [2., 3, 1]
  tril = [[1., 0, 0],
          [2, 1, 0],
          [3, 2, 1]]
  scale = linalg.LinearOperatorLowerTriangular(tril)
  affine = AffineLinearOperator(shift, scale)
  # In this case, `forward` is equivalent to:
  # np.squeeze(np.matmul(tril, np.expand_dims(x, -1)), -1) + shift
  y = affine.forward(x)  # [3., 7, 11]
  ```

  "
  [shift scale  & {:keys [validate_args name]} ]
    (py/call-attr-kw bijectors "AffineLinearOperator" [shift scale] {:validate_args validate_args :name name }))

(defn dtype 
  "dtype of `Tensor`s transformable by this distribution."
  [ self ]
    (py/call-attr self "dtype"))
(defn forward 
  "Returns the forward `Bijector` evaluation, i.e., X = g(Y).

    Args:
      x: `Tensor`. The input to the \"forward\" evaluation.
      name: The name to give this op.

    Returns:
      `Tensor`.

    Raises:
      TypeError: if `self.dtype` is specified and `x.dtype` is not
        `self.dtype`.
      NotImplementedError: if `_forward` is not implemented.
    "
  [self x  & {:keys [name]} ]
    (py/call-attr-kw self "forward" [x] {:name name }))

(defn forward-event-shape 
  "Shape of a single sample from a single batch as a `TensorShape`.

    Same meaning as `forward_event_shape_tensor`. May be only partially defined.

    Args:
      input_shape: `TensorShape` indicating event-portion shape passed into
        `forward` function.

    Returns:
      forward_event_shape_tensor: `TensorShape` indicating event-portion shape
        after applying `forward`. Possibly unknown.
    "
  [ self input_shape ]
  (py/call-attr self "forward_event_shape"  self input_shape ))
(defn forward-event-shape-tensor 
  "Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

    Args:
      input_shape: `Tensor`, `int32` vector indicating event-portion shape
        passed into `forward` function.
      name: name to give to the op

    Returns:
      forward_event_shape_tensor: `Tensor`, `int32` vector indicating
        event-portion shape after applying `forward`.
    "
  [self input_shape  & {:keys [name]} ]
    (py/call-attr-kw self "forward_event_shape_tensor" [input_shape] {:name name }))
(defn forward-log-det-jacobian 
  "Returns both the forward_log_det_jacobian.

    Args:
      x: `Tensor`. The input to the \"forward\" Jacobian determinant evaluation.
      event_ndims: Number of dimensions in the probabilistic events being
        transformed. Must be greater than or equal to
        `self.forward_min_event_ndims`. The result is summed over the final
        dimensions to produce a scalar Jacobian determinant for each event,
        i.e. it has shape `x.shape.ndims - event_ndims` dimensions.
      name: The name to give this op.

    Returns:
      `Tensor`, if this bijector is injective.
        If not injective this is not implemented.

    Raises:
      TypeError: if `self.dtype` is specified and `y.dtype` is not
        `self.dtype`.
      NotImplementedError: if neither `_forward_log_det_jacobian`
        nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented, or
        this is a non-injective bijector.
    "
  [self x event_ndims  & {:keys [name]} ]
    (py/call-attr-kw self "forward_log_det_jacobian" [x event_ndims] {:name name }))

(defn forward-min-event-ndims 
  "Returns the minimal number of dimensions bijector.forward operates on."
  [ self ]
    (py/call-attr self "forward_min_event_ndims"))

(defn graph-parents 
  "Returns this `Bijector`'s graph_parents as a Python list."
  [ self ]
    (py/call-attr self "graph_parents"))
(defn inverse 
  "Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

    Args:
      y: `Tensor`. The input to the \"inverse\" evaluation.
      name: The name to give this op.

    Returns:
      `Tensor`, if this bijector is injective.
        If not injective, returns the k-tuple containing the unique
        `k` points `(x1, ..., xk)` such that `g(xi) = y`.

    Raises:
      TypeError: if `self.dtype` is specified and `y.dtype` is not
        `self.dtype`.
      NotImplementedError: if `_inverse` is not implemented.
    "
  [self y  & {:keys [name]} ]
    (py/call-attr-kw self "inverse" [y] {:name name }))

(defn inverse-event-shape 
  "Shape of a single sample from a single batch as a `TensorShape`.

    Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

    Args:
      output_shape: `TensorShape` indicating event-portion shape passed into
        `inverse` function.

    Returns:
      inverse_event_shape_tensor: `TensorShape` indicating event-portion shape
        after applying `inverse`. Possibly unknown.
    "
  [ self output_shape ]
  (py/call-attr self "inverse_event_shape"  self output_shape ))
(defn inverse-event-shape-tensor 
  "Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

    Args:
      output_shape: `Tensor`, `int32` vector indicating event-portion shape
        passed into `inverse` function.
      name: name to give to the op

    Returns:
      inverse_event_shape_tensor: `Tensor`, `int32` vector indicating
        event-portion shape after applying `inverse`.
    "
  [self output_shape  & {:keys [name]} ]
    (py/call-attr-kw self "inverse_event_shape_tensor" [output_shape] {:name name }))
(defn inverse-log-det-jacobian 
  "Returns the (log o det o Jacobian o inverse)(y).

    Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

    Note that `forward_log_det_jacobian` is the negative of this function,
    evaluated at `g^{-1}(y)`.

    Args:
      y: `Tensor`. The input to the \"inverse\" Jacobian determinant evaluation.
      event_ndims: Number of dimensions in the probabilistic events being
        transformed. Must be greater than or equal to
        `self.inverse_min_event_ndims`. The result is summed over the final
        dimensions to produce a scalar Jacobian determinant for each event,
        i.e. it has shape `y.shape.ndims - event_ndims` dimensions.
      name: The name to give this op.

    Returns:
      `Tensor`, if this bijector is injective.
        If not injective, returns the tuple of local log det
        Jacobians, `log(det(Dg_i^{-1}(y)))`, where `g_i` is the restriction
        of `g` to the `ith` partition `Di`.

    Raises:
      TypeError: if `self.dtype` is specified and `y.dtype` is not
        `self.dtype`.
      NotImplementedError: if `_inverse_log_det_jacobian` is not implemented.
    "
  [self y event_ndims  & {:keys [name]} ]
    (py/call-attr-kw self "inverse_log_det_jacobian" [y event_ndims] {:name name }))

(defn inverse-min-event-ndims 
  "Returns the minimal number of dimensions bijector.inverse operates on."
  [ self ]
    (py/call-attr self "inverse_min_event_ndims"))

(defn is-constant-jacobian 
  "Returns true iff the Jacobian matrix is not a function of x.

    Note: Jacobian matrix is either constant for both forward and inverse or
    neither.

    Returns:
      is_constant_jacobian: Python `bool`.
    "
  [ self ]
    (py/call-attr self "is_constant_jacobian"))

(defn name 
  "Returns the string name of this `Bijector`."
  [ self ]
    (py/call-attr self "name"))

(defn scale 
  "The `scale` `LinearOperator` in `Y = scale @ X + shift`."
  [ self ]
    (py/call-attr self "scale"))

(defn shift 
  "The `shift` `Tensor` in `Y = scale @ X + shift`."
  [ self ]
    (py/call-attr self "shift"))

(defn validate-args 
  "Returns True if Tensor arguments will be validated."
  [ self ]
    (py/call-attr self "validate_args"))
