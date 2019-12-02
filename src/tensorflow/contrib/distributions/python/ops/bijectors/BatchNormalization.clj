(ns tensorflow.contrib.distributions.python.ops.bijectors.BatchNormalization
  "Compute `Y = g(X) s.t.

  X = g^-1(Y) = (Y - mean(Y)) / std(Y)`.

  Applies Batch Normalization [(Ioffe and Szegedy, 2015)][1] to samples from a
  data distribution. This can be used to stabilize training of normalizing
  flows ([Papamakarios et al., 2016][3]; [Dinh et al., 2017][2])

  When training Deep Neural Networks (DNNs), it is common practice to
  normalize or whiten features by shifting them to have zero mean and
  scaling them to have unit variance.

  The `inverse()` method of the `BatchNormalization` bijector, which is used in
  the log-likelihood computation of data samples, implements the normalization
  procedure (shift-and-scale) using the mean and standard deviation of the
  current minibatch.

  Conversely, the `forward()` method of the bijector de-normalizes samples (e.g.
  `X*std(Y) + mean(Y)` with the running-average mean and standard deviation
  computed at training-time. De-normalization is useful for sampling.

  ```python

  dist = tfd.TransformedDistribution(
      distribution=tfd.Normal()),
      bijector=tfb.BatchNorm())

  y = tfd.MultivariateNormalDiag(loc=1., scale=2.).sample(100)  # ~ N(1, 2)
  x = dist.bijector.inverse(y)  # ~ N(0, 1)
  y = dist.sample()  # ~ N(1, 2)
  ```

  During training time, `BatchNorm.inverse` and `BatchNorm.forward` are not
  guaranteed to be inverses of each other because `inverse(y)` uses statistics
  of the current minibatch, while `forward(x)` uses running-average statistics
  accumulated from training. In other words,
  `BatchNorm.inverse(BatchNorm.forward(...))` and
  `BatchNorm.forward(BatchNorm.inverse(...))` will be identical when
  `training=False` but may be different when `training=True`.

  #### References

  [1]: Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating
       Deep Network Training by Reducing Internal Covariate Shift. In
       _International Conference on Machine Learning_, 2015.
       https://arxiv.org/abs/1502.03167

  [2]: Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density Estimation
       using Real NVP. In _International Conference on Learning
       Representations_, 2017. https://arxiv.org/abs/1605.08803

  [3]: George Papamakarios, Theo Pavlakou, and Iain Murray. Masked
       Autoregressive Flow for Density Estimation. In _Neural Information
       Processing Systems_, 2017. https://arxiv.org/abs/1705.07057
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
(defn BatchNormalization 
  "Compute `Y = g(X) s.t.

  X = g^-1(Y) = (Y - mean(Y)) / std(Y)`.

  Applies Batch Normalization [(Ioffe and Szegedy, 2015)][1] to samples from a
  data distribution. This can be used to stabilize training of normalizing
  flows ([Papamakarios et al., 2016][3]; [Dinh et al., 2017][2])

  When training Deep Neural Networks (DNNs), it is common practice to
  normalize or whiten features by shifting them to have zero mean and
  scaling them to have unit variance.

  The `inverse()` method of the `BatchNormalization` bijector, which is used in
  the log-likelihood computation of data samples, implements the normalization
  procedure (shift-and-scale) using the mean and standard deviation of the
  current minibatch.

  Conversely, the `forward()` method of the bijector de-normalizes samples (e.g.
  `X*std(Y) + mean(Y)` with the running-average mean and standard deviation
  computed at training-time. De-normalization is useful for sampling.

  ```python

  dist = tfd.TransformedDistribution(
      distribution=tfd.Normal()),
      bijector=tfb.BatchNorm())

  y = tfd.MultivariateNormalDiag(loc=1., scale=2.).sample(100)  # ~ N(1, 2)
  x = dist.bijector.inverse(y)  # ~ N(0, 1)
  y = dist.sample()  # ~ N(1, 2)
  ```

  During training time, `BatchNorm.inverse` and `BatchNorm.forward` are not
  guaranteed to be inverses of each other because `inverse(y)` uses statistics
  of the current minibatch, while `forward(x)` uses running-average statistics
  accumulated from training. In other words,
  `BatchNorm.inverse(BatchNorm.forward(...))` and
  `BatchNorm.forward(BatchNorm.inverse(...))` will be identical when
  `training=False` but may be different when `training=True`.

  #### References

  [1]: Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating
       Deep Network Training by Reducing Internal Covariate Shift. In
       _International Conference on Machine Learning_, 2015.
       https://arxiv.org/abs/1502.03167

  [2]: Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density Estimation
       using Real NVP. In _International Conference on Learning
       Representations_, 2017. https://arxiv.org/abs/1605.08803

  [3]: George Papamakarios, Theo Pavlakou, and Iain Murray. Masked
       Autoregressive Flow for Density Estimation. In _Neural Information
       Processing Systems_, 2017. https://arxiv.org/abs/1705.07057
  "
  [batchnorm_layer  & {:keys [training validate_args name]} ]
    (py/call-attr-kw bijectors "BatchNormalization" [batchnorm_layer] {:training training :validate_args validate_args :name name }))

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

(defn validate-args 
  "Returns True if Tensor arguments will be validated."
  [ self ]
    (py/call-attr self "validate_args"))
