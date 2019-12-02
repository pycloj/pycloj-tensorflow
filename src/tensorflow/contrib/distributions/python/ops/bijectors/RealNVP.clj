(ns tensorflow.contrib.distributions.python.ops.bijectors.RealNVP
  "RealNVP \"affine coupling layer\" for vector-valued events.

  Real NVP models a normalizing flow on a `D`-dimensional distribution via a
  single `D-d`-dimensional conditional distribution [(Dinh et al., 2017)][1]:

  `y[d:D] = y[d:D] * math_ops.exp(log_scale_fn(y[d:D])) + shift_fn(y[d:D])`
  `y[0:d] = x[0:d]`

  The last `D-d` units are scaled and shifted based on the first `d` units only,
  while the first `d` units are 'masked' and left unchanged. Real NVP's
  `shift_and_log_scale_fn` computes vector-valued quantities. For
  scale-and-shift transforms that do not depend on any masked units, i.e.
  `d=0`, use the `tfb.Affine` bijector with learned parameters instead.

  Masking is currently only supported for base distributions with
  `event_ndims=1`. For more sophisticated masking schemes like checkerboard or
  channel-wise masking [(Papamakarios et al., 2016)[4], use the `tfb.Permute`
  bijector to re-order desired masked units into the first `d` units. For base
  distributions with `event_ndims > 1`, use the `tfb.Reshape` bijector to
  flatten the event shape.

  Recall that the MAF bijector [(Papamakarios et al., 2016)][4] implements a
  normalizing flow via an autoregressive transformation. MAF and IAF have
  opposite computational tradeoffs - MAF can train all units in parallel but
  must sample units sequentially, while IAF must train units sequentially but
  can sample in parallel. In contrast, Real NVP can compute both forward and
  inverse computations in parallel. However, the lack of an autoregressive
  transformations makes it less expressive on a per-bijector basis.

  A \"valid\" `shift_and_log_scale_fn` must compute each `shift` (aka `loc` or
  \"mu\" in [Papamakarios et al. (2016)][4]) and `log(scale)` (aka \"alpha\" in
  [Papamakarios et al. (2016)][4]) such that each are broadcastable with the
  arguments to `forward` and `inverse`, i.e., such that the calculations in
  `forward`, `inverse` [below] are possible. For convenience,
  `real_nvp_default_nvp` is offered as a possible `shift_and_log_scale_fn`
  function.

  NICE [(Dinh et al., 2014)][2] is a special case of the Real NVP bijector
  which discards the scale transformation, resulting in a constant-time
  inverse-log-determinant-Jacobian. To use a NICE bijector instead of Real
  NVP, `shift_and_log_scale_fn` should return `(shift, None)`, and
  `is_constant_jacobian` should be set to `True` in the `RealNVP` constructor.
  Calling `real_nvp_default_template` with `shift_only=True` returns one such
  NICE-compatible `shift_and_log_scale_fn`.

  Caching: the scalar input depth `D` of the base distribution is not known at
  construction time. The first call to any of `forward(x)`, `inverse(x)`,
  `inverse_log_det_jacobian(x)`, or `forward_log_det_jacobian(x)` memoizes
  `D`, which is re-used in subsequent calls. This shape must be known prior to
  graph execution (which is the case if using tf.layers).

  #### Example Use

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions
  tfb = tfp.bijectors

  # A common choice for a normalizing flow is to use a Gaussian for the base
  # distribution. (However, any continuous distribution would work.) E.g.,
  num_dims = 3
  num_samples = 1
  nvp = tfd.TransformedDistribution(
      distribution=tfd.MultivariateNormalDiag(loc=np.zeros(num_dims)),
      bijector=tfb.RealNVP(
          num_masked=2,
          shift_and_log_scale_fn=tfb.real_nvp_default_template(
              hidden_layers=[512, 512])))

  x = nvp.sample(num_samples)
  nvp.log_prob(x)
  nvp.log_prob(np.zeros([num_samples, num_dims]))
  ```

  For more examples, see [Jang (2018)][3].

  #### References

  [1]: Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density Estimation
       using Real NVP. In _International Conference on Learning
       Representations_, 2017. https://arxiv.org/abs/1605.08803

  [2]: Laurent Dinh, David Krueger, and Yoshua Bengio. NICE: Non-linear
       Independent Components Estimation. _arXiv preprint arXiv:1410.8516_,
       2014. https://arxiv.org/abs/1410.8516

  [3]: Eric Jang. Normalizing Flows Tutorial, Part 2: Modern Normalizing Flows.
       _Technical Report_, 2018. http://blog.evjang.com/2018/01/nf2.html

  [4]: George Papamakarios, Theo Pavlakou, and Iain Murray. Masked
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

(defn RealNVP 
  "RealNVP \"affine coupling layer\" for vector-valued events.

  Real NVP models a normalizing flow on a `D`-dimensional distribution via a
  single `D-d`-dimensional conditional distribution [(Dinh et al., 2017)][1]:

  `y[d:D] = y[d:D] * math_ops.exp(log_scale_fn(y[d:D])) + shift_fn(y[d:D])`
  `y[0:d] = x[0:d]`

  The last `D-d` units are scaled and shifted based on the first `d` units only,
  while the first `d` units are 'masked' and left unchanged. Real NVP's
  `shift_and_log_scale_fn` computes vector-valued quantities. For
  scale-and-shift transforms that do not depend on any masked units, i.e.
  `d=0`, use the `tfb.Affine` bijector with learned parameters instead.

  Masking is currently only supported for base distributions with
  `event_ndims=1`. For more sophisticated masking schemes like checkerboard or
  channel-wise masking [(Papamakarios et al., 2016)[4], use the `tfb.Permute`
  bijector to re-order desired masked units into the first `d` units. For base
  distributions with `event_ndims > 1`, use the `tfb.Reshape` bijector to
  flatten the event shape.

  Recall that the MAF bijector [(Papamakarios et al., 2016)][4] implements a
  normalizing flow via an autoregressive transformation. MAF and IAF have
  opposite computational tradeoffs - MAF can train all units in parallel but
  must sample units sequentially, while IAF must train units sequentially but
  can sample in parallel. In contrast, Real NVP can compute both forward and
  inverse computations in parallel. However, the lack of an autoregressive
  transformations makes it less expressive on a per-bijector basis.

  A \"valid\" `shift_and_log_scale_fn` must compute each `shift` (aka `loc` or
  \"mu\" in [Papamakarios et al. (2016)][4]) and `log(scale)` (aka \"alpha\" in
  [Papamakarios et al. (2016)][4]) such that each are broadcastable with the
  arguments to `forward` and `inverse`, i.e., such that the calculations in
  `forward`, `inverse` [below] are possible. For convenience,
  `real_nvp_default_nvp` is offered as a possible `shift_and_log_scale_fn`
  function.

  NICE [(Dinh et al., 2014)][2] is a special case of the Real NVP bijector
  which discards the scale transformation, resulting in a constant-time
  inverse-log-determinant-Jacobian. To use a NICE bijector instead of Real
  NVP, `shift_and_log_scale_fn` should return `(shift, None)`, and
  `is_constant_jacobian` should be set to `True` in the `RealNVP` constructor.
  Calling `real_nvp_default_template` with `shift_only=True` returns one such
  NICE-compatible `shift_and_log_scale_fn`.

  Caching: the scalar input depth `D` of the base distribution is not known at
  construction time. The first call to any of `forward(x)`, `inverse(x)`,
  `inverse_log_det_jacobian(x)`, or `forward_log_det_jacobian(x)` memoizes
  `D`, which is re-used in subsequent calls. This shape must be known prior to
  graph execution (which is the case if using tf.layers).

  #### Example Use

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions
  tfb = tfp.bijectors

  # A common choice for a normalizing flow is to use a Gaussian for the base
  # distribution. (However, any continuous distribution would work.) E.g.,
  num_dims = 3
  num_samples = 1
  nvp = tfd.TransformedDistribution(
      distribution=tfd.MultivariateNormalDiag(loc=np.zeros(num_dims)),
      bijector=tfb.RealNVP(
          num_masked=2,
          shift_and_log_scale_fn=tfb.real_nvp_default_template(
              hidden_layers=[512, 512])))

  x = nvp.sample(num_samples)
  nvp.log_prob(x)
  nvp.log_prob(np.zeros([num_samples, num_dims]))
  ```

  For more examples, see [Jang (2018)][3].

  #### References

  [1]: Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density Estimation
       using Real NVP. In _International Conference on Learning
       Representations_, 2017. https://arxiv.org/abs/1605.08803

  [2]: Laurent Dinh, David Krueger, and Yoshua Bengio. NICE: Non-linear
       Independent Components Estimation. _arXiv preprint arXiv:1410.8516_,
       2014. https://arxiv.org/abs/1410.8516

  [3]: Eric Jang. Normalizing Flows Tutorial, Part 2: Modern Normalizing Flows.
       _Technical Report_, 2018. http://blog.evjang.com/2018/01/nf2.html

  [4]: George Papamakarios, Theo Pavlakou, and Iain Murray. Masked
       Autoregressive Flow for Density Estimation. In _Neural Information
       Processing Systems_, 2017. https://arxiv.org/abs/1705.07057
  "
  [num_masked shift_and_log_scale_fn & {:keys [is_constant_jacobian validate_args name]
                       :or {name None}} ]
    (py/call-attr-kw bijectors "RealNVP" [num_masked shift_and_log_scale_fn] {:is_constant_jacobian is_constant_jacobian :validate_args validate_args :name name }))

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
