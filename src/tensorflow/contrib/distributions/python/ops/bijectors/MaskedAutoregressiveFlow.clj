(ns tensorflow.contrib.distributions.python.ops.bijectors.MaskedAutoregressiveFlow
  "Affine MaskedAutoregressiveFlow bijector for vector-valued events.

  The affine autoregressive flow [(Papamakarios et al., 2016)][3] provides a
  relatively simple framework for user-specified (deep) architectures to learn
  a distribution over vector-valued events. Regarding terminology,

    \"Autoregressive models decompose the joint density as a product of
    conditionals, and model each conditional in turn. Normalizing flows
    transform a base density (e.g. a standard Gaussian) into the target density
    by an invertible transformation with tractable Jacobian.\"
    [(Papamakarios et al., 2016)][3]

  In other words, the \"autoregressive property\" is equivalent to the
  decomposition, `p(x) = prod{ p(x[i] | x[0:i]) : i=0, ..., d }`. The provided
  `shift_and_log_scale_fn`, `masked_autoregressive_default_template`, achieves
  this property by zeroing out weights in its `masked_dense` layers.

  In the `tfp` framework, a \"normalizing flow\" is implemented as a
  `tfp.bijectors.Bijector`. The `forward` \"autoregression\"
  is implemented using a `tf.while_loop` and a deep neural network (DNN) with
  masked weights such that the autoregressive property is automatically met in
  the `inverse`.

  A `TransformedDistribution` using `MaskedAutoregressiveFlow(...)` uses the
  (expensive) forward-mode calculation to draw samples and the (cheap)
  reverse-mode calculation to compute log-probabilities. Conversely, a
  `TransformedDistribution` using `Invert(MaskedAutoregressiveFlow(...))` uses
  the (expensive) forward-mode calculation to compute log-probabilities and the
  (cheap) reverse-mode calculation to compute samples.  See \"Example Use\"
  [below] for more details.

  Given a `shift_and_log_scale_fn`, the forward and inverse transformations are
  (a sequence of) affine transformations. A \"valid\" `shift_and_log_scale_fn`
  must compute each `shift` (aka `loc` or \"mu\" in [Germain et al. (2015)][1])
  and `log(scale)` (aka \"alpha\" in [Germain et al. (2015)][1]) such that each
  are broadcastable with the arguments to `forward` and `inverse`, i.e., such
  that the calculations in `forward`, `inverse` [below] are possible.

  For convenience, `masked_autoregressive_default_template` is offered as a
  possible `shift_and_log_scale_fn` function. It implements the MADE
  architecture [(Germain et al., 2015)][1]. MADE is a feed-forward network that
  computes a `shift` and `log(scale)` using `masked_dense` layers in a deep
  neural network. Weights are masked to ensure the autoregressive property. It
  is possible that this architecture is suboptimal for your task. To build
  alternative networks, either change the arguments to
  `masked_autoregressive_default_template`, use the `masked_dense` function to
  roll-out your own, or use some other architecture, e.g., using `tf.layers`.

  Warning: no attempt is made to validate that the `shift_and_log_scale_fn`
  enforces the \"autoregressive property\".

  Assuming `shift_and_log_scale_fn` has valid shape and autoregressive
  semantics, the forward transformation is

  ```python
  def forward(x):
    y = zeros_like(x)
    event_size = x.shape[-1]
    for _ in range(event_size):
      shift, log_scale = shift_and_log_scale_fn(y)
      y = x * math_ops.exp(log_scale) + shift
    return y
  ```

  and the inverse transformation is

  ```python
  def inverse(y):
    shift, log_scale = shift_and_log_scale_fn(y)
    return (y - shift) / math_ops.exp(log_scale)
  ```

  Notice that the `inverse` does not need a for-loop. This is because in the
  forward pass each calculation of `shift` and `log_scale` is based on the `y`
  calculated so far (not `x`). In the `inverse`, the `y` is fully known, thus is
  equivalent to the scaling used in `forward` after `event_size` passes, i.e.,
  the \"last\" `y` used to compute `shift`, `log_scale`. (Roughly speaking, this
  also proves the transform is bijective.)

  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions
  tfb = tfp.bijectors

  dims = 5

  # A common choice for a normalizing flow is to use a Gaussian for the base
  # distribution. (However, any continuous distribution would work.) E.g.,
  maf = tfd.TransformedDistribution(
      distribution=tfd.Normal(loc=0., scale=1.),
      bijector=tfb.MaskedAutoregressiveFlow(
          shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
              hidden_layers=[512, 512])),
      event_shape=[dims])

  x = maf.sample()  # Expensive; uses `tf.while_loop`, no Bijector caching.
  maf.log_prob(x)   # Almost free; uses Bijector caching.
  maf.log_prob(0.)  # Cheap; no `tf.while_loop` despite no Bijector caching.

  # [Papamakarios et al. (2016)][3] also describe an Inverse Autoregressive
  # Flow [(Kingma et al., 2016)][2]:
  iaf = tfd.TransformedDistribution(
      distribution=tfd.Normal(loc=0., scale=1.),
      bijector=tfb.Invert(tfb.MaskedAutoregressiveFlow(
          shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
              hidden_layers=[512, 512]))),
      event_shape=[dims])

  x = iaf.sample()  # Cheap; no `tf.while_loop` despite no Bijector caching.
  iaf.log_prob(x)   # Almost free; uses Bijector caching.
  iaf.log_prob(0.)  # Expensive; uses `tf.while_loop`, no Bijector caching.

  # In many (if not most) cases the default `shift_and_log_scale_fn` will be a
  # poor choice. Here's an example of using a \"shift only\" version and with a
  # different number/depth of hidden layers.
  shift_only = True
  maf_no_scale_hidden2 = tfd.TransformedDistribution(
      distribution=tfd.Normal(loc=0., scale=1.),
      bijector=tfb.MaskedAutoregressiveFlow(
          tfb.masked_autoregressive_default_template(
              hidden_layers=[32],
              shift_only=shift_only),
          is_constant_jacobian=shift_only),
      event_shape=[dims])
  ```

  #### References

  [1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
       Masked Autoencoder for Distribution Estimation. In _International
       Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509

  [2]: Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya
       Sutskever, and Max Welling. Improving Variational Inference with Inverse
       Autoregressive Flow. In _Neural Information Processing Systems_, 2016.
       https://arxiv.org/abs/1606.04934

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

(defn MaskedAutoregressiveFlow 
  "Affine MaskedAutoregressiveFlow bijector for vector-valued events.

  The affine autoregressive flow [(Papamakarios et al., 2016)][3] provides a
  relatively simple framework for user-specified (deep) architectures to learn
  a distribution over vector-valued events. Regarding terminology,

    \"Autoregressive models decompose the joint density as a product of
    conditionals, and model each conditional in turn. Normalizing flows
    transform a base density (e.g. a standard Gaussian) into the target density
    by an invertible transformation with tractable Jacobian.\"
    [(Papamakarios et al., 2016)][3]

  In other words, the \"autoregressive property\" is equivalent to the
  decomposition, `p(x) = prod{ p(x[i] | x[0:i]) : i=0, ..., d }`. The provided
  `shift_and_log_scale_fn`, `masked_autoregressive_default_template`, achieves
  this property by zeroing out weights in its `masked_dense` layers.

  In the `tfp` framework, a \"normalizing flow\" is implemented as a
  `tfp.bijectors.Bijector`. The `forward` \"autoregression\"
  is implemented using a `tf.while_loop` and a deep neural network (DNN) with
  masked weights such that the autoregressive property is automatically met in
  the `inverse`.

  A `TransformedDistribution` using `MaskedAutoregressiveFlow(...)` uses the
  (expensive) forward-mode calculation to draw samples and the (cheap)
  reverse-mode calculation to compute log-probabilities. Conversely, a
  `TransformedDistribution` using `Invert(MaskedAutoregressiveFlow(...))` uses
  the (expensive) forward-mode calculation to compute log-probabilities and the
  (cheap) reverse-mode calculation to compute samples.  See \"Example Use\"
  [below] for more details.

  Given a `shift_and_log_scale_fn`, the forward and inverse transformations are
  (a sequence of) affine transformations. A \"valid\" `shift_and_log_scale_fn`
  must compute each `shift` (aka `loc` or \"mu\" in [Germain et al. (2015)][1])
  and `log(scale)` (aka \"alpha\" in [Germain et al. (2015)][1]) such that each
  are broadcastable with the arguments to `forward` and `inverse`, i.e., such
  that the calculations in `forward`, `inverse` [below] are possible.

  For convenience, `masked_autoregressive_default_template` is offered as a
  possible `shift_and_log_scale_fn` function. It implements the MADE
  architecture [(Germain et al., 2015)][1]. MADE is a feed-forward network that
  computes a `shift` and `log(scale)` using `masked_dense` layers in a deep
  neural network. Weights are masked to ensure the autoregressive property. It
  is possible that this architecture is suboptimal for your task. To build
  alternative networks, either change the arguments to
  `masked_autoregressive_default_template`, use the `masked_dense` function to
  roll-out your own, or use some other architecture, e.g., using `tf.layers`.

  Warning: no attempt is made to validate that the `shift_and_log_scale_fn`
  enforces the \"autoregressive property\".

  Assuming `shift_and_log_scale_fn` has valid shape and autoregressive
  semantics, the forward transformation is

  ```python
  def forward(x):
    y = zeros_like(x)
    event_size = x.shape[-1]
    for _ in range(event_size):
      shift, log_scale = shift_and_log_scale_fn(y)
      y = x * math_ops.exp(log_scale) + shift
    return y
  ```

  and the inverse transformation is

  ```python
  def inverse(y):
    shift, log_scale = shift_and_log_scale_fn(y)
    return (y - shift) / math_ops.exp(log_scale)
  ```

  Notice that the `inverse` does not need a for-loop. This is because in the
  forward pass each calculation of `shift` and `log_scale` is based on the `y`
  calculated so far (not `x`). In the `inverse`, the `y` is fully known, thus is
  equivalent to the scaling used in `forward` after `event_size` passes, i.e.,
  the \"last\" `y` used to compute `shift`, `log_scale`. (Roughly speaking, this
  also proves the transform is bijective.)

  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions
  tfb = tfp.bijectors

  dims = 5

  # A common choice for a normalizing flow is to use a Gaussian for the base
  # distribution. (However, any continuous distribution would work.) E.g.,
  maf = tfd.TransformedDistribution(
      distribution=tfd.Normal(loc=0., scale=1.),
      bijector=tfb.MaskedAutoregressiveFlow(
          shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
              hidden_layers=[512, 512])),
      event_shape=[dims])

  x = maf.sample()  # Expensive; uses `tf.while_loop`, no Bijector caching.
  maf.log_prob(x)   # Almost free; uses Bijector caching.
  maf.log_prob(0.)  # Cheap; no `tf.while_loop` despite no Bijector caching.

  # [Papamakarios et al. (2016)][3] also describe an Inverse Autoregressive
  # Flow [(Kingma et al., 2016)][2]:
  iaf = tfd.TransformedDistribution(
      distribution=tfd.Normal(loc=0., scale=1.),
      bijector=tfb.Invert(tfb.MaskedAutoregressiveFlow(
          shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
              hidden_layers=[512, 512]))),
      event_shape=[dims])

  x = iaf.sample()  # Cheap; no `tf.while_loop` despite no Bijector caching.
  iaf.log_prob(x)   # Almost free; uses Bijector caching.
  iaf.log_prob(0.)  # Expensive; uses `tf.while_loop`, no Bijector caching.

  # In many (if not most) cases the default `shift_and_log_scale_fn` will be a
  # poor choice. Here's an example of using a \"shift only\" version and with a
  # different number/depth of hidden layers.
  shift_only = True
  maf_no_scale_hidden2 = tfd.TransformedDistribution(
      distribution=tfd.Normal(loc=0., scale=1.),
      bijector=tfb.MaskedAutoregressiveFlow(
          tfb.masked_autoregressive_default_template(
              hidden_layers=[32],
              shift_only=shift_only),
          is_constant_jacobian=shift_only),
      event_shape=[dims])
  ```

  #### References

  [1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
       Masked Autoencoder for Distribution Estimation. In _International
       Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509

  [2]: Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya
       Sutskever, and Max Welling. Improving Variational Inference with Inverse
       Autoregressive Flow. In _Neural Information Processing Systems_, 2016.
       https://arxiv.org/abs/1606.04934

  [3]: George Papamakarios, Theo Pavlakou, and Iain Murray. Masked
       Autoregressive Flow for Density Estimation. In _Neural Information
       Processing Systems_, 2017. https://arxiv.org/abs/1705.07057
  "
  [shift_and_log_scale_fn & {:keys [is_constant_jacobian validate_args unroll_loop name]
                       :or {name None}} ]
    (py/call-attr-kw bijectors "MaskedAutoregressiveFlow" [shift_and_log_scale_fn] {:is_constant_jacobian is_constant_jacobian :validate_args validate_args :unroll_loop unroll_loop :name name }))

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
