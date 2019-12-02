(ns tensorflow.contrib.distributions.python.ops.bijectors
  "Bijector Ops.

Use [tfp.bijectors](/probability/api_docs/python/tfp/bijectors) instead.

@@AbsoluteValue
@@Affine
@@AffineLinearOperator
@@AffineScalar
@@Bijector
@@BatchNormalization
@@Chain
@@CholeskyOuterProduct
@@ConditionalBijector
@@Exp
@@FillTriangular
@@Gumbel
@@Identity
@@Inline
@@Invert
@@Kumaraswamy
@@MaskedAutoregressiveFlow
@@MatrixInverseTriL
@@Ordered
@@Permute
@@PowerTransform
@@RealNVP
@@Reshape
@@ScaleTriL
@@Sigmoid
@@SinhArcsinh
@@SoftmaxCentered
@@Softplus
@@Softsign
@@Square
@@TransformDiagonal
@@Weibull

@@masked_autoregressive_default_template
@@masked_dense
@@real_nvp_default_template
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

(defn masked-autoregressive-default-template 
  "Build the Masked Autoregressive Density Estimator (Germain et al., 2015). (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2018-10-01.
Instructions for updating:
The TensorFlow Distributions library has moved to TensorFlow Probability (https://github.com/tensorflow/probability). You should update all references to use `tfp.distributions` instead of `tf.contrib.distributions`.

This will be wrapped in a make_template to ensure the variables are only
created once. It takes the input and returns the `loc` (\"mu\" in [Germain et
al. (2015)][1]) and `log_scale` (\"alpha\" in [Germain et al. (2015)][1]) from
the MADE network.

Warning: This function uses `masked_dense` to create randomly initialized
`tf.Variables`. It is presumed that these will be fit, just as you would any
other neural architecture which uses `tf.compat.v1.layers.dense`.

#### About Hidden Layers

Each element of `hidden_layers` should be greater than the `input_depth`
(i.e., `input_depth = tf.shape(input)[-1]` where `input` is the input to the
neural network). This is necessary to ensure the autoregressivity property.

#### About Clipping

This function also optionally clips the `log_scale` (but possibly not its
gradient). This is useful because if `log_scale` is too small/large it might
underflow/overflow making it impossible for the `MaskedAutoregressiveFlow`
bijector to implement a bijection. Additionally, the `log_scale_clip_gradient`
`bool` indicates whether the gradient should also be clipped. The default does
not clip the gradient; this is useful because it still provides gradient
information (for fitting) yet solves the numerical stability problem. I.e.,
`log_scale_clip_gradient = False` means
`grad[exp(clip(x))] = grad[x] exp(clip(x))` rather than the usual
`grad[clip(x)] exp(clip(x))`.

Args:
  hidden_layers: Python `list`-like of non-negative integer, scalars
    indicating the number of units in each hidden layer. Default: `[512, 512].
  shift_only: Python `bool` indicating if only the `shift` term shall be
    computed. Default: `False`.
  activation: Activation function (callable). Explicitly setting to `None`
    implies a linear activation.
  log_scale_min_clip: `float`-like scalar `Tensor`, or a `Tensor` with the
    same shape as `log_scale`. The minimum value to clip by. Default: -5.
  log_scale_max_clip: `float`-like scalar `Tensor`, or a `Tensor` with the
    same shape as `log_scale`. The maximum value to clip by. Default: 3.
  log_scale_clip_gradient: Python `bool` indicating that the gradient of
    `tf.clip_by_value` should be preserved. Default: `False`.
  name: A name for ops managed by this function. Default:
    \"masked_autoregressive_default_template\".
  *args: `tf.compat.v1.layers.dense` arguments.
  **kwargs: `tf.compat.v1.layers.dense` keyword arguments.

Returns:
  shift: `Float`-like `Tensor` of shift terms (the \"mu\" in
    [Germain et al.  (2015)][1]).
  log_scale: `Float`-like `Tensor` of log(scale) terms (the \"alpha\" in
    [Germain et al. (2015)][1]).

Raises:
  NotImplementedError: if rightmost dimension of `inputs` is unknown prior to
    graph execution.

#### References

[1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
     Masked Autoencoder for Distribution Estimation. In _International
     Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509"
  [hidden_layers & {:keys [shift_only activation log_scale_min_clip log_scale_max_clip log_scale_clip_gradient name]
                       :or {name None}} ]
    (py/call-attr-kw bijectors "masked_autoregressive_default_template" [hidden_layers] {:shift_only shift_only :activation activation :log_scale_min_clip log_scale_min_clip :log_scale_max_clip log_scale_max_clip :log_scale_clip_gradient log_scale_clip_gradient :name name }))

(defn masked-dense 
  "A autoregressively masked dense layer. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2018-10-01.
Instructions for updating:
The TensorFlow Distributions library has moved to TensorFlow Probability (https://github.com/tensorflow/probability). You should update all references to use `tfp.distributions` instead of `tf.contrib.distributions`.

Analogous to `tf.compat.v1.layers.dense`.

See [Germain et al. (2015)][1] for detailed explanation.

Arguments:
  inputs: Tensor input.
  units: Python `int` scalar representing the dimensionality of the output
    space.
  num_blocks: Python `int` scalar representing the number of blocks for the
    MADE masks.
  exclusive: Python `bool` scalar representing whether to zero the diagonal of
    the mask, used for the first layer of a MADE.
  kernel_initializer: Initializer function for the weight matrix. If `None`
    (default), weights are initialized using the
    `tf.glorot_random_initializer`.
  reuse: Python `bool` scalar representing whether to reuse the weights of a
    previous layer by the same name.
  name: Python `str` used to describe ops managed by this function.
  *args: `tf.compat.v1.layers.dense` arguments.
  **kwargs: `tf.compat.v1.layers.dense` keyword arguments.

Returns:
  Output tensor.

Raises:
  NotImplementedError: if rightmost dimension of `inputs` is unknown prior to
    graph execution.

#### References

[1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
     Masked Autoencoder for Distribution Estimation. In _International
     Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509"
  [inputs units num_blocks & {:keys [exclusive kernel_initializer reuse name]
                       :or {kernel_initializer None reuse None name None}} ]
    (py/call-attr-kw bijectors "masked_dense" [inputs units num_blocks] {:exclusive exclusive :kernel_initializer kernel_initializer :reuse reuse :name name }))

(defn real-nvp-default-template 
  "Build a scale-and-shift function using a multi-layer neural network. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2018-10-01.
Instructions for updating:
The TensorFlow Distributions library has moved to TensorFlow Probability (https://github.com/tensorflow/probability). You should update all references to use `tfp.distributions` instead of `tf.contrib.distributions`.

This will be wrapped in a make_template to ensure the variables are only
created once. It takes the `d`-dimensional input x[0:d] and returns the `D-d`
dimensional outputs `loc` (\"mu\") and `log_scale` (\"alpha\").

Arguments:
  hidden_layers: Python `list`-like of non-negative integer, scalars
    indicating the number of units in each hidden layer. Default: `[512, 512].
  shift_only: Python `bool` indicating if only the `shift` term shall be
    computed (i.e. NICE bijector). Default: `False`.
  activation: Activation function (callable). Explicitly setting to `None`
    implies a linear activation.
  name: A name for ops managed by this function. Default:
    \"real_nvp_default_template\".
  *args: `tf.compat.v1.layers.dense` arguments.
  **kwargs: `tf.compat.v1.layers.dense` keyword arguments.

Returns:
  shift: `Float`-like `Tensor` of shift terms (\"mu\" in
    [Papamakarios et al.  (2016)][1]).
  log_scale: `Float`-like `Tensor` of log(scale) terms (\"alpha\" in
    [Papamakarios et al. (2016)][1]).

Raises:
  NotImplementedError: if rightmost dimension of `inputs` is unknown prior to
    graph execution.

#### References

[1]: George Papamakarios, Theo Pavlakou, and Iain Murray. Masked
     Autoregressive Flow for Density Estimation. In _Neural Information
     Processing Systems_, 2017. https://arxiv.org/abs/1705.07057"
  [hidden_layers & {:keys [shift_only activation name]
                       :or {name None}} ]
    (py/call-attr-kw bijectors "real_nvp_default_template" [hidden_layers] {:shift_only shift_only :activation activation :name name }))
