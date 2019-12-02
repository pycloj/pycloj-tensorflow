(ns tensorflow.contrib.distributions.QuantizedDistribution
  "Distribution representing the quantization `Y = ceiling(X)`.

  #### Definition in Terms of Sampling

  ```
  1. Draw X
  2. Set Y <-- ceiling(X)
  3. If Y < low, reset Y <-- low
  4. If Y > high, reset Y <-- high
  5. Return Y
  ```

  #### Definition in Terms of the Probability Mass Function

  Given scalar random variable `X`, we define a discrete random variable `Y`
  supported on the integers as follows:

  ```
  P[Y = j] := P[X <= low],  if j == low,
           := P[X > high - 1],  j == high,
           := 0, if j < low or j > high,
           := P[j - 1 < X <= j],  all other j.
  ```

  Conceptually, without cutoffs, the quantization process partitions the real
  line `R` into half open intervals, and identifies an integer `j` with the
  right endpoints:

  ```
  R = ... (-2, -1](-1, 0](0, 1](1, 2](2, 3](3, 4] ...
  j = ...      -1      0     1     2     3     4  ...
  ```

  `P[Y = j]` is the mass of `X` within the `jth` interval.
  If `low = 0`, and `high = 2`, then the intervals are redrawn
  and `j` is re-assigned:

  ```
  R = (-infty, 0](0, 1](1, infty)
  j =          0     1     2
  ```

  `P[Y = j]` is still the mass of `X` within the `jth` interval.

  #### Examples

  We illustrate a mixture of discretized logistic distributions
  [(Salimans et al., 2017)][1]. This is used, for example, for capturing 16-bit
  audio in WaveNet [(van den Oord et al., 2017)][2]. The values range in
  a 1-D integer domain of `[0, 2**16-1]`, and the discretization captures
  `P(x - 0.5 < X <= x + 0.5)` for all `x` in the domain excluding the endpoints.
  The lowest value has probability `P(X <= 0.5)` and the highest value has
  probability `P(2**16 - 1.5 < X)`.

  Below we assume a `wavenet` function. It takes as `input` right-shifted audio
  samples of shape `[..., sequence_length]`. It returns a real-valued tensor of
  shape `[..., num_mixtures * 3]`, i.e., each mixture component has a `loc` and
  `scale` parameter belonging to the logistic distribution, and a `logits`
  parameter determining the unnormalized probability of that component.

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions
  tfb = tfp.bijectors

  net = wavenet(inputs)
  loc, unconstrained_scale, logits = tf.split(net,
                                              num_or_size_splits=3,
                                              axis=-1)
  scale = tf.nn.softplus(unconstrained_scale)

  # Form mixture of discretized logistic distributions. Note we shift the
  # logistic distribution by -0.5. This lets the quantization capture \"rounding\"
  # intervals, `(x-0.5, x+0.5]`, and not \"ceiling\" intervals, `(x-1, x]`.
  discretized_logistic_dist = tfd.QuantizedDistribution(
      distribution=tfd.TransformedDistribution(
          distribution=tfd.Logistic(loc=loc, scale=scale),
          bijector=tfb.AffineScalar(shift=-0.5)),
      low=0.,
      high=2**16 - 1.)
  mixture_dist = tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(logits=logits),
      components_distribution=discretized_logistic_dist)

  neg_log_likelihood = -tf.reduce_sum(mixture_dist.log_prob(targets))
  train_op = tf.train.AdamOptimizer().minimize(neg_log_likelihood)
  ```

  After instantiating `mixture_dist`, we illustrate maximum likelihood by
  calculating its log-probability of audio samples as `target` and optimizing.

  #### References

  [1]: Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P. Kingma.
       PixelCNN++: Improving the PixelCNN with discretized logistic mixture
       likelihood and other modifications.
       _International Conference on Learning Representations_, 2017.
       https://arxiv.org/abs/1701.05517
  [2]: Aaron van den Oord et al. Parallel WaveNet: Fast High-Fidelity Speech
       Synthesis. _arXiv preprint arXiv:1711.10433_, 2017.
       https://arxiv.org/abs/1711.10433
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distributions (import-module "tensorflow.contrib.distributions"))
(defn QuantizedDistribution 
  "Distribution representing the quantization `Y = ceiling(X)`.

  #### Definition in Terms of Sampling

  ```
  1. Draw X
  2. Set Y <-- ceiling(X)
  3. If Y < low, reset Y <-- low
  4. If Y > high, reset Y <-- high
  5. Return Y
  ```

  #### Definition in Terms of the Probability Mass Function

  Given scalar random variable `X`, we define a discrete random variable `Y`
  supported on the integers as follows:

  ```
  P[Y = j] := P[X <= low],  if j == low,
           := P[X > high - 1],  j == high,
           := 0, if j < low or j > high,
           := P[j - 1 < X <= j],  all other j.
  ```

  Conceptually, without cutoffs, the quantization process partitions the real
  line `R` into half open intervals, and identifies an integer `j` with the
  right endpoints:

  ```
  R = ... (-2, -1](-1, 0](0, 1](1, 2](2, 3](3, 4] ...
  j = ...      -1      0     1     2     3     4  ...
  ```

  `P[Y = j]` is the mass of `X` within the `jth` interval.
  If `low = 0`, and `high = 2`, then the intervals are redrawn
  and `j` is re-assigned:

  ```
  R = (-infty, 0](0, 1](1, infty)
  j =          0     1     2
  ```

  `P[Y = j]` is still the mass of `X` within the `jth` interval.

  #### Examples

  We illustrate a mixture of discretized logistic distributions
  [(Salimans et al., 2017)][1]. This is used, for example, for capturing 16-bit
  audio in WaveNet [(van den Oord et al., 2017)][2]. The values range in
  a 1-D integer domain of `[0, 2**16-1]`, and the discretization captures
  `P(x - 0.5 < X <= x + 0.5)` for all `x` in the domain excluding the endpoints.
  The lowest value has probability `P(X <= 0.5)` and the highest value has
  probability `P(2**16 - 1.5 < X)`.

  Below we assume a `wavenet` function. It takes as `input` right-shifted audio
  samples of shape `[..., sequence_length]`. It returns a real-valued tensor of
  shape `[..., num_mixtures * 3]`, i.e., each mixture component has a `loc` and
  `scale` parameter belonging to the logistic distribution, and a `logits`
  parameter determining the unnormalized probability of that component.

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions
  tfb = tfp.bijectors

  net = wavenet(inputs)
  loc, unconstrained_scale, logits = tf.split(net,
                                              num_or_size_splits=3,
                                              axis=-1)
  scale = tf.nn.softplus(unconstrained_scale)

  # Form mixture of discretized logistic distributions. Note we shift the
  # logistic distribution by -0.5. This lets the quantization capture \"rounding\"
  # intervals, `(x-0.5, x+0.5]`, and not \"ceiling\" intervals, `(x-1, x]`.
  discretized_logistic_dist = tfd.QuantizedDistribution(
      distribution=tfd.TransformedDistribution(
          distribution=tfd.Logistic(loc=loc, scale=scale),
          bijector=tfb.AffineScalar(shift=-0.5)),
      low=0.,
      high=2**16 - 1.)
  mixture_dist = tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(logits=logits),
      components_distribution=discretized_logistic_dist)

  neg_log_likelihood = -tf.reduce_sum(mixture_dist.log_prob(targets))
  train_op = tf.train.AdamOptimizer().minimize(neg_log_likelihood)
  ```

  After instantiating `mixture_dist`, we illustrate maximum likelihood by
  calculating its log-probability of audio samples as `target` and optimizing.

  #### References

  [1]: Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P. Kingma.
       PixelCNN++: Improving the PixelCNN with discretized logistic mixture
       likelihood and other modifications.
       _International Conference on Learning Representations_, 2017.
       https://arxiv.org/abs/1701.05517
  [2]: Aaron van den Oord et al. Parallel WaveNet: Fast High-Fidelity Speech
       Synthesis. _arXiv preprint arXiv:1711.10433_, 2017.
       https://arxiv.org/abs/1711.10433
  "
  [distribution low high  & {:keys [validate_args name]} ]
    (py/call-attr-kw distributions "QuantizedDistribution" [distribution low high] {:validate_args validate_args :name name }))

(defn allow-nan-stats 
  "Python `bool` describing behavior when a stat is undefined.

    Stats return +/- infinity when it makes sense. E.g., the variance of a
    Cauchy distribution is infinity. However, sometimes the statistic is
    undefined, e.g., if a distribution's pdf does not achieve a maximum within
    the support of the distribution, the mode is undefined. If the mean is
    undefined, then by definition the variance is undefined. E.g. the mean for
    Student's T for df = 1 is undefined (no clear way to say it is either + or -
    infinity), so the variance = E[(X - mean)**2] is also undefined.

    Returns:
      allow_nan_stats: Python `bool`.
    "
  [ self ]
    (py/call-attr self "allow_nan_stats"))

(defn batch-shape 
  "Shape of a single sample from a single event index as a `TensorShape`.

    May be partially defined or unknown.

    The batch dimensions are indexes into independent, non-identical
    parameterizations of this distribution.

    Returns:
      batch_shape: `TensorShape`, possibly unknown.
    "
  [ self ]
    (py/call-attr self "batch_shape"))
(defn batch-shape-tensor 
  "Shape of a single sample from a single event index as a 1-D `Tensor`.

    The batch dimensions are indexes into independent, non-identical
    parameterizations of this distribution.

    Args:
      name: name to give to the op

    Returns:
      batch_shape: `Tensor`.
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "batch_shape_tensor" [] {:name name }))
(defn cdf 
  "Cumulative distribution function.

    Given random variable `X`, the cumulative distribution function `cdf` is:

    ```none
    cdf(x) := P[X <= x]
    ```


    Additional documentation from `QuantizedDistribution`:
    
    For whole numbers `y`,
    
    ```
    cdf(y) := P[Y <= y]
            = 1, if y >= high,
            = 0, if y < low,
            = P[X <= y], otherwise.
    ```
    
    Since `Y` only has mass at whole numbers, `P[Y <= y] = P[Y <= floor(y)]`.
    This dictates that fractional `y` are first floored to a whole number, and
    then above definition applies.
    
    The base distribution's `cdf` method must be defined on `y - 1`.

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      cdf: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    "
  [self value  & {:keys [name]} ]
    (py/call-attr-kw self "cdf" [value] {:name name }))

(defn copy 
  "Creates a deep copy of the distribution.

    Note: the copy distribution may continue to depend on the original
    initialization arguments.

    Args:
      **override_parameters_kwargs: String/value dictionary of initialization
        arguments to override with new values.

    Returns:
      distribution: A new instance of `type(self)` initialized from the union
        of self.parameters and override_parameters_kwargs, i.e.,
        `dict(self.parameters, **override_parameters_kwargs)`.
    "
  [ self  ]
  (py/call-attr self "copy"  self  ))
(defn covariance 
  "Covariance.

    Covariance is (possibly) defined only for non-scalar-event distributions.

    For example, for a length-`k`, vector-valued distribution, it is calculated
    as,

    ```none
    Cov[i, j] = Covariance(X_i, X_j) = E[(X_i - E[X_i]) (X_j - E[X_j])]
    ```

    where `Cov` is a (batch of) `k x k` matrix, `0 <= (i, j) < k`, and `E`
    denotes expectation.

    Alternatively, for non-vector, multivariate distributions (e.g.,
    matrix-valued, Wishart), `Covariance` shall return a (batch of) matrices
    under some vectorization of the events, i.e.,

    ```none
    Cov[i, j] = Covariance(Vec(X)_i, Vec(X)_j) = [as above]
    ```

    where `Cov` is a (batch of) `k' x k'` matrices,
    `0 <= (i, j) < k' = reduce_prod(event_shape)`, and `Vec` is some function
    mapping indices of this distribution's event dimensions to indices of a
    length-`k'` vector.

    Args:
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      covariance: Floating-point `Tensor` with shape `[B1, ..., Bn, k', k']`
        where the first `n` dimensions are batch coordinates and
        `k' = reduce_prod(self.event_shape)`.
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "covariance" [] {:name name }))
(defn cross-entropy 
  "Computes the (Shannon) cross entropy.

    Denote this distribution (`self`) by `P` and the `other` distribution by
    `Q`. Assuming `P, Q` are absolutely continuous with respect to
    one another and permit densities `p(x) dr(x)` and `q(x) dr(x)`, (Shanon)
    cross entropy is defined as:

    ```none
    H[P, Q] = E_p[-log q(X)] = -int_F p(x) log q(x) dr(x)
    ```

    where `F` denotes the support of the random variable `X ~ P`.

    Args:
      other: `tfp.distributions.Distribution` instance.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      cross_entropy: `self.dtype` `Tensor` with shape `[B1, ..., Bn]`
        representing `n` different calculations of (Shanon) cross entropy.
    "
  [self other  & {:keys [name]} ]
    (py/call-attr-kw self "cross_entropy" [other] {:name name }))

(defn distribution 
  "Base distribution, p(x)."
  [ self ]
    (py/call-attr self "distribution"))

(defn dtype 
  "The `DType` of `Tensor`s handled by this `Distribution`."
  [ self ]
    (py/call-attr self "dtype"))
(defn entropy 
  "Shannon entropy in nats."
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "entropy" [] {:name name }))

(defn event-shape 
  "Shape of a single sample from a single batch as a `TensorShape`.

    May be partially defined or unknown.

    Returns:
      event_shape: `TensorShape`, possibly unknown.
    "
  [ self ]
    (py/call-attr self "event_shape"))
(defn event-shape-tensor 
  "Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

    Args:
      name: name to give to the op

    Returns:
      event_shape: `Tensor`.
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "event_shape_tensor" [] {:name name }))

(defn high 
  "Highest value that quantization returns."
  [ self ]
    (py/call-attr self "high"))
(defn is-scalar-batch 
  "Indicates that `batch_shape == []`.

    Args:
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      is_scalar_batch: `bool` scalar `Tensor`.
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "is_scalar_batch" [] {:name name }))
(defn is-scalar-event 
  "Indicates that `event_shape == []`.

    Args:
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      is_scalar_event: `bool` scalar `Tensor`.
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "is_scalar_event" [] {:name name }))
(defn kl-divergence 
  "Computes the Kullback--Leibler divergence.

    Denote this distribution (`self`) by `p` and the `other` distribution by
    `q`. Assuming `p, q` are absolutely continuous with respect to reference
    measure `r`, the KL divergence is defined as:

    ```none
    KL[p, q] = E_p[log(p(X)/q(X))]
             = -int_F p(x) log q(x) dr(x) + int_F p(x) log p(x) dr(x)
             = H[p, q] - H[p]
    ```

    where `F` denotes the support of the random variable `X ~ p`, `H[., .]`
    denotes (Shanon) cross entropy, and `H[.]` denotes (Shanon) entropy.

    Args:
      other: `tfp.distributions.Distribution` instance.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      kl_divergence: `self.dtype` `Tensor` with shape `[B1, ..., Bn]`
        representing `n` different calculations of the Kullback-Leibler
        divergence.
    "
  [self other  & {:keys [name]} ]
    (py/call-attr-kw self "kl_divergence" [other] {:name name }))
(defn log-cdf 
  "Log cumulative distribution function.

    Given random variable `X`, the cumulative distribution function `cdf` is:

    ```none
    log_cdf(x) := Log[ P[X <= x] ]
    ```

    Often, a numerical approximation can be used for `log_cdf(x)` that yields
    a more accurate answer than simply taking the logarithm of the `cdf` when
    `x << -1`.


    Additional documentation from `QuantizedDistribution`:
    
    For whole numbers `y`,
    
    ```
    cdf(y) := P[Y <= y]
            = 1, if y >= high,
            = 0, if y < low,
            = P[X <= y], otherwise.
    ```
    
    Since `Y` only has mass at whole numbers, `P[Y <= y] = P[Y <= floor(y)]`.
    This dictates that fractional `y` are first floored to a whole number, and
    then above definition applies.
    
    The base distribution's `log_cdf` method must be defined on `y - 1`.

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      logcdf: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    "
  [self value  & {:keys [name]} ]
    (py/call-attr-kw self "log_cdf" [value] {:name name }))
(defn log-prob 
  "Log probability density/mass function.


    Additional documentation from `QuantizedDistribution`:
    
    For whole numbers `y`,
    
    ```
    P[Y = y] := P[X <= low],  if y == low,
             := P[X > high - 1],  y == high,
             := 0, if j < low or y > high,
             := P[y - 1 < X <= y],  all other y.
    ```
    
    
    The base distribution's `log_cdf` method must be defined on `y - 1`. If the
    base distribution has a `log_survival_function` method results will be more
    accurate for large values of `y`, and in this case the `log_survival_function`
    must also be defined on `y - 1`.

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      log_prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    "
  [self value  & {:keys [name]} ]
    (py/call-attr-kw self "log_prob" [value] {:name name }))
(defn log-survival-function 
  "Log survival function.

    Given random variable `X`, the survival function is defined:

    ```none
    log_survival_function(x) = Log[ P[X > x] ]
                             = Log[ 1 - P[X <= x] ]
                             = Log[ 1 - cdf(x) ]
    ```

    Typically, different numerical approximations can be used for the log
    survival function, which are more accurate than `1 - cdf(x)` when `x >> 1`.


    Additional documentation from `QuantizedDistribution`:
    
    For whole numbers `y`,
    
    ```
    survival_function(y) := P[Y > y]
                          = 0, if y >= high,
                          = 1, if y < low,
                          = P[X <= y], otherwise.
    ```
    
    Since `Y` only has mass at whole numbers, `P[Y <= y] = P[Y <= floor(y)]`.
    This dictates that fractional `y` are first floored to a whole number, and
    then above definition applies.
    
    The base distribution's `log_cdf` method must be defined on `y - 1`.

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
        `self.dtype`.
    "
  [self value  & {:keys [name]} ]
    (py/call-attr-kw self "log_survival_function" [value] {:name name }))

(defn low 
  "Lowest value that quantization returns."
  [ self ]
    (py/call-attr self "low"))
(defn mean 
  "Mean."
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "mean" [] {:name name }))
(defn mode 
  "Mode."
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "mode" [] {:name name }))

(defn name 
  "Name prepended to all ops created by this `Distribution`."
  [ self ]
    (py/call-attr self "name"))

(defn parameters 
  "Dictionary of parameters used to instantiate this `Distribution`."
  [ self ]
    (py/call-attr self "parameters"))
(defn prob 
  "Probability density/mass function.


    Additional documentation from `QuantizedDistribution`:
    
    For whole numbers `y`,
    
    ```
    P[Y = y] := P[X <= low],  if y == low,
             := P[X > high - 1],  y == high,
             := 0, if j < low or y > high,
             := P[y - 1 < X <= y],  all other y.
    ```
    
    
    The base distribution's `cdf` method must be defined on `y - 1`. If the
    base distribution has a `survival_function` method, results will be more
    accurate for large values of `y`, and in this case the `survival_function` must
    also be defined on `y - 1`.

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    "
  [self value  & {:keys [name]} ]
    (py/call-attr-kw self "prob" [value] {:name name }))
(defn quantile 
  "Quantile function. Aka \"inverse cdf\" or \"percent point function\".

    Given random variable `X` and `p in [0, 1]`, the `quantile` is:

    ```none
    quantile(p) := x such that P[X <= x] == p
    ```

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      quantile: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    "
  [self value  & {:keys [name]} ]
    (py/call-attr-kw self "quantile" [value] {:name name }))

(defn reparameterization-type 
  "Describes how samples from the distribution are reparameterized.

    Currently this is one of the static instances
    `distributions.FULLY_REPARAMETERIZED`
    or `distributions.NOT_REPARAMETERIZED`.

    Returns:
      An instance of `ReparameterizationType`.
    "
  [ self ]
    (py/call-attr self "reparameterization_type"))

(defn sample 
  "Generate samples of the specified shape.

    Note that a call to `sample()` without arguments will generate a single
    sample.

    Args:
      sample_shape: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
      seed: Python integer seed for RNG
      name: name to give to the op.

    Returns:
      samples: a `Tensor` with prepended dimensions `sample_shape`.
    "
  [self  & {:keys [sample_shape seed name]
                       :or {seed None}} ]
    (py/call-attr-kw self "sample" [] {:sample_shape sample_shape :seed seed :name name }))
(defn stddev 
  "Standard deviation.

    Standard deviation is defined as,

    ```none
    stddev = E[(X - E[X])**2]**0.5
    ```

    where `X` is the random variable associated with this distribution, `E`
    denotes expectation, and `stddev.shape = batch_shape + event_shape`.

    Args:
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      stddev: Floating-point `Tensor` with shape identical to
        `batch_shape + event_shape`, i.e., the same shape as `self.mean()`.
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "stddev" [] {:name name }))
(defn survival-function 
  "Survival function.

    Given random variable `X`, the survival function is defined:

    ```none
    survival_function(x) = P[X > x]
                         = 1 - P[X <= x]
                         = 1 - cdf(x).
    ```


    Additional documentation from `QuantizedDistribution`:
    
    For whole numbers `y`,
    
    ```
    survival_function(y) := P[Y > y]
                          = 0, if y >= high,
                          = 1, if y < low,
                          = P[X <= y], otherwise.
    ```
    
    Since `Y` only has mass at whole numbers, `P[Y <= y] = P[Y <= floor(y)]`.
    This dictates that fractional `y` are first floored to a whole number, and
    then above definition applies.
    
    The base distribution's `cdf` method must be defined on `y - 1`.

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
        `self.dtype`.
    "
  [self value  & {:keys [name]} ]
    (py/call-attr-kw self "survival_function" [value] {:name name }))

(defn validate-args 
  "Python `bool` indicating possibly expensive checks are enabled."
  [ self ]
    (py/call-attr self "validate_args"))
(defn variance 
  "Variance.

    Variance is defined as,

    ```none
    Var = E[(X - E[X])**2]
    ```

    where `X` is the random variable associated with this distribution, `E`
    denotes expectation, and `Var.shape = batch_shape + event_shape`.

    Args:
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      variance: Floating-point `Tensor` with shape identical to
        `batch_shape + event_shape`, i.e., the same shape as `self.mean()`.
    "
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "variance" [] {:name name }))
