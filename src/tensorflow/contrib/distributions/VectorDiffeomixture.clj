(ns tensorflow.contrib.distributions.VectorDiffeomixture
  "VectorDiffeomixture distribution.

  A vector diffeomixture (VDM) is a distribution parameterized by a convex
  combination of `K` component `loc` vectors, `loc[k], k = 0,...,K-1`, and `K`
  `scale` matrices `scale[k], k = 0,..., K-1`.  It approximates the following
  [compound distribution]
  (https://en.wikipedia.org/wiki/Compound_probability_distribution)

  ```none
  p(x) = int p(x | z) p(z) dz,
  where z is in the K-simplex, and
  p(x | z) := p(x | loc=sum_k z[k] loc[k], scale=sum_k z[k] scale[k])
  ```

  The integral `int p(x | z) p(z) dz` is approximated with a quadrature scheme
  adapted to the mixture density `p(z)`.  The `N` quadrature points `z_{N, n}`
  and weights `w_{N, n}` (which are non-negative and sum to 1) are chosen
  such that

  ```q_N(x) := sum_{n=1}^N w_{n, N} p(x | z_{N, n}) --> p(x)```

  as `N --> infinity`.

  Since `q_N(x)` is in fact a mixture (of `N` points), we may sample from
  `q_N` exactly.  It is important to note that the VDM is *defined* as `q_N`
  above, and *not* `p(x)`.  Therefore, sampling and pdf may be implemented as
  exact (up to floating point error) methods.

  A common choice for the conditional `p(x | z)` is a multivariate Normal.

  The implemented marginal `p(z)` is the `SoftmaxNormal`, which is a
  `K-1` dimensional Normal transformed by a `SoftmaxCentered` bijector, making
  it a density on the `K`-simplex.  That is,

  ```
  Z = SoftmaxCentered(X),
  X = Normal(mix_loc / temperature, 1 / temperature)
  ```

  The default quadrature scheme chooses `z_{N, n}` as `N` midpoints of
  the quantiles of `p(z)` (generalized quantiles if `K > 2`).

  See [Dillon and Langmore (2018)][1] for more details.

  #### About `Vector` distributions in TensorFlow.

  The `VectorDiffeomixture` is a non-standard distribution that has properties
  particularly useful in [variational Bayesian
  methods](https://en.wikipedia.org/wiki/Variational_Bayesian_methods).

  Conditioned on a draw from the SoftmaxNormal, `X|z` is a vector whose
  components are linear combinations of affine transformations, thus is itself
  an affine transformation.

  Note: The marginals `X_1|v, ..., X_d|v` are *not* generally identical to some
  parameterization of `distribution`.  This is due to the fact that the sum of
  draws from `distribution` are not generally itself the same `distribution`.

  #### About `Diffeomixture`s and reparameterization.

  The `VectorDiffeomixture` is designed to be reparameterized, i.e., its
  parameters are only used to transform samples from a distribution which has no
  trainable parameters. This property is important because backprop stops at
  sources of stochasticity. That is, as long as the parameters are used *after*
  the underlying source of stochasticity, the computed gradient is accurate.

  Reparametrization means that we can use gradient-descent (via backprop) to
  optimize Monte-Carlo objectives. Such objectives are a finite-sample
  approximation of an expectation and arise throughout scientific computing.

  WARNING: If you backprop through a VectorDiffeomixture sample and the \"base\"
  distribution is both: not `FULLY_REPARAMETERIZED` and a function of trainable
  variables, then the gradient is not guaranteed correct!

  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Create two batches of VectorDiffeomixtures, one with mix_loc=[0.],
  # another with mix_loc=[1]. In both cases, `K=2` and the affine
  # transformations involve:
  # k=0: loc=zeros(dims)  scale=LinearOperatorScaledIdentity
  # k=1: loc=[2.]*dims    scale=LinOpDiag
  dims = 5
  vdm = tfd.VectorDiffeomixture(
      mix_loc=[[0.], [1]],
      temperature=[1.],
      distribution=tfd.Normal(loc=0., scale=1.),
      loc=[
          None,  # Equivalent to `np.zeros(dims, dtype=np.float32)`.
          np.float32([2.]*dims),
      ],
      scale=[
          tf.linalg.LinearOperatorScaledIdentity(
            num_rows=dims,
            multiplier=np.float32(1.1),
            is_positive_definite=True),
          tf.linalg.LinearOperatorDiag(
            diag=np.linspace(2.5, 3.5, dims, dtype=np.float32),
            is_positive_definite=True),
      ],
      validate_args=True)
  ```

  #### References

  [1]: Joshua Dillon and Ian Langmore. Quadrature Compound: An approximating
       family of distributions. _arXiv preprint arXiv:1801.03080_, 2018.
       https://arxiv.org/abs/1801.03080
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
(defn VectorDiffeomixture 
  "VectorDiffeomixture distribution.

  A vector diffeomixture (VDM) is a distribution parameterized by a convex
  combination of `K` component `loc` vectors, `loc[k], k = 0,...,K-1`, and `K`
  `scale` matrices `scale[k], k = 0,..., K-1`.  It approximates the following
  [compound distribution]
  (https://en.wikipedia.org/wiki/Compound_probability_distribution)

  ```none
  p(x) = int p(x | z) p(z) dz,
  where z is in the K-simplex, and
  p(x | z) := p(x | loc=sum_k z[k] loc[k], scale=sum_k z[k] scale[k])
  ```

  The integral `int p(x | z) p(z) dz` is approximated with a quadrature scheme
  adapted to the mixture density `p(z)`.  The `N` quadrature points `z_{N, n}`
  and weights `w_{N, n}` (which are non-negative and sum to 1) are chosen
  such that

  ```q_N(x) := sum_{n=1}^N w_{n, N} p(x | z_{N, n}) --> p(x)```

  as `N --> infinity`.

  Since `q_N(x)` is in fact a mixture (of `N` points), we may sample from
  `q_N` exactly.  It is important to note that the VDM is *defined* as `q_N`
  above, and *not* `p(x)`.  Therefore, sampling and pdf may be implemented as
  exact (up to floating point error) methods.

  A common choice for the conditional `p(x | z)` is a multivariate Normal.

  The implemented marginal `p(z)` is the `SoftmaxNormal`, which is a
  `K-1` dimensional Normal transformed by a `SoftmaxCentered` bijector, making
  it a density on the `K`-simplex.  That is,

  ```
  Z = SoftmaxCentered(X),
  X = Normal(mix_loc / temperature, 1 / temperature)
  ```

  The default quadrature scheme chooses `z_{N, n}` as `N` midpoints of
  the quantiles of `p(z)` (generalized quantiles if `K > 2`).

  See [Dillon and Langmore (2018)][1] for more details.

  #### About `Vector` distributions in TensorFlow.

  The `VectorDiffeomixture` is a non-standard distribution that has properties
  particularly useful in [variational Bayesian
  methods](https://en.wikipedia.org/wiki/Variational_Bayesian_methods).

  Conditioned on a draw from the SoftmaxNormal, `X|z` is a vector whose
  components are linear combinations of affine transformations, thus is itself
  an affine transformation.

  Note: The marginals `X_1|v, ..., X_d|v` are *not* generally identical to some
  parameterization of `distribution`.  This is due to the fact that the sum of
  draws from `distribution` are not generally itself the same `distribution`.

  #### About `Diffeomixture`s and reparameterization.

  The `VectorDiffeomixture` is designed to be reparameterized, i.e., its
  parameters are only used to transform samples from a distribution which has no
  trainable parameters. This property is important because backprop stops at
  sources of stochasticity. That is, as long as the parameters are used *after*
  the underlying source of stochasticity, the computed gradient is accurate.

  Reparametrization means that we can use gradient-descent (via backprop) to
  optimize Monte-Carlo objectives. Such objectives are a finite-sample
  approximation of an expectation and arise throughout scientific computing.

  WARNING: If you backprop through a VectorDiffeomixture sample and the \"base\"
  distribution is both: not `FULLY_REPARAMETERIZED` and a function of trainable
  variables, then the gradient is not guaranteed correct!

  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Create two batches of VectorDiffeomixtures, one with mix_loc=[0.],
  # another with mix_loc=[1]. In both cases, `K=2` and the affine
  # transformations involve:
  # k=0: loc=zeros(dims)  scale=LinearOperatorScaledIdentity
  # k=1: loc=[2.]*dims    scale=LinOpDiag
  dims = 5
  vdm = tfd.VectorDiffeomixture(
      mix_loc=[[0.], [1]],
      temperature=[1.],
      distribution=tfd.Normal(loc=0., scale=1.),
      loc=[
          None,  # Equivalent to `np.zeros(dims, dtype=np.float32)`.
          np.float32([2.]*dims),
      ],
      scale=[
          tf.linalg.LinearOperatorScaledIdentity(
            num_rows=dims,
            multiplier=np.float32(1.1),
            is_positive_definite=True),
          tf.linalg.LinearOperatorDiag(
            diag=np.linspace(2.5, 3.5, dims, dtype=np.float32),
            is_positive_definite=True),
      ],
      validate_args=True)
  ```

  #### References

  [1]: Joshua Dillon and Ian Langmore. Quadrature Compound: An approximating
       family of distributions. _arXiv preprint arXiv:1801.03080_, 2018.
       https://arxiv.org/abs/1801.03080
  "
  [mix_loc temperature distribution loc scale  & {:keys [quadrature_size quadrature_fn validate_args allow_nan_stats name]} ]
    (py/call-attr-kw distributions "VectorDiffeomixture" [mix_loc temperature distribution loc scale] {:quadrature_size quadrature_size :quadrature_fn quadrature_fn :validate_args validate_args :allow_nan_stats allow_nan_stats :name name }))

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
  "Base scalar-event, scalar-batch distribution."
  [ self ]
    (py/call-attr self "distribution"))

(defn dtype 
  "The `DType` of `Tensor`s handled by this `Distribution`."
  [ self ]
    (py/call-attr self "dtype"))

(defn endpoint-affine 
  "Affine transformation for each of `K` components."
  [ self ]
    (py/call-attr self "endpoint_affine"))
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

(defn grid 
  "Grid of mixing probabilities, one for each grid point."
  [ self ]
    (py/call-attr self "grid"))

(defn interpolated-affine 
  "Affine transformation for each convex combination of `K` components."
  [ self ]
    (py/call-attr self "interpolated_affine"))
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

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
        `self.dtype`.
    "
  [self value  & {:keys [name]} ]
    (py/call-attr-kw self "log_survival_function" [value] {:name name }))
(defn mean 
  "Mean."
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "mean" [] {:name name }))

(defn mixture-distribution 
  "Distribution used to select a convex combination of affine transforms."
  [ self ]
    (py/call-attr self "mixture_distribution"))
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
