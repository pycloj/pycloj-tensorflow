(ns tensorflow.contrib.distributions.VectorExponentialDiag
  "The vectorization of the Exponential distribution on `R^k`.

  The vector exponential distribution is defined over a subset of `R^k`, and
  parameterized by a (batch of) length-`k` `loc` vector and a (batch of) `k x k`
  `scale` matrix:  `covariance = scale @ scale.T`, where `@` denotes
  matrix-multiplication.

  #### Mathematical Details

  The probability density function (pdf) is defined over the image of the
  `scale` matrix + `loc`, applied to the positive half-space:
  `Supp = {loc + scale @ x : x in R^k, x_1 > 0, ..., x_k > 0}`.  On this set,

  ```none
  pdf(y; loc, scale) = exp(-||x||_1) / Z,  for y in Supp
  x = inv(scale) @ (y - loc),
  Z = |det(scale)|,
  ```

  where:

  * `loc` is a vector in `R^k`,
  * `scale` is a linear operator in `R^{k x k}`, `cov = scale @ scale.T`,
  * `Z` denotes the normalization constant, and,
  * `||x||_1` denotes the `l1` norm of `x`, `sum_i |x_i|`.

  The VectorExponential distribution is a member of the [location-scale
  family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X = (X_1, ..., X_k), each X_i ~ Exponential(rate=1)
  Y = (Y_1, ...,Y_k) = scale @ X + loc
  ```

  #### About `VectorExponential` and `Vector` distributions in TensorFlow.

  The `VectorExponential` is a non-standard distribution that has useful
  properties.

  The marginals `Y_1, ..., Y_k` are *not* Exponential random variables, due to
  the fact that the sum of Exponential random variables is not Exponential.

  Instead, `Y` is a vector whose components are linear combinations of
  Exponential random variables.  Thus, `Y` lives in the vector space generated
  by `vectors` of Exponential distributions.  This allows the user to decide the
  mean and covariance (by setting `loc` and `scale`), while preserving some
  properties of the Exponential distribution.  In particular, the tails of `Y_i`
  will be (up to polynomial factors) exponentially decaying.

  To see this last statement, note that the pdf of `Y_i` is the convolution of
  the pdf of `k` independent Exponential random variables.  One can then show by
  induction that distributions with exponential (up to polynomial factors) tails
  are closed under convolution.


  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Initialize a single 2-variate VectorExponential, supported on
  # {(x, y) in R^2 : x > 0, y > 0}.

  # The first component has pdf exp{-x}, the second 0.5 exp{-x / 2}
  vex = tfd.VectorExponentialDiag(scale_diag=[1., 2.])

  # Compute the pdf of an`R^2` observation; return a scalar.
  vex.prob([3., 4.]).eval()  # shape: []

  # Initialize a 2-batch of 3-variate Vector Exponential's.
  loc = [[1., 2, 3],
         [1., 0, 0]]              # shape: [2, 3]
  scale_diag = [[1., 2, 3],
                [0.5, 1, 1.5]]     # shape: [2, 3]

  vex = tfd.VectorExponentialDiag(loc, scale_diag)

  # Compute the pdf of two `R^3` observations; return a length-2 vector.
  x = [[1.9, 2.2, 3.1],
       [10., 1.0, 9.0]]     # shape: [2, 3]
  vex.prob(x).eval()    # shape: [2]
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
(defonce distributions (import-module "tensorflow.contrib.distributions"))
(defn VectorExponentialDiag 
  "The vectorization of the Exponential distribution on `R^k`.

  The vector exponential distribution is defined over a subset of `R^k`, and
  parameterized by a (batch of) length-`k` `loc` vector and a (batch of) `k x k`
  `scale` matrix:  `covariance = scale @ scale.T`, where `@` denotes
  matrix-multiplication.

  #### Mathematical Details

  The probability density function (pdf) is defined over the image of the
  `scale` matrix + `loc`, applied to the positive half-space:
  `Supp = {loc + scale @ x : x in R^k, x_1 > 0, ..., x_k > 0}`.  On this set,

  ```none
  pdf(y; loc, scale) = exp(-||x||_1) / Z,  for y in Supp
  x = inv(scale) @ (y - loc),
  Z = |det(scale)|,
  ```

  where:

  * `loc` is a vector in `R^k`,
  * `scale` is a linear operator in `R^{k x k}`, `cov = scale @ scale.T`,
  * `Z` denotes the normalization constant, and,
  * `||x||_1` denotes the `l1` norm of `x`, `sum_i |x_i|`.

  The VectorExponential distribution is a member of the [location-scale
  family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X = (X_1, ..., X_k), each X_i ~ Exponential(rate=1)
  Y = (Y_1, ...,Y_k) = scale @ X + loc
  ```

  #### About `VectorExponential` and `Vector` distributions in TensorFlow.

  The `VectorExponential` is a non-standard distribution that has useful
  properties.

  The marginals `Y_1, ..., Y_k` are *not* Exponential random variables, due to
  the fact that the sum of Exponential random variables is not Exponential.

  Instead, `Y` is a vector whose components are linear combinations of
  Exponential random variables.  Thus, `Y` lives in the vector space generated
  by `vectors` of Exponential distributions.  This allows the user to decide the
  mean and covariance (by setting `loc` and `scale`), while preserving some
  properties of the Exponential distribution.  In particular, the tails of `Y_i`
  will be (up to polynomial factors) exponentially decaying.

  To see this last statement, note that the pdf of `Y_i` is the convolution of
  the pdf of `k` independent Exponential random variables.  One can then show by
  induction that distributions with exponential (up to polynomial factors) tails
  are closed under convolution.


  #### Examples

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Initialize a single 2-variate VectorExponential, supported on
  # {(x, y) in R^2 : x > 0, y > 0}.

  # The first component has pdf exp{-x}, the second 0.5 exp{-x / 2}
  vex = tfd.VectorExponentialDiag(scale_diag=[1., 2.])

  # Compute the pdf of an`R^2` observation; return a scalar.
  vex.prob([3., 4.]).eval()  # shape: []

  # Initialize a 2-batch of 3-variate Vector Exponential's.
  loc = [[1., 2, 3],
         [1., 0, 0]]              # shape: [2, 3]
  scale_diag = [[1., 2, 3],
                [0.5, 1, 1.5]]     # shape: [2, 3]

  vex = tfd.VectorExponentialDiag(loc, scale_diag)

  # Compute the pdf of two `R^3` observations; return a length-2 vector.
  x = [[1.9, 2.2, 3.1],
       [10., 1.0, 9.0]]     # shape: [2, 3]
  vex.prob(x).eval()    # shape: [2]
  ```

  "
  [loc scale_diag scale_identity_multiplier  & {:keys [validate_args allow_nan_stats name]} ]
    (py/call-attr-kw distributions "VectorExponentialDiag" [loc scale_diag scale_identity_multiplier] {:validate_args validate_args :allow_nan_stats allow_nan_stats :name name }))

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

(defn bijector 
  "Function transforming x => y."
  [ self ]
    (py/call-attr self "bijector"))
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

(defn loc 
  "The `loc` `Tensor` in `Y = scale @ X + loc`."
  [ self ]
    (py/call-attr self "loc"))
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


    Additional documentation from `VectorExponentialLinearOperator`:
    
    `value` is a batch vector with compatible shape if `value` is a `Tensor` whose
    shape can be broadcast up to either:
    
    ```python
    self.batch_shape + self.event_shape
    ```
    
    or
    
    ```python
    [M1, ..., Mm] + self.batch_shape + self.event_shape
    ```

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


    Additional documentation from `VectorExponentialLinearOperator`:
    
    `value` is a batch vector with compatible shape if `value` is a `Tensor` whose
    shape can be broadcast up to either:
    
    ```python
    self.batch_shape + self.event_shape
    ```
    
    or
    
    ```python
    [M1, ..., Mm] + self.batch_shape + self.event_shape
    ```

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

(defn scale 
  "The `scale` `LinearOperator` in `Y = scale @ X + loc`."
  [ self ]
    (py/call-attr self "scale"))
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
