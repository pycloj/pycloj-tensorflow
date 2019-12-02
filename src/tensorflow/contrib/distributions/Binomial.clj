(ns tensorflow.contrib.distributions.Binomial
  "Binomial distribution.

  This distribution is parameterized by `probs`, a (batch of) probabilities for
  drawing a `1` and `total_count`, the number of trials per draw from the
  Binomial.

  #### Mathematical Details

  The Binomial is a distribution over the number of `1`'s in `total_count`
  independent trials, with each trial having the same probability of `1`, i.e.,
  `probs`.

  The probability mass function (pmf) is,

  ```none
  pmf(k; n, p) = p**k (1 - p)**(n - k) / Z
  Z = k! (n - k)! / n!
  ```

  where:
  * `total_count = n`,
  * `probs = p`,
  * `Z` is the normalizing constant, and,
  * `n!` is the factorial of `n`.

  #### Examples

  Create a single distribution, corresponding to 5 coin flips.

  ```python
  dist = Binomial(total_count=5., probs=.5)
  ```

  Create a single distribution (using logits), corresponding to 5 coin flips.

  ```python
  dist = Binomial(total_count=5., logits=0.)
  ```

  Creates 3 distributions with the third distribution most likely to have
  successes.

  ```python
  p = [.2, .3, .8]
  # n will be broadcast to [4., 4., 4.], to match p.
  dist = Binomial(total_count=4., probs=p)
  ```

  The distribution functions can be evaluated on counts.

  ```python
  # counts same shape as p.
  counts = [1., 2, 3]
  dist.prob(counts)  # Shape [3]

  # p will be broadcast to [[.2, .3, .8], [.2, .3, .8]] to match counts.
  counts = [[1., 2, 1], [2, 2, 4]]
  dist.prob(counts)  # Shape [2, 3]

  # p will be broadcast to shape [5, 7, 3] to match counts.
  counts = [[...]]  # Shape [5, 7, 3]
  dist.prob(counts)  # Shape [5, 7, 3]
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
(defn Binomial 
  "Binomial distribution.

  This distribution is parameterized by `probs`, a (batch of) probabilities for
  drawing a `1` and `total_count`, the number of trials per draw from the
  Binomial.

  #### Mathematical Details

  The Binomial is a distribution over the number of `1`'s in `total_count`
  independent trials, with each trial having the same probability of `1`, i.e.,
  `probs`.

  The probability mass function (pmf) is,

  ```none
  pmf(k; n, p) = p**k (1 - p)**(n - k) / Z
  Z = k! (n - k)! / n!
  ```

  where:
  * `total_count = n`,
  * `probs = p`,
  * `Z` is the normalizing constant, and,
  * `n!` is the factorial of `n`.

  #### Examples

  Create a single distribution, corresponding to 5 coin flips.

  ```python
  dist = Binomial(total_count=5., probs=.5)
  ```

  Create a single distribution (using logits), corresponding to 5 coin flips.

  ```python
  dist = Binomial(total_count=5., logits=0.)
  ```

  Creates 3 distributions with the third distribution most likely to have
  successes.

  ```python
  p = [.2, .3, .8]
  # n will be broadcast to [4., 4., 4.], to match p.
  dist = Binomial(total_count=4., probs=p)
  ```

  The distribution functions can be evaluated on counts.

  ```python
  # counts same shape as p.
  counts = [1., 2, 3]
  dist.prob(counts)  # Shape [3]

  # p will be broadcast to [[.2, .3, .8], [.2, .3, .8]] to match counts.
  counts = [[1., 2, 1], [2, 2, 4]]
  dist.prob(counts)  # Shape [2, 3]

  # p will be broadcast to shape [5, 7, 3] to match counts.
  counts = [[...]]  # Shape [5, 7, 3]
  dist.prob(counts)  # Shape [5, 7, 3]
  ```
  "
  [total_count logits probs  & {:keys [validate_args allow_nan_stats name]} ]
    (py/call-attr-kw distributions "Binomial" [total_count logits probs] {:validate_args validate_args :allow_nan_stats allow_nan_stats :name name }))

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


    Additional documentation from `Binomial`:
    
    For each batch member of counts `value`, `P[value]` is the probability that
    after sampling `self.total_count` draws from this Binomial distribution, the
    number of successes is `value`. Since different sequences of draws can result in
    the same counts, the probability includes a combinatorial coefficient.
    
    Note: `value` must be a non-negative tensor with dtype `dtype` and whose shape
    can be broadcast with `self.probs` and `self.total_count`. `value` is only legal
    if it is less than or equal to `self.total_count` and its components are equal
    to integer values.

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

(defn logits 
  "Log-odds of drawing a `1`."
  [ self ]
    (py/call-attr self "logits"))
(defn mean 
  "Mean."
  [self   & {:keys [name]} ]
    (py/call-attr-kw self "mean" [] {:name name }))
(defn mode 
  "Mode.

    Additional documentation from `Binomial`:
    
    Note that when `(1 + total_count) * probs` is an integer, there are
    actually two modes. Namely, `(1 + total_count) * probs` and
    `(1 + total_count) * probs - 1` are both modes. Here we return only the
    larger of the two modes."
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


    Additional documentation from `Binomial`:
    
    For each batch member of counts `value`, `P[value]` is the probability that
    after sampling `self.total_count` draws from this Binomial distribution, the
    number of successes is `value`. Since different sequences of draws can result in
    the same counts, the probability includes a combinatorial coefficient.
    
    Note: `value` must be a non-negative tensor with dtype `dtype` and whose shape
    can be broadcast with `self.probs` and `self.total_count`. `value` is only legal
    if it is less than or equal to `self.total_count` and its components are equal
    to integer values.

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    "
  [self value  & {:keys [name]} ]
    (py/call-attr-kw self "prob" [value] {:name name }))

(defn probs 
  "Probability of drawing a `1`."
  [ self ]
    (py/call-attr self "probs"))
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

(defn total-count 
  "Number of trials."
  [ self ]
    (py/call-attr self "total_count"))

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
