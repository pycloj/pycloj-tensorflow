(ns tensorflow.-api.v1.random.experimental.Generator
  "Random-number generator.

  It uses Variable to manage its internal state, and allows choosing an
  Random-Number-Generation (RNG) algorithm.

  CPU, GPU and TPU with the same algorithm and seed will generate the same
  integer random numbers. Float-point results (such as the output of `normal`)
  may have small numerical discrepancies between CPU and GPU.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.random.experimental"))

(defn Generator 
  "Random-number generator.

  It uses Variable to manage its internal state, and allows choosing an
  Random-Number-Generation (RNG) algorithm.

  CPU, GPU and TPU with the same algorithm and seed will generate the same
  integer random numbers. Float-point results (such as the output of `normal`)
  may have small numerical discrepancies between CPU and GPU.
  "
  [ copy_from state alg ]
  (py/call-attr experimental "Generator"  copy_from state alg ))

(defn algorithm 
  "The RNG algorithm."
  [ self ]
    (py/call-attr self "algorithm"))

(defn binomial 
  "Outputs random values from a binomial distribution.

    The generated values follow a binomial distribution with specified count and
    probability of success parameters.

    Example:

    ```python
    counts = [10., 20.]
    # Probability of success.
    probs = [0.8, 0.9]

    rng = tf.random.experimental.Generator.from_seed(seed=234)
    binomial_samples = rng.binomial(shape=[2], counts=counts, probs=probs)
    ```


    Args:
      shape: A 1-D integer Tensor or Python array. The shape of the output
        tensor.
      counts: A 0/1-D Tensor or Python value. The counts of the binomial
        distribution.  Must be broadcastable with the leftmost dimension
        defined by `shape`.
      probs: A 0/1-D Tensor or Python value. The probability of success for the
        binomial distribution.  Must be broadcastable with the leftmost
        dimension defined by `shape`.
      dtype: The type of the output. Default: tf.int32
      name: A name for the operation (optional).

    Returns:
      samples: A Tensor of the specified shape filled with random binomial
        values.  For each i, each samples[i, ...] is an independent draw from
        the binomial distribution on counts[i] trials with probability of
        success probs[i].
    "
  [self shape counts probs & {:keys [dtype name]
                       :or {name None}} ]
    (py/call-attr-kw self "binomial" [shape counts probs] {:dtype dtype :name name }))

(defn key 
  "The 'key' part of the state of a counter-based RNG.

    For a counter-base RNG algorithm such as Philox and ThreeFry (as
    described in paper 'Parallel Random Numbers: As Easy as 1, 2, 3'
    [https://www.thesalmons.org/john/random123/papers/random123sc11.pdf]),
    the RNG state consists of two parts: counter and key. The output is
    generated via the formula: output=hash(key, counter), i.e. a hashing of
    the counter parametrized by the key. Two RNGs with two different keys can
    be thought as generating two independent random-number streams (a stream
    is formed by increasing the counter).

    Returns:
      A scalar which is the 'key' part of the state, if the RNG algorithm is
        counter-based; otherwise it raises a ValueError.
    "
  [ self ]
    (py/call-attr self "key"))
(defn make-seeds 
  "Generates seeds for stateless random ops.

    For example:

    ```python
    seeds = get_global_generator().make_seeds(count=10)
    for i in range(10):
      seed = seeds[:, i]
      numbers = stateless_random_normal(shape=[2, 3], seed=seed)
      ...
    ```

    Args:
      count: the number of seed pairs (note that stateless random ops need a
        pair of seeds to invoke).

    Returns:
      A tensor of shape [2, count] and dtype int64.
    "
  [self   & {:keys [count]} ]
    (py/call-attr-kw self "make_seeds" [] {:count count }))

(defn normal 
  "Outputs random values from a normal distribution.

    Args:
      shape: A 1-D integer Tensor or Python array. The shape of the output
        tensor.
      mean: A 0-D Tensor or Python value of type `dtype`. The mean of the normal
        distribution.
      stddev: A 0-D Tensor or Python value of type `dtype`. The standard
        deviation of the normal distribution.
      dtype: The type of the output.
      name: A name for the operation (optional).

    Returns:
      A tensor of the specified shape filled with random normal values.
    "
  [self shape & {:keys [mean stddev dtype name]
                       :or {name None}} ]
    (py/call-attr-kw self "normal" [shape] {:mean mean :stddev stddev :dtype dtype :name name }))

(defn reset 
  "Resets the generator by a new state.

    See `__init__` for the meaning of \"state\".

    Args:
      state: the new state.
    "
  [ self state ]
  (py/call-attr self "reset"  self state ))

(defn reset-from-key-counter 
  "Resets the generator by a new key-counter pair.

    See `from_key_counter` for the meaning of \"key\" and \"counter\".

    Args:
      key: the new key.
      counter: the new counter.
    "
  [ self key counter ]
  (py/call-attr self "reset_from_key_counter"  self key counter ))

(defn reset-from-seed 
  "Resets the generator by a new seed.

    See `from_seed` for the meaning of \"seed\".

    Args:
      seed: the new seed.
    "
  [ self seed ]
  (py/call-attr self "reset_from_seed"  self seed ))

(defn skip 
  "Advance the counter of a counter-based RNG.

    Args:
      delta: the amount of advancement. The state of the RNG after
        `skip(n)` will be the same as that after `normal([n])`
        (or any other distribution). The actual increment added to the
        counter is an unspecified implementation detail.
    "
  [ self delta ]
  (py/call-attr self "skip"  self delta ))
(defn split 
  "Returns a list of independent `Generator` objects.

    Two generators are independent of each other in the sense that the
    random-number streams they generate don't have statistically detectable
    correlations. The new generators are also independent of the old one.
    The old generator's state will be changed (like other random-number
    generating methods), so two calls of `split` will return different
    new generators.

    For example:

    ```python
    gens = get_global_generator().split(count=10)
    for gen in gens:
      numbers = gen.normal(shape=[2, 3])
      # ...
    gens2 = get_global_generator().split(count=10)
    # gens2 will be different from gens
    ```

    The new generators will be put on the current device (possible different
    from the old generator's), for example:

    ```python
    with tf.device(\"/device:CPU:0\"):
      gen = Generator(seed=1234)  # gen is on CPU
    with tf.device(\"/device:GPU:0\"):
      gens = gen.split(count=10)  # gens are on GPU
    ```

    Args:
      count: the number of generators to return.

    Returns:
      A list (length `count`) of `Generator` objects independent of each other.
      The new generators have the same RNG algorithm as the old one.
    "
  [self   & {:keys [count]} ]
    (py/call-attr-kw self "split" [] {:count count }))

(defn state 
  "The internal state of the RNG."
  [ self ]
    (py/call-attr self "state"))

(defn truncated-normal 
  "Outputs random values from a truncated normal distribution.

    The generated values follow a normal distribution with specified mean and
    standard deviation, except that values whose magnitude is more than
    2 standard deviations from the mean are dropped and re-picked.

    Args:
      shape: A 1-D integer Tensor or Python array. The shape of the output
        tensor.
      mean: A 0-D Tensor or Python value of type `dtype`. The mean of the
        truncated normal distribution.
      stddev: A 0-D Tensor or Python value of type `dtype`. The standard
        deviation of the normal distribution, before truncation.
      dtype: The type of the output.
      name: A name for the operation (optional).

    Returns:
      A tensor of the specified shape filled with random truncated normal
        values.
    "
  [self shape & {:keys [mean stddev dtype name]
                       :or {name None}} ]
    (py/call-attr-kw self "truncated_normal" [shape] {:mean mean :stddev stddev :dtype dtype :name name }))

(defn uniform 
  "Outputs random values from a uniform distribution.

    The generated values follow a uniform distribution in the range
    `[minval, maxval)`. The lower bound `minval` is included in the range, while
    the upper bound `maxval` is excluded. (For float numbers especially
    low-precision types like bfloat16, because of
    rounding, the result may sometimes include `maxval`.)

    For floats, the default range is `[0, 1)`.  For ints, at least `maxval` must
    be specified explicitly.

    In the integer case, the random integers are slightly biased unless
    `maxval - minval` is an exact power of two.  The bias is small for values of
    `maxval - minval` significantly smaller than the range of the output (either
    `2**32` or `2**64`).

    Args:
      shape: A 1-D integer Tensor or Python array. The shape of the output
        tensor.
      minval: A 0-D Tensor or Python value of type `dtype`. The lower bound on
        the range of random values to generate.  Defaults to 0.
      maxval: A 0-D Tensor or Python value of type `dtype`. The upper bound on
        the range of random values to generate.  Defaults to 1 if `dtype` is
        floating point.
      dtype: The type of the output.
      name: A name for the operation (optional).

    Returns:
      A tensor of the specified shape filled with random uniform values.

    Raises:
      ValueError: If `dtype` is integral and `maxval` is not specified.
    "
  [self shape & {:keys [minval maxval dtype name]
                       :or {maxval None name None}} ]
    (py/call-attr-kw self "uniform" [shape] {:minval minval :maxval maxval :dtype dtype :name name }))

(defn uniform-full-int 
  "Uniform distribution on an integer type's entire range.

    The other method `uniform` only covers the range [minval, maxval), which
    cannot be `dtype`'s full range because `maxval` is of type `dtype`.

    Args:
      shape: the shape of the output.
      dtype: (optional) the integer type, default to uint64.
      name: (optional) the name of the node.

    Returns:
      A tensor of random numbers of the required shape.
    "
  [self shape & {:keys [dtype name]
                       :or {name None}} ]
    (py/call-attr-kw self "uniform_full_int" [shape] {:dtype dtype :name name }))
