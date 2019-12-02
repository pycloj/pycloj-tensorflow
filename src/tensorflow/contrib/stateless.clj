(ns tensorflow-core.contrib.stateless
  "Stateless random ops which take seed as a tensor input.

DEPRECATED: Use `tf.random.stateless_uniform` rather than
`tf.contrib.stateless.stateless_random_uniform`, and similarly for the other
routines.

Instead of taking `seed` as an attr which initializes a mutable state within
the op, these random ops take `seed` as an input, and the random numbers are
a deterministic function of `shape` and `seed`.

WARNING: These ops are in contrib, and are not stable.  They should be
consistent across multiple runs on the same hardware, but only for the same
version of the code.

@@stateless_multinomial
@@stateless_random_uniform
@@stateless_random_normal
@@stateless_truncated_normal
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce stateless (import-module "tensorflow_core.contrib.stateless"))

(defn stateless-multinomial 
  "Draws deterministic pseudorandom samples from a multinomial distribution. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use `tf.random.stateless_categorical` instead.

This is a stateless version of `tf.random.categorical`: if run twice with the
same seeds, it will produce the same pseudorandom numbers.  The output is
consistent across multiple runs on the same hardware (and between CPU
and GPU), but may change between versions of TensorFlow or on non-CPU/GPU
hardware.

Example:

```python
# samples has shape [1, 5], where each value is either 0 or 1 with equal
# probability.
samples = tf.random.stateless_categorical(
    tf.math.log([[0.5, 0.5]]), 5, seed=[7, 17])
```

Args:
  logits: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice
    `[i, :]` represents the unnormalized log-probabilities for all classes.
  num_samples: 0-D.  Number of independent samples to draw for each row slice.
  seed: A shape [2] integer Tensor of seeds to the random number generator.
  output_dtype: integer type to use for the output. Defaults to int64.
  name: Optional name for the operation.

Returns:
  The drawn samples of shape `[batch_size, num_samples]`."
  [logits num_samples seed & {:keys [output_dtype name]
                       :or {name None}} ]
    (py/call-attr-kw stateless "stateless_multinomial" [logits num_samples seed] {:output_dtype output_dtype :name name }))

(defn stateless-random-normal 
  "Outputs deterministic pseudorandom values from a normal distribution.

  This is a stateless version of `tf.random.normal`: if run twice with the
  same seeds, it will produce the same pseudorandom numbers.  The output is
  consistent across multiple runs on the same hardware (and between CPU
  and GPU), but may change between versions of TensorFlow or on non-CPU/GPU
  hardware.

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    seed: A shape [2] integer Tensor of seeds to the random number generator.
    mean: A 0-D Tensor or Python value of type `dtype`. The mean of the normal
      distribution.
    stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation
      of the normal distribution.
    dtype: The type of the output.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random normal values.
  "
  [shape seed & {:keys [mean stddev dtype name]
                       :or {name None}} ]
    (py/call-attr-kw stateless "stateless_random_normal" [shape seed] {:mean mean :stddev stddev :dtype dtype :name name }))

(defn stateless-random-uniform 
  "Outputs deterministic pseudorandom values from a uniform distribution.

  This is a stateless version of `tf.random.uniform`: if run twice with the
  same seeds, it will produce the same pseudorandom numbers.  The output is
  consistent across multiple runs on the same hardware (and between CPU
  and GPU), but may change between versions of TensorFlow or on non-CPU/GPU
  hardware.

  The generated values follow a uniform distribution in the range
  `[minval, maxval)`. The lower bound `minval` is included in the range, while
  the upper bound `maxval` is excluded.

  For floats, the default range is `[0, 1)`.  For ints, at least `maxval` must
  be specified explicitly.

  In the integer case, the random integers are slightly biased unless
  `maxval - minval` is an exact power of two.  The bias is small for values of
  `maxval - minval` significantly smaller than the range of the output (either
  `2**32` or `2**64`).

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    seed: A shape [2] integer Tensor of seeds to the random number generator.
    minval: A 0-D Tensor or Python value of type `dtype`. The lower bound on the
      range of random values to generate.  Defaults to 0.
    maxval: A 0-D Tensor or Python value of type `dtype`. The upper bound on the
      range of random values to generate.  Defaults to 1 if `dtype` is floating
      point.
    dtype: The type of the output: `float16`, `float32`, `float64`, `int32`, or
      `int64`.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random uniform values.

  Raises:
    ValueError: If `dtype` is integral and `maxval` is not specified.
  "
  [shape seed & {:keys [minval maxval dtype name]
                       :or {maxval None name None}} ]
    (py/call-attr-kw stateless "stateless_random_uniform" [shape seed] {:minval minval :maxval maxval :dtype dtype :name name }))

(defn stateless-truncated-normal 
  "Outputs deterministic pseudorandom values, truncated normally distributed.

  This is a stateless version of `tf.random.truncated_normal`: if run twice with
  the
  same seeds, it will produce the same pseudorandom numbers.  The output is
  consistent across multiple runs on the same hardware (and between CPU
  and GPU), but may change between versions of TensorFlow or on non-CPU/GPU
  hardware.

  The generated values follow a normal distribution with specified mean and
  standard deviation, except that values whose magnitude is more than 2 standard
  deviations from the mean are dropped and re-picked.

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    seed: A shape [2] integer Tensor of seeds to the random number generator.
    mean: A 0-D Tensor or Python value of type `dtype`. The mean of the
      truncated normal distribution.
    stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation
      of the normal distribution, before truncation.
    dtype: The type of the output.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random truncated normal values.
  "
  [shape seed & {:keys [mean stddev dtype name]
                       :or {name None}} ]
    (py/call-attr-kw stateless "stateless_truncated_normal" [shape seed] {:mean mean :stddev stddev :dtype dtype :name name }))
