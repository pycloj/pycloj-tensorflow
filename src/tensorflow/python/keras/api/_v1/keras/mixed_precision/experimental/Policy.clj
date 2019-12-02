(ns tensorflow.python.keras.api.-v1.keras.mixed-precision.experimental.Policy
  "A dtype policy for a Keras layer.

  A dtype policy determines dtype-related aspects of a layer, such as its
  computation and variable dtypes. Each layer has a policy. Policies can be
  passed to the 'dtype' argument of layer constructors, or a global policy can
  be set with 'tf.keras.mixed_precision.experimental.set_policy'. A layer will
  default to the global policy if no policy is passed to it's constructor.

  For most models, each layer will have the same computation dtype and variable
  dtype, which will typically be float32. However, when mixed precision
  training is used, most layers will instead have a float16 computation dtype
  and a float32 variable dtype. See [this
  link](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)
  for more information on mixed precision training. When the variable dtype does
  not match the computation dtype, variables will be automatically casted to the
  computation dtype to avoid type errors.

  Policies also have a `tf.train.experimental.LossScale` instance, which is used
  by Models to performance loss scaling. Layers which are not Models ignore
  the loss scale.

  Policies are constructed by passing a string to the constructor, e.g.
  `tf.keras.mixed_precision.experimental.Policy('float32')`. The string
  determines the compute and variable dtypes. Currently, it can be one of
  in one of the following forms:

    * Any dtype name, such as 'float32' or 'float64'. Both the variable and
      compute dtypes will be that dtype.
    * '<dtype>_with_float32_vars', where <dtype> is any dtype. The compute dtype
      will be <dtype>, while the variable dtype is float32. This can be used for
      mixed precision, which uses float16 or bfloat16 for most computations, and
      float32 for variables, but it is recommended to use the 'mixed_float16' or
      'mixed_bfloat16' policies instead.
    * 'mixed_float16' or 'mixed_bfloat16': Similar to
      'float16_with_float32_vars' or 'bfloat16_with_float32_vars' respectively.
      'mixed_float16' is identical to 'float16_with_float32_vars' except the
      loss_scale is dynamic by default. 'mixed_bfloat16' is currently identical
      to 'bfloat16_with_float32_vars'. More changes may be added to these mixed
      policies in the future, to further differentiate them from
      [b]float16_with_float32_vars.

  ### How to use mixed precision in layers with Policies

  To use mixed precision in a model, the 'mixed_float16' policy can
  be used. `tf.keras.mixed_precision.experimental.set_policy` can be used to set
  the default policy for layers if no policy is passed to them. For example:

  ```python
  tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
  model = tf.keras.models.Sequential(
      tf.keras.layers.Input((100,)),
      # Dense layers use global policy of 'mixed_float16', which does
      # computations in float16 while keeping variables in float32.
      tf.keras.layers.Dense(10),
      tf.keras.layers.Dense(10),
      # Softmax should be done in float32 for numeric stability. We pass
      # dtype='float32' to use float32 instead of the global policy.
      tf.keras.layers.Activation('Softmax', dtype='float32')
  )
  model.fit(...)  # Train `model`
  ```

  Alternatively, the policy can be passed to individual layers instead of
  setting the global policy with `set_policy`:

  ```python
  policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
  model = tf.keras.models.Sequential(
      tf.keras.layers.Input((100,)),
      tf.keras.layers.Dense(10, dtype=policy),
      tf.keras.layers.Dense(10, dtype=policy),
      # Softmax should be done in float32 for numeric stability.
      tf.keras.layers.Activation('Softmax', dtype='float32')
  )
  model.fit(...)  # Train `model`
  ```

  As the above example shows, strings can be directly passed to layer
  constructors in the `dtype` argument instead of policies, but only if the
  string is convertible to a dtype.

  ### The deprecated \"infer\" policy

  In addition to a dtype or \"<dtype>_with_float32_vars\", a policy can also be
  \"infer\". This Policy is deprecated, and it is not recommended. When a layer
  has an infer policy, it will infer the computation and variable dtype from
  the first input the first time the layer is called.

  Once the layer is called for the first time, the layer's policy will change to
  the dtype of the first input.

  Similarly to \"infer\", there is a deprecated \"infer_with_float32_vars\" policy
  that infers the compute dtype, but not the variable dtype.

  In TensorFlow 1, only the \"infer\" and \"infer_with_float32_vars\" policies are
  available.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow.python.keras.api._v1.keras.mixed_precision.experimental"))
(defn Policy 
  "A dtype policy for a Keras layer.

  A dtype policy determines dtype-related aspects of a layer, such as its
  computation and variable dtypes. Each layer has a policy. Policies can be
  passed to the 'dtype' argument of layer constructors, or a global policy can
  be set with 'tf.keras.mixed_precision.experimental.set_policy'. A layer will
  default to the global policy if no policy is passed to it's constructor.

  For most models, each layer will have the same computation dtype and variable
  dtype, which will typically be float32. However, when mixed precision
  training is used, most layers will instead have a float16 computation dtype
  and a float32 variable dtype. See [this
  link](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)
  for more information on mixed precision training. When the variable dtype does
  not match the computation dtype, variables will be automatically casted to the
  computation dtype to avoid type errors.

  Policies also have a `tf.train.experimental.LossScale` instance, which is used
  by Models to performance loss scaling. Layers which are not Models ignore
  the loss scale.

  Policies are constructed by passing a string to the constructor, e.g.
  `tf.keras.mixed_precision.experimental.Policy('float32')`. The string
  determines the compute and variable dtypes. Currently, it can be one of
  in one of the following forms:

    * Any dtype name, such as 'float32' or 'float64'. Both the variable and
      compute dtypes will be that dtype.
    * '<dtype>_with_float32_vars', where <dtype> is any dtype. The compute dtype
      will be <dtype>, while the variable dtype is float32. This can be used for
      mixed precision, which uses float16 or bfloat16 for most computations, and
      float32 for variables, but it is recommended to use the 'mixed_float16' or
      'mixed_bfloat16' policies instead.
    * 'mixed_float16' or 'mixed_bfloat16': Similar to
      'float16_with_float32_vars' or 'bfloat16_with_float32_vars' respectively.
      'mixed_float16' is identical to 'float16_with_float32_vars' except the
      loss_scale is dynamic by default. 'mixed_bfloat16' is currently identical
      to 'bfloat16_with_float32_vars'. More changes may be added to these mixed
      policies in the future, to further differentiate them from
      [b]float16_with_float32_vars.

  ### How to use mixed precision in layers with Policies

  To use mixed precision in a model, the 'mixed_float16' policy can
  be used. `tf.keras.mixed_precision.experimental.set_policy` can be used to set
  the default policy for layers if no policy is passed to them. For example:

  ```python
  tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
  model = tf.keras.models.Sequential(
      tf.keras.layers.Input((100,)),
      # Dense layers use global policy of 'mixed_float16', which does
      # computations in float16 while keeping variables in float32.
      tf.keras.layers.Dense(10),
      tf.keras.layers.Dense(10),
      # Softmax should be done in float32 for numeric stability. We pass
      # dtype='float32' to use float32 instead of the global policy.
      tf.keras.layers.Activation('Softmax', dtype='float32')
  )
  model.fit(...)  # Train `model`
  ```

  Alternatively, the policy can be passed to individual layers instead of
  setting the global policy with `set_policy`:

  ```python
  policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
  model = tf.keras.models.Sequential(
      tf.keras.layers.Input((100,)),
      tf.keras.layers.Dense(10, dtype=policy),
      tf.keras.layers.Dense(10, dtype=policy),
      # Softmax should be done in float32 for numeric stability.
      tf.keras.layers.Activation('Softmax', dtype='float32')
  )
  model.fit(...)  # Train `model`
  ```

  As the above example shows, strings can be directly passed to layer
  constructors in the `dtype` argument instead of policies, but only if the
  string is convertible to a dtype.

  ### The deprecated \"infer\" policy

  In addition to a dtype or \"<dtype>_with_float32_vars\", a policy can also be
  \"infer\". This Policy is deprecated, and it is not recommended. When a layer
  has an infer policy, it will infer the computation and variable dtype from
  the first input the first time the layer is called.

  Once the layer is called for the first time, the layer's policy will change to
  the dtype of the first input.

  Similarly to \"infer\", there is a deprecated \"infer_with_float32_vars\" policy
  that infers the compute dtype, but not the variable dtype.

  In TensorFlow 1, only the \"infer\" and \"infer_with_float32_vars\" policies are
  available.
  "
  [name  & {:keys [loss_scale]} ]
    (py/call-attr-kw experimental "Policy" [name] {:loss_scale loss_scale }))

(defn compute-dtype 
  "The compute dtype of this policy.

    This is the dtype layers will do their computations in.

    If this is None, the policy is \"infer\" or \"infer_with_float32_vars\" and
    `variable_dtype` is either None or float32 respectively.

    Note that even if the compute dtype is float16 or bfloat16, hardware devices
    may not do individual adds, multiplies, and other fundamental operations in
    [b]float16, but instead may do some of them in float32 for numeric
    stability. The compute dtype is the dtype of the inputs and outputs of the
    TensorFlow ops that the layer executes. Internally, many TensorFlow ops will
    do certain internal calculations in float32, or some other device-internal
    intermediate format with higher precision than [b]float16, to increase
    numeric stability.

    For example, a `tf.keras.layers.Dense` layer, when run on a GPU with a
    float16 compute dtype, will pass float16 inputs to tf.matmul. But, tf.matmul
    will do use float32 intermediate math. The performance benefit of float16 is
    still apparent, due to increased memory bandwidth and the fact GPUs have
    specialized hardware for computating matmuls on float16 while still keeping
    intermediate computations in float32.

    Returns:
      The variable dtype of this policy, or None if the variable dtype should be
      inferred from the inputs.
    "
  [ self ]
    (py/call-attr self "compute_dtype"))

(defn loss-scale 
  "Returns the loss scale of this Policy.

    Returns:
      A `tf.train.experimental.LossScale`, or None.
    "
  [ self ]
    (py/call-attr self "loss_scale"))

(defn name 
  "Returns the name of this policy."
  [ self ]
    (py/call-attr self "name"))

(defn should-cast-variables 
  "Returns True if variables should be casted.

    This is true if the variable dtype is not the same as the compute dtype.

    Returns:
      True, if variables should be casted.
    "
  [ self ]
    (py/call-attr self "should_cast_variables"))

(defn variable-dtype 
  "The variable dtype of this policy.

    This is the dtype layers will create their variables in, unless a layer
    explicit chooses a different dtype. If this is different than
    `Policy.compute_dtype` and both are non-None, Layers will cast variables to
    the compute dtype to avoid type errors.

    If this is None, the policy is \"infer\" and the `compute_dtype` is also None.
    If `compute_dtype` is None, this is either None or float32.

    Returns:
      The variable dtype of this policy, or None if the variable dtype should be
      inferred from the inputs.
    "
  [ self ]
    (py/call-attr self "variable_dtype"))
