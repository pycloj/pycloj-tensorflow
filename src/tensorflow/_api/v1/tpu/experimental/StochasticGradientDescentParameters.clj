(ns tensorflow.-api.v1.tpu.experimental.StochasticGradientDescentParameters
  "Optimization parameters for stochastic gradient descent for TPU embeddings.

  Pass this to `tf.estimator.tpu.experimental.EmbeddingConfigSpec` via the
  `optimization_parameters` argument to set the optimizer and its parameters.
  See the documentation for `tf.estimator.tpu.experimental.EmbeddingConfigSpec`
  for more details.

  ```
  estimator = tf.estimator.tpu.TPUEstimator(
      ...
      embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          ...
          optimization_parameters=(
              tf.tpu.experimental.StochasticGradientDescentParameters(0.1))))
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
(defonce experimental (import-module "tensorflow._api.v1.tpu.experimental"))

(defn StochasticGradientDescentParameters 
  "Optimization parameters for stochastic gradient descent for TPU embeddings.

  Pass this to `tf.estimator.tpu.experimental.EmbeddingConfigSpec` via the
  `optimization_parameters` argument to set the optimizer and its parameters.
  See the documentation for `tf.estimator.tpu.experimental.EmbeddingConfigSpec`
  for more details.

  ```
  estimator = tf.estimator.tpu.TPUEstimator(
      ...
      embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          ...
          optimization_parameters=(
              tf.tpu.experimental.StochasticGradientDescentParameters(0.1))))
  ```

  "
  [ learning_rate clip_weight_min clip_weight_max ]
  (py/call-attr experimental "StochasticGradientDescentParameters"  learning_rate clip_weight_min clip_weight_max ))
