(ns tensorflow.-api.v1.tpu.experimental.AdagradParameters
  "Optimization parameters for Adagrad with TPU embeddings.

  Pass this to `tf.estimator.tpu.experimental.EmbeddingConfigSpec` via the
  `optimization_parameters` argument to set the optimizer and its parameters.
  See the documentation for `tf.estimator.tpu.experimental.EmbeddingConfigSpec`
  for more details.

  ```
  estimator = tf.estimator.tpu.TPUEstimator(
      ...
      embedding_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          ...
          optimization_parameters=tf.tpu.experimental.AdagradParameters(0.1),
          ...))
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

(defn AdagradParameters 
  "Optimization parameters for Adagrad with TPU embeddings.

  Pass this to `tf.estimator.tpu.experimental.EmbeddingConfigSpec` via the
  `optimization_parameters` argument to set the optimizer and its parameters.
  See the documentation for `tf.estimator.tpu.experimental.EmbeddingConfigSpec`
  for more details.

  ```
  estimator = tf.estimator.tpu.TPUEstimator(
      ...
      embedding_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          ...
          optimization_parameters=tf.tpu.experimental.AdagradParameters(0.1),
          ...))
  ```

  "
  [learning_rate & {:keys [initial_accumulator use_gradient_accumulation clip_weight_min clip_weight_max]
                       :or {clip_weight_min None clip_weight_max None}} ]
    (py/call-attr-kw experimental "AdagradParameters" [learning_rate] {:initial_accumulator initial_accumulator :use_gradient_accumulation use_gradient_accumulation :clip_weight_min clip_weight_min :clip_weight_max clip_weight_max }))
