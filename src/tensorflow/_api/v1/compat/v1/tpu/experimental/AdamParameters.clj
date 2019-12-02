(ns tensorflow.-api.v1.compat.v1.tpu.experimental.AdamParameters
  "Optimization parameters for Adam with TPU embeddings.

  Pass this to `tf.estimator.tpu.experimental.EmbeddingConfigSpec` via the
  `optimization_parameters` argument to set the optimizer and its parameters.
  See the documentation for `tf.estimator.tpu.experimental.EmbeddingConfigSpec`
  for more details.

  ```
  estimator = tf.estimator.tpu.TPUEstimator(
      ...
      embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          ...
          optimization_parameters=tf.tpu.experimental.AdamParameters(0.1),
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
(defonce experimental (import-module "tensorflow._api.v1.compat.v1.tpu.experimental"))

(defn AdamParameters 
  "Optimization parameters for Adam with TPU embeddings.

  Pass this to `tf.estimator.tpu.experimental.EmbeddingConfigSpec` via the
  `optimization_parameters` argument to set the optimizer and its parameters.
  See the documentation for `tf.estimator.tpu.experimental.EmbeddingConfigSpec`
  for more details.

  ```
  estimator = tf.estimator.tpu.TPUEstimator(
      ...
      embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          ...
          optimization_parameters=tf.tpu.experimental.AdamParameters(0.1),
          ...))
  ```

  "
  [learning_rate & {:keys [beta1 beta2 epsilon lazy_adam sum_inside_sqrt use_gradient_accumulation clip_weight_min clip_weight_max]
                       :or {clip_weight_min None clip_weight_max None}} ]
    (py/call-attr-kw experimental "AdamParameters" [learning_rate] {:beta1 beta1 :beta2 beta2 :epsilon epsilon :lazy_adam lazy_adam :sum_inside_sqrt sum_inside_sqrt :use_gradient_accumulation use_gradient_accumulation :clip_weight_min clip_weight_min :clip_weight_max clip_weight_max }))
