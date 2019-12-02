(ns tensorflow-estimator.python.estimator.api.-v1.estimator.tpu.experimental.EmbeddingConfigSpec
  "Class to keep track of the specification for TPU embeddings.

  Pass this class to `tf.estimator.tpu.TPUEstimator` via the
  `embedding_config_spec` parameter. At minimum you need to specify
  `feature_columns` and `optimization_parameters`. The feature columns passed
  should be created with some combination of
  `tf.tpu.experimental.embedding_column` and
  `tf.tpu.experimental.shared_embedding_columns`.

  TPU embeddings do not support arbitrary Tensorflow optimizers and the
  main optimizer you use for your model will be ignored for the embedding table
  variables. Instead TPU embeddigns support a fixed set of predefined optimizers
  that you can select from and set the parameters of. These include adagrad,
  adam and stochastic gradient descent. Each supported optimizer has a
  `Parameters` class in the `tf.tpu.experimental` namespace.

  ```
  column_a = tf.feature_column.categorical_column_with_identity(...)
  column_b = tf.feature_column.categorical_column_with_identity(...)
  column_c = tf.feature_column.categorical_column_with_identity(...)
  tpu_shared_columns = tf.tpu.experimental.shared_embedding_columns(
      [column_a, column_b], 10)
  tpu_non_shared_column = tf.tpu.experimental.embedding_column(
      column_c, 10)
  tpu_columns = [tpu_non_shared_column] + tpu_shared_columns
  ...
  def model_fn(features):
    dense_features = tf.keras.layers.DenseFeature(tpu_columns)
    embedded_feature = dense_features(features)
    ...

  estimator = tf.estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      ...
      embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          column=tpu_columns,
          optimization_parameters=(
              tf.estimator.tpu.experimental.AdagradParameters(0.1))))
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow_estimator.python.estimator.api._v1.estimator.tpu.experimental"))

(defn EmbeddingConfigSpec 
  "Class to keep track of the specification for TPU embeddings.

  Pass this class to `tf.estimator.tpu.TPUEstimator` via the
  `embedding_config_spec` parameter. At minimum you need to specify
  `feature_columns` and `optimization_parameters`. The feature columns passed
  should be created with some combination of
  `tf.tpu.experimental.embedding_column` and
  `tf.tpu.experimental.shared_embedding_columns`.

  TPU embeddings do not support arbitrary Tensorflow optimizers and the
  main optimizer you use for your model will be ignored for the embedding table
  variables. Instead TPU embeddigns support a fixed set of predefined optimizers
  that you can select from and set the parameters of. These include adagrad,
  adam and stochastic gradient descent. Each supported optimizer has a
  `Parameters` class in the `tf.tpu.experimental` namespace.

  ```
  column_a = tf.feature_column.categorical_column_with_identity(...)
  column_b = tf.feature_column.categorical_column_with_identity(...)
  column_c = tf.feature_column.categorical_column_with_identity(...)
  tpu_shared_columns = tf.tpu.experimental.shared_embedding_columns(
      [column_a, column_b], 10)
  tpu_non_shared_column = tf.tpu.experimental.embedding_column(
      column_c, 10)
  tpu_columns = [tpu_non_shared_column] + tpu_shared_columns
  ...
  def model_fn(features):
    dense_features = tf.keras.layers.DenseFeature(tpu_columns)
    embedded_feature = dense_features(features)
    ...

  estimator = tf.estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      ...
      embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          column=tpu_columns,
          optimization_parameters=(
              tf.estimator.tpu.experimental.AdagradParameters(0.1))))
  "
  [feature_columns optimization_parameters clipping_limit & {:keys [pipeline_execution_with_tensor_core experimental_gradient_multiplier_fn feature_to_config_dict table_to_config_dict partition_strategy]
                       :or {experimental_gradient_multiplier_fn None feature_to_config_dict None table_to_config_dict None}} ]
    (py/call-attr-kw experimental "EmbeddingConfigSpec" [feature_columns optimization_parameters clipping_limit] {:pipeline_execution_with_tensor_core pipeline_execution_with_tensor_core :experimental_gradient_multiplier_fn experimental_gradient_multiplier_fn :feature_to_config_dict feature_to_config_dict :table_to_config_dict table_to_config_dict :partition_strategy partition_strategy }))

(defn clipping-limit 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "clipping_limit"))

(defn experimental-gradient-multiplier-fn 
  "Alias for field number 4"
  [ self ]
    (py/call-attr self "experimental_gradient_multiplier_fn"))

(defn feature-columns 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "feature_columns"))

(defn feature-to-config-dict 
  "Alias for field number 5"
  [ self ]
    (py/call-attr self "feature_to_config_dict"))

(defn optimization-parameters 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "optimization_parameters"))

(defn partition-strategy 
  "Alias for field number 7"
  [ self ]
    (py/call-attr self "partition_strategy"))

(defn pipeline-execution-with-tensor-core 
  "Alias for field number 3"
  [ self ]
    (py/call-attr self "pipeline_execution_with_tensor_core"))

(defn table-to-config-dict 
  "Alias for field number 6"
  [ self ]
    (py/call-attr self "table_to_config_dict"))
