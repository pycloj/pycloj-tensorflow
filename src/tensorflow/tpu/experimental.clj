(ns tensorflow.-api.v1.tpu.experimental
  "Public API for tf.tpu.experimental namespace.
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

(defn embedding-column 
  "TPU version of `tf.compat.v1.feature_column.embedding_column`.

  Note that the interface for `tf.tpu.experimental.embedding_column` is
  different from that of `tf.compat.v1.feature_column.embedding_column`: The
  following arguments are NOT supported: `ckpt_to_load_from`,
  `tensor_name_in_ckpt`, `max_norm` and `trainable`.

  Use this function in place of `tf.compat.v1.feature_column.embedding_column`
  when you want to use the TPU to accelerate your embedding lookups via TPU
  embeddings.

  ```
  column = tf.feature_column.categorical_column_with_identity(...)
  tpu_column = tf.tpu.experimental.embedding_column(column, 10)
  ...
  def model_fn(features):
    dense_feature = tf.keras.layers.DenseFeature(tpu_column)
    embedded_feature = dense_feature(features)
    ...

  estimator = tf.estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      ...
      embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
        column=[tpu_column],
        ...))
  ```

  Args:
    categorical_column: A categorical column returned from
        `categorical_column_with_identity`, `weighted_categorical_column`,
        `categorical_column_with_vocabulary_file`,
        `categorical_column_with_vocabulary_list`,
        `sequence_categorical_column_with_identity`,
        `sequence_categorical_column_with_vocabulary_file`,
        `sequence_categorical_column_with_vocabulary_list`
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row for a non-sequence column. For more information, see
      `tf.feature_column.embedding_column`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.compat.v1.truncated_normal_initializer` with mean `0.0` and
      standard deviation `1/sqrt(dimension)`.
    max_sequence_length: An non-negative integer specifying the max sequence
      length. Any sequence shorter then this will be padded with 0 embeddings
      and any sequence longer will be truncated. This must be positive for
      sequence features and 0 for non-sequence features.
    learning_rate_fn: A function that takes global step and returns learning
      rate for the embedding table.

  Returns:
    A  `_TPUEmbeddingColumnV2`.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if `initializer` is specified but not callable.
  "
  [categorical_column dimension & {:keys [combiner initializer max_sequence_length learning_rate_fn]
                       :or {initializer None learning_rate_fn None}} ]
    (py/call-attr-kw experimental "embedding_column" [categorical_column dimension] {:combiner combiner :initializer initializer :max_sequence_length max_sequence_length :learning_rate_fn learning_rate_fn }))

(defn initialize-tpu-system 
  "Initialize the TPU devices.

  Args:
    cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,
        which provides information about the TPU cluster.
  Returns:
    The tf.tpu.Topology object for the topology of the TPU cluster.

  Raises:
    RuntimeError: If no TPU devices found for eager execution.
  "
  [ cluster_resolver ]
  (py/call-attr experimental "initialize_tpu_system"  cluster_resolver ))

(defn shared-embedding-columns 
  "TPU version of `tf.compat.v1.feature_column.shared_embedding_columns`.

  Note that the interface for `tf.tpu.experimental.shared_embedding_columns` is
  different from that of `tf.compat.v1.feature_column.shared_embedding_columns`:
  The following arguments are NOT supported: `ckpt_to_load_from`,
  `tensor_name_in_ckpt`, `max_norm` and `trainable`.

  Use this function in place of
  tf.compat.v1.feature_column.shared_embedding_columns` when you want to use the
  TPU to accelerate your embedding lookups via TPU embeddings.

  ```
  column_a = tf.feature_column.categorical_column_with_identity(...)
  column_b = tf.feature_column.categorical_column_with_identity(...)
  tpu_columns = tf.tpu.experimental.shared_embedding_columns(
      [column_a, column_b], 10)
  ...
  def model_fn(features):
    dense_feature = tf.keras.layers.DenseFeature(tpu_columns)
    embedded_feature = dense_feature(features)
    ...

  estimator = tf.estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      ...
      embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
          column=tpu_columns,
          ...))
  ```

  Args:
    categorical_columns: A list of categorical columns returned from
        `categorical_column_with_identity`, `weighted_categorical_column`,
        `categorical_column_with_vocabulary_file`,
        `categorical_column_with_vocabulary_list`,
        `sequence_categorical_column_with_identity`,
        `sequence_categorical_column_with_vocabulary_file`,
        `sequence_categorical_column_with_vocabulary_list`
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row for a non-sequence column. For more information, see
      `tf.feature_column.embedding_column`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.truncated_normal_initializer` with mean `0.0` and standard deviation
      `1/sqrt(dimension)`.
    shared_embedding_collection_name: Optional name of the collection where
      shared embedding weights are added. If not given, a reasonable name will
      be chosen based on the names of `categorical_columns`. This is also used
      in `variable_scope` when creating shared embedding weights.
    max_sequence_lengths: An list of non-negative integers, either None or
      empty or the same length as the argument categorical_columns. Entries
      corresponding to non-sequence columns must be 0 and entries corresponding
      to sequence columns specify the max sequence length for the column. Any
      sequence shorter then this will be padded with 0 embeddings and any
      sequence longer will be truncated.
    learning_rate_fn: A function that takes global step and returns learning
      rate for the embedding table.

  Returns:
    A  list of `_TPUSharedEmbeddingColumnV2`.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if `initializer` is specified but not callable.
    ValueError: if `max_sequence_lengths` is specified and not the same length
      as `categorical_columns`.
    ValueError: if `max_sequence_lengths` is positive for a non sequence column
      or 0 for a sequence column.
  "
  [categorical_columns dimension & {:keys [combiner initializer shared_embedding_collection_name max_sequence_lengths learning_rate_fn]
                       :or {initializer None shared_embedding_collection_name None max_sequence_lengths None learning_rate_fn None}} ]
    (py/call-attr-kw experimental "shared_embedding_columns" [categorical_columns dimension] {:combiner combiner :initializer initializer :shared_embedding_collection_name shared_embedding_collection_name :max_sequence_lengths max_sequence_lengths :learning_rate_fn learning_rate_fn }))
