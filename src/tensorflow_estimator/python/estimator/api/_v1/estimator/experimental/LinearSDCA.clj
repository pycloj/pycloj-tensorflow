(ns tensorflow-estimator.python.estimator.api.-v1.estimator.experimental.LinearSDCA
  "Stochastic Dual Coordinate Ascent helper for linear estimators.

  Objects of this class are intended to be provided as the optimizer argument
  (though LinearSDCA objects do not implement the `tf.train.Optimizer` interface)
  when creating `tf.estimator.LinearClassifier` or `tf.estimator.LinearRegressor`.

  SDCA can only be used with `LinearClassifier` and `LinearRegressor` under the
  following conditions:
  
    - Feature columns are of type V2.
    - Multivalent categorical columns are not normalized. In other words the
      `sparse_combiner` argument in the estimator constructor should be \"sum\".
    - For classification: binary label.
    - For regression: one-dimensional label.

  Example usage:

  ```python
  real_feature_column = numeric_column(...)
  sparse_feature_column = categorical_column_with_hash_bucket(...)
  linear_sdca = tf.estimator.experimental.LinearSDCA(
      example_id_column='example_id',
      num_loss_partitions=1,
      num_table_shards=1,
      symmetric_l2_regularization=2.0)
  classifier = tf.estimator.LinearClassifier(
      feature_columns=[real_feature_column, sparse_feature_column],
      weight_column=...,
      optimizer=linear_sdca)
  classifier.train(input_fn_train, steps=50)
  classifier.evaluate(input_fn=input_fn_eval)
  ```

  Here the expectation is that the `input_fn_*` functions passed to train and
  evaluate return a pair (dict, label_tensor) where dict has `example_id_column`
  as `key` whose value is a `Tensor` of shape [batch_size] and dtype string.
  num_loss_partitions defines sigma' in eq (11) of [3]. Convergence of (global)
  loss is guaranteed if `num_loss_partitions` is larger or equal to the product
  `(#concurrent train ops/per worker) x (#workers)`. Larger values for
  `num_loss_partitions` lead to slower convergence. The recommended value for
  `num_loss_partitions` in `tf.estimator` (where currently there is one process
  per worker) is the number of workers running the train steps. It defaults to 1
  (single machine).
  `num_table_shards` defines the number of shards for the internal state
  table, typically set to match the number of parameter servers for large
  data sets.

  The SDCA algorithm was originally introduced in [1] and it was followed by
  the L1 proximal step [2], a distributed version [3] and adaptive sampling [4].
  [1] www.jmlr.org/papers/volume14/shalev-shwartz13a/shalev-shwartz13a.pdf
  [2] https://arxiv.org/pdf/1309.2375.pdf
  [3] https://arxiv.org/pdf/1502.03508.pdf
  [4] https://arxiv.org/pdf/1502.08053.pdf
  Details specific to this implementation are provided in:
  https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator/python/estimator/canned/linear_optimizer/doc/sdca.ipynb
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow_estimator.python.estimator.api._v1.estimator.experimental"))

(defn LinearSDCA 
  "Stochastic Dual Coordinate Ascent helper for linear estimators.

  Objects of this class are intended to be provided as the optimizer argument
  (though LinearSDCA objects do not implement the `tf.train.Optimizer` interface)
  when creating `tf.estimator.LinearClassifier` or `tf.estimator.LinearRegressor`.

  SDCA can only be used with `LinearClassifier` and `LinearRegressor` under the
  following conditions:
  
    - Feature columns are of type V2.
    - Multivalent categorical columns are not normalized. In other words the
      `sparse_combiner` argument in the estimator constructor should be \"sum\".
    - For classification: binary label.
    - For regression: one-dimensional label.

  Example usage:

  ```python
  real_feature_column = numeric_column(...)
  sparse_feature_column = categorical_column_with_hash_bucket(...)
  linear_sdca = tf.estimator.experimental.LinearSDCA(
      example_id_column='example_id',
      num_loss_partitions=1,
      num_table_shards=1,
      symmetric_l2_regularization=2.0)
  classifier = tf.estimator.LinearClassifier(
      feature_columns=[real_feature_column, sparse_feature_column],
      weight_column=...,
      optimizer=linear_sdca)
  classifier.train(input_fn_train, steps=50)
  classifier.evaluate(input_fn=input_fn_eval)
  ```

  Here the expectation is that the `input_fn_*` functions passed to train and
  evaluate return a pair (dict, label_tensor) where dict has `example_id_column`
  as `key` whose value is a `Tensor` of shape [batch_size] and dtype string.
  num_loss_partitions defines sigma' in eq (11) of [3]. Convergence of (global)
  loss is guaranteed if `num_loss_partitions` is larger or equal to the product
  `(#concurrent train ops/per worker) x (#workers)`. Larger values for
  `num_loss_partitions` lead to slower convergence. The recommended value for
  `num_loss_partitions` in `tf.estimator` (where currently there is one process
  per worker) is the number of workers running the train steps. It defaults to 1
  (single machine).
  `num_table_shards` defines the number of shards for the internal state
  table, typically set to match the number of parameter servers for large
  data sets.

  The SDCA algorithm was originally introduced in [1] and it was followed by
  the L1 proximal step [2], a distributed version [3] and adaptive sampling [4].
  [1] www.jmlr.org/papers/volume14/shalev-shwartz13a/shalev-shwartz13a.pdf
  [2] https://arxiv.org/pdf/1309.2375.pdf
  [3] https://arxiv.org/pdf/1502.03508.pdf
  [4] https://arxiv.org/pdf/1502.08053.pdf
  Details specific to this implementation are provided in:
  https://github.com/tensorflow/estimator/tree/master/tensorflow_estimator/python/estimator/canned/linear_optimizer/doc/sdca.ipynb
  "
  [example_id_column & {:keys [num_loss_partitions num_table_shards symmetric_l1_regularization symmetric_l2_regularization adaptive]
                       :or {num_table_shards None}} ]
    (py/call-attr-kw experimental "LinearSDCA" [example_id_column] {:num_loss_partitions num_loss_partitions :num_table_shards num_table_shards :symmetric_l1_regularization symmetric_l1_regularization :symmetric_l2_regularization symmetric_l2_regularization :adaptive adaptive }))

(defn get-train-step 
  "Returns the training operation of an SdcaModel optimizer."
  [ self state_manager weight_column_name loss_type feature_columns features targets bias_var global_step ]
  (py/call-attr self "get_train_step"  self state_manager weight_column_name loss_type feature_columns features targets bias_var global_step ))
