(ns tensorflow.python.keras.api.-v1.keras.losses.CosineSimilarity
  "Computes the cosine similarity between `y_true` and `y_pred`.

  Usage:

  ```python
  cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
  loss = cosine_loss([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]])
  # l2_norm(y_true) = [[0., 1.], [1./1.414], 1./1.414]]]
  # l2_norm(y_pred) = [[1., 0.], [1./1.414], 1./1.414]]]
  # l2_norm(y_true) . l2_norm(y_pred) = [[0., 0.], [0.5, 0.5]]
  # loss = mean(sum(l2_norm(y_true) . l2_norm(y_pred), axis=1))
         = ((0. + 0.) +  (0.5 + 0.5)) / 2

  print('Loss: ', loss.numpy())  # Loss: 0.5
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.CosineSimilarity(axis=1))
  ```

  Args:
    axis: (Optional) Defaults to -1. The dimension along which the cosine
      similarity is computed.
    reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `AUTO`. `AUTO` indicates that the reduction option will
      be determined by the usage context. For almost all cases this defaults to
      `SUM_OVER_BATCH_SIZE`.
      When used with `tf.distribute.Strategy`, outside of built-in training
      loops such as `tf.keras` `compile` and `fit`, using `AUTO` or
      `SUM_OVER_BATCH_SIZE` will raise an error. Please see
      https://www.tensorflow.org/alpha/tutorials/distribute/training_loops
      for more details on this.
    name: Optional name for the op.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce losses (import-module "tensorflow.python.keras.api._v1.keras.losses"))

(defn CosineSimilarity 
  "Computes the cosine similarity between `y_true` and `y_pred`.

  Usage:

  ```python
  cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
  loss = cosine_loss([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]])
  # l2_norm(y_true) = [[0., 1.], [1./1.414], 1./1.414]]]
  # l2_norm(y_pred) = [[1., 0.], [1./1.414], 1./1.414]]]
  # l2_norm(y_true) . l2_norm(y_pred) = [[0., 0.], [0.5, 0.5]]
  # loss = mean(sum(l2_norm(y_true) . l2_norm(y_pred), axis=1))
         = ((0. + 0.) +  (0.5 + 0.5)) / 2

  print('Loss: ', loss.numpy())  # Loss: 0.5
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.CosineSimilarity(axis=1))
  ```

  Args:
    axis: (Optional) Defaults to -1. The dimension along which the cosine
      similarity is computed.
    reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `AUTO`. `AUTO` indicates that the reduction option will
      be determined by the usage context. For almost all cases this defaults to
      `SUM_OVER_BATCH_SIZE`.
      When used with `tf.distribute.Strategy`, outside of built-in training
      loops such as `tf.keras` `compile` and `fit`, using `AUTO` or
      `SUM_OVER_BATCH_SIZE` will raise an error. Please see
      https://www.tensorflow.org/alpha/tutorials/distribute/training_loops
      for more details on this.
    name: Optional name for the op.
  "
  [ & {:keys [axis reduction name]} ]
   (py/call-attr-kw losses "CosineSimilarity" [] {:axis axis :reduction reduction :name name }))

(defn call 
  "Invokes the `LossFunctionWrapper` instance.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.

    Returns:
      Loss values per sample.
    "
  [ self y_true y_pred ]
  (py/call-attr self "call"  self y_true y_pred ))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
