(ns tensorflow.python.keras.api.-v1.keras.losses.BinaryCrossentropy
  "Computes the cross-entropy loss between true labels and predicted labels.

  Use this cross-entropy loss when there are only two label classes (assumed to
  be 0 and 1). For each example, there should be a single floating-point value
  per prediction.

  In the snippet below, each of the four examples has only a single
  floating-pointing value, and both `y_pred` and `y_true` have the shape
  `[batch_size]`.

  Usage:

  ```python
  bce = tf.keras.losses.BinaryCrossentropy()
  loss = bce([0., 0., 1., 1.], [1., 1., 1., 0.])
  print('Loss: ', loss.numpy())  # Loss: 11.522857
  ```

  Usage with the `tf.keras` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.BinaryCrossentropy())
  ```

  Args:
    from_logits: Whether to interpret `y_pred` as a tensor of
      [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we assume
        that `y_pred` contains probabilities (i.e., values in [0, 1]).
      Note: Using from_logits=True may be more numerically stable.
    label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When > 0, we
      compute the loss between the predicted labels and a smoothed version of
      the true labels, where the smoothing squeezes the labels towards 0.5.
      Larger values of `label_smoothing` correspond to heavier smoothing.
    reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `AUTO`. `AUTO` indicates that the reduction option will
      be determined by the usage context. For almost all cases this defaults to
      `SUM_OVER_BATCH_SIZE`.
      When used with `tf.distribute.Strategy`, outside of built-in training
      loops such as `tf.keras` `compile` and `fit`, using `AUTO` or
      `SUM_OVER_BATCH_SIZE` will raise an error. Please see
      https://www.tensorflow.org/alpha/tutorials/distribute/training_loops
      for more details on this.
    name: (Optional) Name for the op.
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

(defn BinaryCrossentropy 
  "Computes the cross-entropy loss between true labels and predicted labels.

  Use this cross-entropy loss when there are only two label classes (assumed to
  be 0 and 1). For each example, there should be a single floating-point value
  per prediction.

  In the snippet below, each of the four examples has only a single
  floating-pointing value, and both `y_pred` and `y_true` have the shape
  `[batch_size]`.

  Usage:

  ```python
  bce = tf.keras.losses.BinaryCrossentropy()
  loss = bce([0., 0., 1., 1.], [1., 1., 1., 0.])
  print('Loss: ', loss.numpy())  # Loss: 11.522857
  ```

  Usage with the `tf.keras` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.BinaryCrossentropy())
  ```

  Args:
    from_logits: Whether to interpret `y_pred` as a tensor of
      [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we assume
        that `y_pred` contains probabilities (i.e., values in [0, 1]).
      Note: Using from_logits=True may be more numerically stable.
    label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When > 0, we
      compute the loss between the predicted labels and a smoothed version of
      the true labels, where the smoothing squeezes the labels towards 0.5.
      Larger values of `label_smoothing` correspond to heavier smoothing.
    reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `AUTO`. `AUTO` indicates that the reduction option will
      be determined by the usage context. For almost all cases this defaults to
      `SUM_OVER_BATCH_SIZE`.
      When used with `tf.distribute.Strategy`, outside of built-in training
      loops such as `tf.keras` `compile` and `fit`, using `AUTO` or
      `SUM_OVER_BATCH_SIZE` will raise an error. Please see
      https://www.tensorflow.org/alpha/tutorials/distribute/training_loops
      for more details on this.
    name: (Optional) Name for the op.
  "
  [ & {:keys [from_logits label_smoothing reduction name]} ]
   (py/call-attr-kw losses "BinaryCrossentropy" [] {:from_logits from_logits :label_smoothing label_smoothing :reduction reduction :name name }))

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
