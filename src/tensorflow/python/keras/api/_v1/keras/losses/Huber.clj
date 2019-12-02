(ns tensorflow.python.keras.api.-v1.keras.losses.Huber
  "Computes the Huber loss between `y_true` and `y_pred`.

  For each value x in `error = y_true - y_pred`:

  ```
  loss = 0.5 * x^2                  if |x| <= d
  loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
  ```
  where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

  Usage:

  ```python
  l = tf.keras.losses.Huber()
  loss = l([0., 1., 1.], [1., 0., 1.])
  print('Loss: ', loss.numpy())  # Loss: 0.333
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.Huber())
  ```

  Args:
    delta: A float, the point where the Huber loss function changes from a
      quadratic to linear.
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

(defn Huber 
  "Computes the Huber loss between `y_true` and `y_pred`.

  For each value x in `error = y_true - y_pred`:

  ```
  loss = 0.5 * x^2                  if |x| <= d
  loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
  ```
  where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

  Usage:

  ```python
  l = tf.keras.losses.Huber()
  loss = l([0., 1., 1.], [1., 0., 1.])
  print('Loss: ', loss.numpy())  # Loss: 0.333
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.Huber())
  ```

  Args:
    delta: A float, the point where the Huber loss function changes from a
      quadratic to linear.
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
  [ & {:keys [delta reduction name]} ]
   (py/call-attr-kw losses "Huber" [] {:delta delta :reduction reduction :name name }))

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
