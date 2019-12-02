(ns tensorflow.python.keras.api.-v1.keras.losses.SquaredHinge
  "Computes the squared hinge loss between `y_true` and `y_pred`.

  `loss = square(maximum(1 - y_true * y_pred, 0))`

  `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
  provided we will convert them to -1 or 1.

  Usage:

  ```python
  sh = tf.keras.losses.SquaredHinge()
  loss = sh([-1., 1., 1.], [0.6, -0.7, -0.5])

  # loss = (max(0, 1 - y_true * y_pred))^2 = [1.6^2 + 1.7^2 + 1.5^2] / 3

  print('Loss: ', loss.numpy())  # Loss: 2.566666
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.SquaredHinge())
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
(defonce losses (import-module "tensorflow.python.keras.api._v1.keras.losses"))

(defn SquaredHinge 
  "Computes the squared hinge loss between `y_true` and `y_pred`.

  `loss = square(maximum(1 - y_true * y_pred, 0))`

  `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
  provided we will convert them to -1 or 1.

  Usage:

  ```python
  sh = tf.keras.losses.SquaredHinge()
  loss = sh([-1., 1., 1.], [0.6, -0.7, -0.5])

  # loss = (max(0, 1 - y_true * y_pred))^2 = [1.6^2 + 1.7^2 + 1.5^2] / 3

  print('Loss: ', loss.numpy())  # Loss: 2.566666
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.SquaredHinge())
  ```
  "
  [ & {:keys [reduction name]} ]
   (py/call-attr-kw losses "SquaredHinge" [] {:reduction reduction :name name }))

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
