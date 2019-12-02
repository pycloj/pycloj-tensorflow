(ns tensorflow.python.keras.api.-v1.keras.losses.Hinge
  "Computes the hinge loss between `y_true` and `y_pred`.

  `loss = maximum(1 - y_true * y_pred, 0)`

  `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
  provided we will convert them to -1 or 1.

  Usage:

  ```python
  h = tf.keras.losses.Hinge()
  loss = h([-1., 1., 1.], [0.6, -0.7, -0.5])

  # loss = max(0, 1 - y_true * y_pred) = [1.6 + 1.7 + 1.5] / 3

  print('Loss: ', loss.numpy())  # Loss: 1.6
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.Hinge())
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

(defn Hinge 
  "Computes the hinge loss between `y_true` and `y_pred`.

  `loss = maximum(1 - y_true * y_pred, 0)`

  `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
  provided we will convert them to -1 or 1.

  Usage:

  ```python
  h = tf.keras.losses.Hinge()
  loss = h([-1., 1., 1.], [0.6, -0.7, -0.5])

  # loss = max(0, 1 - y_true * y_pred) = [1.6 + 1.7 + 1.5] / 3

  print('Loss: ', loss.numpy())  # Loss: 1.6
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.Hinge())
  ```
  "
  [ & {:keys [reduction name]} ]
   (py/call-attr-kw losses "Hinge" [] {:reduction reduction :name name }))

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
