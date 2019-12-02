(ns tensorflow.python.keras.api.-v1.keras.losses.CategoricalHinge
  "Computes the categorical hinge loss between `y_true` and `y_pred`.

  `loss = maximum(neg - pos + 1, 0)`
  where `neg = sum(y_true * y_pred)` and `pos = maximum(1 - y_true)`

  Usage:

  ```python
  ch = tf.keras.losses.CategoricalHinge()
  loss = ch([0., 1., 1.], [1., 0., 1.])
  print('Loss: ', loss.numpy())  # Loss: 1.0
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.CategoricalHinge())
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

(defn CategoricalHinge 
  "Computes the categorical hinge loss between `y_true` and `y_pred`.

  `loss = maximum(neg - pos + 1, 0)`
  where `neg = sum(y_true * y_pred)` and `pos = maximum(1 - y_true)`

  Usage:

  ```python
  ch = tf.keras.losses.CategoricalHinge()
  loss = ch([0., 1., 1.], [1., 0., 1.])
  print('Loss: ', loss.numpy())  # Loss: 1.0
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.CategoricalHinge())
  ```
  "
  [ & {:keys [reduction name]} ]
   (py/call-attr-kw losses "CategoricalHinge" [] {:reduction reduction :name name }))

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
