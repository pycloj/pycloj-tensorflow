(ns tensorflow.python.keras.api.-v1.keras.losses.Poisson
  "Computes the Poisson loss between `y_true` and `y_pred`.

  `loss = y_pred - y_true * log(y_pred)`

  Usage:

  ```python
  p = tf.keras.losses.Poisson()
  loss = p([1., 9., 2.], [4., 8., 12.])
  print('Loss: ', loss.numpy())  # Loss: -0.35702705
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.Poisson())
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

(defn Poisson 
  "Computes the Poisson loss between `y_true` and `y_pred`.

  `loss = y_pred - y_true * log(y_pred)`

  Usage:

  ```python
  p = tf.keras.losses.Poisson()
  loss = p([1., 9., 2.], [4., 8., 12.])
  print('Loss: ', loss.numpy())  # Loss: -0.35702705
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.Poisson())
  ```
  "
  [ & {:keys [reduction name]} ]
   (py/call-attr-kw losses "Poisson" [] {:reduction reduction :name name }))

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
