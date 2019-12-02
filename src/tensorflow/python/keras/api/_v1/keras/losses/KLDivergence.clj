(ns tensorflow.python.keras.api.-v1.keras.losses.KLDivergence
  "Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

  `loss = y_true * log(y_true / y_pred)`

  See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

  Usage:

  ```python
  k = tf.keras.losses.KLDivergence()
  loss = k([.4, .9, .2], [.5, .8, .12])
  print('Loss: ', loss.numpy())  # Loss: 0.11891246
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.KLDivergence())
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

(defn KLDivergence 
  "Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

  `loss = y_true * log(y_true / y_pred)`

  See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

  Usage:

  ```python
  k = tf.keras.losses.KLDivergence()
  loss = k([.4, .9, .2], [.5, .8, .12])
  print('Loss: ', loss.numpy())  # Loss: 0.11891246
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.KLDivergence())
  ```
  "
  [ & {:keys [reduction name]} ]
   (py/call-attr-kw losses "KLDivergence" [] {:reduction reduction :name name }))

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
