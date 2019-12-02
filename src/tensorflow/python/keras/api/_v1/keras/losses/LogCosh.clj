(ns tensorflow.python.keras.api.-v1.keras.losses.LogCosh
  "Computes the logarithm of the hyperbolic cosine of the prediction error.

  `logcosh = log((exp(x) + exp(-x))/2)`,
  where x is the error `y_pred - y_true`.

  Usage:

  ```python
  l = tf.keras.losses.LogCosh()
  loss = l([0., 1., 1.], [1., 0., 1.])
  print('Loss: ', loss.numpy())  # Loss: 0.289
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.LogCosh())
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

(defn LogCosh 
  "Computes the logarithm of the hyperbolic cosine of the prediction error.

  `logcosh = log((exp(x) + exp(-x))/2)`,
  where x is the error `y_pred - y_true`.

  Usage:

  ```python
  l = tf.keras.losses.LogCosh()
  loss = l([0., 1., 1.], [1., 0., 1.])
  print('Loss: ', loss.numpy())  # Loss: 0.289
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.LogCosh())
  ```
  "
  [ & {:keys [reduction name]} ]
   (py/call-attr-kw losses "LogCosh" [] {:reduction reduction :name name }))

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
