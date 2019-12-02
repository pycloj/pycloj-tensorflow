(ns tensorflow.python.keras.api.-v1.keras.losses.MeanSquaredError
  "Computes the mean of squares of errors between labels and predictions.

  `loss = square(y_true - y_pred)`

  Usage:

  ```python
  mse = tf.keras.losses.MeanSquaredError()
  loss = mse([0., 0., 1., 1.], [1., 1., 1., 0.])
  print('Loss: ', loss.numpy())  # Loss: 0.75
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.MeanSquaredError())
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

(defn MeanSquaredError 
  "Computes the mean of squares of errors between labels and predictions.

  `loss = square(y_true - y_pred)`

  Usage:

  ```python
  mse = tf.keras.losses.MeanSquaredError()
  loss = mse([0., 0., 1., 1.], [1., 1., 1., 0.])
  print('Loss: ', loss.numpy())  # Loss: 0.75
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.MeanSquaredError())
  ```
  "
  [ & {:keys [reduction name]} ]
   (py/call-attr-kw losses "MeanSquaredError" [] {:reduction reduction :name name }))

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
