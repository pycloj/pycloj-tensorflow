(ns tensorflow.python.keras.api.-v1.keras.losses.MeanAbsolutePercentageError
  "Computes the mean absolute percentage error between `y_true` and `y_pred`.

  `loss = 100 * abs(y_true - y_pred) / y_true`

  Usage:

  ```python
  mape = tf.keras.losses.MeanAbsolutePercentageError()
  loss = mape([0., 0., 1., 1.], [1., 1., 1., 0.])
  print('Loss: ', loss.numpy())  # Loss: 5e+08
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.MeanAbsolutePercentageError())
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

(defn MeanAbsolutePercentageError 
  "Computes the mean absolute percentage error between `y_true` and `y_pred`.

  `loss = 100 * abs(y_true - y_pred) / y_true`

  Usage:

  ```python
  mape = tf.keras.losses.MeanAbsolutePercentageError()
  loss = mape([0., 0., 1., 1.], [1., 1., 1., 0.])
  print('Loss: ', loss.numpy())  # Loss: 5e+08
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.MeanAbsolutePercentageError())
  ```
  "
  [ & {:keys [reduction name]} ]
   (py/call-attr-kw losses "MeanAbsolutePercentageError" [] {:reduction reduction :name name }))

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
