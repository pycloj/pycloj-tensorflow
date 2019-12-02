(ns tensorflow.python.keras.api.-v1.keras.losses.Loss
  "Loss base class.

  To be implemented by subclasses:
  * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.

  Example subclass implementation:
  ```
  class MeanSquaredError(Loss):
    def call(self, y_true, y_pred):
      y_pred = ops.convert_to_tensor(y_pred)
      y_true = math_ops.cast(y_true, y_pred.dtype)
      return K.mean(math_ops.square(y_pred - y_true), axis=-1)
  ```

  When used with `tf.distribute.Strategy`, outside of built-in training loops
  such as `tf.keras` `compile` and `fit`, please use 'SUM' or 'NONE' reduction
  types, and reduce losses explicitly in your training loop. Using 'AUTO' or
  'SUM_OVER_BATCH_SIZE' will raise an error.

  Please see
  https://www.tensorflow.org/alpha/tutorials/distribute/training_loops for more
  details on this.

  You can implement 'SUM_OVER_BATCH_SIZE' using global batch size like:
  ```
  with strategy.scope():
    loss_obj = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    ....
    loss = (tf.reduce_sum(loss_obj(labels, predictions)) *
            (1. / global_batch_size))
  ```

  Args:
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

(defn Loss 
  "Loss base class.

  To be implemented by subclasses:
  * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.

  Example subclass implementation:
  ```
  class MeanSquaredError(Loss):
    def call(self, y_true, y_pred):
      y_pred = ops.convert_to_tensor(y_pred)
      y_true = math_ops.cast(y_true, y_pred.dtype)
      return K.mean(math_ops.square(y_pred - y_true), axis=-1)
  ```

  When used with `tf.distribute.Strategy`, outside of built-in training loops
  such as `tf.keras` `compile` and `fit`, please use 'SUM' or 'NONE' reduction
  types, and reduce losses explicitly in your training loop. Using 'AUTO' or
  'SUM_OVER_BATCH_SIZE' will raise an error.

  Please see
  https://www.tensorflow.org/alpha/tutorials/distribute/training_loops for more
  details on this.

  You can implement 'SUM_OVER_BATCH_SIZE' using global batch size like:
  ```
  with strategy.scope():
    loss_obj = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    ....
    loss = (tf.reduce_sum(loss_obj(labels, predictions)) *
            (1. / global_batch_size))
  ```

  Args:
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
  [ & {:keys [reduction name]
       :or {name None}} ]
  
   (py/call-attr-kw losses "Loss" [] {:reduction reduction :name name }))

(defn call 
  "Invokes the `Loss` instance.

    Args:
      y_true: Ground truth values, with the same shape as 'y_pred'.
      y_pred: The predicted values.
    "
  [ self y_true y_pred ]
  (py/call-attr self "call"  self y_true y_pred ))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
