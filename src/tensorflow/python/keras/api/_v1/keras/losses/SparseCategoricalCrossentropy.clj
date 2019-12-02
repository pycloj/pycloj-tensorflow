(ns tensorflow.python.keras.api.-v1.keras.losses.SparseCategoricalCrossentropy
  "Computes the crossentropy loss between the labels and predictions.

  Use this crossentropy loss function when there are two or more label classes.
  We expect labels to be provided as integers. If you want to provide labels
  using `one-hot` representation, please use `CategoricalCrossentropy` loss.
  There should be `# classes` floating point values per feature for `y_pred`
  and a single floating point value per feature for `y_true`.

  In the snippet below, there is a single floating point value per example for
  `y_true` and `# classes` floating pointing values per example for `y_pred`.
  The shape of `y_true` is `[batch_size]` and the shape of `y_pred` is
  `[batch_size, num_classes]`.

  Usage:

  ```python
  cce = tf.keras.losses.SparseCategoricalCrossentropy()
  loss = cce(
    tf.convert_to_tensor([0, 1, 2]),
    tf.convert_to_tensor([[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]]))
  print('Loss: ', loss.numpy())  # Loss: 0.3239
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy())
  ```

  Args:
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
      Note: Using from_logits=True may be more numerically stable.
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

(defn SparseCategoricalCrossentropy 
  "Computes the crossentropy loss between the labels and predictions.

  Use this crossentropy loss function when there are two or more label classes.
  We expect labels to be provided as integers. If you want to provide labels
  using `one-hot` representation, please use `CategoricalCrossentropy` loss.
  There should be `# classes` floating point values per feature for `y_pred`
  and a single floating point value per feature for `y_true`.

  In the snippet below, there is a single floating point value per example for
  `y_true` and `# classes` floating pointing values per example for `y_pred`.
  The shape of `y_true` is `[batch_size]` and the shape of `y_pred` is
  `[batch_size, num_classes]`.

  Usage:

  ```python
  cce = tf.keras.losses.SparseCategoricalCrossentropy()
  loss = cce(
    tf.convert_to_tensor([0, 1, 2]),
    tf.convert_to_tensor([[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]]))
  print('Loss: ', loss.numpy())  # Loss: 0.3239
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy())
  ```

  Args:
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
      Note: Using from_logits=True may be more numerically stable.
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
  [ & {:keys [from_logits reduction name]} ]
   (py/call-attr-kw losses "SparseCategoricalCrossentropy" [] {:from_logits from_logits :reduction reduction :name name }))

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
