(ns tensorflow.python.keras.api.-v1.keras.losses.CategoricalCrossentropy
  "Computes the crossentropy loss between the labels and predictions.

  Use this crossentropy loss function when there are two or more label classes.
  We expect labels to be provided in a `one_hot` representation. If you want to
  provide labels as integers, please use `SparseCategoricalCrossentropy` loss.
  There should be `# classes` floating point values per feature.

  In the snippet below, there is `# classes` floating pointing values per
  example. The shape of both `y_pred` and `y_true` are
  `[batch_size, num_classes]`.

  Usage:

  ```python
  cce = tf.keras.losses.CategoricalCrossentropy()
  loss = cce(
    [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    [[.9, .05, .05], [.05, .89, .06], [.05, .01, .94]])
  print('Loss: ', loss.numpy())  # Loss: 0.0945
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.CategoricalCrossentropy())
  ```

  Args:
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
      Note: Using from_logits=True may be more numerically stable.
    label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
      meaning the confidence on label values are relaxed. e.g.
      `label_smoothing=0.2` means that we will use a value of `0.1` for label
      `0` and `0.9` for label `1`\"
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

(defn CategoricalCrossentropy 
  "Computes the crossentropy loss between the labels and predictions.

  Use this crossentropy loss function when there are two or more label classes.
  We expect labels to be provided in a `one_hot` representation. If you want to
  provide labels as integers, please use `SparseCategoricalCrossentropy` loss.
  There should be `# classes` floating point values per feature.

  In the snippet below, there is `# classes` floating pointing values per
  example. The shape of both `y_pred` and `y_true` are
  `[batch_size, num_classes]`.

  Usage:

  ```python
  cce = tf.keras.losses.CategoricalCrossentropy()
  loss = cce(
    [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    [[.9, .05, .05], [.05, .89, .06], [.05, .01, .94]])
  print('Loss: ', loss.numpy())  # Loss: 0.0945
  ```

  Usage with the `compile` API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.CategoricalCrossentropy())
  ```

  Args:
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
      Note: Using from_logits=True may be more numerically stable.
    label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
      meaning the confidence on label values are relaxed. e.g.
      `label_smoothing=0.2` means that we will use a value of `0.1` for label
      `0` and `0.9` for label `1`\"
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
  [ & {:keys [from_logits label_smoothing reduction name]} ]
   (py/call-attr-kw losses "CategoricalCrossentropy" [] {:from_logits from_logits :label_smoothing label_smoothing :reduction reduction :name name }))

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
