(ns tensorflow.python.keras.api.-v1.keras.metrics
  "Built-in metrics.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce metrics (import-module "tensorflow.python.keras.api._v1.keras.metrics"))

(defn KLD 
  "Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

  `loss = y_true * log(y_true / y_pred)`

  See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

  Usage:

  ```python
  loss = tf.keras.losses.KLD([.4, .9, .2], [.5, .8, .12])
  print('Loss: ', loss.numpy())  # Loss: 0.11891246
  ```

  Args:
    y_true: Tensor of true targets.
    y_pred: Tensor of predicted targets.

  Returns:
    A `Tensor` with loss.

  Raises:
      TypeError: If `y_true` cannot be cast to the `y_pred.dtype`.

  "
  [ y_true y_pred ]
  (py/call-attr metrics "KLD"  y_true y_pred ))

(defn MAE 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "MAE"  y_true y_pred ))

(defn MAPE 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "MAPE"  y_true y_pred ))

(defn MSE 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "MSE"  y_true y_pred ))

(defn MSLE 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "MSLE"  y_true y_pred ))
(defn binary-accuracy 
  ""
  [y_true y_pred  & {:keys [threshold]} ]
    (py/call-attr-kw metrics "binary_accuracy" [y_true y_pred] {:threshold threshold }))
(defn binary-crossentropy 
  ""
  [y_true y_pred  & {:keys [from_logits label_smoothing]} ]
    (py/call-attr-kw metrics "binary_crossentropy" [y_true y_pred] {:from_logits from_logits :label_smoothing label_smoothing }))

(defn categorical-accuracy 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "categorical_accuracy"  y_true y_pred ))
(defn categorical-crossentropy 
  "Computes the categorical crossentropy loss.

  Args:
    y_true: tensor of true targets.
    y_pred: tensor of predicted targets.
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.

  Returns:
    Categorical crossentropy loss value.
  "
  [y_true y_pred  & {:keys [from_logits label_smoothing]} ]
    (py/call-attr-kw metrics "categorical_crossentropy" [y_true y_pred] {:from_logits from_logits :label_smoothing label_smoothing }))
(defn cosine 
  "Computes the cosine similarity between labels and predictions."
  [y_true y_pred  & {:keys [axis]} ]
    (py/call-attr-kw metrics "cosine" [y_true y_pred] {:axis axis }))
(defn cosine-proximity 
  "Computes the cosine similarity between labels and predictions."
  [y_true y_pred  & {:keys [axis]} ]
    (py/call-attr-kw metrics "cosine_proximity" [y_true y_pred] {:axis axis }))

(defn deserialize 
  ""
  [ config custom_objects ]
  (py/call-attr metrics "deserialize"  config custom_objects ))

(defn get 
  ""
  [ identifier ]
  (py/call-attr metrics "get"  identifier ))

(defn hinge 
  "Computes the hinge loss between `y_true` and `y_pred`.

  Args:
    y_true: The ground truth values. `y_true` values are expected to be -1 or 1.
      If binary (0 or 1) labels are provided they will be converted to -1 or 1.
    y_pred: The predicted values.

  Returns:
    Tensor with one scalar loss entry per sample.
  "
  [ y_true y_pred ]
  (py/call-attr metrics "hinge"  y_true y_pred ))

(defn kld 
  "Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

  `loss = y_true * log(y_true / y_pred)`

  See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

  Usage:

  ```python
  loss = tf.keras.losses.KLD([.4, .9, .2], [.5, .8, .12])
  print('Loss: ', loss.numpy())  # Loss: 0.11891246
  ```

  Args:
    y_true: Tensor of true targets.
    y_pred: Tensor of predicted targets.

  Returns:
    A `Tensor` with loss.

  Raises:
      TypeError: If `y_true` cannot be cast to the `y_pred.dtype`.

  "
  [ y_true y_pred ]
  (py/call-attr metrics "kld"  y_true y_pred ))

(defn kullback-leibler-divergence 
  "Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

  `loss = y_true * log(y_true / y_pred)`

  See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

  Usage:

  ```python
  loss = tf.keras.losses.KLD([.4, .9, .2], [.5, .8, .12])
  print('Loss: ', loss.numpy())  # Loss: 0.11891246
  ```

  Args:
    y_true: Tensor of true targets.
    y_pred: Tensor of predicted targets.

  Returns:
    A `Tensor` with loss.

  Raises:
      TypeError: If `y_true` cannot be cast to the `y_pred.dtype`.

  "
  [ y_true y_pred ]
  (py/call-attr metrics "kullback_leibler_divergence"  y_true y_pred ))

(defn mae 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "mae"  y_true y_pred ))

(defn mape 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "mape"  y_true y_pred ))

(defn mean-absolute-error 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "mean_absolute_error"  y_true y_pred ))

(defn mean-absolute-percentage-error 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "mean_absolute_percentage_error"  y_true y_pred ))

(defn mean-squared-error 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "mean_squared_error"  y_true y_pred ))

(defn mean-squared-logarithmic-error 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "mean_squared_logarithmic_error"  y_true y_pred ))

(defn mse 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "mse"  y_true y_pred ))

(defn msle 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "msle"  y_true y_pred ))

(defn poisson 
  "Computes the Poisson loss between y_true and y_pred.

  The Poisson loss is the mean of the elements of the `Tensor`
  `y_pred - y_true * log(y_pred)`.

  Usage:

  ```python
  loss = tf.keras.losses.poisson([1.4, 9.3, 2.2], [4.3, 8.2, 12.2])
  print('Loss: ', loss.numpy())  # Loss: -0.8045559
  ```

  Args:
    y_true: Tensor of true targets.
    y_pred: Tensor of predicted targets.

  Returns:
    A `Tensor` with the mean Poisson loss.

  Raises:
      InvalidArgumentError: If `y_true` and `y_pred` have incompatible shapes.
  "
  [ y_true y_pred ]
  (py/call-attr metrics "poisson"  y_true y_pred ))

(defn serialize 
  ""
  [ metric ]
  (py/call-attr metrics "serialize"  metric ))

(defn sparse-categorical-accuracy 
  ""
  [ y_true y_pred ]
  (py/call-attr metrics "sparse_categorical_accuracy"  y_true y_pred ))
(defn sparse-categorical-crossentropy 
  ""
  [y_true y_pred  & {:keys [from_logits axis]} ]
    (py/call-attr-kw metrics "sparse_categorical_crossentropy" [y_true y_pred] {:from_logits from_logits :axis axis }))
(defn sparse-top-k-categorical-accuracy 
  ""
  [y_true y_pred  & {:keys [k]} ]
    (py/call-attr-kw metrics "sparse_top_k_categorical_accuracy" [y_true y_pred] {:k k }))

(defn squared-hinge 
  "Computes the squared hinge loss between `y_true` and `y_pred`.

  Args:
    y_true: The ground truth values. `y_true` values are expected to be -1 or 1.
      If binary (0 or 1) labels are provided we will convert them to -1 or 1.
    y_pred: The predicted values.

  Returns:
    Tensor with one scalar loss entry per sample.
  "
  [ y_true y_pred ]
  (py/call-attr metrics "squared_hinge"  y_true y_pred ))
(defn top-k-categorical-accuracy 
  ""
  [y_true y_pred  & {:keys [k]} ]
    (py/call-attr-kw metrics "top_k_categorical_accuracy" [y_true y_pred] {:k k }))
