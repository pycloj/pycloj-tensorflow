(ns tensorflow.contrib.keras.api.keras.losses
  "Keras built-in loss functions."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce losses (import-module "tensorflow.contrib.keras.api.keras.losses"))
(defn binary-crossentropy 
  ""
  [y_true y_pred  & {:keys [from_logits label_smoothing]} ]
    (py/call-attr-kw losses "binary_crossentropy" [y_true y_pred] {:from_logits from_logits :label_smoothing label_smoothing }))
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
    (py/call-attr-kw losses "categorical_crossentropy" [y_true y_pred] {:from_logits from_logits :label_smoothing label_smoothing }))

(defn categorical-hinge 
  "Computes the categorical hinge loss between `y_true` and `y_pred`.

  Args:
    y_true: The ground truth values. `y_true` values are expected to be -1 or 1.
      If binary (0 or 1) labels are provided they will be converted to -1 or 1.
    y_pred: The predicted values.

  Returns:
    A tensor.
  "
  [ y_true y_pred ]
  (py/call-attr losses "categorical_hinge"  y_true y_pred ))
(defn cosine-similarity 
  "Computes the cosine similarity between labels and predictions."
  [y_true y_pred  & {:keys [axis]} ]
    (py/call-attr-kw losses "cosine_similarity" [y_true y_pred] {:axis axis }))

(defn deserialize 
  ""
  [ name custom_objects ]
  (py/call-attr losses "deserialize"  name custom_objects ))

(defn get 
  ""
  [ identifier ]
  (py/call-attr losses "get"  identifier ))

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
  (py/call-attr losses "hinge"  y_true y_pred ))

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
  (py/call-attr losses "kullback_leibler_divergence"  y_true y_pred ))

(defn logcosh 
  "Logarithm of the hyperbolic cosine of the prediction error.

  `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
  to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
  like the mean squared error, but will not be so strongly affected by the
  occasional wildly incorrect prediction.

  Arguments:
      y_true: tensor of true targets.
      y_pred: tensor of predicted targets.

  Returns:
      Tensor with one scalar loss entry per sample.
  "
  [ y_true y_pred ]
  (py/call-attr losses "logcosh"  y_true y_pred ))

(defn mean-absolute-error 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "mean_absolute_error"  y_true y_pred ))

(defn mean-absolute-percentage-error 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "mean_absolute_percentage_error"  y_true y_pred ))

(defn mean-squared-error 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "mean_squared_error"  y_true y_pred ))

(defn mean-squared-logarithmic-error 
  ""
  [ y_true y_pred ]
  (py/call-attr losses "mean_squared_logarithmic_error"  y_true y_pred ))

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
  (py/call-attr losses "poisson"  y_true y_pred ))

(defn serialize 
  ""
  [ loss ]
  (py/call-attr losses "serialize"  loss ))
(defn sparse-categorical-crossentropy 
  ""
  [y_true y_pred  & {:keys [from_logits axis]} ]
    (py/call-attr-kw losses "sparse_categorical_crossentropy" [y_true y_pred] {:from_logits from_logits :axis axis }))

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
  (py/call-attr losses "squared_hinge"  y_true y_pred ))
