(ns tensorflow.contrib.learn.python.learn.ops.losses-ops
  "TensorFlow Ops for loss computation (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce losses-ops (import-module "tensorflow.contrib.learn.python.learn.ops.losses_ops"))
(defn deprecated 
  "Decorator for marking functions or methods deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called. It has the following format:

    <function> (from <module>) is deprecated and will be removed after <date>.
    Instructions for updating:
    <instructions>

  If `date` is None, 'after <date>' is replaced with 'in a future version'.
  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated)' is appended
  to the first line of the docstring and a deprecation notice is prepended
  to the rest of the docstring.

  Args:
    date: String or None. The date the function is scheduled to be removed.
      Must be ISO 8601 (YYYY-MM-DD), or None.
    instructions: String. Instructions on how to update code using the
      deprecated function.
    warn_once: Boolean. Set to `True` to warn only the first time the decorated
      function is called. Otherwise, every call will log a warning.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not None or in ISO 8601 format, or instructions are
      empty.
  "
  [date instructions  & {:keys [warn_once]} ]
    (py/call-attr-kw losses-ops "deprecated" [date instructions] {:warn_once warn_once }))

(defn mean-squared-error-regressor 
  "Returns prediction and loss for mean squared error regression. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-01.
Instructions for updating:
Use `tf.losses.mean_squared_error` and explicit logits computation."
  [ tensor_in labels weights biases name ]
  (py/call-attr losses-ops "mean_squared_error_regressor"  tensor_in labels weights biases name ))

(defn softmax-classifier 
  "Returns prediction and loss for softmax classifier. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-01.
Instructions for updating:
Use `tf.losses.softmax_cross_entropy` and explicit logits computation.

This function returns \"probabilities\" and a cross entropy loss. To obtain
predictions, use `tf.argmax` on the returned probabilities.

This function requires labels to be passed in one-hot encoding.

Args:
  tensor_in: Input tensor, [batch_size, feature_size], features.
  labels: Tensor, [batch_size, n_classes], one-hot labels of the output
    classes.
  weights: Tensor, [batch_size, feature_size], linear transformation
    matrix.
  biases: Tensor, [batch_size], biases.
  class_weight: Tensor, optional, [n_classes], weight for each class.
    If not given, all classes are supposed to have weight one.
  name: Operation name.

Returns:
  `tuple` of softmax predictions and loss `Tensor`s."
  [ tensor_in labels weights biases class_weight name ]
  (py/call-attr losses-ops "softmax_classifier"  tensor_in labels weights biases class_weight name ))
