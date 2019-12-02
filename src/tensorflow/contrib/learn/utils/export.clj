(ns tensorflow.contrib.learn.python.learn.utils.export
  "Export utilities (deprecated).

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
(defonce export (import-module "tensorflow.contrib.learn.python.learn.utils.export"))

(defn classification-signature-fn 
  "Creates classification signature from given examples and predictions. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2017-03-25.
Instructions for updating:
signature_fns are deprecated. For canned Estimators they are no longer needed. For custom Estimators, please return output_alternatives from your model_fn via ModelFnOps.

Args:
  examples: `Tensor`.
  unused_features: `dict` of `Tensor`s.
  predictions: `Tensor` or dict of tensors that contains the classes tensor
    as in {'classes': `Tensor`}.

Returns:
  Tuple of default classification signature and empty named signatures.

Raises:
  ValueError: If examples is `None`."
  [ examples unused_features predictions ]
  (py/call-attr export "classification_signature_fn"  examples unused_features predictions ))

(defn classification-signature-fn-with-prob 
  "Classification signature from given examples and predicted probabilities. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2017-03-25.
Instructions for updating:
signature_fns are deprecated. For canned Estimators they are no longer needed. For custom Estimators, please return output_alternatives from your model_fn via ModelFnOps.

Args:
  examples: `Tensor`.
  unused_features: `dict` of `Tensor`s.
  predictions: `Tensor` of predicted probabilities or dict that contains the
    probabilities tensor as in {'probabilities', `Tensor`}.

Returns:
  Tuple of default classification signature and empty named signatures.

Raises:
  ValueError: If examples is `None`."
  [ examples unused_features predictions ]
  (py/call-attr export "classification_signature_fn_with_prob"  examples unused_features predictions ))
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
    (py/call-attr-kw export "deprecated" [date instructions] {:warn_once warn_once }))

(defn export-estimator 
  "Deprecated, please use Estimator.export_savedmodel(). (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2016-09-23.
Instructions for updating:
Please use Estimator.export_savedmodel() instead."
  [estimator export_dir signature_fn & {:keys [input_fn default_batch_size exports_to_keep]
                       :or {exports_to_keep None}} ]
    (py/call-attr-kw export "export_estimator" [estimator export_dir signature_fn] {:input_fn input_fn :default_batch_size default_batch_size :exports_to_keep exports_to_keep }))

(defn generic-signature-fn 
  "Creates generic signature from given examples and predictions. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2017-03-25.
Instructions for updating:
signature_fns are deprecated. For canned Estimators they are no longer needed. For custom Estimators, please return output_alternatives from your model_fn via ModelFnOps.

This is needed for backward compatibility with default behavior of
export_estimator.

Args:
  examples: `Tensor`.
  unused_features: `dict` of `Tensor`s.
  predictions: `Tensor` or `dict` of `Tensor`s.

Returns:
  Tuple of default signature and empty named signatures.

Raises:
  ValueError: If examples is `None`."
  [ examples unused_features predictions ]
  (py/call-attr export "generic_signature_fn"  examples unused_features predictions ))

(defn logistic-regression-signature-fn 
  "Creates logistic regression signature from given examples and predictions. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2017-03-25.
Instructions for updating:
signature_fns are deprecated. For canned Estimators they are no longer needed. For custom Estimators, please return output_alternatives from your model_fn via ModelFnOps.

Args:
  examples: `Tensor`.
  unused_features: `dict` of `Tensor`s.
  predictions: `Tensor` of shape [batch_size, 2] of predicted probabilities or
    dict that contains the probabilities tensor as in
    {'probabilities', `Tensor`}.

Returns:
  Tuple of default regression signature and named signature.

Raises:
  ValueError: If examples is `None`."
  [ examples unused_features predictions ]
  (py/call-attr export "logistic_regression_signature_fn"  examples unused_features predictions ))

(defn regression-signature-fn 
  "Creates regression signature from given examples and predictions. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2017-03-25.
Instructions for updating:
signature_fns are deprecated. For canned Estimators they are no longer needed. For custom Estimators, please return output_alternatives from your model_fn via ModelFnOps.

Args:
  examples: `Tensor`.
  unused_features: `dict` of `Tensor`s.
  predictions: `Tensor`.

Returns:
  Tuple of default regression signature and empty named signatures.

Raises:
  ValueError: If examples is `None`."
  [ examples unused_features predictions ]
  (py/call-attr export "regression_signature_fn"  examples unused_features predictions ))
