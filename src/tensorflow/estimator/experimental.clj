(ns tensorflow-estimator.python.estimator.api.-v1.estimator.experimental
  "Public API for tf.estimator.experimental namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow_estimator.python.estimator.api._v1.estimator.experimental"))

(defn build-raw-supervised-input-receiver-fn 
  "Build a supervised_input_receiver_fn for raw features and labels.

  This function wraps tensor placeholders in a supervised_receiver_fn
  with the expectation that the features and labels appear precisely as
  the model_fn expects them. Features and labels can therefore be dicts of
  tensors, or raw tensors.

  Args:
    features: a dict of string to `Tensor` or `Tensor`.
    labels: a dict of string to `Tensor` or `Tensor`.
    default_batch_size: the number of query examples expected per batch.
        Leave unset for variable batch size (recommended).

  Returns:
    A supervised_input_receiver_fn.

  Raises:
    ValueError: if features and labels have overlapping keys.
  "
  [ features labels default_batch_size ]
  (py/call-attr experimental "build_raw_supervised_input_receiver_fn"  features labels default_batch_size ))

(defn call-logit-fn 
  "Calls logit_fn (experimental).

  THIS FUNCTION IS EXPERIMENTAL. Keras layers/models are the recommended APIs
  for logit and model composition.

  A utility function that calls the provided logit_fn with the relevant subset
  of provided arguments. Similar to tf.estimator._call_model_fn().

  Args:
    logit_fn: A logit_fn as defined above.
    features: The features dict.
    mode: TRAIN / EVAL / PREDICT ModeKeys.
    params: The hyperparameter dict.
    config: The configuration object.

  Returns:
    A logit Tensor, the output of logit_fn.

  Raises:
    ValueError: if logit_fn does not return a Tensor or a dictionary mapping
      strings to Tensors.
  "
  [ logit_fn features mode params config ]
  (py/call-attr experimental "call_logit_fn"  logit_fn features mode params config ))

(defn dnn-logit-fn-builder 
  "Function builder for a dnn logit_fn.

  Args:
    units: An int indicating the dimension of the logit layer.  In the
      MultiHead case, this should be the sum of all component Heads' logit
      dimensions.
    hidden_units: Iterable of integer number of hidden units per layer.
    feature_columns: Iterable of `feature_column._FeatureColumn` model inputs.
    activation_fn: Activation function applied to each layer.
    dropout: When not `None`, the probability we will drop out a given
      coordinate.
    input_layer_partitioner: Partitioner for input layer.
    batch_norm: Whether to use batch normalization after each hidden layer.

  Returns:
    A logit_fn (see below).

  Raises:
    ValueError: If units is not an int.
  "
  [ units hidden_units feature_columns activation_fn dropout input_layer_partitioner batch_norm ]
  (py/call-attr experimental "dnn_logit_fn_builder"  units hidden_units feature_columns activation_fn dropout input_layer_partitioner batch_norm ))
(defn linear-logit-fn-builder 
  "Function builder for a linear logit_fn.

  Args:
    units: An int indicating the dimension of the logit layer.
    feature_columns: An iterable containing all the feature columns used by
      the model.
    sparse_combiner: A string specifying how to reduce if a categorical column
      is multivalent.  One of \"mean\", \"sqrtn\", and \"sum\".

  Returns:
    A logit_fn (see below).

  "
  [units feature_columns  & {:keys [sparse_combiner]} ]
    (py/call-attr-kw experimental "linear_logit_fn_builder" [units feature_columns] {:sparse_combiner sparse_combiner }))

(defn make-early-stopping-hook 
  "Creates early-stopping hook.

  Returns a `SessionRunHook` that stops training when `should_stop_fn` returns
  `True`.

  Usage example:

  ```python
  estimator = ...
  hook = early_stopping.make_early_stopping_hook(
      estimator, should_stop_fn=make_stop_fn(...))
  train_spec = tf.estimator.TrainSpec(..., hooks=[hook])
  tf.estimator.train_and_evaluate(estimator, train_spec, ...)
  ```

  Caveat: Current implementation supports early-stopping both training and
  evaluation in local mode. In distributed mode, training can be stopped but
  evaluation (where it's a separate job) will indefinitely wait for new model
  checkpoints to evaluate, so you will need other means to detect and stop it.
  Early-stopping evaluation in distributed mode requires changes in
  `train_and_evaluate` API and will be addressed in a future revision.

  Args:
    estimator: A `tf.estimator.Estimator` instance.
    should_stop_fn: `callable`, function that takes no arguments and returns a
      `bool`. If the function returns `True`, stopping will be initiated by the
      chief.
    run_every_secs: If specified, calls `should_stop_fn` at an interval of
      `run_every_secs` seconds. Defaults to 60 seconds. Either this or
      `run_every_steps` must be set.
    run_every_steps: If specified, calls `should_stop_fn` every
      `run_every_steps` steps. Either this or `run_every_secs` must be set.

  Returns:
    A `SessionRunHook` that periodically executes `should_stop_fn` and initiates
    early stopping if the function returns `True`.

  Raises:
    TypeError: If `estimator` is not of type `tf.estimator.Estimator`.
    ValueError: If both `run_every_secs` and `run_every_steps` are set.
  "
  [estimator should_stop_fn & {:keys [run_every_secs run_every_steps]
                       :or {run_every_steps None}} ]
    (py/call-attr-kw experimental "make_early_stopping_hook" [estimator should_stop_fn] {:run_every_secs run_every_secs :run_every_steps run_every_steps }))
(defn make-stop-at-checkpoint-step-hook 
  "Creates a proper StopAtCheckpointStepHook based on chief status."
  [estimator last_step  & {:keys [wait_after_file_check_secs]} ]
    (py/call-attr-kw experimental "make_stop_at_checkpoint_step_hook" [estimator last_step] {:wait_after_file_check_secs wait_after_file_check_secs }))

(defn stop-if-higher-hook 
  "Creates hook to stop if the given metric is higher than the threshold.

  Usage example:

  ```python
  estimator = ...
  # Hook to stop training if accuracy becomes higher than 0.9.
  hook = early_stopping.stop_if_higher_hook(estimator, \"accuracy\", 0.9)
  train_spec = tf.estimator.TrainSpec(..., hooks=[hook])
  tf.estimator.train_and_evaluate(estimator, train_spec, ...)
  ```

  Caveat: Current implementation supports early-stopping both training and
  evaluation in local mode. In distributed mode, training can be stopped but
  evaluation (where it's a separate job) will indefinitely wait for new model
  checkpoints to evaluate, so you will need other means to detect and stop it.
  Early-stopping evaluation in distributed mode requires changes in
  `train_and_evaluate` API and will be addressed in a future revision.

  Args:
    estimator: A `tf.estimator.Estimator` instance.
    metric_name: `str`, metric to track. \"loss\", \"accuracy\", etc.
    threshold: Numeric threshold for the given metric.
    eval_dir: If set, directory containing summary files with eval metrics. By
      default, `estimator.eval_dir()` will be used.
    min_steps: `int`, stop is never requested if global step is less than this
      value. Defaults to 0.
    run_every_secs: If specified, calls `should_stop_fn` at an interval of
      `run_every_secs` seconds. Defaults to 60 seconds. Either this or
      `run_every_steps` must be set.
    run_every_steps: If specified, calls `should_stop_fn` every
      `run_every_steps` steps. Either this or `run_every_secs` must be set.

  Returns:
    An early-stopping hook of type `SessionRunHook` that periodically checks
    if the given metric is higher than specified threshold and initiates
    early stopping if true.
  "
  [estimator metric_name threshold eval_dir & {:keys [min_steps run_every_secs run_every_steps]
                       :or {run_every_steps None}} ]
    (py/call-attr-kw experimental "stop_if_higher_hook" [estimator metric_name threshold eval_dir] {:min_steps min_steps :run_every_secs run_every_secs :run_every_steps run_every_steps }))

(defn stop-if-lower-hook 
  "Creates hook to stop if the given metric is lower than the threshold.

  Usage example:

  ```python
  estimator = ...
  # Hook to stop training if loss becomes lower than 100.
  hook = early_stopping.stop_if_lower_hook(estimator, \"loss\", 100)
  train_spec = tf.estimator.TrainSpec(..., hooks=[hook])
  tf.estimator.train_and_evaluate(estimator, train_spec, ...)
  ```

  Caveat: Current implementation supports early-stopping both training and
  evaluation in local mode. In distributed mode, training can be stopped but
  evaluation (where it's a separate job) will indefinitely wait for new model
  checkpoints to evaluate, so you will need other means to detect and stop it.
  Early-stopping evaluation in distributed mode requires changes in
  `train_and_evaluate` API and will be addressed in a future revision.

  Args:
    estimator: A `tf.estimator.Estimator` instance.
    metric_name: `str`, metric to track. \"loss\", \"accuracy\", etc.
    threshold: Numeric threshold for the given metric.
    eval_dir: If set, directory containing summary files with eval metrics. By
      default, `estimator.eval_dir()` will be used.
    min_steps: `int`, stop is never requested if global step is less than this
      value. Defaults to 0.
    run_every_secs: If specified, calls `should_stop_fn` at an interval of
      `run_every_secs` seconds. Defaults to 60 seconds. Either this or
      `run_every_steps` must be set.
    run_every_steps: If specified, calls `should_stop_fn` every
      `run_every_steps` steps. Either this or `run_every_secs` must be set.

  Returns:
    An early-stopping hook of type `SessionRunHook` that periodically checks
    if the given metric is lower than specified threshold and initiates
    early stopping if true.
  "
  [estimator metric_name threshold eval_dir & {:keys [min_steps run_every_secs run_every_steps]
                       :or {run_every_steps None}} ]
    (py/call-attr-kw experimental "stop_if_lower_hook" [estimator metric_name threshold eval_dir] {:min_steps min_steps :run_every_secs run_every_secs :run_every_steps run_every_steps }))

(defn stop-if-no-decrease-hook 
  "Creates hook to stop if metric does not decrease within given max steps.

  Usage example:

  ```python
  estimator = ...
  # Hook to stop training if loss does not decrease in over 100000 steps.
  hook = early_stopping.stop_if_no_decrease_hook(estimator, \"loss\", 100000)
  train_spec = tf.estimator.TrainSpec(..., hooks=[hook])
  tf.estimator.train_and_evaluate(estimator, train_spec, ...)
  ```

  Caveat: Current implementation supports early-stopping both training and
  evaluation in local mode. In distributed mode, training can be stopped but
  evaluation (where it's a separate job) will indefinitely wait for new model
  checkpoints to evaluate, so you will need other means to detect and stop it.
  Early-stopping evaluation in distributed mode requires changes in
  `train_and_evaluate` API and will be addressed in a future revision.

  Args:
    estimator: A `tf.estimator.Estimator` instance.
    metric_name: `str`, metric to track. \"loss\", \"accuracy\", etc.
    max_steps_without_decrease: `int`, maximum number of training steps with no
      decrease in the given metric.
    eval_dir: If set, directory containing summary files with eval metrics. By
      default, `estimator.eval_dir()` will be used.
    min_steps: `int`, stop is never requested if global step is less than this
      value. Defaults to 0.
    run_every_secs: If specified, calls `should_stop_fn` at an interval of
      `run_every_secs` seconds. Defaults to 60 seconds. Either this or
      `run_every_steps` must be set.
    run_every_steps: If specified, calls `should_stop_fn` every
      `run_every_steps` steps. Either this or `run_every_secs` must be set.

  Returns:
    An early-stopping hook of type `SessionRunHook` that periodically checks
    if the given metric shows no decrease over given maximum number of
    training steps, and initiates early stopping if true.
  "
  [estimator metric_name max_steps_without_decrease eval_dir & {:keys [min_steps run_every_secs run_every_steps]
                       :or {run_every_steps None}} ]
    (py/call-attr-kw experimental "stop_if_no_decrease_hook" [estimator metric_name max_steps_without_decrease eval_dir] {:min_steps min_steps :run_every_secs run_every_secs :run_every_steps run_every_steps }))

(defn stop-if-no-increase-hook 
  "Creates hook to stop if metric does not increase within given max steps.

  Usage example:

  ```python
  estimator = ...
  # Hook to stop training if accuracy does not increase in over 100000 steps.
  hook = early_stopping.stop_if_no_increase_hook(estimator, \"accuracy\", 100000)
  train_spec = tf.estimator.TrainSpec(..., hooks=[hook])
  tf.estimator.train_and_evaluate(estimator, train_spec, ...)
  ```

  Caveat: Current implementation supports early-stopping both training and
  evaluation in local mode. In distributed mode, training can be stopped but
  evaluation (where it's a separate job) will indefinitely wait for new model
  checkpoints to evaluate, so you will need other means to detect and stop it.
  Early-stopping evaluation in distributed mode requires changes in
  `train_and_evaluate` API and will be addressed in a future revision.

  Args:
    estimator: A `tf.estimator.Estimator` instance.
    metric_name: `str`, metric to track. \"loss\", \"accuracy\", etc.
    max_steps_without_increase: `int`, maximum number of training steps with no
      increase in the given metric.
    eval_dir: If set, directory containing summary files with eval metrics. By
      default, `estimator.eval_dir()` will be used.
    min_steps: `int`, stop is never requested if global step is less than this
      value. Defaults to 0.
    run_every_secs: If specified, calls `should_stop_fn` at an interval of
      `run_every_secs` seconds. Defaults to 60 seconds. Either this or
      `run_every_steps` must be set.
    run_every_steps: If specified, calls `should_stop_fn` every
      `run_every_steps` steps. Either this or `run_every_secs` must be set.

  Returns:
    An early-stopping hook of type `SessionRunHook` that periodically checks
    if the given metric shows no increase over given maximum number of
    training steps, and initiates early stopping if true.
  "
  [estimator metric_name max_steps_without_increase eval_dir & {:keys [min_steps run_every_secs run_every_steps]
                       :or {run_every_steps None}} ]
    (py/call-attr-kw experimental "stop_if_no_increase_hook" [estimator metric_name max_steps_without_increase eval_dir] {:min_steps min_steps :run_every_secs run_every_secs :run_every_steps run_every_steps }))
