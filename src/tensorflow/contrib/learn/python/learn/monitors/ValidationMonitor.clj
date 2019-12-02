(ns tensorflow.contrib.learn.python.learn.monitors.ValidationMonitor
  "Runs evaluation of a given estimator, at most every N steps.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Note that the evaluation is done based on the saved checkpoint, which will
  usually be older than the current step.

  Can do early stopping on validation metrics if `early_stopping_rounds` is
  provided.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce monitors (import-module "tensorflow.contrib.learn.python.learn.monitors"))

(defn ValidationMonitor 
  "Runs evaluation of a given estimator, at most every N steps.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Note that the evaluation is done based on the saved checkpoint, which will
  usually be older than the current step.

  Can do early stopping on validation metrics if `early_stopping_rounds` is
  provided.
  "
  [x y input_fn batch_size eval_steps & {:keys [every_n_steps metrics hooks early_stopping_rounds early_stopping_metric early_stopping_metric_minimize name check_interval_secs]
                       :or {metrics None hooks None early_stopping_rounds None name None}} ]
    (py/call-attr-kw monitors "ValidationMonitor" [x y input_fn batch_size eval_steps] {:every_n_steps every_n_steps :metrics metrics :hooks hooks :early_stopping_rounds early_stopping_rounds :early_stopping_metric early_stopping_metric :early_stopping_metric_minimize early_stopping_metric_minimize :name name :check_interval_secs check_interval_secs }))

(defn begin 
  "Called at the beginning of training.

    When called, the default graph is the one we are executing.

    Args:
      max_steps: `int`, the maximum global step this training will run until.

    Raises:
      ValueError: if we've already begun a run.
    "
  [ self max_steps ]
  (py/call-attr self "begin"  self max_steps ))

(defn best-metrics 
  "Returns all eval metrics computed with the best early stopping metric.

    For instance, if the metrics computed in two successive evals are
    1. {'loss':40, 'auc':0.5}
    2. {'loss':50, 'auc':0.6}
    this function would return the first dict {'loss':40, 'auc':0.5} after both
    first and second eval (if `early_stopping_metric` is 'loss' and
    `early_stopping_metric_minimize` is True).

    Returns:
      The output dict of estimator.evaluate which contains the best value of
      the early stopping metric seen so far.
    "
  [ self ]
    (py/call-attr self "best_metrics"))

(defn best-step 
  "Returns the step at which the best early stopping metric was found."
  [ self ]
    (py/call-attr self "best_step"))

(defn best-value 
  "Returns the best early stopping metric value found so far."
  [ self ]
    (py/call-attr self "best_value"))

(defn early-stopped 
  "Returns True if this monitor caused an early stop."
  [ self ]
    (py/call-attr self "early_stopped"))

(defn end 
  ""
  [ self session ]
  (py/call-attr self "end"  self session ))

(defn epoch-begin 
  "Begin epoch.

    Args:
      epoch: `int`, the epoch number.

    Raises:
      ValueError: if we've already begun an epoch, or `epoch` < 0.
    "
  [ self epoch ]
  (py/call-attr self "epoch_begin"  self epoch ))

(defn epoch-end 
  "End epoch.

    Args:
      epoch: `int`, the epoch number.

    Raises:
      ValueError: if we've not begun an epoch, or `epoch` number does not match.
    "
  [ self epoch ]
  (py/call-attr self "epoch_end"  self epoch ))

(defn every-n-post-step 
  "Callback after a step is finished or `end()` is called.

    Args:
      step: `int`, the current value of the global step.
      session: `Session` object.
    "
  [ self step session ]
  (py/call-attr self "every_n_post_step"  self step session ))

(defn every-n-step-begin 
  "Callback before every n'th step begins.

    Args:
      step: `int`, the current value of the global step.

    Returns:
      A `list` of tensors that will be evaluated at this step.
    "
  [ self step ]
  (py/call-attr self "every_n_step_begin"  self step ))

(defn every-n-step-end 
  ""
  [ self step outputs ]
  (py/call-attr self "every_n_step_end"  self step outputs ))

(defn post-step 
  ""
  [ self step session ]
  (py/call-attr self "post_step"  self step session ))

(defn run-on-all-workers 
  ""
  [ self ]
    (py/call-attr self "run_on_all_workers"))

(defn set-estimator 
  "A setter called automatically by the target estimator.

    If the estimator is locked, this method does nothing.

    Args:
      estimator: the estimator that this monitor monitors.

    Raises:
      ValueError: if the estimator is None.
    "
  [ self estimator ]
  (py/call-attr self "set_estimator"  self estimator ))

(defn step-begin 
  "Overrides `BaseMonitor.step_begin`.

    When overriding this method, you must call the super implementation.

    Args:
      step: `int`, the current value of the global step.
    Returns:
      A `list`, the result of every_n_step_begin, if that was called this step,
      or an empty list otherwise.

    Raises:
      ValueError: if called more than once during a step.
    "
  [ self step ]
  (py/call-attr self "step_begin"  self step ))

(defn step-end 
  "Overrides `BaseMonitor.step_end`.

    When overriding this method, you must call the super implementation.

    Args:
      step: `int`, the current value of the global step.
      output: `dict` mapping `string` values representing tensor names to
        the value resulted from running these tensors. Values may be either
        scalars, for scalar tensors, or Numpy `array`, for non-scalar tensors.
    Returns:
      `bool`, the result of every_n_step_end, if that was called this step,
      or `False` otherwise.
    "
  [ self step output ]
  (py/call-attr self "step_end"  self step output ))
