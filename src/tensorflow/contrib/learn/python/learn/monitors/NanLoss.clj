(ns tensorflow.contrib.learn.python.learn.monitors.NanLoss
  "NaN Loss monitor.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Monitors loss and stops training if loss is NaN.
  Can either fail with exception or just stop training.
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
(defn NanLoss 
  "NaN Loss monitor.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Monitors loss and stops training if loss is NaN.
  Can either fail with exception or just stop training.
  "
  [loss_tensor  & {:keys [every_n_steps fail_on_nan_loss]} ]
    (py/call-attr-kw monitors "NanLoss" [loss_tensor] {:every_n_steps every_n_steps :fail_on_nan_loss fail_on_nan_loss }))

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
  ""
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
