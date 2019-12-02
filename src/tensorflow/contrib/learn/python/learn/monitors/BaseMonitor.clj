(ns tensorflow.contrib.learn.python.learn.monitors.BaseMonitor
  "Base class for Monitors.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Defines basic interfaces of Monitors.
  Monitors can either be run on all workers or, more commonly, restricted
  to run exclusively on the elected chief worker.
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

(defn BaseMonitor 
  "Base class for Monitors.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Defines basic interfaces of Monitors.
  Monitors can either be run on all workers or, more commonly, restricted
  to run exclusively on the elected chief worker.
  "
  [  ]
  (py/call-attr monitors "BaseMonitor"  ))

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
  "Callback at the end of training/evaluation.

    Args:
      session: A `tf.compat.v1.Session` object that can be used to run ops.

    Raises:
      ValueError: if we've not begun a run.
    "
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

(defn post-step 
  "Callback after the step is finished.

    Called after step_end and receives session to perform extra session.run
    calls. If failure occurred in the process, will be called as well.

    Args:
      step: `int`, global step of the model.
      session: `Session` object.
    "
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
  "Callback before training step begins.

    You may use this callback to request evaluation of additional tensors
    in the graph.

    Args:
      step: `int`, the current value of the global step.

    Returns:
      List of `Tensor` objects or string tensor names to be run.

    Raises:
      ValueError: if we've already begun a step, or `step` < 0, or
          `step` > `max_steps`.
    "
  [ self step ]
  (py/call-attr self "step_begin"  self step ))

(defn step-end 
  "Callback after training step finished.

    This callback provides access to the tensors/ops evaluated at this step,
    including the additional tensors for which evaluation was requested in
    `step_begin`.

    In addition, the callback has the opportunity to stop training by returning
    `True`. This is useful for early stopping, for example.

    Note that this method is not called if the call to `Session.run()` that
    followed the last call to `step_begin()` failed.

    Args:
      step: `int`, the current value of the global step.
      output: `dict` mapping `string` values representing tensor names to
        the value resulted from running these tensors. Values may be either
        scalars, for scalar tensors, or Numpy `array`, for non-scalar tensors.

    Returns:
      `bool`. True if training should stop.

    Raises:
      ValueError: if we've not begun a step, or `step` number does not match.
    "
  [ self step output ]
  (py/call-attr self "step_end"  self step output ))
