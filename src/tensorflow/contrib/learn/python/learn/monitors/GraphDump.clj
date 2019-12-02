(ns tensorflow.contrib.learn.python.learn.monitors.GraphDump
  "Dumps almost all tensors in the graph at every step.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Note, this is very expensive, prefer `PrintTensor` in production.
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

(defn GraphDump 
  "Dumps almost all tensors in the graph at every step.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Note, this is very expensive, prefer `PrintTensor` in production.
  "
  [ ignore_ops ]
  (py/call-attr monitors "GraphDump"  ignore_ops ))

(defn begin 
  ""
  [ self max_steps ]
  (py/call-attr self "begin"  self max_steps ))
(defn compare 
  "Compares two `GraphDump` monitors and returns differences.

    Args:
      other_dump: Another `GraphDump` monitor.
      step: `int`, step to compare on.
      atol: `float`, absolute tolerance in comparison of floating arrays.

    Returns:
      Returns tuple:
        matched: `list` of keys that matched.
        non_matched: `dict` of keys to tuple of 2 mismatched values.

    Raises:
      ValueError: if a key in `data` is missing from `other_dump` at `step`.
    "
  [self other_dump step  & {:keys [atol]} ]
    (py/call-attr-kw self "compare" [other_dump step] {:atol atol }))

(defn data 
  ""
  [ self ]
    (py/call-attr self "data"))

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
  ""
  [ self step ]
  (py/call-attr self "step_begin"  self step ))

(defn step-end 
  ""
  [ self step output ]
  (py/call-attr self "step_end"  self step output ))
