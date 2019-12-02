(ns tensorflow.contrib.learn.python.learn.monitors
  "Monitors instrument the training process (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.

@@get_default_monitors
@@BaseMonitor
@@CaptureVariable
@@CheckpointSaver
@@EveryN
@@ExportMonitor
@@GraphDump
@@LoggingTrainable
@@NanLoss
@@PrintTensor
@@StepCounter
@@StopAtStep
@@SummarySaver
@@ValidationMonitor
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

(defn get-default-monitors 
  "Returns a default set of typically-used monitors. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.train.MonitoredTrainingSession.

Args:
  loss_op: `Tensor`, the loss tensor. This will be printed using `PrintTensor`
      at the default interval.
  summary_op: See `SummarySaver`.
  save_summary_steps: See `SummarySaver`.
  output_dir:  See `SummarySaver`.
  summary_writer:  See `SummarySaver`.
Returns:
  `list` of monitors."
  [loss_op summary_op & {:keys [save_summary_steps output_dir summary_writer]
                       :or {output_dir None summary_writer None}} ]
    (py/call-attr-kw monitors "get_default_monitors" [loss_op summary_op] {:save_summary_steps save_summary_steps :output_dir output_dir :summary_writer summary_writer }))

(defn replace-monitors-with-hooks 
  "Wraps monitors with a hook.

  `Monitor` is deprecated in favor of `SessionRunHook`. If you're using a
  monitor, you can wrap it with a hook using function. It is recommended to
  implement hook version of your monitor.

  Args:
    monitors_or_hooks: A `list` may contain both monitors and hooks.
    estimator: An `Estimator` that monitor will be used with.

  Returns:
    Returns a list of hooks. If there is any monitor in the given list, it is
    replaced by a hook.
  "
  [ monitors_or_hooks estimator ]
  (py/call-attr monitors "replace_monitors_with_hooks"  monitors_or_hooks estimator ))
