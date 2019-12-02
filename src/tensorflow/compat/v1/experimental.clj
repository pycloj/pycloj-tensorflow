(ns tensorflow.-api.v1.compat.v1.experimental
  "Public API for tf.experimental namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.compat.v1.experimental"))

(defn function-executor-type 
  "Context manager for setting the executor of eager defined functions.

  Eager defined functions are functions decorated by tf.contrib.eager.defun.

  Args:
    executor_type: a string for the name of the executor to be used to execute
      functions defined by tf.contrib.eager.defun.

  Yields:
    Context manager for setting the executor of eager defined functions.
  "
  [ executor_type ]
  (py/call-attr experimental "function_executor_type"  executor_type ))

(defn output-all-intermediates 
  "Whether to output all intermediates from functional control flow ops.

  The \"default\" behavior to is to output all intermediates when using v2 control
  flow inside Keras models in graph mode (possibly inside Estimators). This is
  needed to support taking gradients of v2 control flow. In graph mode, Keras
  can sometimes freeze the forward graph before the gradient computation which
  does not work for v2 control flow since it requires updating the forward ops
  to output the needed intermediates. We work around this by proactively
  outputting the needed intermediates when building the forward pass itself.
  Ideally any such extra tensors should be pruned out at runtime. However, if
  for any reason this doesn't work for you or if you have an infernce-only model
  you can turn this behavior off using
  `tf.compat.v1.experimental.output_all_intermediates(False)`.

  If with the default behavior you are still seeing errors of the form
  \"Connecting to invalid output X of source node Y which has Z outputs\" try
  setting `tf.compat.v1.experimental.output_all_intermediates(True)` and
  please file an issue at https://github.com/tensorflow/tensorflow/issues.

  Args:
    state: True, False or None. None restores the default behavior.
  "
  [ state ]
  (py/call-attr experimental "output_all_intermediates"  state ))
