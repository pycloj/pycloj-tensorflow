(ns tensorflow.contrib.training.SummaryAtEndHook
  "A run hook that saves a summary with the results of evaluation."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce training (import-module "tensorflow.contrib.training"))

(defn SummaryAtEndHook 
  "A run hook that saves a summary with the results of evaluation."
  [ log_dir summary_writer summary_op feed_dict ]
  (py/call-attr training "SummaryAtEndHook"  log_dir summary_writer summary_op feed_dict ))

(defn after-create-session 
  ""
  [ self session coord ]
  (py/call-attr self "after_create_session"  self session coord ))

(defn after-run 
  "Called after each call to run().

    The `run_values` argument contains results of requested ops/tensors by
    `before_run()`.

    The `run_context` argument is the same one send to `before_run` call.
    `run_context.request_stop()` can be called to stop the iteration.

    If `session.run()` raises any exceptions then `after_run()` is not called.

    Args:
      run_context: A `SessionRunContext` object.
      run_values: A SessionRunValues object.
    "
  [ self run_context run_values ]
  (py/call-attr self "after_run"  self run_context run_values ))

(defn before-run 
  "Called before each call to run().

    You can return from this call a `SessionRunArgs` object indicating ops or
    tensors to add to the upcoming `run()` call.  These ops/tensors will be run
    together with the ops/tensors originally passed to the original run() call.
    The run args you return can also contain feeds to be added to the run()
    call.

    The `run_context` argument is a `SessionRunContext` that provides
    information about the upcoming `run()` call: the originally requested
    op/tensors, the TensorFlow Session.

    At this point graph is finalized and you can not add ops.

    Args:
      run_context: A `SessionRunContext` object.

    Returns:
      None or a `SessionRunArgs` object.
    "
  [ self run_context ]
  (py/call-attr self "before_run"  self run_context ))

(defn begin 
  ""
  [ self  ]
  (py/call-attr self "begin"  self  ))

(defn end 
  ""
  [ self session ]
  (py/call-attr self "end"  self session ))
