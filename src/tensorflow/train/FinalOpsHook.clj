(ns tensorflow.train.FinalOpsHook
  "A hook which evaluates `Tensors` at the end of a session."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce train (import-module "tensorflow.train"))

(defn FinalOpsHook 
  "A hook which evaluates `Tensors` at the end of a session."
  [ final_ops final_ops_feed_dict ]
  (py/call-attr train "FinalOpsHook"  final_ops final_ops_feed_dict ))

(defn after-create-session 
  "Called when new TensorFlow session is created.

    This is called to signal the hooks that a new session has been created. This
    has two essential differences with the situation in which `begin` is called:

    * When this is called, the graph is finalized and ops can no longer be added
        to the graph.
    * This method will also be called as a result of recovering a wrapped
        session, not only at the beginning of the overall session.

    Args:
      session: A TensorFlow Session that has been created.
      coord: A Coordinator object which keeps track of all threads.
    "
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
  "Called once before using the session.

    When called, the default graph is the one that will be launched in the
    session.  The hook can modify the graph by adding new operations to it.
    After the `begin()` call the graph will be finalized and the other callbacks
    can not modify the graph anymore. Second call of `begin()` on the same
    graph, should not change the graph.
    "
  [ self  ]
  (py/call-attr self "begin"  self  ))

(defn end 
  ""
  [ self session ]
  (py/call-attr self "end"  self session ))

(defn final-ops-values 
  ""
  [ self ]
    (py/call-attr self "final_ops_values"))
