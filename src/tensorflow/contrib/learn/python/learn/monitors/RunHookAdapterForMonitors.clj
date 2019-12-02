(ns tensorflow.contrib.learn.python.learn.monitors.RunHookAdapterForMonitors
  "Wraps monitors into a SessionRunHook."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce monitors (import-module "tensorflow.contrib.learn.python.learn.monitors"))

(defn RunHookAdapterForMonitors 
  "Wraps monitors into a SessionRunHook."
  [ monitors ]
  (py/call-attr monitors "RunHookAdapterForMonitors"  monitors ))

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
  ""
  [ self run_context run_values ]
  (py/call-attr self "after_run"  self run_context run_values ))

(defn before-run 
  ""
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
