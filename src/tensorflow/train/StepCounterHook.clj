(ns tensorflow.train.StepCounterHook
  "Hook that counts steps per second."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce train (import-module "tensorflow.train"))

(defn StepCounterHook 
  "Hook that counts steps per second."
  [ & {:keys [every_n_steps every_n_secs output_dir summary_writer]
       :or {every_n_secs None output_dir None summary_writer None}} ]
  
   (py/call-attr-kw train "StepCounterHook" [] {:every_n_steps every_n_steps :every_n_secs every_n_secs :output_dir output_dir :summary_writer summary_writer }))

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
  "Called at the end of session.

    The `session` argument can be used in case the hook wants to run final ops,
    such as saving a last checkpoint.

    If `session.run()` raises exception other than OutOfRangeError or
    StopIteration then `end()` is not called.
    Note the difference between `end()` and `after_run()` behavior when
    `session.run()` raises OutOfRangeError or StopIteration. In that case
    `end()` is called but `after_run()` is not called.

    Args:
      session: A TensorFlow Session that will be soon closed.
    "
  [ self session ]
  (py/call-attr self "end"  self session ))
