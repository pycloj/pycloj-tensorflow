(ns tensorflow.train.SessionRunValues
  "Contains the results of `Session.run()`.

  In the future we may use this object to add more information about result of
  run without changing the Hook API.

  Args:
    results: The return values from `Session.run()` corresponding to the fetches
      attribute returned in the RunArgs. Note that this has the same shape as
      the RunArgs fetches.  For example:
        fetches = global_step_tensor
        => results = nparray(int)
        fetches = [train_op, summary_op, global_step_tensor]
        => results = [None, nparray(string), nparray(int)]
        fetches = {'step': global_step_tensor, 'summ': summary_op}
        => results = {'step': nparray(int), 'summ': nparray(string)}
    options: `RunOptions` from the `Session.run()` call.
    run_metadata: `RunMetadata` from the `Session.run()` call.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce train (import-module "tensorflow.train"))

(defn SessionRunValues 
  "Contains the results of `Session.run()`.

  In the future we may use this object to add more information about result of
  run without changing the Hook API.

  Args:
    results: The return values from `Session.run()` corresponding to the fetches
      attribute returned in the RunArgs. Note that this has the same shape as
      the RunArgs fetches.  For example:
        fetches = global_step_tensor
        => results = nparray(int)
        fetches = [train_op, summary_op, global_step_tensor]
        => results = [None, nparray(string), nparray(int)]
        fetches = {'step': global_step_tensor, 'summ': summary_op}
        => results = {'step': nparray(int), 'summ': nparray(string)}
    options: `RunOptions` from the `Session.run()` call.
    run_metadata: `RunMetadata` from the `Session.run()` call.
  "
  [ results options run_metadata ]
  (py/call-attr train "SessionRunValues"  results options run_metadata ))

(defn options 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "options"))

(defn results 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "results"))

(defn run-metadata 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "run_metadata"))