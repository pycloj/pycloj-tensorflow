(ns tensorflow.train.SessionRunContext
  "Provides information about the `session.run()` call being made.

  Provides information about original request to `Session.Run()` function.
  SessionRunHook objects can stop the loop by calling `request_stop()` of
  `run_context`. In the future we may use this object to add more information
  about run without changing the Hook API.
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

(defn SessionRunContext 
  "Provides information about the `session.run()` call being made.

  Provides information about original request to `Session.Run()` function.
  SessionRunHook objects can stop the loop by calling `request_stop()` of
  `run_context`. In the future we may use this object to add more information
  about run without changing the Hook API.
  "
  [ original_args session ]
  (py/call-attr train "SessionRunContext"  original_args session ))

(defn original-args 
  "A `SessionRunArgs` object holding the original arguments of `run()`.

    If user called `MonitoredSession.run(fetches=a, feed_dict=b)`, then this
    field is equal to SessionRunArgs(a, b).

    Returns:
     A `SessionRunArgs` object
    "
  [ self ]
    (py/call-attr self "original_args"))

(defn request-stop 
  "Sets stop requested field.

    Hooks can use this function to request stop of iterations.
    `MonitoredSession` checks whether this is called or not.
    "
  [ self  ]
  (py/call-attr self "request_stop"  self  ))

(defn session 
  "A TensorFlow session object which will execute the `run`."
  [ self ]
    (py/call-attr self "session"))

(defn stop-requested 
  "Returns whether a stop is requested or not.

    If true, `MonitoredSession` stops iterations.
    Returns:
      A `bool`
    "
  [ self ]
    (py/call-attr self "stop_requested"))
