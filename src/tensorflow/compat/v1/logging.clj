(ns tensorflow.-api.v1.compat.v1.logging
  "Logging and Summary Operations.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce logging (import-module "tensorflow._api.v1.compat.v1.logging"))

(defn TaskLevelStatusMessage 
  ""
  [ msg ]
  (py/call-attr logging "TaskLevelStatusMessage"  msg ))

(defn debug 
  ""
  [ msg ]
  (py/call-attr logging "debug"  msg ))

(defn error 
  ""
  [ msg ]
  (py/call-attr logging "error"  msg ))

(defn fatal 
  ""
  [ msg ]
  (py/call-attr logging "fatal"  msg ))

(defn flush 
  ""
  [  ]
  (py/call-attr logging "flush"  ))

(defn get-verbosity 
  "Return how much logging output will be produced."
  [  ]
  (py/call-attr logging "get_verbosity"  ))

(defn info 
  ""
  [ msg ]
  (py/call-attr logging "info"  msg ))

(defn log 
  ""
  [ level msg ]
  (py/call-attr logging "log"  level msg ))

(defn log-every-n 
  "Log 'msg % args' at level 'level' once per 'n' times.

  Logs the 1st call, (N+1)st call, (2N+1)st call,  etc.
  Not threadsafe.

  Args:
    level: The level at which to log.
    msg: The message to be logged.
    n: The number of times this should be called before it is logged.
    *args: The args to be substituted into the msg.
  "
  [ level msg n ]
  (py/call-attr logging "log_every_n"  level msg n ))

(defn log-first-n 
  "Log 'msg % args' at level 'level' only first 'n' times.

  Not threadsafe.

  Args:
    level: The level at which to log.
    msg: The message to be logged.
    n: The number of times this should be called before it is logged.
    *args: The args to be substituted into the msg.
  "
  [ level msg n ]
  (py/call-attr logging "log_first_n"  level msg n ))

(defn log-if 
  "Log 'msg % args' at level 'level' only if condition is fulfilled."
  [ level msg condition ]
  (py/call-attr logging "log_if"  level msg condition ))

(defn set-verbosity 
  "Sets the threshold for what messages will be logged."
  [ v ]
  (py/call-attr logging "set_verbosity"  v ))

(defn vlog 
  ""
  [ level msg ]
  (py/call-attr logging "vlog"  level msg ))

(defn warn 
  ""
  [ msg ]
  (py/call-attr logging "warn"  msg ))

(defn warning 
  ""
  [ msg ]
  (py/call-attr logging "warning"  msg ))
