(ns tensorflow.-api.v1.compat.v1.errors.AbortedError
  "The operation was aborted, typically due to a concurrent action.

  For example, running a
  `tf.QueueBase.enqueue`
  operation may raise `AbortedError` if a
  `tf.QueueBase.close` operation
  previously ran.

  @@__init__
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce errors (import-module "tensorflow._api.v1.compat.v1.errors"))

(defn AbortedError 
  "The operation was aborted, typically due to a concurrent action.

  For example, running a
  `tf.QueueBase.enqueue`
  operation may raise `AbortedError` if a
  `tf.QueueBase.close` operation
  previously ran.

  @@__init__
  "
  [ node_def op message ]
  (py/call-attr errors "AbortedError"  node_def op message ))

(defn error-code 
  "The integer error code that describes the error."
  [ self ]
    (py/call-attr self "error_code"))

(defn message 
  "The error message that describes the error."
  [ self ]
    (py/call-attr self "message"))

(defn node-def 
  "The `NodeDef` proto representing the op that failed."
  [ self ]
    (py/call-attr self "node_def"))

(defn op 
  "The operation that failed, if known.

    *N.B.* If the failed op was synthesized at runtime, e.g. a `Send`
    or `Recv` op, there will be no corresponding
    `tf.Operation`
    object.  In that case, this will return `None`, and you should
    instead use the `tf.errors.OpError.node_def` to
    discover information about the op.

    Returns:
      The `Operation` that failed, or None.
    "
  [ self ]
    (py/call-attr self "op"))
