(ns tensorflow.-api.v1.compat.v2.errors.OutOfRangeError
  "Raised when an operation iterates past the valid input range.

  This exception is raised in \"end-of-file\" conditions, such as when a
  `tf.QueueBase.dequeue`
  operation is blocked on an empty queue, and a
  `tf.QueueBase.close`
  operation executes.

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
(defonce errors (import-module "tensorflow._api.v1.compat.v2.errors"))

(defn OutOfRangeError 
  "Raised when an operation iterates past the valid input range.

  This exception is raised in \"end-of-file\" conditions, such as when a
  `tf.QueueBase.dequeue`
  operation is blocked on an empty queue, and a
  `tf.QueueBase.close`
  operation executes.

  @@__init__
  "
  [ node_def op message ]
  (py/call-attr errors "OutOfRangeError"  node_def op message ))

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
