(ns tensorflow.-api.v1.compat.v1.errors.DataLossError
  "Raised when unrecoverable data loss or corruption is encountered.

  For example, this may be raised by running a
  `tf.WholeFileReader.read`
  operation, if the file is truncated while it is being read.

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

(defn DataLossError 
  "Raised when unrecoverable data loss or corruption is encountered.

  For example, this may be raised by running a
  `tf.WholeFileReader.read`
  operation, if the file is truncated while it is being read.

  @@__init__
  "
  [ node_def op message ]
  (py/call-attr errors "DataLossError"  node_def op message ))

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
