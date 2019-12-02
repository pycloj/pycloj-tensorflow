(ns tensorflow.-api.v1.compat.v1.errors.UnknownError
  "Unknown error.

  An example of where this error may be returned is if a Status value
  received from another address space belongs to an error-space that
  is not known to this address space. Also errors raised by APIs that
  do not return enough error information may be converted to this
  error.

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
(defn UnknownError 
  "Unknown error.

  An example of where this error may be returned is if a Status value
  received from another address space belongs to an error-space that
  is not known to this address space. Also errors raised by APIs that
  do not return enough error information may be converted to this
  error.

  @@__init__
  "
  [node_def op message  & {:keys [error_code]} ]
    (py/call-attr-kw errors "UnknownError" [node_def op message] {:error_code error_code }))

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
