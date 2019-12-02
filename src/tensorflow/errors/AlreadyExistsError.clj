(ns tensorflow.errors.AlreadyExistsError
  "Raised when an entity that we attempted to create already exists.

  For example, running an operation that saves a file
  (e.g. `tf.train.Saver.save`)
  could potentially raise this exception if an explicit filename for an
  existing file was passed.

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
(defonce errors (import-module "tensorflow.errors"))

(defn AlreadyExistsError 
  "Raised when an entity that we attempted to create already exists.

  For example, running an operation that saves a file
  (e.g. `tf.train.Saver.save`)
  could potentially raise this exception if an explicit filename for an
  existing file was passed.

  @@__init__
  "
  [ node_def op message ]
  (py/call-attr errors "AlreadyExistsError"  node_def op message ))

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
