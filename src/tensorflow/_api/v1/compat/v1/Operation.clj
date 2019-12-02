(ns tensorflow.-api.v1.compat.v1.Operation
  "Represents a graph node that performs computation on tensors.

  An `Operation` is a node in a TensorFlow `Graph` that takes zero or
  more `Tensor` objects as input, and produces zero or more `Tensor`
  objects as output. Objects of type `Operation` are created by
  calling a Python op constructor (such as
  `tf.matmul`)
  or `tf.Graph.create_op`.

  For example `c = tf.matmul(a, b)` creates an `Operation` of type
  \"MatMul\" that takes tensors `a` and `b` as input, and produces `c`
  as output.

  After the graph has been launched in a session, an `Operation` can
  be executed by passing it to
  `tf.Session.run`.
  `op.run()` is a shortcut for calling
  `tf.compat.v1.get_default_session().run(op)`.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v1 (import-module "tensorflow._api.v1.compat.v1"))

(defn Operation 
  "Represents a graph node that performs computation on tensors.

  An `Operation` is a node in a TensorFlow `Graph` that takes zero or
  more `Tensor` objects as input, and produces zero or more `Tensor`
  objects as output. Objects of type `Operation` are created by
  calling a Python op constructor (such as
  `tf.matmul`)
  or `tf.Graph.create_op`.

  For example `c = tf.matmul(a, b)` creates an `Operation` of type
  \"MatMul\" that takes tensors `a` and `b` as input, and produces `c`
  as output.

  After the graph has been launched in a session, an `Operation` can
  be executed by passing it to
  `tf.Session.run`.
  `op.run()` is a shortcut for calling
  `tf.compat.v1.get_default_session().run(op)`.
  "
  [ node_def g inputs output_types control_inputs input_types original_op op_def ]
  (py/call-attr v1 "Operation"  node_def g inputs output_types control_inputs input_types original_op op_def ))

(defn colocation-groups 
  "Returns the list of colocation groups of the op."
  [ self  ]
  (py/call-attr self "colocation_groups"  self  ))

(defn control-inputs 
  "The `Operation` objects on which this op has a control dependency.

    Before this op is executed, TensorFlow will ensure that the
    operations in `self.control_inputs` have finished executing. This
    mechanism can be used to run ops sequentially for performance
    reasons, or to ensure that the side effects of an op are observed
    in the correct order.

    Returns:
      A list of `Operation` objects.

    "
  [ self ]
    (py/call-attr self "control_inputs"))

(defn device 
  "The name of the device to which this op has been assigned, if any.

    Returns:
      The string name of the device to which this op has been
      assigned, or an empty string if it has not been assigned to a
      device.
    "
  [ self ]
    (py/call-attr self "device"))

(defn get-attr 
  "Returns the value of the attr of this op with the given `name`.

    Args:
      name: The name of the attr to fetch.

    Returns:
      The value of the attr, as a Python object.

    Raises:
      ValueError: If this op does not have an attr with the given `name`.
    "
  [ self name ]
  (py/call-attr self "get_attr"  self name ))

(defn graph 
  "The `Graph` that contains this operation."
  [ self ]
    (py/call-attr self "graph"))

(defn inputs 
  "The list of `Tensor` objects representing the data inputs of this op."
  [ self ]
    (py/call-attr self "inputs"))

(defn name 
  "The full name of this operation."
  [ self ]
    (py/call-attr self "name"))

(defn node-def 
  "Returns the `NodeDef` representation of this operation.

    Returns:
      A
      [`NodeDef`](https://www.tensorflow.org/code/tensorflow/core/framework/node_def.proto)
      protocol buffer.
    "
  [ self ]
    (py/call-attr self "node_def"))

(defn op-def 
  "Returns the `OpDef` proto that represents the type of this op.

    Returns:
      An
      [`OpDef`](https://www.tensorflow.org/code/tensorflow/core/framework/op_def.proto)
      protocol buffer.
    "
  [ self ]
    (py/call-attr self "op_def"))

(defn outputs 
  "The list of `Tensor` objects representing the outputs of this op."
  [ self ]
    (py/call-attr self "outputs"))

(defn run 
  "Runs this operation in a `Session`.

    Calling this method will execute all preceding operations that
    produce the inputs needed for this operation.

    *N.B.* Before invoking `Operation.run()`, its graph must have been
    launched in a session, and either a default session must be
    available, or `session` must be specified explicitly.

    Args:
      feed_dict: A dictionary that maps `Tensor` objects to feed values. See
        `tf.Session.run` for a description of the valid feed values.
      session: (Optional.) The `Session` to be used to run to this operation. If
        none, the default session will be used.
    "
  [ self feed_dict session ]
  (py/call-attr self "run"  self feed_dict session ))

(defn traceback 
  "Returns the call stack from when this operation was constructed."
  [ self ]
    (py/call-attr self "traceback"))

(defn traceback-with-start-lines 
  "Same as traceback but includes start line of function definition.

    Returns:
      A list of 5-tuples (filename, lineno, name, code, func_start_lineno).
    "
  [ self ]
    (py/call-attr self "traceback_with_start_lines"))

(defn type 
  "The type of the op (e.g. `\"MatMul\"`)."
  [ self ]
    (py/call-attr self "type"))

(defn values 
  "DEPRECATED: Use outputs."
  [ self  ]
  (py/call-attr self "values"  self  ))
