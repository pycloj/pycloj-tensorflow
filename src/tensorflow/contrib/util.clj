(ns tensorflow.contrib.util
  "Utilities for dealing with Tensors.

@@constant_value
@@make_tensor_proto
@@make_ndarray
@@ops_used_by_graph_def
@@stripped_op_list_for_graph

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce util (import-module "tensorflow.contrib.util"))
(defn constant-value 
  "Returns the constant value of the given tensor, if efficiently calculable.

  This function attempts to partially evaluate the given tensor, and
  returns its value as a numpy ndarray if this succeeds.

  Compatibility(V1): If `constant_value(tensor)` returns a non-`None` result, it
  will no longer be possible to feed a different value for `tensor`. This allows
  the result of this function to influence the graph that is constructed, and
  permits static shape optimizations.

  Args:
    tensor: The Tensor to be evaluated.
    partial: If True, the returned numpy array is allowed to have partially
      evaluated values. Values that can't be evaluated will be None.

  Returns:
    A numpy ndarray containing the constant value of the given `tensor`,
    or None if it cannot be calculated.

  Raises:
    TypeError: if tensor is not an ops.Tensor.
  "
  [tensor  & {:keys [partial]} ]
    (py/call-attr-kw util "constant_value" [tensor] {:partial partial }))

(defn make-ndarray 
  "Create a numpy ndarray from a tensor.

  Create a numpy ndarray with the same shape and data as the tensor.

  Args:
    tensor: A TensorProto.

  Returns:
    A numpy array with the tensor contents.

  Raises:
    TypeError: if tensor has unsupported type.

  "
  [ tensor ]
  (py/call-attr util "make_ndarray"  tensor ))
(defn make-tensor-proto 
  "Create a TensorProto.

  In TensorFlow 2.0, representing tensors as protos should no longer be a
  common workflow. That said, this utility function is still useful for
  generating TF Serving request protos:

    request = tensorflow_serving.apis.predict_pb2.PredictRequest()
    request.model_spec.name = \"my_model\"
    request.model_spec.signature_name = \"serving_default\"
    request.inputs[\"images\"].CopyFrom(tf.make_tensor_proto(X_new))

  make_tensor_proto accepts \"values\" of a python scalar, a python list, a
  numpy ndarray, or a numpy scalar.

  If \"values\" is a python scalar or a python list, make_tensor_proto
  first convert it to numpy ndarray. If dtype is None, the
  conversion tries its best to infer the right numpy data
  type. Otherwise, the resulting numpy array has a compatible data
  type with the given dtype.

  In either case above, the numpy ndarray (either the caller provided
  or the auto converted) must have the compatible type with dtype.

  make_tensor_proto then converts the numpy array to a tensor proto.

  If \"shape\" is None, the resulting tensor proto represents the numpy
  array precisely.

  Otherwise, \"shape\" specifies the tensor's shape and the numpy array
  can not have more elements than what \"shape\" specifies.

  Args:
    values:         Values to put in the TensorProto.
    dtype:          Optional tensor_pb2 DataType value.
    shape:          List of integers representing the dimensions of tensor.
    verify_shape:   Boolean that enables verification of a shape of values.
    allow_broadcast:  Boolean that enables allowing scalars and 1 length vector
        broadcasting. Cannot be true when verify_shape is true.

  Returns:
    A `TensorProto`. Depending on the type, it may contain data in the
    \"tensor_content\" attribute, which is not directly useful to Python programs.
    To access the values you should convert the proto back to a numpy ndarray
    with `tf.make_ndarray(proto)`.

    If `values` is a `TensorProto`, it is immediately returned; `dtype` and
    `shape` are ignored.

  Raises:
    TypeError:  if unsupported types are provided.
    ValueError: if arguments have inappropriate values or if verify_shape is
     True and shape of values is not equals to a shape from the argument.

  "
  [values dtype shape  & {:keys [verify_shape allow_broadcast]} ]
    (py/call-attr-kw util "make_tensor_proto" [values dtype shape] {:verify_shape verify_shape :allow_broadcast allow_broadcast }))

(defn ops-used-by-graph-def 
  "Collect the list of ops used by a graph.

  Does not validate that the ops are all registered.

  Args:
    graph_def: A `GraphDef` proto, as from `graph.as_graph_def()`.

  Returns:
    A list of strings, each naming an op used by the graph.
  "
  [ graph_def ]
  (py/call-attr util "ops_used_by_graph_def"  graph_def ))

(defn stripped-op-list-for-graph 
  "Collect the stripped OpDefs for ops used by a graph.

  This function computes the `stripped_op_list` field of `MetaGraphDef` and
  similar protos.  The result can be communicated from the producer to the
  consumer, which can then use the C++ function
  `RemoveNewDefaultAttrsFromGraphDef` to improve forwards compatibility.

  Args:
    graph_def: A `GraphDef` proto, as from `graph.as_graph_def()`.

  Returns:
    An `OpList` of ops used by the graph.

  Raises:
    ValueError: If an unregistered op is used.
  "
  [ graph_def ]
  (py/call-attr util "stripped_op_list_for_graph"  graph_def ))
