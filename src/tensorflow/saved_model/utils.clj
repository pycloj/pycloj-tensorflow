(ns tensorflow.-api.v1.saved-model.utils
  "SavedModel utility functions.

Utility functions to assist with setup and construction of the SavedModel proto.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce utils (import-module "tensorflow._api.v1.saved_model.utils"))

(defn build-tensor-info 
  "Utility function to build TensorInfo proto from a Tensor. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
This function will only be available through the v1 compatibility library as tf.compat.v1.saved_model.utils.build_tensor_info or tf.compat.v1.saved_model.build_tensor_info.

Args:
  tensor: Tensor or SparseTensor whose name, dtype and shape are used to
      build the TensorInfo. For SparseTensors, the names of the three
      constituent Tensors are used.

Returns:
  A TensorInfo protocol buffer constructed based on the supplied argument.

Raises:
  RuntimeError: If eager execution is enabled."
  [ tensor ]
  (py/call-attr utils "build_tensor_info"  tensor ))

(defn get-tensor-from-tensor-info 
  "Returns the Tensor or CompositeTensor described by a TensorInfo proto. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
This function will only be available through the v1 compatibility library as tf.compat.v1.saved_model.utils.get_tensor_from_tensor_info or tf.compat.v1.saved_model.get_tensor_from_tensor_info.

Args:
  tensor_info: A TensorInfo proto describing a Tensor or SparseTensor or
    CompositeTensor.
  graph: The tf.Graph in which tensors are looked up. If None, the
      current default graph is used.
  import_scope: If not None, names in `tensor_info` are prefixed with this
      string before lookup.

Returns:
  The Tensor or SparseTensor or CompositeTensor in `graph` described by
  `tensor_info`.

Raises:
  KeyError: If `tensor_info` does not correspond to a tensor in `graph`.
  ValueError: If `tensor_info` is malformed."
  [ tensor_info graph import_scope ]
  (py/call-attr utils "get_tensor_from_tensor_info"  tensor_info graph import_scope ))
