(ns tensorflow-estimator.python.estimator.api.-v1.estimator.export.ServingInputReceiver
  "A return type for a serving_input_receiver_fn.

  The expected return values are:
    features: A `Tensor`, `SparseTensor`, or dict of string or int to `Tensor`
      or `SparseTensor`, specifying the features to be passed to the model.
      Note: if `features` passed is not a dict, it will be wrapped in a dict
      with a single entry, using 'feature' as the key.  Consequently, the model
      must accept a feature dict of the form {'feature': tensor}.  You may use
      `TensorServingInputReceiver` if you want the tensor to be passed as is.
    receiver_tensors: A `Tensor`, `SparseTensor`, or dict of string to `Tensor`
      or `SparseTensor`, specifying input nodes where this receiver expects to
      be fed by default.  Typically, this is a single placeholder expecting
      serialized `tf.Example` protos.
    receiver_tensors_alternatives: a dict of string to additional
      groups of receiver tensors, each of which may be a `Tensor`,
      `SparseTensor`, or dict of string to `Tensor` or`SparseTensor`.
      These named receiver tensor alternatives generate additional serving
      signatures, which may be used to feed inputs at different points within
      the input receiver subgraph.  A typical usage is to allow feeding raw
      feature `Tensor`s *downstream* of the tf.parse_example() op.
      Defaults to None.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce export (import-module "tensorflow_estimator.python.estimator.api._v1.estimator.export"))

(defn ServingInputReceiver 
  "A return type for a serving_input_receiver_fn.

  The expected return values are:
    features: A `Tensor`, `SparseTensor`, or dict of string or int to `Tensor`
      or `SparseTensor`, specifying the features to be passed to the model.
      Note: if `features` passed is not a dict, it will be wrapped in a dict
      with a single entry, using 'feature' as the key.  Consequently, the model
      must accept a feature dict of the form {'feature': tensor}.  You may use
      `TensorServingInputReceiver` if you want the tensor to be passed as is.
    receiver_tensors: A `Tensor`, `SparseTensor`, or dict of string to `Tensor`
      or `SparseTensor`, specifying input nodes where this receiver expects to
      be fed by default.  Typically, this is a single placeholder expecting
      serialized `tf.Example` protos.
    receiver_tensors_alternatives: a dict of string to additional
      groups of receiver tensors, each of which may be a `Tensor`,
      `SparseTensor`, or dict of string to `Tensor` or`SparseTensor`.
      These named receiver tensor alternatives generate additional serving
      signatures, which may be used to feed inputs at different points within
      the input receiver subgraph.  A typical usage is to allow feeding raw
      feature `Tensor`s *downstream* of the tf.parse_example() op.
      Defaults to None.
  "
  [ features receiver_tensors receiver_tensors_alternatives ]
  (py/call-attr export "ServingInputReceiver"  features receiver_tensors receiver_tensors_alternatives ))

(defn features 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "features"))

(defn receiver-tensors 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "receiver_tensors"))

(defn receiver-tensors-alternatives 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "receiver_tensors_alternatives"))
