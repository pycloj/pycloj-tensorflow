(ns tensorflow-estimator.python.estimator.api.-v1.estimator.export.TensorServingInputReceiver
  "A return type for a serving_input_receiver_fn.

  This is for use with models that expect a single `Tensor` or `SparseTensor`
  as an input feature, as opposed to a dict of features.

  The normal `ServingInputReceiver` always returns a feature dict, even if it
  contains only one entry, and so can be used only with models that accept such
  a dict.  For models that accept only a single raw feature, the
  `serving_input_receiver_fn` provided to `Estimator.export_saved_model()`
  should return this `TensorServingInputReceiver` instead.  See:
  https://github.com/tensorflow/tensorflow/issues/11674

  Note that the receiver_tensors and receiver_tensor_alternatives arguments
  will be automatically converted to the dict representation in either case,
  because the SavedModel format requires each input `Tensor` to have a name
  (provided by the dict key).

  The expected return values are:
    features: A single `Tensor` or `SparseTensor`, representing the feature
      to be passed to the model.
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

(defn TensorServingInputReceiver 
  "A return type for a serving_input_receiver_fn.

  This is for use with models that expect a single `Tensor` or `SparseTensor`
  as an input feature, as opposed to a dict of features.

  The normal `ServingInputReceiver` always returns a feature dict, even if it
  contains only one entry, and so can be used only with models that accept such
  a dict.  For models that accept only a single raw feature, the
  `serving_input_receiver_fn` provided to `Estimator.export_saved_model()`
  should return this `TensorServingInputReceiver` instead.  See:
  https://github.com/tensorflow/tensorflow/issues/11674

  Note that the receiver_tensors and receiver_tensor_alternatives arguments
  will be automatically converted to the dict representation in either case,
  because the SavedModel format requires each input `Tensor` to have a name
  (provided by the dict key).

  The expected return values are:
    features: A single `Tensor` or `SparseTensor`, representing the feature
      to be passed to the model.
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
  (py/call-attr export "TensorServingInputReceiver"  features receiver_tensors receiver_tensors_alternatives ))

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
