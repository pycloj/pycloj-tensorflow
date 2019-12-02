(ns tensorflow.contrib.learn.InputFnOps
  "A return type for an input_fn (deprecated).

  THIS CLASS IS DEPRECATED. Please use tf.estimator.export.ServingInputReceiver
  instead.

  This return type is currently only supported for serving input_fn.
  Training and eval input_fn should return a `(features, labels)` tuple.

  The expected return values are:
    features: A dict of string to `Tensor` or `SparseTensor`, specifying the
      features to be passed to the model.
    labels: A `Tensor`, `SparseTensor`, or a dict of string to `Tensor` or
      `SparseTensor`, specifying labels for training or eval. For serving, set
      `labels` to `None`.
    default_inputs: a dict of string to `Tensor` or `SparseTensor`, specifying
      the input placeholders (if any) that this input_fn expects to be fed.
      Typically, this is used by a serving input_fn, which expects to be fed
      serialized `tf.Example` protos.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce learn (import-module "tensorflow.contrib.learn"))

(defn InputFnOps 
  "A return type for an input_fn (deprecated).

  THIS CLASS IS DEPRECATED. Please use tf.estimator.export.ServingInputReceiver
  instead.

  This return type is currently only supported for serving input_fn.
  Training and eval input_fn should return a `(features, labels)` tuple.

  The expected return values are:
    features: A dict of string to `Tensor` or `SparseTensor`, specifying the
      features to be passed to the model.
    labels: A `Tensor`, `SparseTensor`, or a dict of string to `Tensor` or
      `SparseTensor`, specifying labels for training or eval. For serving, set
      `labels` to `None`.
    default_inputs: a dict of string to `Tensor` or `SparseTensor`, specifying
      the input placeholders (if any) that this input_fn expects to be fed.
      Typically, this is used by a serving input_fn, which expects to be fed
      serialized `tf.Example` protos.
  "
  [ features labels default_inputs ]
  (py/call-attr learn "InputFnOps"  features labels default_inputs ))

(defn default-inputs 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "default_inputs"))

(defn features 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "features"))

(defn labels 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "labels"))
