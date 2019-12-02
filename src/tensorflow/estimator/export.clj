(ns tensorflow-estimator.python.estimator.api.-v1.estimator.export
  "All public utility methods for exporting Estimator to SavedModel.

This file includes functions and constants from core (model_utils) and export.py

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

(defn build-parsing-serving-input-receiver-fn 
  "Build a serving_input_receiver_fn expecting fed tf.Examples.

  Creates a serving_input_receiver_fn that expects a serialized tf.Example fed
  into a string placeholder.  The function parses the tf.Example according to
  the provided feature_spec, and returns all parsed Tensors as features.

  Args:
    feature_spec: a dict of string to `VarLenFeature`/`FixedLenFeature`.
    default_batch_size: the number of query examples expected per batch.
        Leave unset for variable batch size (recommended).

  Returns:
    A serving_input_receiver_fn suitable for use in serving.
  "
  [ feature_spec default_batch_size ]
  (py/call-attr export "build_parsing_serving_input_receiver_fn"  feature_spec default_batch_size ))

(defn build-raw-serving-input-receiver-fn 
  "Build a serving_input_receiver_fn expecting feature Tensors.

  Creates an serving_input_receiver_fn that expects all features to be fed
  directly.

  Args:
    features: a dict of string to `Tensor`.
    default_batch_size: the number of query examples expected per batch.
        Leave unset for variable batch size (recommended).

  Returns:
    A serving_input_receiver_fn.
  "
  [ features default_batch_size ]
  (py/call-attr export "build_raw_serving_input_receiver_fn"  features default_batch_size ))
