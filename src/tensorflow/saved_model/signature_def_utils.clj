(ns tensorflow.-api.v1.saved-model.signature-def-utils
  "SignatureDef utility functions.

Utility functions for building and inspecting SignatureDef protos.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce signature-def-utils (import-module "tensorflow._api.v1.saved_model.signature_def_utils"))

(defn build-signature-def 
  "Utility function to build a SignatureDef protocol buffer.

  Args:
    inputs: Inputs of the SignatureDef defined as a proto map of string to
        tensor info.
    outputs: Outputs of the SignatureDef defined as a proto map of string to
        tensor info.
    method_name: Method name of the SignatureDef as a string.

  Returns:
    A SignatureDef protocol buffer constructed based on the supplied arguments.
  "
  [ inputs outputs method_name ]
  (py/call-attr signature-def-utils "build_signature_def"  inputs outputs method_name ))

(defn classification-signature-def 
  "Creates classification signature from given examples and predictions.

  This function produces signatures intended for use with the TensorFlow Serving
  Classify API (tensorflow_serving/apis/prediction_service.proto), and so
  constrains the input and output types to those allowed by TensorFlow Serving.

  Args:
    examples: A string `Tensor`, expected to accept serialized tf.Examples.
    classes: A string `Tensor`.  Note that the ClassificationResponse message
      requires that class labels are strings, not integers or anything else.
    scores: a float `Tensor`.

  Returns:
    A classification-flavored signature_def.

  Raises:
    ValueError: If examples is `None`.
  "
  [ examples classes scores ]
  (py/call-attr signature-def-utils "classification_signature_def"  examples classes scores ))

(defn is-valid-signature 
  "Determine whether a SignatureDef can be served by TensorFlow Serving."
  [ signature_def ]
  (py/call-attr signature-def-utils "is_valid_signature"  signature_def ))

(defn predict-signature-def 
  "Creates prediction signature from given inputs and outputs.

  This function produces signatures intended for use with the TensorFlow Serving
  Predict API (tensorflow_serving/apis/prediction_service.proto). This API
  imposes no constraints on the input and output types.

  Args:
    inputs: dict of string to `Tensor`.
    outputs: dict of string to `Tensor`.

  Returns:
    A prediction-flavored signature_def.

  Raises:
    ValueError: If inputs or outputs is `None`.
  "
  [ inputs outputs ]
  (py/call-attr signature-def-utils "predict_signature_def"  inputs outputs ))

(defn regression-signature-def 
  "Creates regression signature from given examples and predictions.

  This function produces signatures intended for use with the TensorFlow Serving
  Regress API (tensorflow_serving/apis/prediction_service.proto), and so
  constrains the input and output types to those allowed by TensorFlow Serving.

  Args:
    examples: A string `Tensor`, expected to accept serialized tf.Examples.
    predictions: A float `Tensor`.

  Returns:
    A regression-flavored signature_def.

  Raises:
    ValueError: If examples is `None`.
  "
  [ examples predictions ]
  (py/call-attr signature-def-utils "regression_signature_def"  examples predictions ))
