(ns tensorflow.contrib.predictor
  "Modules for `Predictor`s.

@@from_contrib_estimator
@@from_estimator
@@from_saved_model
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce predictor (import-module "tensorflow.contrib.predictor"))

(defn from-contrib-estimator 
  "Constructs a `Predictor` from a `tf.contrib.learn.Estimator`.

  Args:
    estimator: an instance of `tf.contrib.learn.Estimator`.
    prediction_input_fn: a function that takes no arguments and returns an
      instance of `InputFnOps`.
    input_alternative_key: Optional. Specify the input alternative used for
      prediction.
    output_alternative_key: Specify the output alternative used for
      prediction. Not needed for single-headed models but required for
      multi-headed models.
    graph: Optional. The Tensorflow `graph` in which prediction should be
      done.
    config: `ConfigProto` proto used to configure the session.

  Returns:
    An initialized `Predictor`.

  Raises:
    TypeError: if `estimator` is a core `Estimator` instead of a contrib
      `Estimator`.
  "
  [ estimator prediction_input_fn input_alternative_key output_alternative_key graph config ]
  (py/call-attr predictor "from_contrib_estimator"  estimator prediction_input_fn input_alternative_key output_alternative_key graph config ))

(defn from-estimator 
  "Constructs a `Predictor` from a `tf.python.estimator.Estimator`.

  Args:
    estimator: an instance of `learn.python.estimator.Estimator`.
    serving_input_receiver_fn: a function that takes no arguments and returns
      an instance of `ServingInputReceiver` compatible with `estimator`.
    output_key: Optional string specifying the export output to use. If
      `None`, then `DEFAULT_SERVING_SIGNATURE_DEF_KEY` is used.
    graph: Optional. The Tensorflow `graph` in which prediction should be
      done.
    config: `ConfigProto` proto used to configure the session.

  Returns:
    An initialized `Predictor`.

  Raises:
    TypeError: if `estimator` is a contrib `Estimator` instead of a core
      `Estimator`.
  "
  [ estimator serving_input_receiver_fn output_key graph config ]
  (py/call-attr predictor "from_estimator"  estimator serving_input_receiver_fn output_key graph config ))

(defn from-saved-model 
  "Constructs a `Predictor` from a `SavedModel` on disk.

  Args:
    export_dir: a path to a directory containing a `SavedModel`.
    signature_def_key: Optional string specifying the signature to use. If
      `None`, then `DEFAULT_SERVING_SIGNATURE_DEF_KEY` is used. Only one of
    `signature_def_key` and `signature_def`
    signature_def: A `SignatureDef` proto specifying the inputs and outputs
      for prediction. Only one of `signature_def_key` and `signature_def`
      should be specified.
      input_names: A dictionary mapping strings to `Tensor`s in the `SavedModel`
        that represent the input. The keys can be any string of the user's
        choosing.
      output_names: A dictionary mapping strings to `Tensor`s in the
        `SavedModel` that represent the output. The keys can be any string of
        the user's choosing.
    tags: Optional. Tags that will be used to retrieve the correct
      `SignatureDef`. Defaults to `DEFAULT_TAGS`.
    graph: Optional. The Tensorflow `graph` in which prediction should be
      done.
    config: `ConfigProto` proto used to configure the session.

  Returns:
    An initialized `Predictor`.

  Raises:
    ValueError: More than one of `signature_def_key` and `signature_def` is
      specified.
  "
  [ export_dir signature_def_key signature_def input_names output_names tags graph config ]
  (py/call-attr predictor "from_saved_model"  export_dir signature_def_key signature_def input_names output_names tags graph config ))
