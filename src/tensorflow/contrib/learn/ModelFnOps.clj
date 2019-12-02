(ns tensorflow.contrib.learn.ModelFnOps
  "Ops returned from a model_fn.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
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

(defn ModelFnOps 
  "Ops returned from a model_fn.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
  "
  [ mode predictions loss train_op eval_metric_ops output_alternatives training_chief_hooks training_hooks scaffold ]
  (py/call-attr learn "ModelFnOps"  mode predictions loss train_op eval_metric_ops output_alternatives training_chief_hooks training_hooks scaffold ))

(defn estimator-spec 
  "Creates an equivalent `EstimatorSpec`.

    Args:
      default_serving_output_alternative_key: Required for multiple heads. If
        you have multiple entries in `output_alternatives` dict (comparable to
        multiple heads), `EstimatorSpec` requires a default head that will be
        used if a Servo request does not explicitly mention which head to infer
        on. Pass the key of the output alternative here that you want to
        designate as default. A separate ExportOutpout for this default head
        will be added to the export_outputs dict with the special key
        saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY, unless there is
        already an enry in output_alternatives with this special key.

    Returns:
      Instance of `EstimatorSpec` that is equivalent to this `ModelFnOps`

    Raises:
      ValueError: If problem type is unknown.
    "
  [ self default_serving_output_alternative_key ]
  (py/call-attr self "estimator_spec"  self default_serving_output_alternative_key ))

(defn eval-metric-ops 
  "Alias for field number 3"
  [ self ]
    (py/call-attr self "eval_metric_ops"))

(defn loss 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "loss"))

(defn mode 
  "Alias for field number 8"
  [ self ]
    (py/call-attr self "mode"))

(defn output-alternatives 
  "Alias for field number 4"
  [ self ]
    (py/call-attr self "output_alternatives"))

(defn predictions 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "predictions"))

(defn scaffold 
  "Alias for field number 7"
  [ self ]
    (py/call-attr self "scaffold"))

(defn train-op 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "train_op"))

(defn training-chief-hooks 
  "Alias for field number 5"
  [ self ]
    (py/call-attr self "training_chief_hooks"))

(defn training-hooks 
  "Alias for field number 6"
  [ self ]
    (py/call-attr self "training_hooks"))
