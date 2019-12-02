(ns tensorflow-estimator.python.estimator.api.-v1.estimator.EstimatorSpec
  "Ops and objects returned from a `model_fn` and passed to an `Estimator`.

  `EstimatorSpec` fully defines the model to be run by an `Estimator`.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce estimator (import-module "tensorflow_estimator.python.estimator.api._v1.estimator"))

(defn EstimatorSpec 
  "Ops and objects returned from a `model_fn` and passed to an `Estimator`.

  `EstimatorSpec` fully defines the model to be run by an `Estimator`.
  "
  [ mode predictions loss train_op eval_metric_ops export_outputs training_chief_hooks training_hooks scaffold evaluation_hooks prediction_hooks ]
  (py/call-attr estimator "EstimatorSpec"  mode predictions loss train_op eval_metric_ops export_outputs training_chief_hooks training_hooks scaffold evaluation_hooks prediction_hooks ))

(defn eval-metric-ops 
  "Alias for field number 4"
  [ self ]
    (py/call-attr self "eval_metric_ops"))

(defn evaluation-hooks 
  "Alias for field number 9"
  [ self ]
    (py/call-attr self "evaluation_hooks"))

(defn export-outputs 
  "Alias for field number 5"
  [ self ]
    (py/call-attr self "export_outputs"))

(defn loss 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "loss"))

(defn mode 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "mode"))

(defn prediction-hooks 
  "Alias for field number 10"
  [ self ]
    (py/call-attr self "prediction_hooks"))

(defn predictions 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "predictions"))

(defn scaffold 
  "Alias for field number 8"
  [ self ]
    (py/call-attr self "scaffold"))

(defn train-op 
  "Alias for field number 3"
  [ self ]
    (py/call-attr self "train_op"))

(defn training-chief-hooks 
  "Alias for field number 6"
  [ self ]
    (py/call-attr self "training_chief_hooks"))

(defn training-hooks 
  "Alias for field number 7"
  [ self ]
    (py/call-attr self "training_hooks"))
