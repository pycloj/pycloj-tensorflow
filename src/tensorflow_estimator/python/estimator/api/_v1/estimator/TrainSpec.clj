(ns tensorflow-estimator.python.estimator.api.-v1.estimator.TrainSpec
  "Configuration for the \"train\" part for the `train_and_evaluate` call.

  `TrainSpec` determines the input data for the training, as well as the
  duration. Optional hooks run at various stages of training.
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

(defn TrainSpec 
  "Configuration for the \"train\" part for the `train_and_evaluate` call.

  `TrainSpec` determines the input data for the training, as well as the
  duration. Optional hooks run at various stages of training.
  "
  [ input_fn max_steps hooks ]
  (py/call-attr estimator "TrainSpec"  input_fn max_steps hooks ))

(defn hooks 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "hooks"))

(defn input-fn 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "input_fn"))

(defn max-steps 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "max_steps"))
