(ns tensorflow-estimator.python.estimator.api.-v1.estimator.ModeKeys
  "Standard names for Estimator model modes.

  The following standard keys are defined:

  * `TRAIN`: training/fitting mode.
  * `EVAL`: testing/evaluation mode.
  * `PREDICT`: predication/inference mode.
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

(defn ModeKeys 
  "Standard names for Estimator model modes.

  The following standard keys are defined:

  * `TRAIN`: training/fitting mode.
  * `EVAL`: testing/evaluation mode.
  * `PREDICT`: predication/inference mode.
  "
  [  ]
  (py/call-attr estimator "ModeKeys"  ))
