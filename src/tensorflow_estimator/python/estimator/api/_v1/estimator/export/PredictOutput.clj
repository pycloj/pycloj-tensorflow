(ns tensorflow-estimator.python.estimator.api.-v1.estimator.export.PredictOutput
  "Represents the output of a generic prediction head.

  A generic prediction need not be either a classification or a regression.

  Named outputs must be provided as a dict from string to `Tensor`,
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

(defn PredictOutput 
  "Represents the output of a generic prediction head.

  A generic prediction need not be either a classification or a regression.

  Named outputs must be provided as a dict from string to `Tensor`,
  "
  [ outputs ]
  (py/call-attr export "PredictOutput"  outputs ))

(defn as-signature-def 
  ""
  [ self receiver_tensors ]
  (py/call-attr self "as_signature_def"  self receiver_tensors ))

(defn outputs 
  ""
  [ self ]
    (py/call-attr self "outputs"))
