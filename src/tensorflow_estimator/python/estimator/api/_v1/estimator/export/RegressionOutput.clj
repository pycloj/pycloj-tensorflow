(ns tensorflow-estimator.python.estimator.api.-v1.estimator.export.RegressionOutput
  "Represents the output of a regression head."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce export (import-module "tensorflow_estimator.python.estimator.api._v1.estimator.export"))

(defn RegressionOutput 
  "Represents the output of a regression head."
  [ value ]
  (py/call-attr export "RegressionOutput"  value ))

(defn as-signature-def 
  ""
  [ self receiver_tensors ]
  (py/call-attr self "as_signature_def"  self receiver_tensors ))

(defn value 
  ""
  [ self ]
    (py/call-attr self "value"))
