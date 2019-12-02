(ns tensorflow-estimator.python.estimator.api.-v1.estimator.export.ClassificationOutput
  "Represents the output of a classification head.

  Either classes or scores or both must be set.

  The classes `Tensor` must provide string labels, not integer class IDs.

  If only classes is set, it is interpreted as providing top-k results in
  descending order.

  If only scores is set, it is interpreted as providing a score for every class
  in order of class ID.

  If both classes and scores are set, they are interpreted as zipped, so each
  score corresponds to the class at the same index.  Clients should not depend
  on the order of the entries.
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

(defn ClassificationOutput 
  "Represents the output of a classification head.

  Either classes or scores or both must be set.

  The classes `Tensor` must provide string labels, not integer class IDs.

  If only classes is set, it is interpreted as providing top-k results in
  descending order.

  If only scores is set, it is interpreted as providing a score for every class
  in order of class ID.

  If both classes and scores are set, they are interpreted as zipped, so each
  score corresponds to the class at the same index.  Clients should not depend
  on the order of the entries.
  "
  [ scores classes ]
  (py/call-attr export "ClassificationOutput"  scores classes ))

(defn as-signature-def 
  ""
  [ self receiver_tensors ]
  (py/call-attr self "as_signature_def"  self receiver_tensors ))

(defn classes 
  ""
  [ self ]
    (py/call-attr self "classes"))

(defn scores 
  ""
  [ self ]
    (py/call-attr self "scores"))
