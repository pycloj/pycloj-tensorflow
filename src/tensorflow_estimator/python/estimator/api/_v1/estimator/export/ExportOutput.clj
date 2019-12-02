(ns tensorflow-estimator.python.estimator.api.-v1.estimator.export.ExportOutput
  "Represents an output of a model that can be served.

  These typically correspond to model heads.
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

(defn ExportOutput 
  "Represents an output of a model that can be served.

  These typically correspond to model heads.
  "
  [  ]
  (py/call-attr export "ExportOutput"  ))

(defn as-signature-def 
  "Generate a SignatureDef proto for inclusion in a MetaGraphDef.

    The SignatureDef will specify outputs as described in this ExportOutput,
    and will use the provided receiver_tensors as inputs.

    Args:
      receiver_tensors: a `Tensor`, or a dict of string to `Tensor`, specifying
        input nodes that will be fed.
    "
  [ self receiver_tensors ]
  (py/call-attr self "as_signature_def"  self receiver_tensors ))
