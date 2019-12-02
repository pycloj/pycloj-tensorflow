(ns tensorflow-estimator.python.estimator.api.-v1.estimator.FinalExporter
  "This class exports the serving graph and checkpoints at the end.

  This class performs a single export at the end of training.
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
(defn FinalExporter 
  "This class exports the serving graph and checkpoints at the end.

  This class performs a single export at the end of training.
  "
  [name serving_input_receiver_fn assets_extra  & {:keys [as_text]} ]
    (py/call-attr-kw estimator "FinalExporter" [name serving_input_receiver_fn assets_extra] {:as_text as_text }))

(defn export 
  ""
  [ self estimator export_path checkpoint_path eval_result is_the_final_export ]
  (py/call-attr self "export"  self estimator export_path checkpoint_path eval_result is_the_final_export ))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))
