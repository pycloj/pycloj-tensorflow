(ns tensorflow-estimator.python.estimator.api.-v1.estimator.LatestExporter
  "This class regularly exports the serving graph and checkpoints.

  In addition to exporting, this class also garbage collects stale exports.
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
(defn LatestExporter 
  "This class regularly exports the serving graph and checkpoints.

  In addition to exporting, this class also garbage collects stale exports.
  "
  [name serving_input_receiver_fn assets_extra  & {:keys [as_text exports_to_keep]} ]
    (py/call-attr-kw estimator "LatestExporter" [name serving_input_receiver_fn assets_extra] {:as_text as_text :exports_to_keep exports_to_keep }))

(defn export 
  ""
  [ self estimator export_path checkpoint_path eval_result is_the_final_export ]
  (py/call-attr self "export"  self estimator export_path checkpoint_path eval_result is_the_final_export ))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))
