(ns tensorflow-estimator.python.estimator.api.-v1.estimator.BestExporter
  "This class exports the serving graph and checkpoints of the best models.

  This class performs a model export everytime the new model is better than any
  existing model.
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

(defn BestExporter 
  "This class exports the serving graph and checkpoints of the best models.

  This class performs a model export everytime the new model is better than any
  existing model.
  "
  [ & {:keys [name serving_input_receiver_fn event_file_pattern compare_fn assets_extra as_text exports_to_keep]
       :or {serving_input_receiver_fn None assets_extra None}} ]
  
   (py/call-attr-kw estimator "BestExporter" [] {:name name :serving_input_receiver_fn serving_input_receiver_fn :event_file_pattern event_file_pattern :compare_fn compare_fn :assets_extra assets_extra :as_text as_text :exports_to_keep exports_to_keep }))

(defn export 
  ""
  [ self estimator export_path checkpoint_path eval_result is_the_final_export ]
  (py/call-attr self "export"  self estimator export_path checkpoint_path eval_result is_the_final_export ))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))
