(ns tensorflow-estimator.python.estimator.api.-v1.estimator.Exporter
  "A class representing a type of model export."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce estimator (import-module "tensorflow_estimator.python.estimator.api._v1.estimator"))

(defn Exporter 
  "A class representing a type of model export."
  [  ]
  (py/call-attr estimator "Exporter"  ))

(defn export 
  "Exports the given `Estimator` to a specific format.

    Args:
      estimator: the `Estimator` to export.
      export_path: A string containing a directory where to write the export.
      checkpoint_path: The checkpoint path to export.
      eval_result: The output of `Estimator.evaluate` on this checkpoint.
      is_the_final_export: This boolean is True when this is an export in the
        end of training.  It is False for the intermediate exports during
        the training.
        When passing `Exporter` to `tf.estimator.train_and_evaluate`
        `is_the_final_export` is always False if `TrainSpec.max_steps` is
        `None`.

    Returns:
      The string path to the exported directory or `None` if export is skipped.
    "
  [ self estimator export_path checkpoint_path eval_result is_the_final_export ]
  (py/call-attr self "export"  self estimator export_path checkpoint_path eval_result is_the_final_export ))

(defn name 
  "Directory name.

    A directory name under the export base directory where exports of
    this type are written.  Should not be `None` nor empty.
    "
  [ self ]
    (py/call-attr self "name"))
