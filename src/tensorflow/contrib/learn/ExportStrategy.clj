(ns tensorflow.contrib.learn.ExportStrategy
  "A class representing a type of model export.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Typically constructed by a utility function specific to the exporter, such as
  `saved_model_export_utils.make_export_strategy()`.

  Attributes:
    name: The directory name under the export base directory where exports of
      this type will be written.
    export_fn: A function that writes an export, given an estimator, a
      destination path, and optionally a checkpoint path and an evaluation
      result for that checkpoint.  This export_fn() may be run repeatedly during
      continuous training, or just once at the end of fixed-length training.
      Note the export_fn() may choose whether or not to export based on the eval
      result or based on an internal timer or any other criterion, if exports
      are not desired for every checkpoint.

    The signature of this function must be one of:

      * `(estimator, export_path) -> export_path`
      * `(estimator, export_path, checkpoint_path) -> export_path`
      * `(estimator, export_path, checkpoint_path, eval_result) -> export_path`
      * `(estimator, export_path, checkpoint_path, eval_result,
          strip_default_attrs) -> export_path`
    strip_default_attrs: (Optional) Boolean. If set as True, default attrs in
        the `GraphDef` will be stripped on write. This is recommended for better
        forward compatibility of the resulting `SavedModel`.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce learn (import-module "tensorflow.contrib.learn"))

(defn ExportStrategy 
  "A class representing a type of model export.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Typically constructed by a utility function specific to the exporter, such as
  `saved_model_export_utils.make_export_strategy()`.

  Attributes:
    name: The directory name under the export base directory where exports of
      this type will be written.
    export_fn: A function that writes an export, given an estimator, a
      destination path, and optionally a checkpoint path and an evaluation
      result for that checkpoint.  This export_fn() may be run repeatedly during
      continuous training, or just once at the end of fixed-length training.
      Note the export_fn() may choose whether or not to export based on the eval
      result or based on an internal timer or any other criterion, if exports
      are not desired for every checkpoint.

    The signature of this function must be one of:

      * `(estimator, export_path) -> export_path`
      * `(estimator, export_path, checkpoint_path) -> export_path`
      * `(estimator, export_path, checkpoint_path, eval_result) -> export_path`
      * `(estimator, export_path, checkpoint_path, eval_result,
          strip_default_attrs) -> export_path`
    strip_default_attrs: (Optional) Boolean. If set as True, default attrs in
        the `GraphDef` will be stripped on write. This is recommended for better
        forward compatibility of the resulting `SavedModel`.
  "
  [ name export_fn strip_default_attrs ]
  (py/call-attr learn "ExportStrategy"  name export_fn strip_default_attrs ))

(defn export 
  "Exports the given Estimator to a specific format.

    Args:
      estimator: the Estimator to export.
      export_path: A string containing a directory where to write the export.
      checkpoint_path: The checkpoint path to export.  If None (the default),
        the strategy may locate a checkpoint (e.g. the most recent) by itself.
      eval_result: The output of Estimator.evaluate on this checkpoint.  This
        should be set only if checkpoint_path is provided (otherwise it is
        unclear which checkpoint this eval refers to).

    Returns:
      The string path to the exported directory.

    Raises:
      ValueError: if the export_fn does not have the required signature
    "
  [ self estimator export_path checkpoint_path eval_result ]
  (py/call-attr self "export"  self estimator export_path checkpoint_path eval_result ))

(defn export-fn 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "export_fn"))

(defn name 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "name"))

(defn strip-default-attrs 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "strip_default_attrs"))
