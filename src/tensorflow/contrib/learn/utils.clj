(ns tensorflow.contrib.learn.python.learn.utils
  "TensorFlow Learn Utils (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce utils (import-module "tensorflow.contrib.learn.python.learn.utils"))

(defn build-default-serving-input-fn 
  "Build an input_fn appropriate for serving, expecting feature Tensors. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use tf.estimator.export.build_raw_serving_input_receiver_fn.

Creates an input_fn that expects all features to be fed directly.
This input_fn is for use at serving time, so the labels return value is always
None.

Args:
  features: a dict of string to `Tensor`.
  default_batch_size: the number of query examples expected per batch.
      Leave unset for variable batch size (recommended).

Returns:
  An input_fn suitable for use in serving."
  [ features default_batch_size ]
  (py/call-attr utils "build_default_serving_input_fn"  features default_batch_size ))

(defn build-parsing-serving-input-fn 
  "Build an input_fn appropriate for serving, expecting fed tf.Examples. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use tf.estimator.export.build_parsing_serving_input_receiver_fn.

Creates an input_fn that expects a serialized tf.Example fed into a string
placeholder.  The function parses the tf.Example according to the provided
feature_spec, and returns all parsed Tensors as features.  This input_fn is
for use at serving time, so the labels return value is always None.

Args:
  feature_spec: a dict of string to `VarLenFeature`/`FixedLenFeature`.
  default_batch_size: the number of query examples expected per batch.
      Leave unset for variable batch size (recommended).

Returns:
  An input_fn suitable for use in serving."
  [ feature_spec default_batch_size ]
  (py/call-attr utils "build_parsing_serving_input_fn"  feature_spec default_batch_size ))

(defn export-estimator 
  "Deprecated, please use Estimator.export_savedmodel(). (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2016-09-23.
Instructions for updating:
Please use Estimator.export_savedmodel() instead."
  [estimator export_dir signature_fn & {:keys [input_fn default_batch_size exports_to_keep]
                       :or {exports_to_keep None}} ]
    (py/call-attr-kw utils "export_estimator" [estimator export_dir signature_fn] {:input_fn input_fn :default_batch_size default_batch_size :exports_to_keep exports_to_keep }))

(defn make-export-strategy 
  "Create an ExportStrategy for use with Experiment. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Switch to tf.estimator.Exporter and associated utilities.

Args:
  serving_input_fn: A function that takes no arguments and returns an
    `InputFnOps`.
  default_output_alternative_key: the name of the head to serve when an
    incoming serving request does not explicitly request a specific head.
    Must be `None` if the estimator inherits from `tf.estimator.Estimator`
    or for single-headed models.
  assets_extra: A dict specifying how to populate the assets.extra directory
    within the exported SavedModel.  Each key should give the destination
    path (including the filename) relative to the assets.extra directory.
    The corresponding value gives the full path of the source file to be
    copied.  For example, the simple case of copying a single file without
    renaming it is specified as
    `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
  as_text: whether to write the SavedModel proto in text format.
  exports_to_keep: Number of exports to keep.  Older exports will be
    garbage-collected.  Defaults to 5.  Set to None to disable garbage
    collection.
  strip_default_attrs: Boolean. If True, default attrs in the
    `GraphDef` will be stripped on write. This is recommended for better
    forward compatibility of the resulting `SavedModel`.

Returns:
  An ExportStrategy that can be passed to the Experiment constructor."
  [serving_input_fn default_output_alternative_key assets_extra & {:keys [as_text exports_to_keep strip_default_attrs]
                       :or {strip_default_attrs None}} ]
    (py/call-attr-kw utils "make_export_strategy" [serving_input_fn default_output_alternative_key assets_extra] {:as_text as_text :exports_to_keep exports_to_keep :strip_default_attrs strip_default_attrs }))
