(ns tensorflow.-api.v1.compat.v1.saved-model.loader
  "Loader functionality for SavedModel with hermetic, language-neutral exports.

Load and restore capability for a SavedModel, which may include multiple meta
graph defs. Each SavedModel is associated with a single checkpoint. Each meta
graph def is saved with one or more tags, which are used to identify the exact
meta graph def to load.

The `load` operation requires the session in which to restore the graph
definition and variables, the tags used to identify the meta graph def to
load and the location of the SavedModel.

Upon a load, the subset of variables and assets supplied as part of the specific
meta graph def, will be restored into the supplied session. The values of the
variables though will correspond to the saved values from the first meta graph
added to the SavedModel using `add_meta_graph_and_variables(...)` in
`builder.py`.

Typical usage:

```python
...
builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph_and_variables(sess,
                                       [\"foo-tag\"],
                                       signature_def_map=foo_signatures,
                                       assets_collection=foo_assets)
...

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph([\"bar-tag\", \"baz-tag\"],
                         assets_collection=bar_baz_assets)
...

builder.save()

...
with tf.compat.v1.Session(graph=tf.Graph()) as sess:
  tf.compat.v1.saved_model.loader.load(sess, [\"foo-tag\"], export_dir)
  ...

```

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce loader (import-module "tensorflow._api.v1.compat.v1.saved_model.loader"))

(defn load 
  "Loads the model from a SavedModel as specified by tags. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
This function will only be available through the v1 compatibility library as tf.compat.v1.saved_model.loader.load or tf.compat.v1.saved_model.load. There will be a new function for importing SavedModels in Tensorflow 2.0.

Args:
  sess: The TensorFlow session to restore the variables.
  tags: Set of string tags to identify the required MetaGraphDef. These should
      correspond to the tags used when saving the variables using the
      SavedModel `save()` API.
  export_dir: Directory in which the SavedModel protocol buffer and variables
      to be loaded are located.
  import_scope: Optional `string` -- if specified, prepend this string
      followed by '/' to all loaded tensor names. This scope is applied to
      tensor instances loaded into the passed session, but it is *not* written
      through to the static `MetaGraphDef` protocol buffer that is returned.
  **saver_kwargs: Optional keyword arguments passed through to Saver.

Returns:
  The `MetaGraphDef` protocol buffer loaded in the provided session. This
  can be used to further extract signature-defs, collection-defs, etc.

Raises:
  RuntimeError: MetaGraphDef associated with the tags cannot be found."
  [ sess tags export_dir import_scope ]
  (py/call-attr loader "load"  sess tags export_dir import_scope ))

(defn maybe-saved-model-directory 
  "Checks whether the provided export directory could contain a SavedModel.

  Note that the method does not load any data by itself. If the method returns
  `false`, the export directory definitely does not contain a SavedModel. If the
  method returns `true`, the export directory may contain a SavedModel but
  provides no guarantee that it can be loaded.

  Args:
    export_dir: Absolute string path to possible export location. For example,
                '/my/foo/model'.

  Returns:
    True if the export directory contains SavedModel files, False otherwise.
  "
  [ export_dir ]
  (py/call-attr loader "maybe_saved_model_directory"  export_dir ))
