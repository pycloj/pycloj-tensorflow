(ns tensorflow.saved-model
  "Public API for tf.saved_model namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce saved-model (import-module "tensorflow.saved_model"))

(defn build-signature-def 
  "Utility function to build a SignatureDef protocol buffer.

  Args:
    inputs: Inputs of the SignatureDef defined as a proto map of string to
        tensor info.
    outputs: Outputs of the SignatureDef defined as a proto map of string to
        tensor info.
    method_name: Method name of the SignatureDef as a string.

  Returns:
    A SignatureDef protocol buffer constructed based on the supplied arguments.
  "
  [ inputs outputs method_name ]
  (py/call-attr saved-model "build_signature_def"  inputs outputs method_name ))

(defn build-tensor-info 
  "Utility function to build TensorInfo proto from a Tensor. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
This function will only be available through the v1 compatibility library as tf.compat.v1.saved_model.utils.build_tensor_info or tf.compat.v1.saved_model.build_tensor_info.

Args:
  tensor: Tensor or SparseTensor whose name, dtype and shape are used to
      build the TensorInfo. For SparseTensors, the names of the three
      constituent Tensors are used.

Returns:
  A TensorInfo protocol buffer constructed based on the supplied argument.

Raises:
  RuntimeError: If eager execution is enabled."
  [ tensor ]
  (py/call-attr saved-model "build_tensor_info"  tensor ))

(defn classification-signature-def 
  "Creates classification signature from given examples and predictions.

  This function produces signatures intended for use with the TensorFlow Serving
  Classify API (tensorflow_serving/apis/prediction_service.proto), and so
  constrains the input and output types to those allowed by TensorFlow Serving.

  Args:
    examples: A string `Tensor`, expected to accept serialized tf.Examples.
    classes: A string `Tensor`.  Note that the ClassificationResponse message
      requires that class labels are strings, not integers or anything else.
    scores: a float `Tensor`.

  Returns:
    A classification-flavored signature_def.

  Raises:
    ValueError: If examples is `None`.
  "
  [ examples classes scores ]
  (py/call-attr saved-model "classification_signature_def"  examples classes scores ))

(defn contains-saved-model 
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
  (py/call-attr saved-model "contains_saved_model"  export_dir ))

(defn get-tensor-from-tensor-info 
  "Returns the Tensor or CompositeTensor described by a TensorInfo proto. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
This function will only be available through the v1 compatibility library as tf.compat.v1.saved_model.utils.get_tensor_from_tensor_info or tf.compat.v1.saved_model.get_tensor_from_tensor_info.

Args:
  tensor_info: A TensorInfo proto describing a Tensor or SparseTensor or
    CompositeTensor.
  graph: The tf.Graph in which tensors are looked up. If None, the
      current default graph is used.
  import_scope: If not None, names in `tensor_info` are prefixed with this
      string before lookup.

Returns:
  The Tensor or SparseTensor or CompositeTensor in `graph` described by
  `tensor_info`.

Raises:
  KeyError: If `tensor_info` does not correspond to a tensor in `graph`.
  ValueError: If `tensor_info` is malformed."
  [ tensor_info graph import_scope ]
  (py/call-attr saved-model "get_tensor_from_tensor_info"  tensor_info graph import_scope ))

(defn is-valid-signature 
  "Determine whether a SignatureDef can be served by TensorFlow Serving."
  [ signature_def ]
  (py/call-attr saved-model "is_valid_signature"  signature_def ))

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
  (py/call-attr saved-model "load"  sess tags export_dir import_scope ))

(defn load-v2 
  "Load a SavedModel from `export_dir`.

  Signatures associated with the SavedModel are available as functions:

  ```python
  imported = tf.saved_model.load(path)
  f = imported.signatures[\"serving_default\"]
  print(f(x=tf.constant([[1.]])))
  ```

  Objects exported with `tf.saved_model.save` additionally have trackable
  objects and functions assigned to attributes:

  ```python
  exported = tf.train.Checkpoint(v=tf.Variable(3.))
  exported.f = tf.function(
      lambda x: exported.v * x,
      input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
  tf.saved_model.save(exported, path)
  imported = tf.saved_model.load(path)
  assert 3. == imported.v.numpy()
  assert 6. == imported.f(x=tf.constant(2.)).numpy()
  ```

  _Loading Keras models_

  Keras models are trackable, so they can be saved to SavedModel. The object
  returned by `tf.saved_model.load` is not a Keras object (i.e. doesn't have
  `.fit`, `.predict`, etc. methods). A few attributes and functions are still
  available: `.variables`, `.trainable_variables` and `.__call__`.

  ```python
  model = tf.keras.Model(...)
  tf.saved_model.save(model, path)
  imported = tf.saved_model.load(path)
  outputs = imported(inputs)
  ```

  Use `tf.keras.models.load_model` to restore the Keras model.

  _Importing SavedModels from TensorFlow 1.x_

  SavedModels from `tf.estimator.Estimator` or 1.x SavedModel APIs have a flat
  graph instead of `tf.function` objects. These SavedModels will have functions
  corresponding to their signatures in the `.signatures` attribute, but also
  have a `.prune` method which allows you to extract functions for new
  subgraphs. This is equivalent to importing the SavedModel and naming feeds and
  fetches in a Session from TensorFlow 1.x.

  ```python
  imported = tf.saved_model.load(path_to_v1_saved_model)
  pruned = imported.prune(\"x:0\", \"out:0\")
  pruned(tf.ones([]))
  ```

  See `tf.compat.v1.wrap_function` for details. These SavedModels also have a
  `.variables` attribute containing imported variables, and a `.graph` attribute
  representing the whole imported graph. For SavedModels exported from
  `tf.saved_model.save`, variables are instead assigned to whichever attributes
  they were assigned before export.

  Args:
    export_dir: The SavedModel directory to load from.
    tags: A tag or sequence of tags identifying the MetaGraph to load. Optional
      if the SavedModel contains a single MetaGraph, as for those exported from
      `tf.saved_model.load`.

  Returns:
    A trackable object with a `signatures` attribute mapping from signature
    keys to functions. If the SavedModel was exported by `tf.saved_model.load`,
    it also points to trackable objects and functions which were attached
    to the exported object.

  Raises:
    ValueError: If `tags` don't match a MetaGraph in the SavedModel.
  "
  [ export_dir tags ]
  (py/call-attr saved-model "load_v2"  export_dir tags ))

(defn main-op-with-restore 
  "Returns a main op to init variables, tables and restore the graph. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
This function will only be available through the v1 compatibility library as tf.compat.v1.saved_model.main_op_with_restore or tf.compat.v1.saved_model.main_op.main_op_with_restore.

Returns the main op including the group of ops that initializes all
variables, initialize local variables, initialize all tables and the restore
op name.

Args:
  restore_op_name: Name of the op to use to restore the graph.

Returns:
  The set of ops to be run as part of the main op upon the load operation."
  [ restore_op_name ]
  (py/call-attr saved-model "main_op_with_restore"  restore_op_name ))

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
  (py/call-attr saved-model "maybe_saved_model_directory"  export_dir ))

(defn predict-signature-def 
  "Creates prediction signature from given inputs and outputs.

  This function produces signatures intended for use with the TensorFlow Serving
  Predict API (tensorflow_serving/apis/prediction_service.proto). This API
  imposes no constraints on the input and output types.

  Args:
    inputs: dict of string to `Tensor`.
    outputs: dict of string to `Tensor`.

  Returns:
    A prediction-flavored signature_def.

  Raises:
    ValueError: If inputs or outputs is `None`.
  "
  [ inputs outputs ]
  (py/call-attr saved-model "predict_signature_def"  inputs outputs ))

(defn regression-signature-def 
  "Creates regression signature from given examples and predictions.

  This function produces signatures intended for use with the TensorFlow Serving
  Regress API (tensorflow_serving/apis/prediction_service.proto), and so
  constrains the input and output types to those allowed by TensorFlow Serving.

  Args:
    examples: A string `Tensor`, expected to accept serialized tf.Examples.
    predictions: A float `Tensor`.

  Returns:
    A regression-flavored signature_def.

  Raises:
    ValueError: If examples is `None`.
  "
  [ examples predictions ]
  (py/call-attr saved-model "regression_signature_def"  examples predictions ))

(defn save 
  "Exports the Trackable object `obj` to [SavedModel format](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md).

  Example usage:

  ```python
  class Adder(tf.Module):

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    def add(self, x):
      return x + x + 1.

  to_export = Adder()
  tf.saved_model.save(to_export, '/tmp/adder')
  ```

  The resulting SavedModel is then servable with an input named \"x\", its value
  having any shape and dtype float32.

  The optional `signatures` argument controls which methods in `obj` will be
  available to programs which consume `SavedModel`s, for example serving
  APIs. Python functions may be decorated with
  `@tf.function(input_signature=...)` and passed as signatures directly, or
  lazily with a call to `get_concrete_function` on the method decorated with
  `@tf.function`.

  If the `signatures` argument is omitted, `obj` will be searched for
  `@tf.function`-decorated methods. If exactly one `@tf.function` is found, that
  method will be used as the default signature for the SavedModel. This behavior
  is expected to change in the future, when a corresponding
  `tf.saved_model.load` symbol is added. At that point signatures will be
  completely optional, and any `@tf.function` attached to `obj` or its
  dependencies will be exported for use with `load`.

  When invoking a signature in an exported SavedModel, `Tensor` arguments are
  identified by name. These names will come from the Python function's argument
  names by default. They may be overridden by specifying a `name=...` argument
  in the corresponding `tf.TensorSpec` object. Explicit naming is required if
  multiple `Tensor`s are passed through a single argument to the Python
  function.

  The outputs of functions used as `signatures` must either be flat lists, in
  which case outputs will be numbered, or a dictionary mapping string keys to
  `Tensor`, in which case the keys will be used to name outputs.

  Signatures are available in objects returned by `tf.saved_model.load` as a
  `.signatures` attribute. This is a reserved attribute: `tf.saved_model.save`
  on an object with a custom `.signatures` attribute will raise an exception.

  Since `tf.keras.Model` objects are also Trackable, this function can be
  used to export Keras models. For example, exporting with a signature
  specified:

  ```python
  class Model(tf.keras.Model):

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def serve(self, serialized):
      ...

  m = Model()
  tf.saved_model.save(m, '/tmp/saved_model/')
  ```

  Exporting from a function without a fixed signature:

  ```python
  class Model(tf.keras.Model):

    @tf.function
    def call(self, x):
      ...

  m = Model()
  tf.saved_model.save(
      m, '/tmp/saved_model/',
      signatures=m.call.get_concrete_function(
          tf.TensorSpec(shape=[None, 3], dtype=tf.float32, name=\"inp\")))
  ```

  `tf.keras.Model` instances constructed from inputs and outputs already have a
  signature and so do not require a `@tf.function` decorator or a `signatures`
  argument. If neither are specified, the model's forward pass is exported.

  ```python
  x = input_layer.Input((4,), name=\"x\")
  y = core.Dense(5, name=\"out\")(x)
  model = training.Model(x, y)
  tf.saved_model.save(model, '/tmp/saved_model/')
  # The exported SavedModel takes \"x\" with shape [None, 4] and returns \"out\"
  # with shape [None, 5]
  ```

  Variables must be tracked by assigning them to an attribute of a tracked
  object or to an attribute of `obj` directly. TensorFlow objects (e.g. layers
  from `tf.keras.layers`, optimizers from `tf.train`) track their variables
  automatically. This is the same tracking scheme that `tf.train.Checkpoint`
  uses, and an exported `Checkpoint` object may be restored as a training
  checkpoint by pointing `tf.train.Checkpoint.restore` to the SavedModel's
  \"variables/\" subdirectory. Currently variables are the only stateful objects
  supported by `tf.saved_model.save`, but others (e.g. tables) will be supported
  in the future.

  `tf.function` does not hard-code device annotations from outside the function
  body, instead using the calling context's device. This means for example that
  exporting a model which runs on a GPU and serving it on a CPU will generally
  work, with some exceptions. `tf.device` annotations inside the body of the
  function will be hard-coded in the exported model; this type of annotation is
  discouraged. Device-specific operations, e.g. with \"cuDNN\" in the name or with
  device-specific layouts, may cause issues. Currently a `DistributionStrategy`
  is another exception: active distribution strategies will cause device
  placements to be hard-coded in a function. Exporting a single-device
  computation and importing under a `DistributionStrategy` is not currently
  supported, but may be in the future.

  SavedModels exported with `tf.saved_model.save` [strip default-valued
  attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes)
  automatically, which removes one source of incompatibilities when the consumer
  of a SavedModel is running an older TensorFlow version than the
  producer. There are however other sources of incompatibilities which are not
  handled automatically, such as when the exported model contains operations
  which the consumer does not have definitions for.

  Args:
    obj: A trackable object to export.
    export_dir: A directory in which to write the SavedModel.
    signatures: Optional, either a `tf.function` with an input signature
      specified or the result of `f.get_concrete_function` on a
      `@tf.function`-decorated function `f`, in which case `f` will be used to
      generate a signature for the SavedModel under the default serving
      signature key. `signatures` may also be a dictionary, in which case it
      maps from signature keys to either `tf.function` instances with input
      signatures or concrete functions. The keys of such a dictionary may be
      arbitrary strings, but will typically be from the
      `tf.saved_model.signature_constants` module.

  Raises:
    ValueError: If `obj` is not trackable.

  @compatibility(eager)
  Not well supported when graph building. From TensorFlow 1.x,
  `tf.compat.v1.enable_eager_execution()` should run first. Calling
  tf.saved_model.save in a loop when graph building from TensorFlow 1.x will
  add new save operations to the default graph each iteration.

  May not be called from within a function body.
  @end_compatibility
  "
  [ obj export_dir signatures ]
  (py/call-attr saved-model "save"  obj export_dir signatures ))

(defn simple-save 
  "Convenience function to build a SavedModel suitable for serving. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
This function will only be available through the v1 compatibility library as tf.compat.v1.saved_model.simple_save.

In many common cases, saving models for serving will be as simple as:

    simple_save(session,
                export_dir,
                inputs={\"x\": x, \"y\": y},
                outputs={\"z\": z})

Although in many cases it's not necessary to understand all of the many ways
    to configure a SavedModel, this method has a few practical implications:
  - It will be treated as a graph for inference / serving (i.e. uses the tag
    `saved_model.SERVING`)
  - The SavedModel will load in TensorFlow Serving and supports the
    [Predict
    API](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/predict.proto).
    To use the Classify, Regress, or MultiInference APIs, please
    use either
    [tf.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)
    or the lower level
    [SavedModel
    APIs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md).
  - Some TensorFlow ops depend on information on disk or other information
    called \"assets\". These are generally handled automatically by adding the
    assets to the `GraphKeys.ASSET_FILEPATHS` collection. Only assets in that
    collection are exported; if you need more custom behavior, you'll need to
    use the
    [SavedModelBuilder](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/builder.py).

More information about SavedModel and signatures can be found here:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md.

Args:
  session: The TensorFlow session from which to save the meta graph and
      variables.
  export_dir: The path to which the SavedModel will be stored.
  inputs: dict mapping string input names to tensors. These are added
      to the SignatureDef as the inputs.
  outputs:  dict mapping string output names to tensors. These are added
      to the SignatureDef as the outputs.
  legacy_init_op: Legacy support for op or group of ops to execute after the
      restore op upon a load."
  [ session export_dir inputs outputs legacy_init_op ]
  (py/call-attr saved-model "simple_save"  session export_dir inputs outputs legacy_init_op ))
