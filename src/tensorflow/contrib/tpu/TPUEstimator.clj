(ns tensorflow.contrib.tpu.TPUEstimator
  "Estimator with TPU support.

  TPUEstimator also supports training on CPU and GPU. You don't need to define
  a separate `tf.estimator.Estimator`.

  TPUEstimator handles many of the details of running on TPU devices, such as
  replicating inputs and models for each core, and returning to host
  periodically to run hooks.

  TPUEstimator transforms a global batch size in params to a per-shard batch
  size when calling the `input_fn` and `model_fn`. Users should specify
  global batch size in constructor, and then get the batch size for each shard
  in `input_fn` and `model_fn` by `params['batch_size']`.

  - For training, `model_fn` gets per-core batch size; `input_fn` may get
    per-core or per-host batch size depending on `per_host_input_for_training`
    in `TPUConfig` (See docstring for TPUConfig for details).

  - For evaluation and prediction, `model_fn` gets per-core batch size and
    `input_fn` get per-host batch size.

  Evaluation
  ==========

  `model_fn` should return `TPUEstimatorSpec`, which expects the `eval_metrics`
  for TPU evaluation. If eval_on_tpu is False, the evaluation will execute on
  CPU or GPU; in this case the following discussion on TPU evaluation does not
  apply.

  `TPUEstimatorSpec.eval_metrics` is a tuple of `metric_fn` and `tensors`, where
  `tensors` could be a list of any nested structure of `Tensor`s (See
  `TPUEstimatorSpec` for details).  `metric_fn` takes the `tensors` and returns
  a dict from metric string name to the result of calling a metric function,
  namely a `(metric_tensor, update_op)` tuple.

  One can set `use_tpu` to `False` for testing. All training, evaluation, and
  predict will be executed on CPU. `input_fn` and `model_fn` will receive
  `train_batch_size` or `eval_batch_size` unmodified as `params['batch_size']`.

  Current limitations:
  --------------------

  1. TPU evaluation only works on a single host (one TPU worker) except
     BROADCAST mode.

  2. `input_fn` for evaluation should **NOT** raise an end-of-input exception
     (`OutOfRangeError` or `StopIteration`). And all evaluation steps and all
     batches should have the same size.

  Example (MNIST):
  ----------------

  ```
  # The metric Fn which runs on CPU.
  def metric_fn(labels, logits):
    predictions = tf.argmax(logits, 1)
    return {
      'accuracy': tf.compat.v1.metrics.precision(
          labels=labels, predictions=predictions),
    }

  # Your model Fn which runs on TPU (eval_metrics is list in this example)
  def model_fn(features, labels, mode, config, params):
    ...
    logits = ...

    if mode = tf.estimator.ModeKeys.EVAL:
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(metric_fn, [labels, logits]))

  # or specify the eval_metrics tensors as dict.
  def model_fn(features, labels, mode, config, params):
    ...
    final_layer_output = ...

    if mode = tf.estimator.ModeKeys.EVAL:
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(metric_fn, {
              'labels': labels,
              'logits': final_layer_output,
          }))
  ```

  Prediction
  ==========

  Prediction on TPU is an experimental feature to support large batch inference.
  It is not designed for latency-critical system. In addition, due to some
  usability issues, for prediction with small dataset, CPU `.predict`, i.e.,
  creating a new `TPUEstimator` instance with `use_tpu=False`, might be more
  convenient.

  Note: In contrast to TPU training/evaluation, the `input_fn` for prediction
  *should* raise an end-of-input exception (`OutOfRangeError` or
  `StopIteration`), which serves as the stopping signal to `TPUEstimator`. To be
  precise, the ops created by `input_fn` produce one batch of the data.
  The `predict()` API processes one batch at a time. When reaching the end of
  the data source, an end-of-input exception should be raised by one of these
  operations. The user usually does not need to do this manually. As long as the
  dataset is not repeated forever, the `tf.data` API will raise an end-of-input
  exception automatically after the last batch has been produced.

  Note: Estimator.predict returns a Python generator. Please consume all the
  data from the generator so that TPUEstimator can shutdown the TPU system
  properly for user.

  Current limitations:
  --------------------
  1. TPU prediction only works on a single host (one TPU worker).

  2. `input_fn` must return a `Dataset` instance rather than `features`. In
  fact, .train() and .evaluate() also support Dataset as return value.

  Example (MNIST):
  ----------------
  ```
  height = 32
  width = 32
  total_examples = 100

  def predict_input_fn(params):
    batch_size = params['batch_size']

    images = tf.random.uniform(
        [total_examples, height, width, 3], minval=-1, maxval=1)

    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.map(lambda images: {'image': images})

    dataset = dataset.batch(batch_size)
    return dataset

  def model_fn(features, labels, params, mode):
     # Generate predictions, called 'output', from features['image']

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
              'predictions': output,
              'is_padding': features['is_padding']
          })

  tpu_est = TPUEstimator(
      model_fn=model_fn,
      ...,
      predict_batch_size=16)

  # Fully consume the generator so that TPUEstimator can shutdown the TPU
  # system.
  for item in tpu_est.predict(input_fn=input_fn):
    # Filter out item if the `is_padding` is 1.
    # Process the 'predictions'
  ```

  Exporting
  =========

  `export_saved_model` exports 2 metagraphs, one with `saved_model.SERVING`, and
  another with `saved_model.SERVING` and `saved_model.TPU` tags. At serving
  time, these tags are used to select the appropriate metagraph to load.

  Before running the graph on TPU, the TPU system needs to be initialized. If
  TensorFlow Serving model-server is used, this is done automatically. If not,
  please use `session.run(tpu.initialize_system())`.

  There are two versions of the API: ExportSavedModelApiVersion.V1 and V2.

  In V1, the exported CPU graph is `model_fn` as it is. The exported TPU graph
  wraps `tpu.rewrite()` and `TPUPartitionedCallOp` around `model_fn` so
  `model_fn` is on TPU by default. To place ops on CPU,
  `tpu.outside_compilation(host_call, logits)` can be used.

  Example:
  ----------------

  ```
  def model_fn(features, labels, mode, config, params):
    ...
    logits = ...
    export_outputs = {
      'logits': export_output_lib.PredictOutput(
        {'logits': logits})
    }

    def host_call(logits):
      class_ids = math_ops.argmax(logits)
      classes = string_ops.as_string(class_ids)
      export_outputs['classes'] =
        export_output_lib.ClassificationOutput(classes=classes)

    tpu.outside_compilation(host_call, logits)

    ...
  ```

  In V2, `export_saved_model()` sets up `params['use_tpu']` flag to let the user
  know if the code is exporting to TPU (or not). When `params['use_tpu']` is
  `True`, users need to call `tpu.rewrite()`, `TPUPartitionedCallOp` and/or
  `batch_function()`. Alternatively use `inference_on_tpu()` which is a
  convenience wrapper of the three.

  ```
    def model_fn(features, labels, mode, config, params):
      ...
      # This could be some pre-processing on CPU like calls to input layer with
      # embedding columns.
      x2 = features['x'] * 2

      def computation(input_tensor):
        return layers.dense(
            input_tensor, 1, kernel_initializer=init_ops.zeros_initializer())

      inputs = [x2]
      if params['use_tpu']:
        predictions = array_ops.identity(
            tpu_estimator.inference_on_tpu(computation, inputs,
            num_batch_threads=1, max_batch_size=2, batch_timeout_micros=100),
            name='predictions')
      else:
        predictions = array_ops.identity(
            computation(*inputs), name='predictions')
      key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
      export_outputs = {
          key: export_lib.PredictOutput({'prediction': predictions})
      }
      ...
  ```

  TIP: V2 is recommended as it is more flexible (eg: batching, etc).
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tpu (import-module "tensorflow.contrib.tpu"))

(defn TPUEstimator 
  "Estimator with TPU support.

  TPUEstimator also supports training on CPU and GPU. You don't need to define
  a separate `tf.estimator.Estimator`.

  TPUEstimator handles many of the details of running on TPU devices, such as
  replicating inputs and models for each core, and returning to host
  periodically to run hooks.

  TPUEstimator transforms a global batch size in params to a per-shard batch
  size when calling the `input_fn` and `model_fn`. Users should specify
  global batch size in constructor, and then get the batch size for each shard
  in `input_fn` and `model_fn` by `params['batch_size']`.

  - For training, `model_fn` gets per-core batch size; `input_fn` may get
    per-core or per-host batch size depending on `per_host_input_for_training`
    in `TPUConfig` (See docstring for TPUConfig for details).

  - For evaluation and prediction, `model_fn` gets per-core batch size and
    `input_fn` get per-host batch size.

  Evaluation
  ==========

  `model_fn` should return `TPUEstimatorSpec`, which expects the `eval_metrics`
  for TPU evaluation. If eval_on_tpu is False, the evaluation will execute on
  CPU or GPU; in this case the following discussion on TPU evaluation does not
  apply.

  `TPUEstimatorSpec.eval_metrics` is a tuple of `metric_fn` and `tensors`, where
  `tensors` could be a list of any nested structure of `Tensor`s (See
  `TPUEstimatorSpec` for details).  `metric_fn` takes the `tensors` and returns
  a dict from metric string name to the result of calling a metric function,
  namely a `(metric_tensor, update_op)` tuple.

  One can set `use_tpu` to `False` for testing. All training, evaluation, and
  predict will be executed on CPU. `input_fn` and `model_fn` will receive
  `train_batch_size` or `eval_batch_size` unmodified as `params['batch_size']`.

  Current limitations:
  --------------------

  1. TPU evaluation only works on a single host (one TPU worker) except
     BROADCAST mode.

  2. `input_fn` for evaluation should **NOT** raise an end-of-input exception
     (`OutOfRangeError` or `StopIteration`). And all evaluation steps and all
     batches should have the same size.

  Example (MNIST):
  ----------------

  ```
  # The metric Fn which runs on CPU.
  def metric_fn(labels, logits):
    predictions = tf.argmax(logits, 1)
    return {
      'accuracy': tf.compat.v1.metrics.precision(
          labels=labels, predictions=predictions),
    }

  # Your model Fn which runs on TPU (eval_metrics is list in this example)
  def model_fn(features, labels, mode, config, params):
    ...
    logits = ...

    if mode = tf.estimator.ModeKeys.EVAL:
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(metric_fn, [labels, logits]))

  # or specify the eval_metrics tensors as dict.
  def model_fn(features, labels, mode, config, params):
    ...
    final_layer_output = ...

    if mode = tf.estimator.ModeKeys.EVAL:
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(metric_fn, {
              'labels': labels,
              'logits': final_layer_output,
          }))
  ```

  Prediction
  ==========

  Prediction on TPU is an experimental feature to support large batch inference.
  It is not designed for latency-critical system. In addition, due to some
  usability issues, for prediction with small dataset, CPU `.predict`, i.e.,
  creating a new `TPUEstimator` instance with `use_tpu=False`, might be more
  convenient.

  Note: In contrast to TPU training/evaluation, the `input_fn` for prediction
  *should* raise an end-of-input exception (`OutOfRangeError` or
  `StopIteration`), which serves as the stopping signal to `TPUEstimator`. To be
  precise, the ops created by `input_fn` produce one batch of the data.
  The `predict()` API processes one batch at a time. When reaching the end of
  the data source, an end-of-input exception should be raised by one of these
  operations. The user usually does not need to do this manually. As long as the
  dataset is not repeated forever, the `tf.data` API will raise an end-of-input
  exception automatically after the last batch has been produced.

  Note: Estimator.predict returns a Python generator. Please consume all the
  data from the generator so that TPUEstimator can shutdown the TPU system
  properly for user.

  Current limitations:
  --------------------
  1. TPU prediction only works on a single host (one TPU worker).

  2. `input_fn` must return a `Dataset` instance rather than `features`. In
  fact, .train() and .evaluate() also support Dataset as return value.

  Example (MNIST):
  ----------------
  ```
  height = 32
  width = 32
  total_examples = 100

  def predict_input_fn(params):
    batch_size = params['batch_size']

    images = tf.random.uniform(
        [total_examples, height, width, 3], minval=-1, maxval=1)

    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.map(lambda images: {'image': images})

    dataset = dataset.batch(batch_size)
    return dataset

  def model_fn(features, labels, params, mode):
     # Generate predictions, called 'output', from features['image']

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
              'predictions': output,
              'is_padding': features['is_padding']
          })

  tpu_est = TPUEstimator(
      model_fn=model_fn,
      ...,
      predict_batch_size=16)

  # Fully consume the generator so that TPUEstimator can shutdown the TPU
  # system.
  for item in tpu_est.predict(input_fn=input_fn):
    # Filter out item if the `is_padding` is 1.
    # Process the 'predictions'
  ```

  Exporting
  =========

  `export_saved_model` exports 2 metagraphs, one with `saved_model.SERVING`, and
  another with `saved_model.SERVING` and `saved_model.TPU` tags. At serving
  time, these tags are used to select the appropriate metagraph to load.

  Before running the graph on TPU, the TPU system needs to be initialized. If
  TensorFlow Serving model-server is used, this is done automatically. If not,
  please use `session.run(tpu.initialize_system())`.

  There are two versions of the API: ExportSavedModelApiVersion.V1 and V2.

  In V1, the exported CPU graph is `model_fn` as it is. The exported TPU graph
  wraps `tpu.rewrite()` and `TPUPartitionedCallOp` around `model_fn` so
  `model_fn` is on TPU by default. To place ops on CPU,
  `tpu.outside_compilation(host_call, logits)` can be used.

  Example:
  ----------------

  ```
  def model_fn(features, labels, mode, config, params):
    ...
    logits = ...
    export_outputs = {
      'logits': export_output_lib.PredictOutput(
        {'logits': logits})
    }

    def host_call(logits):
      class_ids = math_ops.argmax(logits)
      classes = string_ops.as_string(class_ids)
      export_outputs['classes'] =
        export_output_lib.ClassificationOutput(classes=classes)

    tpu.outside_compilation(host_call, logits)

    ...
  ```

  In V2, `export_saved_model()` sets up `params['use_tpu']` flag to let the user
  know if the code is exporting to TPU (or not). When `params['use_tpu']` is
  `True`, users need to call `tpu.rewrite()`, `TPUPartitionedCallOp` and/or
  `batch_function()`. Alternatively use `inference_on_tpu()` which is a
  convenience wrapper of the three.

  ```
    def model_fn(features, labels, mode, config, params):
      ...
      # This could be some pre-processing on CPU like calls to input layer with
      # embedding columns.
      x2 = features['x'] * 2

      def computation(input_tensor):
        return layers.dense(
            input_tensor, 1, kernel_initializer=init_ops.zeros_initializer())

      inputs = [x2]
      if params['use_tpu']:
        predictions = array_ops.identity(
            tpu_estimator.inference_on_tpu(computation, inputs,
            num_batch_threads=1, max_batch_size=2, batch_timeout_micros=100),
            name='predictions')
      else:
        predictions = array_ops.identity(
            computation(*inputs), name='predictions')
      key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
      export_outputs = {
          key: export_lib.PredictOutput({'prediction': predictions})
      }
      ...
  ```

  TIP: V2 is recommended as it is more flexible (eg: batching, etc).
  "
  [model_fn model_dir config params & {:keys [use_tpu train_batch_size eval_batch_size predict_batch_size batch_axis eval_on_tpu export_to_tpu export_to_cpu warm_start_from embedding_config_spec export_saved_model_api_version]
                       :or {train_batch_size None eval_batch_size None predict_batch_size None batch_axis None warm_start_from None embedding_config_spec None}} ]
    (py/call-attr-kw tpu "TPUEstimator" [model_fn model_dir config params] {:use_tpu use_tpu :train_batch_size train_batch_size :eval_batch_size eval_batch_size :predict_batch_size predict_batch_size :batch_axis batch_axis :eval_on_tpu eval_on_tpu :export_to_tpu export_to_tpu :export_to_cpu export_to_cpu :warm_start_from warm_start_from :embedding_config_spec embedding_config_spec :export_saved_model_api_version export_saved_model_api_version }))

(defn config 
  ""
  [ self ]
    (py/call-attr self "config"))

(defn eval-dir 
  "Shows the directory name where evaluation metrics are dumped.

    Args:
      name: Name of the evaluation if user needs to run multiple evaluations on
        different data sets, such as on training data vs test data. Metrics for
        different evaluations are saved in separate folders, and appear
        separately in tensorboard.

    Returns:
      A string which is the path of directory contains evaluation metrics.
    "
  [ self name ]
  (py/call-attr self "eval_dir"  self name ))

(defn evaluate 
  ""
  [ self input_fn steps hooks checkpoint_path name ]
  (py/call-attr self "evaluate"  self input_fn steps hooks checkpoint_path name ))

(defn experimental-export-all-saved-models 
  "Exports a `SavedModel` with `tf.MetaGraphDefs` for each requested mode.

    For each mode passed in via the `input_receiver_fn_map`,
    this method builds a new graph by calling the `input_receiver_fn` to obtain
    feature and label `Tensor`s. Next, this method calls the `Estimator`'s
    `model_fn` in the passed mode to generate the model graph based on
    those features and labels, and restores the given checkpoint
    (or, lacking that, the most recent checkpoint) into the graph.
    Only one of the modes is used for saving variables to the `SavedModel`
    (order of preference: `tf.estimator.ModeKeys.TRAIN`,
    `tf.estimator.ModeKeys.EVAL`, then
    `tf.estimator.ModeKeys.PREDICT`), such that up to three
    `tf.MetaGraphDefs` are saved with a single set of variables in a single
    `SavedModel` directory.

    For the variables and `tf.MetaGraphDefs`, a timestamped export directory
    below
    `export_dir_base`, and writes a `SavedModel` into it containing
    the `tf.MetaGraphDef` for the given mode and its associated signatures.

    For prediction, the exported `MetaGraphDef` will provide one `SignatureDef`
    for each element of the `export_outputs` dict returned from the `model_fn`,
    named using the same keys.  One of these keys is always
    `tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY`,
    indicating which
    signature will be served when a serving request does not specify one.
    For each signature, the outputs are provided by the corresponding
    `tf.estimator.export.ExportOutput`s, and the inputs are always the input
    receivers provided by
    the `serving_input_receiver_fn`.

    For training and evaluation, the `train_op` is stored in an extra
    collection,
    and loss, metrics, and predictions are included in a `SignatureDef` for the
    mode in question.

    Extra assets may be written into the `SavedModel` via the `assets_extra`
    argument.  This should be a dict, where each key gives a destination path
    (including the filename) relative to the assets.extra directory.  The
    corresponding value gives the full path of the source file to be copied.
    For example, the simple case of copying a single file without renaming it
    is specified as `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.

    Args:
      export_dir_base: A string containing a directory in which to create
        timestamped subdirectories containing exported `SavedModel`s.
      input_receiver_fn_map: dict of `tf.estimator.ModeKeys` to
        `input_receiver_fn` mappings, where the `input_receiver_fn` is a
        function that takes no arguments and returns the appropriate subclass of
        `InputReceiver`.
      assets_extra: A dict specifying how to populate the assets.extra directory
        within the exported `SavedModel`, or `None` if no extra assets are
        needed.
      as_text: whether to write the `SavedModel` proto in text format.
      checkpoint_path: The checkpoint path to export.  If `None` (the default),
        the most recent checkpoint found within the model directory is chosen.

    Returns:
      The string path to the exported directory.

    Raises:
      ValueError: if any `input_receiver_fn` is `None`, no `export_outputs`
        are provided, or no checkpoint can be found.
    "
  [self export_dir_base input_receiver_fn_map assets_extra & {:keys [as_text checkpoint_path]
                       :or {checkpoint_path None}} ]
    (py/call-attr-kw self "experimental_export_all_saved_models" [export_dir_base input_receiver_fn_map assets_extra] {:as_text as_text :checkpoint_path checkpoint_path }))

(defn export-saved-model 
  "Exports inference graph as a `SavedModel` into the given dir.

    For a detailed guide, see
    [Using SavedModel with Estimators](https://tensorflow.org/guide/saved_model#using_savedmodel_with_estimators).

    This method builds a new graph by first calling the
    `serving_input_receiver_fn` to obtain feature `Tensor`s, and then calling
    this `Estimator`'s `model_fn` to generate the model graph based on those
    features. It restores the given checkpoint (or, lacking that, the most
    recent checkpoint) into this graph in a fresh session.  Finally it creates
    a timestamped export directory below the given `export_dir_base`, and writes
    a `SavedModel` into it containing a single `tf.MetaGraphDef` saved from this
    session.

    The exported `MetaGraphDef` will provide one `SignatureDef` for each
    element of the `export_outputs` dict returned from the `model_fn`, named
    using
    the same keys.  One of these keys is always
    `tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY`,
    indicating which
    signature will be served when a serving request does not specify one.
    For each signature, the outputs are provided by the corresponding
    `tf.estimator.export.ExportOutput`s, and the inputs are always the input
    receivers provided by
    the `serving_input_receiver_fn`.

    Extra assets may be written into the `SavedModel` via the `assets_extra`
    argument.  This should be a dict, where each key gives a destination path
    (including the filename) relative to the assets.extra directory.  The
    corresponding value gives the full path of the source file to be copied.
    For example, the simple case of copying a single file without renaming it
    is specified as `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.

    The experimental_mode parameter can be used to export a single
    train/eval/predict graph as a `SavedModel`.
    See `experimental_export_all_saved_models` for full docs.

    Args:
      export_dir_base: A string containing a directory in which to create
        timestamped subdirectories containing exported `SavedModel`s.
      serving_input_receiver_fn: A function that takes no argument and returns a
        `tf.estimator.export.ServingInputReceiver` or
        `tf.estimator.export.TensorServingInputReceiver`.
      assets_extra: A dict specifying how to populate the assets.extra directory
        within the exported `SavedModel`, or `None` if no extra assets are
        needed.
      as_text: whether to write the `SavedModel` proto in text format.
      checkpoint_path: The checkpoint path to export.  If `None` (the default),
        the most recent checkpoint found within the model directory is chosen.
      experimental_mode: `tf.estimator.ModeKeys` value indicating with mode
        will be exported. Note that this feature is experimental.

    Returns:
      The string path to the exported directory.

    Raises:
      ValueError: if no `serving_input_receiver_fn` is provided, no
      `export_outputs` are provided, or no checkpoint can be found.
    "
  [self export_dir_base serving_input_receiver_fn assets_extra & {:keys [as_text checkpoint_path experimental_mode]
                       :or {checkpoint_path None}} ]
    (py/call-attr-kw self "export_saved_model" [export_dir_base serving_input_receiver_fn assets_extra] {:as_text as_text :checkpoint_path checkpoint_path :experimental_mode experimental_mode }))

(defn export-savedmodel 
  "Exports inference graph as a `SavedModel` into the given dir. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
This function has been renamed, use `export_saved_model` instead.

For a detailed guide, see
[Using SavedModel with Estimators](https://tensorflow.org/guide/saved_model#using_savedmodel_with_estimators).

This method builds a new graph by first calling the
`serving_input_receiver_fn` to obtain feature `Tensor`s, and then calling
this `Estimator`'s `model_fn` to generate the model graph based on those
features. It restores the given checkpoint (or, lacking that, the most
recent checkpoint) into this graph in a fresh session.  Finally it creates
a timestamped export directory below the given `export_dir_base`, and writes
a `SavedModel` into it containing a single `tf.MetaGraphDef` saved from this
session.

The exported `MetaGraphDef` will provide one `SignatureDef` for each
element of the `export_outputs` dict returned from the `model_fn`, named
using
the same keys.  One of these keys is always
`tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY`,
indicating which
signature will be served when a serving request does not specify one.
For each signature, the outputs are provided by the corresponding
`tf.estimator.export.ExportOutput`s, and the inputs are always the input
receivers provided by
the `serving_input_receiver_fn`.

Extra assets may be written into the `SavedModel` via the `assets_extra`
argument.  This should be a dict, where each key gives a destination path
(including the filename) relative to the assets.extra directory.  The
corresponding value gives the full path of the source file to be copied.
For example, the simple case of copying a single file without renaming it
is specified as `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.

Args:
  export_dir_base: A string containing a directory in which to create
    timestamped subdirectories containing exported `SavedModel`s.
  serving_input_receiver_fn: A function that takes no argument and returns a
    `tf.estimator.export.ServingInputReceiver` or
    `tf.estimator.export.TensorServingInputReceiver`.
  assets_extra: A dict specifying how to populate the assets.extra directory
    within the exported `SavedModel`, or `None` if no extra assets are
    needed.
  as_text: whether to write the `SavedModel` proto in text format.
  checkpoint_path: The checkpoint path to export.  If `None` (the default),
    the most recent checkpoint found within the model directory is chosen.
  strip_default_attrs: Boolean. If `True`, default-valued attributes will be
    removed from the `NodeDef`s. For a detailed guide, see [Stripping
    Default-Valued Attributes](
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).

Returns:
  The string path to the exported directory.

Raises:
  ValueError: if no `serving_input_receiver_fn` is provided, no
  `export_outputs` are provided, or no checkpoint can be found."
  [self export_dir_base serving_input_receiver_fn assets_extra & {:keys [as_text checkpoint_path strip_default_attrs]
                       :or {checkpoint_path None}} ]
    (py/call-attr-kw self "export_savedmodel" [export_dir_base serving_input_receiver_fn assets_extra] {:as_text as_text :checkpoint_path checkpoint_path :strip_default_attrs strip_default_attrs }))

(defn get-variable-names 
  "Returns list of all variable names in this model.

    Returns:
      List of names.

    Raises:
      ValueError: If the `Estimator` has not produced a checkpoint yet.
    "
  [ self  ]
  (py/call-attr self "get_variable_names"  self  ))

(defn get-variable-value 
  "Returns value of the variable given by name.

    Args:
      name: string or a list of string, name of the tensor.

    Returns:
      Numpy array - value of the tensor.

    Raises:
      ValueError: If the `Estimator` has not produced a checkpoint yet.
    "
  [ self name ]
  (py/call-attr self "get_variable_value"  self name ))

(defn latest-checkpoint 
  "Finds the filename of the latest saved checkpoint file in `model_dir`.

    Returns:
      The full path to the latest checkpoint or `None` if no checkpoint was
      found.
    "
  [ self  ]
  (py/call-attr self "latest_checkpoint"  self  ))

(defn model-dir 
  ""
  [ self ]
    (py/call-attr self "model_dir"))

(defn model-fn 
  "Returns the `model_fn` which is bound to `self.params`.

    Returns:
      The `model_fn` with following signature:
        `def model_fn(features, labels, mode, config)`
    "
  [ self ]
    (py/call-attr self "model_fn"))

(defn params 
  ""
  [ self ]
    (py/call-attr self "params"))
(defn predict 
  ""
  [self input_fn predict_keys hooks checkpoint_path  & {:keys [yield_single_examples]} ]
    (py/call-attr-kw self "predict" [input_fn predict_keys hooks checkpoint_path] {:yield_single_examples yield_single_examples }))

(defn train 
  ""
  [ self input_fn hooks steps max_steps saving_listeners ]
  (py/call-attr self "train"  self input_fn hooks steps max_steps saving_listeners ))
