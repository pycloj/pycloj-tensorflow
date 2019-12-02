(ns tensorflow-estimator.python.estimator.api.-v1.estimator.experimental.KMeans
  "An Estimator for K-Means clustering.

  Example:
  ```
  import numpy as np
  import tensorflow as tf

  num_points = 100
  dimensions = 2
  points = np.random.uniform(0, 1000, [num_points, dimensions])

  def input_fn():
    return tf.train.limit_epochs(
        tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=1)

  num_clusters = 5
  kmeans = tf.estimator.experimental.KMeans(
      num_clusters=num_clusters, use_mini_batch=False)

  # train
  num_iterations = 10
  previous_centers = None
  for _ in xrange(num_iterations):
    kmeans.train(input_fn)
    cluster_centers = kmeans.cluster_centers()
    if previous_centers is not None:
      print 'delta:', cluster_centers - previous_centers
    previous_centers = cluster_centers
    print 'score:', kmeans.score(input_fn)
  print 'cluster centers:', cluster_centers

  # map the input points to their clusters
  cluster_indices = list(kmeans.predict_cluster_index(input_fn))
  for i, point in enumerate(points):
    cluster_index = cluster_indices[i]
    center = cluster_centers[cluster_index]
    print 'point:', point, 'is in cluster', cluster_index, 'centered at', center
  ```

  The `SavedModel` saved by the `export_saved_model` method does not include the
  cluster centers. However, the cluster centers may be retrieved by the
  latest checkpoint saved during training. Specifically,
  ```
  kmeans.cluster_centers()
  ```
  is equivalent to
  ```
  tf.train.load_variable(
      kmeans.model_dir, KMeansClustering.CLUSTER_CENTERS_VAR_NAME)
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
(defonce experimental (import-module "tensorflow_estimator.python.estimator.api._v1.estimator.experimental"))

(defn KMeans 
  "An Estimator for K-Means clustering.

  Example:
  ```
  import numpy as np
  import tensorflow as tf

  num_points = 100
  dimensions = 2
  points = np.random.uniform(0, 1000, [num_points, dimensions])

  def input_fn():
    return tf.train.limit_epochs(
        tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=1)

  num_clusters = 5
  kmeans = tf.estimator.experimental.KMeans(
      num_clusters=num_clusters, use_mini_batch=False)

  # train
  num_iterations = 10
  previous_centers = None
  for _ in xrange(num_iterations):
    kmeans.train(input_fn)
    cluster_centers = kmeans.cluster_centers()
    if previous_centers is not None:
      print 'delta:', cluster_centers - previous_centers
    previous_centers = cluster_centers
    print 'score:', kmeans.score(input_fn)
  print 'cluster centers:', cluster_centers

  # map the input points to their clusters
  cluster_indices = list(kmeans.predict_cluster_index(input_fn))
  for i, point in enumerate(points):
    cluster_index = cluster_indices[i]
    center = cluster_centers[cluster_index]
    print 'point:', point, 'is in cluster', cluster_index, 'centered at', center
  ```

  The `SavedModel` saved by the `export_saved_model` method does not include the
  cluster centers. However, the cluster centers may be retrieved by the
  latest checkpoint saved during training. Specifically,
  ```
  kmeans.cluster_centers()
  ```
  is equivalent to
  ```
  tf.train.load_variable(
      kmeans.model_dir, KMeansClustering.CLUSTER_CENTERS_VAR_NAME)
  ```
  "
  [num_clusters model_dir & {:keys [initial_clusters distance_metric seed use_mini_batch mini_batch_steps_per_iteration kmeans_plus_plus_num_retries relative_tolerance config feature_columns]
                       :or {seed None relative_tolerance None config None feature_columns None}} ]
    (py/call-attr-kw experimental "KMeans" [num_clusters model_dir] {:initial_clusters initial_clusters :distance_metric distance_metric :seed seed :use_mini_batch use_mini_batch :mini_batch_steps_per_iteration mini_batch_steps_per_iteration :kmeans_plus_plus_num_retries kmeans_plus_plus_num_retries :relative_tolerance relative_tolerance :config config :feature_columns feature_columns }))

(defn cluster-centers 
  "Returns the cluster centers."
  [ self  ]
  (py/call-attr self "cluster_centers"  self  ))

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
  "Evaluates the model given evaluation data `input_fn`.

    For each step, calls `input_fn`, which returns one batch of data.
    Evaluates until:
    - `steps` batches are processed, or
    - `input_fn` raises an end-of-input exception (`tf.errors.OutOfRangeError`
    or
    `StopIteration`).

    Args:
      input_fn: A function that constructs the input data for evaluation. See
        [Premade Estimators](
        https://tensorflow.org/guide/premade_estimators#create_input_functions)
        for more information. The
        function should construct and return one of the following:  * A
        `tf.data.Dataset` object: Outputs of `Dataset` object must be a tuple
        `(features, labels)` with same constraints as below. * A tuple
        `(features, labels)`: Where `features` is a `tf.Tensor` or a dictionary
        of string feature name to `Tensor` and `labels` is a `Tensor` or a
        dictionary of string label name to `Tensor`. Both `features` and
        `labels` are consumed by `model_fn`. They should satisfy the expectation
        of `model_fn` from inputs.
      steps: Number of steps for which to evaluate model. If `None`, evaluates
        until `input_fn` raises an end-of-input exception.
      hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
        callbacks inside the evaluation call.
      checkpoint_path: Path of a specific checkpoint to evaluate. If `None`, the
        latest checkpoint in `model_dir` is used.  If there are no checkpoints
        in `model_dir`, evaluation is run with newly initialized `Variables`
        instead of ones restored from checkpoint.
      name: Name of the evaluation if user needs to run multiple evaluations on
        different data sets, such as on training data vs test data. Metrics for
        different evaluations are saved in separate folders, and appear
        separately in tensorboard.

    Returns:
      A dict containing the evaluation metrics specified in `model_fn` keyed by
      name, as well as an entry `global_step` which contains the value of the
      global step for which this evaluation was performed. For canned
      estimators, the dict contains the `loss` (mean loss per mini-batch) and
      the `average_loss` (mean loss per sample). Canned classifiers also return
      the `accuracy`. Canned regressors also return the `label/mean` and the
      `prediction/mean`.

    Raises:
      ValueError: If `steps <= 0`.
    "
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
  "Yields predictions for given features.

    Please note that interleaving two predict outputs does not work. See:
    [issue/20506](
    https://github.com/tensorflow/tensorflow/issues/20506#issuecomment-422208517)

    Args:
      input_fn: A function that constructs the features. Prediction continues
        until `input_fn` raises an end-of-input exception
        (`tf.errors.OutOfRangeError` or `StopIteration`).
        See [Premade Estimators](
        https://tensorflow.org/guide/premade_estimators#create_input_functions)
        for more information. The function should construct and return one of
        the following:

          * A `tf.data.Dataset` object: Outputs of `Dataset` object must have
            same constraints as below.
          * features: A `tf.Tensor` or a dictionary of string feature name to
            `Tensor`. features are consumed by `model_fn`. They should satisfy
            the expectation of `model_fn` from inputs.
          * A tuple, in which case the first item is extracted as features.

      predict_keys: list of `str`, name of the keys to predict. It is used if
        the `tf.estimator.EstimatorSpec.predictions` is a `dict`. If
        `predict_keys` is used then rest of the predictions will be filtered
        from the dictionary. If `None`, returns all.
      hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
        callbacks inside the prediction call.
      checkpoint_path: Path of a specific checkpoint to predict. If `None`, the
        latest checkpoint in `model_dir` is used.  If there are no checkpoints
        in `model_dir`, prediction is run with newly initialized `Variables`
        instead of ones restored from checkpoint.
      yield_single_examples: If `False`, yields the whole batch as returned by
        the `model_fn` instead of decomposing the batch into individual
        elements. This is useful if `model_fn` returns some tensors whose first
        dimension is not equal to the batch size.

    Yields:
      Evaluated values of `predictions` tensors.

    Raises:
      ValueError: If batch length of predictions is not the same and
        `yield_single_examples` is `True`.
      ValueError: If there is a conflict between `predict_keys` and
        `predictions`. For example if `predict_keys` is not `None` but
        `tf.estimator.EstimatorSpec.predictions` is not a `dict`.
    "
  [self input_fn predict_keys hooks checkpoint_path  & {:keys [yield_single_examples]} ]
    (py/call-attr-kw self "predict" [input_fn predict_keys hooks checkpoint_path] {:yield_single_examples yield_single_examples }))

(defn predict-cluster-index 
  "Finds the index of the closest cluster center to each input point.

    Args:
      input_fn: Input points. See `tf.estimator.Estimator.predict`.

    Yields:
      The index of the closest cluster center for each input point.
    "
  [ self input_fn ]
  (py/call-attr self "predict_cluster_index"  self input_fn ))

(defn score 
  "Returns the sum of squared distances to nearest clusters.

    Note that this function is different from the corresponding one in sklearn
    which returns the negative sum.

    Args:
      input_fn: Input points. See `tf.estimator.Estimator.evaluate`. Only one
          batch is retrieved.

    Returns:
      The sum of the squared distance from each point in the first batch of
      inputs to its nearest cluster center.
    "
  [ self input_fn ]
  (py/call-attr self "score"  self input_fn ))

(defn train 
  "Trains a model given training data `input_fn`.

    Args:
      input_fn: A function that provides input data for training as minibatches.
        See [Premade Estimators](
        https://tensorflow.org/guide/premade_estimators#create_input_functions)
        for more information. The function should construct and return one of
        the following:
          * A `tf.data.Dataset` object: Outputs of `Dataset` object must be
            a tuple `(features, labels)` with same constraints as below.
          * A tuple `(features, labels)`: Where `features` is a `tf.Tensor` or
            a dictionary of string feature name to `Tensor` and `labels` is a
            `Tensor` or a dictionary of string label name to `Tensor`. Both
            `features` and `labels` are consumed by `model_fn`. They should
            satisfy the expectation of `model_fn` from inputs.
      hooks: List of `tf.train.SessionRunHook` subclass instances. Used for
        callbacks inside the training loop.
      steps: Number of steps for which to train the model. If `None`, train
        forever or train until `input_fn` generates the `tf.errors.OutOfRange`
        error or `StopIteration` exception. `steps` works incrementally. If you
        call two times `train(steps=10)` then training occurs in total 20 steps.
        If `OutOfRange` or `StopIteration` occurs in the middle, training stops
        before 20 steps. If you don't want to have incremental behavior please
        set `max_steps` instead. If set, `max_steps` must be `None`.
      max_steps: Number of total steps for which to train model. If `None`,
        train forever or train until `input_fn` generates the
        `tf.errors.OutOfRange` error or `StopIteration` exception. If set,
        `steps` must be `None`. If `OutOfRange` or `StopIteration` occurs in the
        middle, training stops before `max_steps` steps. Two calls to
        `train(steps=100)` means 200 training iterations. On the other hand, two
        calls to `train(max_steps=100)` means that the second call will not do
        any iteration since first call did all 100 steps.
      saving_listeners: list of `CheckpointSaverListener` objects. Used for
        callbacks that run immediately before or after checkpoint savings.

    Returns:
      `self`, for chaining.

    Raises:
      ValueError: If both `steps` and `max_steps` are not `None`.
      ValueError: If either `steps` or `max_steps <= 0`.
    "
  [ self input_fn hooks steps max_steps saving_listeners ]
  (py/call-attr self "train"  self input_fn hooks steps max_steps saving_listeners ))

(defn transform 
  "Transforms each input point to its distances to all cluster centers.

    Note that if `distance_metric=KMeansClustering.SQUARED_EUCLIDEAN_DISTANCE`,
    this
    function returns the squared Euclidean distance while the corresponding
    sklearn function returns the Euclidean distance.

    Args:
      input_fn: Input points. See `tf.estimator.Estimator.predict`.

    Yields:
      The distances from each input point to each cluster center.
    "
  [ self input_fn ]
  (py/call-attr self "transform"  self input_fn ))