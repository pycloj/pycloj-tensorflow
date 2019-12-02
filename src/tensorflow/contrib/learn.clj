(ns tensorflow.contrib.learn
  "High level API for learning (DEPRECATED).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.

@@BaseEstimator
@@Estimator
@@Trainable
@@Evaluable
@@KMeansClustering
@@ModeKeys
@@ModelFnOps
@@MetricSpec
@@PredictionKey
@@DNNClassifier
@@DNNEstimator
@@DNNRegressor
@@DNNLinearCombinedRegressor
@@DNNLinearCombinedEstimator
@@DNNLinearCombinedClassifier
@@DynamicRnnEstimator
@@LinearClassifier
@@LinearEstimator
@@LinearRegressor
@@LogisticRegressor
@@StateSavingRnnEstimator
@@SVM
@@SKCompat

@@Head
@@multi_class_head
@@multi_label_head
@@binary_svm_head
@@regression_head
@@poisson_regression_head
@@multi_head
@@no_op_train_fn

@@Experiment
@@ExportStrategy
@@TaskType

@@NanLossDuringTrainingError
@@RunConfig
@@evaluate
@@infer
@@run_feeds
@@run_n
@@train

@@extract_dask_data
@@extract_dask_labels
@@extract_pandas_data
@@extract_pandas_labels
@@extract_pandas_matrix
@@infer_real_valued_columns_from_input
@@infer_real_valued_columns_from_input_fn
@@read_batch_examples
@@read_batch_features
@@read_batch_record_features
@@read_keyed_batch_examples
@@read_keyed_batch_examples_shared_queue
@@read_keyed_batch_features
@@read_keyed_batch_features_shared_queue

@@InputFnOps
@@ProblemType
@@build_parsing_serving_input_fn
@@make_export_strategy
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

(defn LogisticRegressor 
  "Builds a logistic regression Estimator for binary classification.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  This method provides a basic Estimator with some additional metrics for custom
  binary classification models, including AUC, precision/recall and accuracy.

  Example:

  ```python
    # See tf.contrib.learn.Estimator(...) for details on model_fn structure
    def my_model_fn(...):
      pass

    estimator = LogisticRegressor(model_fn=my_model_fn)

    # Input builders
    def input_fn_train:
      pass

    estimator.fit(input_fn=input_fn_train)
    estimator.predict(x=x)
  ```

  Args:
    model_fn: Model function with the signature:
      `(features, labels, mode) -> (predictions, loss, train_op)`.
      Expects the returned predictions to be probabilities in [0.0, 1.0].
    thresholds: List of floating point thresholds to use for accuracy,
      precision, and recall metrics. If `None`, defaults to `[0.5]`.
    model_dir: Directory to save model parameters, graphs, etc. This can also
      be used to load checkpoints from the directory into a estimator to
      continue training a previously saved model.
    config: A RunConfig configuration object.
    feature_engineering_fn: Feature engineering function. Takes features and
                      labels which are the output of `input_fn` and
                      returns features and labels which will be fed
                      into the model.

  Returns:
    An `Estimator` instance.
  "
  [ model_fn thresholds model_dir config feature_engineering_fn ]
  (py/call-attr learn "LogisticRegressor"  model_fn thresholds model_dir config feature_engineering_fn ))

(defn binary-svm-head 
  "Creates a `Head` for binary classification with SVMs. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please switch to tf.contrib.estimator.*_head.

The head uses binary hinge loss.

Args:
  label_name: String, name of the key in label dict. Can be null if label
    is a tensor (single headed models).
  weight_column_name: A string defining feature column name representing
    weights. It is used to down weight or boost examples during training. It
    will be multiplied by the loss of the example.
  enable_centered_bias: A bool. If True, estimator will learn a centered
    bias variable for each class. Rest of the model structure learns the
    residual after centered bias.
  head_name: name of the head. If provided, predictions, summary and metrics
    keys will be suffixed by `\"/\" + head_name` and the default variable scope
    will be `head_name`.
  thresholds: thresholds for eval metrics, defaults to [.5]

Returns:
  An instance of `Head` for binary classification with SVM."
  [label_name weight_column_name & {:keys [enable_centered_bias head_name thresholds]
                       :or {head_name None thresholds None}} ]
    (py/call-attr-kw learn "binary_svm_head" [label_name weight_column_name] {:enable_centered_bias enable_centered_bias :head_name head_name :thresholds thresholds }))

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
  (py/call-attr learn "build_parsing_serving_input_fn"  feature_spec default_batch_size ))

(defn evaluate 
  "Evaluate a model loaded from a checkpoint. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2017-02-15.
Instructions for updating:
graph_actions.py will be deleted. Use tf.train.* utilities instead. You can use learn/estimators/estimator.py as an example.

Given `graph`, a directory to write summaries to (`output_dir`), a checkpoint
to restore variables from, and a `dict` of `Tensor`s to evaluate, run an eval
loop for `max_steps` steps, or until an exception (generally, an
end-of-input signal from a reader operation) is raised from running
`eval_dict`.

In each step of evaluation, all tensors in the `eval_dict` are evaluated, and
every `log_every_steps` steps, they are logged. At the very end of evaluation,
a summary is evaluated (finding the summary ops using `Supervisor`'s logic)
and written to `output_dir`.

Args:
  graph: A `Graph` to train. It is expected that this graph is not in use
    elsewhere.
  output_dir: A string containing the directory to write a summary to.
  checkpoint_path: A string containing the path to a checkpoint to restore.
    Can be `None` if the graph doesn't require loading any variables.
  eval_dict: A `dict` mapping string names to tensors to evaluate. It is
    evaluated in every logging step. The result of the final evaluation is
    returned. If `update_op` is None, then it's evaluated in every step. If
    `max_steps` is `None`, this should depend on a reader that will raise an
    end-of-input exception when the inputs are exhausted.
  update_op: A `Tensor` which is run in every step.
  global_step_tensor: A `Variable` containing the global step. If `None`,
    one is extracted from the graph using the same logic as in `Supervisor`.
    Used to place eval summaries on training curves.
  supervisor_master: The master string to use when preparing the session.
  log_every_steps: Integer. Output logs every `log_every_steps` evaluation
    steps. The logs contain the `eval_dict` and timing information.
  feed_fn: A function that is called every iteration to produce a `feed_dict`
    passed to `session.run` calls. Optional.
  max_steps: Integer. Evaluate `eval_dict` this many times.

Returns:
  A tuple `(eval_results, global_step)`:
  eval_results: A `dict` mapping `string` to numeric values (`int`, `float`)
    that are the result of running eval_dict in the last step. `None` if no
    eval steps were run.
  global_step: The global step this evaluation corresponds to.

Raises:
  ValueError: if `output_dir` is empty."
  [graph output_dir checkpoint_path eval_dict update_op global_step_tensor & {:keys [supervisor_master log_every_steps feed_fn max_steps]
                       :or {feed_fn None max_steps None}} ]
    (py/call-attr-kw learn "evaluate" [graph output_dir checkpoint_path eval_dict update_op global_step_tensor] {:supervisor_master supervisor_master :log_every_steps log_every_steps :feed_fn feed_fn :max_steps max_steps }))

(defn extract-dask-data 
  "Extract data from dask.Series or dask.DataFrame for predictors. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please feed input to tf.data to support dask.

Given a distributed dask.DataFrame or dask.Series containing columns or names
for one or more predictors, this operation returns a single dask.DataFrame or
dask.Series that can be iterated over.

Args:
  data: A distributed dask.DataFrame or dask.Series.

Returns:
  A dask.DataFrame or dask.Series that can be iterated over.
  If the supplied argument is neither a dask.DataFrame nor a dask.Series this
  operation returns it without modification."
  [ data ]
  (py/call-attr learn "extract_dask_data"  data ))

(defn extract-dask-labels 
  "Extract data from dask.Series or dask.DataFrame for labels. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please feed input to tf.data to support dask.

Given a distributed dask.DataFrame or dask.Series containing exactly one
column or name, this operation returns a single dask.DataFrame or dask.Series
that can be iterated over.

Args:
  labels: A distributed dask.DataFrame or dask.Series with exactly one
          column or name.

Returns:
  A dask.DataFrame or dask.Series that can be iterated over.
  If the supplied argument is neither a dask.DataFrame nor a dask.Series this
  operation returns it without modification.

Raises:
  ValueError: If the supplied dask.DataFrame contains more than one
              column or the supplied dask.Series contains more than
              one name."
  [ labels ]
  (py/call-attr learn "extract_dask_labels"  labels ))

(defn extract-pandas-data 
  "Extract data from pandas.DataFrame for predictors. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please access pandas data directly.

Given a DataFrame, will extract the values and cast them to float. The
DataFrame is expected to contain values of type int, float or bool.

Args:
  data: `pandas.DataFrame` containing the data to be extracted.

Returns:
  A numpy `ndarray` of the DataFrame's values as floats.

Raises:
  ValueError: if data contains types other than int, float or bool."
  [ data ]
  (py/call-attr learn "extract_pandas_data"  data ))

(defn extract-pandas-labels 
  "Extract data from pandas.DataFrame for labels. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please access pandas data directly.

Args:
  labels: `pandas.DataFrame` or `pandas.Series` containing one column of
    labels to be extracted.

Returns:
  A numpy `ndarray` of labels from the DataFrame.

Raises:
  ValueError: if more than one column is found or type is not int, float or
    bool."
  [ labels ]
  (py/call-attr learn "extract_pandas_labels"  labels ))

(defn extract-pandas-matrix 
  "Extracts numpy matrix from pandas DataFrame. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please access pandas data directly.

Args:
  data: `pandas.DataFrame` containing the data to be extracted.

Returns:
  A numpy `ndarray` of the DataFrame's values."
  [ data ]
  (py/call-attr learn "extract_pandas_matrix"  data ))

(defn infer 
  "Restore graph from `restore_checkpoint_path` and run `output_dict` tensors. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2017-02-15.
Instructions for updating:
graph_actions.py will be deleted. Use tf.train.* utilities instead. You can use learn/estimators/estimator.py as an example.

If `restore_checkpoint_path` is supplied, restore from checkpoint. Otherwise,
init all variables.

Args:
  restore_checkpoint_path: A string containing the path to a checkpoint to
    restore.
  output_dict: A `dict` mapping string names to `Tensor` objects to run.
    Tensors must all be from the same graph.
  feed_dict: `dict` object mapping `Tensor` objects to input values to feed.

Returns:
  Dict of values read from `output_dict` tensors. Keys are the same as
  `output_dict`, values are the results read from the corresponding `Tensor`
  in `output_dict`.

Raises:
  ValueError: if `output_dict` or `feed_dicts` is None or empty."
  [ restore_checkpoint_path output_dict feed_dict ]
  (py/call-attr learn "infer"  restore_checkpoint_path output_dict feed_dict ))

(defn infer-real-valued-columns-from-input 
  "Creates `FeatureColumn` objects for inputs defined by input `x`. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please specify feature columns explicitly.

This interprets all inputs as dense, fixed-length float values.

Args:
  x: Real-valued matrix of shape [n_samples, n_features...]. Can be
     iterator that returns arrays of features.

Returns:
  List of `FeatureColumn` objects."
  [ x ]
  (py/call-attr learn "infer_real_valued_columns_from_input"  x ))

(defn infer-real-valued-columns-from-input-fn 
  "Creates `FeatureColumn` objects for inputs defined by `input_fn`. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please specify feature columns explicitly.

This interprets all inputs as dense, fixed-length float values. This creates
a local graph in which it calls `input_fn` to build the tensors, then discards
it.

Args:
  input_fn: Input function returning a tuple of:
      features - Dictionary of string feature name to `Tensor` or `Tensor`.
      labels - `Tensor` of label values.

Returns:
  List of `FeatureColumn` objects."
  [ input_fn ]
  (py/call-attr learn "infer_real_valued_columns_from_input_fn"  input_fn ))

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
    (py/call-attr-kw learn "make_export_strategy" [serving_input_fn default_output_alternative_key assets_extra] {:as_text as_text :exports_to_keep exports_to_keep :strip_default_attrs strip_default_attrs }))

(defn multi-class-head 
  "Creates a `Head` for multi class single label classification. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please switch to tf.contrib.estimator.*_head.

The Head uses softmax cross entropy loss.

This head expects to be fed integer labels specifying the class index. But
if `label_keys` is specified, then labels must be strings from this
vocabulary, and the predicted classes will be strings from the same
vocabulary.

Args:
  n_classes: Integer, number of classes, must be >= 2
  label_name: String, name of the key in label dict. Can be null if label
      is a tensor (single headed models).
  weight_column_name: A string defining feature column name representing
    weights. It is used to down weight or boost examples during training. It
    will be multiplied by the loss of the example.
  enable_centered_bias: A bool. If True, estimator will learn a centered
    bias variable for each class. Rest of the model structure learns the
    residual after centered bias.
  head_name: name of the head. If provided, predictions, summary and metrics
    keys will be suffixed by `\"/\" + head_name` and the default variable scope
    will be `head_name`.
  thresholds: thresholds for eval metrics, defaults to [.5]
  metric_class_ids: List of class IDs for which we should report per-class
    metrics. Must all be in the range `[0, n_classes)`. Invalid if
    `n_classes` is 2.
  loss_fn: Optional function that takes (`labels`, `logits`, `weights`) as
    parameter and returns a weighted scalar loss. `weights` should be
    optional. See `tf.losses`
  label_keys: Optional list of strings with size `[n_classes]` defining the
    label vocabulary. Only supported for `n_classes` > 2.

Returns:
  An instance of `Head` for multi class classification.

Raises:
  ValueError: if `n_classes` is < 2.
  ValueError: If `metric_class_ids` is provided when `n_classes` is 2.
  ValueError: If `len(label_keys) != n_classes`."
  [n_classes label_name weight_column_name & {:keys [enable_centered_bias head_name thresholds metric_class_ids loss_fn label_keys]
                       :or {head_name None thresholds None metric_class_ids None loss_fn None label_keys None}} ]
    (py/call-attr-kw learn "multi_class_head" [n_classes label_name weight_column_name] {:enable_centered_bias enable_centered_bias :head_name head_name :thresholds thresholds :metric_class_ids metric_class_ids :loss_fn loss_fn :label_keys label_keys }))

(defn multi-head 
  "Creates a MultiHead stemming from same logits/hidden layer. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please switch to tf.contrib.estimator.*_head.

Args:
  heads: list of Head objects.
  loss_weights: optional list of weights to be used to merge losses from
      each head. All losses are weighted equally if not provided.

Returns:
  A instance of `Head` that merges multiple heads.

Raises:
  ValueError: if heads and loss_weights have different size."
  [ heads loss_weights ]
  (py/call-attr learn "multi_head"  heads loss_weights ))

(defn multi-label-head 
  "Creates a Head for multi label classification. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please switch to tf.contrib.estimator.*_head.

Multi-label classification handles the case where each example may have zero
or more associated labels, from a discrete set.  This is distinct from
`multi_class_head` which has exactly one label from a discrete set.

This head by default uses sigmoid cross entropy loss, which expects as input
a multi-hot tensor of shape `(batch_size, num_classes)`.

Args:
  n_classes: Integer, number of classes, must be >= 2
  label_name: String, name of the key in label dict. Can be null if label
      is a tensor (single headed models).
  weight_column_name: A string defining feature column name representing
    weights. It is used to down weight or boost examples during training. It
    will be multiplied by the loss of the example.
  enable_centered_bias: A bool. If True, estimator will learn a centered
    bias variable for each class. Rest of the model structure learns the
    residual after centered bias.
  head_name: name of the head. If provided, predictions, summary and metrics
    keys will be suffixed by `\"/\" + head_name` and the default variable scope
    will be `head_name`.
  thresholds: thresholds for eval metrics, defaults to [.5]
  metric_class_ids: List of class IDs for which we should report per-class
    metrics. Must all be in the range `[0, n_classes)`.
  loss_fn: Optional function that takes (`labels`, `logits`, `weights`) as
    parameter and returns a weighted scalar loss. `weights` should be
    optional. See `tf.losses`

Returns:
  An instance of `Head` for multi label classification.

Raises:
  ValueError: If n_classes is < 2
  ValueError: If loss_fn does not have expected signature."
  [n_classes label_name weight_column_name & {:keys [enable_centered_bias head_name thresholds metric_class_ids loss_fn]
                       :or {head_name None thresholds None metric_class_ids None loss_fn None}} ]
    (py/call-attr-kw learn "multi_label_head" [n_classes label_name weight_column_name] {:enable_centered_bias enable_centered_bias :head_name head_name :thresholds thresholds :metric_class_ids metric_class_ids :loss_fn loss_fn }))

(defn no-op-train-fn 
  "DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use 'lambda _: tf.no_op()'."
  [ loss ]
  (py/call-attr learn "no_op_train_fn"  loss ))

(defn poisson-regression-head 
  "Creates a `Head` for poisson regression. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please switch to tf.contrib.estimator.*_head.

Args:
  label_name: String, name of the key in label dict. Can be null if label
      is a tensor (single headed models).
  weight_column_name: A string defining feature column name representing
    weights. It is used to down weight or boost examples during training. It
    will be multiplied by the loss of the example.
  label_dimension: Number of regression labels per example. This is the size
    of the last dimension of the labels `Tensor` (typically, this has shape
    `[batch_size, label_dimension]`).
  enable_centered_bias: A bool. If True, estimator will learn a centered
    bias variable for each class. Rest of the model structure learns the
    residual after centered bias.
  head_name: name of the head. If provided, predictions, summary and metrics
    keys will be suffixed by `\"/\" + head_name` and the default variable scope
    will be `head_name`.

Returns:
  An instance of `Head` for poisson regression."
  [label_name weight_column_name & {:keys [label_dimension enable_centered_bias head_name]
                       :or {head_name None}} ]
    (py/call-attr-kw learn "poisson_regression_head" [label_name weight_column_name] {:label_dimension label_dimension :enable_centered_bias enable_centered_bias :head_name head_name }))

(defn read-batch-examples 
  "Adds operations to read, queue, batch `Example` protos. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.data.

Given file pattern (or list of files), will setup a queue for file names,
read `Example` proto using provided `reader`, use batch queue to create
batches of examples of size `batch_size`.

All queue runners are added to the queue runners collection, and may be
started via `start_queue_runners`.

All ops are added to the default graph.

Use `parse_fn` if you need to do parsing / processing on single examples.

Args:
  file_pattern: List of files or patterns of file paths containing
      `Example` records. See `tf.io.gfile.glob` for pattern rules.
  batch_size: An int or scalar `Tensor` specifying the batch size to use.
  reader: A function or class that returns an object with
    `read` method, (filename tensor) -> (example tensor).
  randomize_input: Whether the input should be randomized.
  num_epochs: Integer specifying the number of times to read through the
    dataset. If `None`, cycles through the dataset forever.
    NOTE - If specified, creates a variable that must be initialized, so call
    `tf.compat.v1.local_variables_initializer()` and run the op in a session.
  queue_capacity: Capacity for input queue.
  num_threads: The number of threads enqueuing examples. In order to have
    predictable and repeatable order of reading and enqueueing, such as in
    prediction and evaluation mode, `num_threads` should be 1.
  read_batch_size: An int or scalar `Tensor` specifying the number of
    records to read at once.
  parse_fn: Parsing function, takes `Example` Tensor returns parsed
    representation. If `None`, no parsing is done.
  name: Name of resulting op.
  seed: An integer (optional). Seed used if randomize_input == True.

Returns:
  String `Tensor` of batched `Example` proto.

Raises:
  ValueError: for invalid inputs."
  [file_pattern batch_size reader & {:keys [randomize_input num_epochs queue_capacity num_threads read_batch_size parse_fn name seed]
                       :or {num_epochs None parse_fn None name None seed None}} ]
    (py/call-attr-kw learn "read_batch_examples" [file_pattern batch_size reader] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :num_threads num_threads :read_batch_size read_batch_size :parse_fn parse_fn :name name :seed seed }))

(defn read-batch-features 
  "Adds operations to read, queue, batch and parse `Example` protos. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.data.

Given file pattern (or list of files), will setup a queue for file names,
read `Example` proto using provided `reader`, use batch queue to create
batches of examples of size `batch_size` and parse example given `features`
specification.

All queue runners are added to the queue runners collection, and may be
started via `start_queue_runners`.

All ops are added to the default graph.

Args:
  file_pattern: List of files or patterns of file paths containing
      `Example` records. See `tf.io.gfile.glob` for pattern rules.
  batch_size: An int or scalar `Tensor` specifying the batch size to use.
  features: A `dict` mapping feature keys to `FixedLenFeature` or
    `VarLenFeature` values.
  reader: A function or class that returns an object with
    `read` method, (filename tensor) -> (example tensor).
  randomize_input: Whether the input should be randomized.
  num_epochs: Integer specifying the number of times to read through the
    dataset. If None, cycles through the dataset forever. NOTE - If specified,
    creates a variable that must be initialized, so call
    tf.compat.v1.local_variables_initializer() and run the op in a session.
  queue_capacity: Capacity for input queue.
  feature_queue_capacity: Capacity of the parsed features queue. Set this
    value to a small number, for example 5 if the parsed features are large.
  reader_num_threads: The number of threads to read examples. In order to have
    predictable and repeatable order of reading and enqueueing, such as in
    prediction and evaluation mode, `reader_num_threads` should be 1.
  num_enqueue_threads: Number of threads to enqueue the parsed example queue.
    Using multiple threads to enqueue the parsed example queue helps maintain
    a full queue when the subsequent computations overall are cheaper than
    parsing. In order to have predictable and repeatable order of reading and
    enqueueing, such as in prediction and evaluation mode,
    `num_enqueue_threads` should be 1.
  parse_fn: Parsing function, takes `Example` Tensor returns parsed
    representation. If `None`, no parsing is done.
  name: Name of resulting op.
  read_batch_size: An int or scalar `Tensor` specifying the number of
    records to read at once. If `None`, defaults to `batch_size`.

Returns:
  A dict of `Tensor` or `SparseTensor` objects for each in `features`.

Raises:
  ValueError: for invalid inputs."
  [file_pattern batch_size features reader & {:keys [randomize_input num_epochs queue_capacity feature_queue_capacity reader_num_threads num_enqueue_threads parse_fn name read_batch_size]
                       :or {num_epochs None parse_fn None name None read_batch_size None}} ]
    (py/call-attr-kw learn "read_batch_features" [file_pattern batch_size features reader] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :feature_queue_capacity feature_queue_capacity :reader_num_threads reader_num_threads :num_enqueue_threads num_enqueue_threads :parse_fn parse_fn :name name :read_batch_size read_batch_size }))

(defn read-batch-record-features 
  "Reads TFRecord, queues, batches and parses `Example` proto. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.data.

See more detailed description in `read_examples`.

Args:
  file_pattern: List of files or patterns of file paths containing
      `Example` records. See `tf.io.gfile.glob` for pattern rules.
  batch_size: An int or scalar `Tensor` specifying the batch size to use.
  features: A `dict` mapping feature keys to `FixedLenFeature` or
    `VarLenFeature` values.
  randomize_input: Whether the input should be randomized.
  num_epochs: Integer specifying the number of times to read through the
    dataset. If None, cycles through the dataset forever. NOTE - If specified,
    creates a variable that must be initialized, so call
    tf.compat.v1.local_variables_initializer() and run the op in a session.
  queue_capacity: Capacity for input queue.
  reader_num_threads: The number of threads to read examples. In order to have
    predictable and repeatable order of reading and enqueueing, such as in
    prediction and evaluation mode, `reader_num_threads` should be 1.
  name: Name of resulting op.

Returns:
  A dict of `Tensor` or `SparseTensor` objects for each in `features`.

Raises:
  ValueError: for invalid inputs."
  [file_pattern batch_size features & {:keys [randomize_input num_epochs queue_capacity reader_num_threads name]
                       :or {num_epochs None}} ]
    (py/call-attr-kw learn "read_batch_record_features" [file_pattern batch_size features] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :reader_num_threads reader_num_threads :name name }))

(defn read-keyed-batch-examples 
  "Adds operations to read, queue, batch `Example` protos. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.data.

Given file pattern (or list of files), will setup a queue for file names,
read `Example` proto using provided `reader`, use batch queue to create
batches of examples of size `batch_size`.

All queue runners are added to the queue runners collection, and may be
started via `start_queue_runners`.

All ops are added to the default graph.

Use `parse_fn` if you need to do parsing / processing on single examples.

Args:
  file_pattern: List of files or patterns of file paths containing
      `Example` records. See `tf.io.gfile.glob` for pattern rules.
  batch_size: An int or scalar `Tensor` specifying the batch size to use.
  reader: A function or class that returns an object with
    `read` method, (filename tensor) -> (example tensor).
  randomize_input: Whether the input should be randomized.
  num_epochs: Integer specifying the number of times to read through the
    dataset. If `None`, cycles through the dataset forever.
    NOTE - If specified, creates a variable that must be initialized, so call
    `tf.compat.v1.local_variables_initializer()` and run the op in a session.
  queue_capacity: Capacity for input queue.
  num_threads: The number of threads enqueuing examples. In order to have
    predictable and repeatable order of reading and enqueueing, such as in
    prediction and evaluation mode, `num_threads` should be 1.
  read_batch_size: An int or scalar `Tensor` specifying the number of
    records to read at once.
  parse_fn: Parsing function, takes `Example` Tensor returns parsed
    representation. If `None`, no parsing is done.
  name: Name of resulting op.
  seed: An integer (optional). Seed used if randomize_input == True.

Returns:
  Returns tuple of:
  - `Tensor` of string keys.
  - String `Tensor` of batched `Example` proto.

Raises:
  ValueError: for invalid inputs."
  [file_pattern batch_size reader & {:keys [randomize_input num_epochs queue_capacity num_threads read_batch_size parse_fn name seed]
                       :or {num_epochs None parse_fn None name None seed None}} ]
    (py/call-attr-kw learn "read_keyed_batch_examples" [file_pattern batch_size reader] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :num_threads num_threads :read_batch_size read_batch_size :parse_fn parse_fn :name name :seed seed }))

(defn read-keyed-batch-examples-shared-queue 
  "Adds operations to read, queue, batch `Example` protos. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.data.

Given file pattern (or list of files), will setup a shared queue for file
names, setup a worker queue that pulls from the shared queue, read `Example`
protos using provided `reader`, use batch queue to create batches of examples
of size `batch_size`. This provides at most once visit guarantees. Note that
this only works if the parameter servers are not pre-empted or restarted or
the session is not restored from a checkpoint since the state of a queue
is not checkpointed and we will end up restarting from the entire list of
files.

All queue runners are added to the queue runners collection, and may be
started via `start_queue_runners`.

All ops are added to the default graph.

Use `parse_fn` if you need to do parsing / processing on single examples.

Args:
  file_pattern: List of files or patterns of file paths containing
      `Example` records. See `tf.io.gfile.glob` for pattern rules.
  batch_size: An int or scalar `Tensor` specifying the batch size to use.
  reader: A function or class that returns an object with
    `read` method, (filename tensor) -> (example tensor).
  randomize_input: Whether the input should be randomized.
  num_epochs: Integer specifying the number of times to read through the
    dataset. If `None`, cycles through the dataset forever.
    NOTE - If specified, creates a variable that must be initialized, so call
    `tf.compat.v1.local_variables_initializer()` and run the op in a session.
  queue_capacity: Capacity for input queue.
  num_threads: The number of threads enqueuing examples.
  read_batch_size: An int or scalar `Tensor` specifying the number of
    records to read at once.
  parse_fn: Parsing function, takes `Example` Tensor returns parsed
    representation. If `None`, no parsing is done.
  name: Name of resulting op.
  seed: An integer (optional). Seed used if randomize_input == True.

Returns:
  Returns tuple of:
  - `Tensor` of string keys.
  - String `Tensor` of batched `Example` proto.

Raises:
  ValueError: for invalid inputs."
  [file_pattern batch_size reader & {:keys [randomize_input num_epochs queue_capacity num_threads read_batch_size parse_fn name seed]
                       :or {num_epochs None parse_fn None name None seed None}} ]
    (py/call-attr-kw learn "read_keyed_batch_examples_shared_queue" [file_pattern batch_size reader] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :num_threads num_threads :read_batch_size read_batch_size :parse_fn parse_fn :name name :seed seed }))

(defn read-keyed-batch-features 
  "Adds operations to read, queue, batch and parse `Example` protos. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.data.

Given file pattern (or list of files), will setup a queue for file names,
read `Example` proto using provided `reader`, use batch queue to create
batches of examples of size `batch_size` and parse example given `features`
specification.

All queue runners are added to the queue runners collection, and may be
started via `start_queue_runners`.

All ops are added to the default graph.

Args:
  file_pattern: List of files or patterns of file paths containing
      `Example` records. See `tf.io.gfile.glob` for pattern rules.
  batch_size: An int or scalar `Tensor` specifying the batch size to use.
  features: A `dict` mapping feature keys to `FixedLenFeature` or
    `VarLenFeature` values.
  reader: A function or class that returns an object with
    `read` method, (filename tensor) -> (example tensor).
  randomize_input: Whether the input should be randomized.
  num_epochs: Integer specifying the number of times to read through the
    dataset. If None, cycles through the dataset forever. NOTE - If specified,
    creates a variable that must be initialized, so call
    tf.compat.v1.local_variables_initializer() and run the op in a session.
  queue_capacity: Capacity for input queue.
  reader_num_threads: The number of threads to read examples. In order to have
    predictable and repeatable order of reading and enqueueing, such as in
    prediction and evaluation mode, `reader_num_threads` should be 1.
  feature_queue_capacity: Capacity of the parsed features queue.
  num_enqueue_threads: Number of threads to enqueue the parsed example queue.
    Using multiple threads to enqueue the parsed example queue helps maintain
    a full queue when the subsequent computations overall are cheaper than
    parsing. In order to have predictable and repeatable order of reading and
    enqueueing, such as in prediction and evaluation mode,
    `num_enqueue_threads` should be 1.
  parse_fn: Parsing function, takes `Example` Tensor returns parsed
    representation. If `None`, no parsing is done.
  name: Name of resulting op.
  read_batch_size: An int or scalar `Tensor` specifying the number of
    records to read at once. If `None`, defaults to `batch_size`.

Returns:
  Returns tuple of:
  - `Tensor` of string keys.
  - A dict of `Tensor` or `SparseTensor` objects for each in `features`.

Raises:
  ValueError: for invalid inputs."
  [file_pattern batch_size features reader & {:keys [randomize_input num_epochs queue_capacity reader_num_threads feature_queue_capacity num_enqueue_threads parse_fn name read_batch_size]
                       :or {num_epochs None parse_fn None name None read_batch_size None}} ]
    (py/call-attr-kw learn "read_keyed_batch_features" [file_pattern batch_size features reader] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :reader_num_threads reader_num_threads :feature_queue_capacity feature_queue_capacity :num_enqueue_threads num_enqueue_threads :parse_fn parse_fn :name name :read_batch_size read_batch_size }))

(defn read-keyed-batch-features-shared-queue 
  "Adds operations to read, queue, batch and parse `Example` protos. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.data.

Given file pattern (or list of files), will setup a shared queue for file
names, setup a worker queue that gets filenames from the shared queue,
read `Example` proto using provided `reader`, use batch queue to create
batches of examples of size `batch_size` and parse example given `features`
specification.

All queue runners are added to the queue runners collection, and may be
started via `start_queue_runners`.

All ops are added to the default graph.

Args:
  file_pattern: List of files or patterns of file paths containing
      `Example` records. See `tf.io.gfile.glob` for pattern rules.
  batch_size: An int or scalar `Tensor` specifying the batch size to use.
  features: A `dict` mapping feature keys to `FixedLenFeature` or
    `VarLenFeature` values.
  reader: A function or class that returns an object with
    `read` method, (filename tensor) -> (example tensor).
  randomize_input: Whether the input should be randomized.
  num_epochs: Integer specifying the number of times to read through the
    dataset. If None, cycles through the dataset forever. NOTE - If specified,
    creates a variable that must be initialized, so call
    tf.compat.v1.local_variables_initializer() and run the op in a session.
  queue_capacity: Capacity for input queue.
  reader_num_threads: The number of threads to read examples.
  feature_queue_capacity: Capacity of the parsed features queue.
  num_queue_runners: Number of threads to enqueue the parsed example queue.
    Using multiple threads to enqueue the parsed example queue helps maintain
    a full queue when the subsequent computations overall are cheaper than
    parsing.
  parse_fn: Parsing function, takes `Example` Tensor returns parsed
    representation. If `None`, no parsing is done.
  name: Name of resulting op.

Returns:
  Returns tuple of:
  - `Tensor` of string keys.
  - A dict of `Tensor` or `SparseTensor` objects for each in `features`.

Raises:
  ValueError: for invalid inputs."
  [file_pattern batch_size features reader & {:keys [randomize_input num_epochs queue_capacity reader_num_threads feature_queue_capacity num_queue_runners parse_fn name]
                       :or {num_epochs None parse_fn None name None}} ]
    (py/call-attr-kw learn "read_keyed_batch_features_shared_queue" [file_pattern batch_size features reader] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :reader_num_threads reader_num_threads :feature_queue_capacity feature_queue_capacity :num_queue_runners num_queue_runners :parse_fn parse_fn :name name }))

(defn regression-head 
  "Creates a `Head` for linear regression. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please switch to tf.contrib.estimator.*_head.

Args:
  label_name: String, name of the key in label dict. Can be null if label
      is a tensor (single headed models).
  weight_column_name: A string defining feature column name representing
    weights. It is used to down weight or boost examples during training. It
    will be multiplied by the loss of the example.
  label_dimension: Number of regression labels per example. This is the size
    of the last dimension of the labels `Tensor` (typically, this has shape
    `[batch_size, label_dimension]`).
  enable_centered_bias: A bool. If True, estimator will learn a centered
    bias variable for each class. Rest of the model structure learns the
    residual after centered bias.
  head_name: name of the head. If provided, predictions, summary and metrics
    keys will be suffixed by `\"/\" + head_name` and the default variable scope
    will be `head_name`.
  link_fn: link function to convert logits to predictions. If provided,
    this link function will be used instead of identity.

Returns:
  An instance of `Head` for linear regression."
  [label_name weight_column_name & {:keys [label_dimension enable_centered_bias head_name link_fn]
                       :or {head_name None link_fn None}} ]
    (py/call-attr-kw learn "regression_head" [label_name weight_column_name] {:label_dimension label_dimension :enable_centered_bias enable_centered_bias :head_name head_name :link_fn link_fn }))

(defn run-feeds 
  "See run_feeds_iter(). Returns a `list` instead of an iterator. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2017-02-15.
Instructions for updating:
graph_actions.py will be deleted. Use tf.train.* utilities instead. You can use learn/estimators/estimator.py as an example."
  [  ]
  (py/call-attr learn "run_feeds"  ))
(defn run-n 
  "Run `output_dict` tensors `n` times, with the same `feed_dict` each run. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2017-02-15.
Instructions for updating:
graph_actions.py will be deleted. Use tf.train.* utilities instead. You can use learn/estimators/estimator.py as an example.

Args:
  output_dict: A `dict` mapping string names to tensors to run. Must all be
    from the same graph.
  feed_dict: `dict` of input values to feed each run.
  restore_checkpoint_path: A string containing the path to a checkpoint to
    restore.
  n: Number of times to repeat.

Returns:
  A list of `n` `dict` objects, each containing values read from `output_dict`
  tensors."
  [output_dict feed_dict restore_checkpoint_path  & {:keys [n]} ]
    (py/call-attr-kw learn "run_n" [output_dict feed_dict restore_checkpoint_path] {:n n }))

(defn train 
  "Train a model. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2017-02-15.
Instructions for updating:
graph_actions.py will be deleted. Use tf.train.* utilities instead. You can use learn/estimators/estimator.py as an example.

Given `graph`, a directory to write outputs to (`output_dir`), and some ops,
run a training loop. The given `train_op` performs one step of training on the
model. The `loss_op` represents the objective function of the training. It is
expected to increment the `global_step_tensor`, a scalar integer tensor
counting training steps. This function uses `Supervisor` to initialize the
graph (from a checkpoint if one is available in `output_dir`), write summaries
defined in the graph, and write regular checkpoints as defined by
`supervisor_save_model_secs`.

Training continues until `global_step_tensor` evaluates to `max_steps`, or, if
`fail_on_nan_loss`, until `loss_op` evaluates to `NaN`. In that case the
program is terminated with exit code 1.

Args:
  graph: A graph to train. It is expected that this graph is not in use
    elsewhere.
  output_dir: A directory to write outputs to.
  train_op: An op that performs one training step when run.
  loss_op: A scalar loss tensor.
  global_step_tensor: A tensor representing the global step. If none is given,
    one is extracted from the graph using the same logic as in `Supervisor`.
  init_op: An op that initializes the graph. If `None`, use `Supervisor`'s
    default.
  init_feed_dict: A dictionary that maps `Tensor` objects to feed values.
    This feed dictionary will be used when `init_op` is evaluated.
  init_fn: Optional callable passed to Supervisor to initialize the model.
  log_every_steps: Output logs regularly. The logs contain timing data and the
    current loss.
  supervisor_is_chief: Whether the current process is the chief supervisor in
    charge of restoring the model and running standard services.
  supervisor_master: The master string to use when preparing the session.
  supervisor_save_model_secs: Save a checkpoint every
    `supervisor_save_model_secs` seconds when training.
  keep_checkpoint_max: The maximum number of recent checkpoint files to
    keep. As new files are created, older files are deleted. If None or 0,
    all checkpoint files are kept. This is simply passed as the max_to_keep
    arg to tf.compat.v1.train.Saver constructor.
  supervisor_save_summaries_steps: Save summaries every
    `supervisor_save_summaries_steps` seconds when training.
  feed_fn: A function that is called every iteration to produce a `feed_dict`
    passed to `session.run` calls. Optional.
  steps: Trains for this many steps (e.g. current global step + `steps`).
  fail_on_nan_loss: If true, raise `NanLossDuringTrainingError` if `loss_op`
    evaluates to `NaN`. If false, continue training as if nothing happened.
  monitors: List of `BaseMonitor` subclass instances. Used for callbacks
    inside the training loop.
  max_steps: Number of total steps for which to train model. If `None`,
    train forever. Two calls fit(steps=100) means 200 training iterations.
    On the other hand two calls of fit(max_steps=100) means, second call
    will not do any iteration since first call did all 100 steps.

Returns:
  The final loss value.

Raises:
  ValueError: If `output_dir`, `train_op`, `loss_op`, or `global_step_tensor`
    is not provided. See `tf.contrib.framework.get_global_step` for how we
    look up the latter if not provided explicitly.
  NanLossDuringTrainingError: If `fail_on_nan_loss` is `True`, and loss ever
    evaluates to `NaN`.
  ValueError: If both `steps` and `max_steps` are not `None`."
  [graph output_dir train_op loss_op global_step_tensor init_op init_feed_dict init_fn & {:keys [log_every_steps supervisor_is_chief supervisor_master supervisor_save_model_secs keep_checkpoint_max supervisor_save_summaries_steps feed_fn steps fail_on_nan_loss monitors max_steps]
                       :or {feed_fn None steps None monitors None max_steps None}} ]
    (py/call-attr-kw learn "train" [graph output_dir train_op loss_op global_step_tensor init_op init_feed_dict init_fn] {:log_every_steps log_every_steps :supervisor_is_chief supervisor_is_chief :supervisor_master supervisor_master :supervisor_save_model_secs supervisor_save_model_secs :keep_checkpoint_max keep_checkpoint_max :supervisor_save_summaries_steps supervisor_save_summaries_steps :feed_fn feed_fn :steps steps :fail_on_nan_loss fail_on_nan_loss :monitors monitors :max_steps max_steps }))
