(ns tensorflow.contrib.learn.SVM
  "Support Vector Machine (SVM) model for binary classification.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Currently, only linear SVMs are supported. For the underlying optimization
  problem, the `SDCAOptimizer` is used. For performance and convergence tuning,
  the num_loss_partitions parameter passed to `SDCAOptimizer` (see `__init__()`
  method), should be set to (#concurrent train ops per worker) x (#workers). If
  num_loss_partitions is larger or equal to this value, convergence is
  guaranteed but becomes slower as num_loss_partitions increases. If it is set
  to a smaller value, the optimizer is more aggressive in reducing the global
  loss but convergence is not guaranteed. The recommended value in an
  `Estimator` (where there is one process per worker) is the number of workers
  running the train steps. It defaults to 1 (single machine).

  Example:

  ```python
  real_feature_column = real_valued_column(...)
  sparse_feature_column = sparse_column_with_hash_bucket(...)

  estimator = SVM(
      example_id_column='example_id',
      feature_columns=[real_feature_column, sparse_feature_column],
      l2_regularization=10.0)

  # Input builders
  def input_fn_train: # returns x, y
    ...
  def input_fn_eval: # returns x, y
    ...

  estimator.fit(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(x=x)
  ```

  Input of `fit` and `evaluate` should have following features, otherwise there
  will be a `KeyError`:
    a feature with `key=example_id_column` whose value is a `Tensor` of dtype
    string.
    if `weight_column_name` is not `None`, a feature with
    `key=weight_column_name` whose value is a `Tensor`.
    for each `column` in `feature_columns`:
      - if `column` is a `SparseColumn`, a feature with `key=column.name`
        whose `value` is a `SparseTensor`.
      - if `column` is a `RealValuedColumn, a feature with `key=column.name`
        whose `value` is a `Tensor`.
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

(defn SVM 
  "Support Vector Machine (SVM) model for binary classification.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Currently, only linear SVMs are supported. For the underlying optimization
  problem, the `SDCAOptimizer` is used. For performance and convergence tuning,
  the num_loss_partitions parameter passed to `SDCAOptimizer` (see `__init__()`
  method), should be set to (#concurrent train ops per worker) x (#workers). If
  num_loss_partitions is larger or equal to this value, convergence is
  guaranteed but becomes slower as num_loss_partitions increases. If it is set
  to a smaller value, the optimizer is more aggressive in reducing the global
  loss but convergence is not guaranteed. The recommended value in an
  `Estimator` (where there is one process per worker) is the number of workers
  running the train steps. It defaults to 1 (single machine).

  Example:

  ```python
  real_feature_column = real_valued_column(...)
  sparse_feature_column = sparse_column_with_hash_bucket(...)

  estimator = SVM(
      example_id_column='example_id',
      feature_columns=[real_feature_column, sparse_feature_column],
      l2_regularization=10.0)

  # Input builders
  def input_fn_train: # returns x, y
    ...
  def input_fn_eval: # returns x, y
    ...

  estimator.fit(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(x=x)
  ```

  Input of `fit` and `evaluate` should have following features, otherwise there
  will be a `KeyError`:
    a feature with `key=example_id_column` whose value is a `Tensor` of dtype
    string.
    if `weight_column_name` is not `None`, a feature with
    `key=weight_column_name` whose value is a `Tensor`.
    for each `column` in `feature_columns`:
      - if `column` is a `SparseColumn`, a feature with `key=column.name`
        whose `value` is a `SparseTensor`.
      - if `column` is a `RealValuedColumn, a feature with `key=column.name`
        whose `value` is a `Tensor`.
  "
  [example_id_column feature_columns weight_column_name model_dir & {:keys [l1_regularization l2_regularization num_loss_partitions kernels config feature_engineering_fn]
                       :or {kernels None config None feature_engineering_fn None}} ]
    (py/call-attr-kw learn "SVM" [example_id_column feature_columns weight_column_name model_dir] {:l1_regularization l1_regularization :l2_regularization l2_regularization :num_loss_partitions num_loss_partitions :kernels kernels :config config :feature_engineering_fn feature_engineering_fn }))

(defn config 
  ""
  [ self ]
    (py/call-attr self "config"))
(defn evaluate 
  "See `Evaluable`. (deprecated arguments)

Warning: SOME ARGUMENTS ARE DEPRECATED: `(batch_size, x, y)`. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.
Example conversion:
  est = Estimator(...) -> est = SKCompat(Estimator(...))

Raises:
  ValueError: If at least one of `x` or `y` is provided, and at least one of
      `input_fn` or `feed_fn` is provided.
      Or if `metrics` is not `None` or `dict`."
  [self x y input_fn feed_fn batch_size steps metrics name checkpoint_path hooks  & {:keys [log_progress]} ]
    (py/call-attr-kw self "evaluate" [x y input_fn feed_fn batch_size steps metrics name checkpoint_path hooks] {:log_progress log_progress }))

(defn export 
  "See BaseEstimator.export. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2017-03-25.
Instructions for updating:
Please use Estimator.export_savedmodel() instead."
  [self export_dir signature_fn input_fn & {:keys [default_batch_size exports_to_keep]
                       :or {exports_to_keep None}} ]
    (py/call-attr-kw self "export" [export_dir signature_fn input_fn] {:default_batch_size default_batch_size :exports_to_keep exports_to_keep }))

(defn export-savedmodel 
  "Exports inference graph as a SavedModel into given dir.

    Args:
      export_dir_base: A string containing a directory to write the exported
        graph and checkpoints.
      serving_input_fn: A function that takes no argument and
        returns an `InputFnOps`.
      default_output_alternative_key: the name of the head to serve when none is
        specified.  Not needed for single-headed models.
      assets_extra: A dict specifying how to populate the assets.extra directory
        within the exported SavedModel.  Each key should give the destination
        path (including the filename) relative to the assets.extra directory.
        The corresponding value gives the full path of the source file to be
        copied.  For example, the simple case of copying a single file without
        renaming it is specified as
        `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
      as_text: whether to write the SavedModel proto in text format.
      checkpoint_path: The checkpoint path to export.  If None (the default),
        the most recent checkpoint found within the model directory is chosen.
      graph_rewrite_specs: an iterable of `GraphRewriteSpec`.  Each element will
        produce a separate MetaGraphDef within the exported SavedModel, tagged
        and rewritten as specified.  Defaults to a single entry using the
        default serving tag (\"serve\") and no rewriting.
      strip_default_attrs: Boolean. If `True`, default-valued attributes will be
        removed from the NodeDefs. For a detailed guide, see
        [Stripping Default-Valued
          Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).

    Returns:
      The string path to the exported directory.

    Raises:
      ValueError: if an unrecognized export_type is requested.
    "
  [self export_dir_base serving_input_fn default_output_alternative_key assets_extra & {:keys [as_text checkpoint_path graph_rewrite_specs strip_default_attrs]
                       :or {checkpoint_path None}} ]
    (py/call-attr-kw self "export_savedmodel" [export_dir_base serving_input_fn default_output_alternative_key assets_extra] {:as_text as_text :checkpoint_path checkpoint_path :graph_rewrite_specs graph_rewrite_specs :strip_default_attrs strip_default_attrs }))

(defn export-with-defaults 
  "Same as BaseEstimator.export, but uses some defaults. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2017-03-25.
Instructions for updating:
Please use Estimator.export_savedmodel() instead."
  [self export_dir signature_fn input_fn & {:keys [default_batch_size exports_to_keep]
                       :or {exports_to_keep None}} ]
    (py/call-attr-kw self "export_with_defaults" [export_dir signature_fn input_fn] {:default_batch_size default_batch_size :exports_to_keep exports_to_keep }))

(defn fit 
  "See `Trainable`. (deprecated arguments)

Warning: SOME ARGUMENTS ARE DEPRECATED: `(batch_size, x, y)`. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.
Example conversion:
  est = Estimator(...) -> est = SKCompat(Estimator(...))

Raises:
  ValueError: If `x` or `y` are not `None` while `input_fn` is not `None`.
  ValueError: If both `steps` and `max_steps` are not `None`."
  [ self x y input_fn steps batch_size monitors max_steps ]
  (py/call-attr self "fit"  self x y input_fn steps batch_size monitors max_steps ))
(defn get-params 
  "Get parameters for this estimator.

    Args:
      deep: boolean, optional

        If `True`, will return the parameters for this estimator and
        contained subobjects that are estimators.

    Returns:
      params : mapping of string to any
      Parameter names mapped to their values.
    "
  [self   & {:keys [deep]} ]
    (py/call-attr-kw self "get_params" [] {:deep deep }))

(defn get-variable-names 
  "Returns list of all variable names in this model.

    Returns:
      List of names.
    "
  [ self  ]
  (py/call-attr self "get_variable_names"  self  ))

(defn get-variable-value 
  "Returns value of the variable given by name.

    Args:
      name: string, name of the tensor.

    Returns:
      Numpy array - value of the tensor.
    "
  [ self name ]
  (py/call-attr self "get_variable_value"  self name ))

(defn model-dir 
  ""
  [ self ]
    (py/call-attr self "model_dir"))

(defn model-fn 
  "Returns the model_fn which is bound to self.params.

    Returns:
      The model_fn with the following signature:
        `def model_fn(features, labels, mode, metrics)`
    "
  [ self ]
    (py/call-attr self "model_fn"))

(defn partial-fit 
  "Incremental fit on a batch of samples. (deprecated arguments)

Warning: SOME ARGUMENTS ARE DEPRECATED: `(batch_size, x, y)`. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.
Example conversion:
  est = Estimator(...) -> est = SKCompat(Estimator(...))

This method is expected to be called several times consecutively
on different or the same chunks of the dataset. This either can
implement iterative training or out-of-core/online training.

This is especially useful when the whole dataset is too big to
fit in memory at the same time. Or when model is taking long time
to converge, and you want to split up training into subparts.

Args:
  x: Matrix of shape [n_samples, n_features...]. Can be iterator that
     returns arrays of features. The training input samples for fitting the
     model. If set, `input_fn` must be `None`.
  y: Vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
     iterator that returns array of labels. The training label values
     (class labels in classification, real numbers in regression). If set,
     `input_fn` must be `None`.
  input_fn: Input function. If set, `x`, `y`, and `batch_size` must be
    `None`.
  steps: Number of steps for which to train model. If `None`, train forever.
  batch_size: minibatch size to use on the input, defaults to first
    dimension of `x`. Must be `None` if `input_fn` is provided.
  monitors: List of `BaseMonitor` subclass instances. Used for callbacks
    inside the training loop.

Returns:
  `self`, for chaining.

Raises:
  ValueError: If at least one of `x` and `y` is provided, and `input_fn` is
      provided."
  [self x y input_fn & {:keys [steps batch_size monitors]
                       :or {batch_size None monitors None}} ]
    (py/call-attr-kw self "partial_fit" [x y input_fn] {:steps steps :batch_size batch_size :monitors monitors }))
(defn predict 
  "Returns predictions for given features. (deprecated arguments)

Warning: SOME ARGUMENTS ARE DEPRECATED: `(as_iterable, batch_size, x)`. They will be removed after 2016-12-01.
Instructions for updating:
Estimator is decoupled from Scikit Learn interface by moving into
separate class SKCompat. Arguments x, y and batch_size are only
available in the SKCompat class, Estimator will only accept input_fn.
Example conversion:
  est = Estimator(...) -> est = SKCompat(Estimator(...))

Args:
  x: Matrix of shape [n_samples, n_features...]. Can be iterator that
     returns arrays of features. The training input samples for fitting the
     model. If set, `input_fn` must be `None`.
  input_fn: Input function. If set, `x` and 'batch_size' must be `None`.
  batch_size: Override default batch size. If set, 'input_fn' must be
    'None'.
  outputs: list of `str`, name of the output to predict.
    If `None`, returns all.
  as_iterable: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).
  iterate_batches: If True, yield the whole batch at once instead of
    decomposing the batch into individual samples. Only relevant when
    as_iterable is True.

Returns:
  A numpy array of predicted classes or regression values if the
  constructor's `model_fn` returns a `Tensor` for `predictions` or a `dict`
  of numpy arrays if `model_fn` returns a `dict`. Returns an iterable of
  predictions if as_iterable is True.

Raises:
  ValueError: If x and input_fn are both provided or both `None`."
  [self x input_fn batch_size outputs  & {:keys [as_iterable iterate_batches]} ]
    (py/call-attr-kw self "predict" [x input_fn batch_size outputs] {:as_iterable as_iterable :iterate_batches iterate_batches }))
(defn predict-classes 
  "Runs inference to determine the predicted class. (deprecated argument values)

Warning: SOME ARGUMENT VALUES ARE DEPRECATED: `(as_iterable=False)`. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below."
  [self x input_fn batch_size  & {:keys [as_iterable]} ]
    (py/call-attr-kw self "predict_classes" [x input_fn batch_size] {:as_iterable as_iterable }))
(defn predict-proba 
  "Runs inference to determine the class probability predictions. (deprecated argument values)

Warning: SOME ARGUMENT VALUES ARE DEPRECATED: `(as_iterable=False)`. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below."
  [self x input_fn batch_size outputs  & {:keys [as_iterable]} ]
    (py/call-attr-kw self "predict_proba" [x input_fn batch_size outputs] {:as_iterable as_iterable }))

(defn set-params 
  "Set the parameters of this estimator.

    The method works on simple estimators as well as on nested objects
    (such as pipelines). The former have parameters of the form
    ``<component>__<parameter>`` so that it's possible to update each
    component of a nested object.

    Args:
      **params: Parameters.

    Returns:
      self

    Raises:
      ValueError: If params contain invalid names.
    "
  [ self  ]
  (py/call-attr self "set_params"  self  ))
