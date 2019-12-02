(ns tensorflow.contrib.learn.LinearClassifier
  "Linear classifier model.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Train a linear model to classify instances into one of multiple possible
  classes. When number of possible classes is 2, this is binary classification.

  Example:

  ```python
  sparse_column_a = sparse_column_with_hash_bucket(...)
  sparse_column_b = sparse_column_with_hash_bucket(...)

  sparse_feature_a_x_sparse_feature_b = crossed_column(...)

  # Estimator using the default optimizer.
  estimator = LinearClassifier(
      feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b])

  # Or estimator using the FTRL optimizer with regularization.
  estimator = LinearClassifier(
      feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b],
      optimizer=tf.compat.v1.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
      ))

  # Or estimator using the SDCAOptimizer.
  estimator = LinearClassifier(
     feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b],
     optimizer=tf.contrib.linear_optimizer.SDCAOptimizer(
       example_id_column='example_id',
       num_loss_partitions=...,
       symmetric_l2_regularization=2.0
     ))

  # Input builders
  def input_fn_train: # returns x, y (where y represents label's class index).
    ...
  def input_fn_eval: # returns x, y (where y represents label's class index).
    ...
  def input_fn_predict: # returns x, None.
    ...
  estimator.fit(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  # predict_classes returns class indices.
  estimator.predict_classes(input_fn=input_fn_predict)
  ```

  If the user specifies `label_keys` in constructor, labels must be strings from
  the `label_keys` vocabulary. Example:

  ```python
  label_keys = ['label0', 'label1', 'label2']
  estimator = LinearClassifier(
      n_classes=n_classes,
      feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b],
      label_keys=label_keys)

  def input_fn_train: # returns x, y (where y is one of label_keys).
    pass
  estimator.fit(input_fn=input_fn_train)

  def input_fn_eval: # returns x, y (where y is one of label_keys).
    pass
  estimator.evaluate(input_fn=input_fn_eval)
  def input_fn_predict: # returns x, None
  # predict_classes returns one of label_keys.
  estimator.predict_classes(input_fn=input_fn_predict)
  ```

  Input of `fit` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:

  * if `weight_column_name` is not `None`, a feature with
    `key=weight_column_name` whose value is a `Tensor`.
  * for each `column` in `feature_columns`:
    - if `column` is a `SparseColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `WeightedSparseColumn`, two features: the first with
      `key` the id column name, the second with `key` the weight column name.
      Both features' `value` must be a `SparseTensor`.
    - if `column` is a `RealValuedColumn`, a feature with `key=column.name`
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

(defn LinearClassifier 
  "Linear classifier model.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Train a linear model to classify instances into one of multiple possible
  classes. When number of possible classes is 2, this is binary classification.

  Example:

  ```python
  sparse_column_a = sparse_column_with_hash_bucket(...)
  sparse_column_b = sparse_column_with_hash_bucket(...)

  sparse_feature_a_x_sparse_feature_b = crossed_column(...)

  # Estimator using the default optimizer.
  estimator = LinearClassifier(
      feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b])

  # Or estimator using the FTRL optimizer with regularization.
  estimator = LinearClassifier(
      feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b],
      optimizer=tf.compat.v1.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
      ))

  # Or estimator using the SDCAOptimizer.
  estimator = LinearClassifier(
     feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b],
     optimizer=tf.contrib.linear_optimizer.SDCAOptimizer(
       example_id_column='example_id',
       num_loss_partitions=...,
       symmetric_l2_regularization=2.0
     ))

  # Input builders
  def input_fn_train: # returns x, y (where y represents label's class index).
    ...
  def input_fn_eval: # returns x, y (where y represents label's class index).
    ...
  def input_fn_predict: # returns x, None.
    ...
  estimator.fit(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  # predict_classes returns class indices.
  estimator.predict_classes(input_fn=input_fn_predict)
  ```

  If the user specifies `label_keys` in constructor, labels must be strings from
  the `label_keys` vocabulary. Example:

  ```python
  label_keys = ['label0', 'label1', 'label2']
  estimator = LinearClassifier(
      n_classes=n_classes,
      feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b],
      label_keys=label_keys)

  def input_fn_train: # returns x, y (where y is one of label_keys).
    pass
  estimator.fit(input_fn=input_fn_train)

  def input_fn_eval: # returns x, y (where y is one of label_keys).
    pass
  estimator.evaluate(input_fn=input_fn_eval)
  def input_fn_predict: # returns x, None
  # predict_classes returns one of label_keys.
  estimator.predict_classes(input_fn=input_fn_predict)
  ```

  Input of `fit` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:

  * if `weight_column_name` is not `None`, a feature with
    `key=weight_column_name` whose value is a `Tensor`.
  * for each `column` in `feature_columns`:
    - if `column` is a `SparseColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `WeightedSparseColumn`, two features: the first with
      `key` the id column name, the second with `key` the weight column name.
      Both features' `value` must be a `SparseTensor`.
    - if `column` is a `RealValuedColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`.
  "
  [feature_columns model_dir & {:keys [n_classes weight_column_name optimizer gradient_clip_norm enable_centered_bias _joint_weight config feature_engineering_fn label_keys]
                       :or {weight_column_name None optimizer None gradient_clip_norm None config None feature_engineering_fn None label_keys None}} ]
    (py/call-attr-kw learn "LinearClassifier" [feature_columns model_dir] {:n_classes n_classes :weight_column_name weight_column_name :optimizer optimizer :gradient_clip_norm gradient_clip_norm :enable_centered_bias enable_centered_bias :_joint_weight _joint_weight :config config :feature_engineering_fn feature_engineering_fn :label_keys label_keys }))

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
  [self export_dir input_fn input_feature_key & {:keys [use_deprecated_input_fn signature_fn default_batch_size exports_to_keep]
                       :or {signature_fn None exports_to_keep None}} ]
    (py/call-attr-kw self "export" [export_dir input_fn input_feature_key] {:use_deprecated_input_fn use_deprecated_input_fn :signature_fn signature_fn :default_batch_size default_batch_size :exports_to_keep exports_to_keep }))

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
  "Returns predictions for given features. (deprecated argument values) (deprecated argument values)

Warning: SOME ARGUMENT VALUES ARE DEPRECATED: `(as_iterable=False)`. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

Warning: SOME ARGUMENT VALUES ARE DEPRECATED: `(outputs=None)`. They will be removed after 2017-03-01.
Instructions for updating:
Please switch to predict_classes, or set `outputs` argument.

By default, returns predicted classes. But this default will be dropped
soon. Users should either pass `outputs`, or call `predict_classes` method.

Args:
  x: features.
  input_fn: Input function. If set, x must be None.
  batch_size: Override default batch size.
  outputs: list of `str`, name of the output to predict.
    If `None`, returns classes.
  as_iterable: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

Returns:
  Numpy array of predicted classes with shape [batch_size] (or an iterable
  of predicted classes if as_iterable is True). Each predicted class is
  represented by its class index (i.e. integer from 0 to n_classes-1).
  If `outputs` is set, returns a dict of predictions."
  [self x input_fn batch_size outputs  & {:keys [as_iterable]} ]
    (py/call-attr-kw self "predict" [x input_fn batch_size outputs] {:as_iterable as_iterable }))
(defn predict-classes 
  "Returns predicted classes for given features. (deprecated argument values)

Warning: SOME ARGUMENT VALUES ARE DEPRECATED: `(as_iterable=False)`. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

Args:
  x: features.
  input_fn: Input function. If set, x must be None.
  batch_size: Override default batch size.
  as_iterable: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

Returns:
  Numpy array of predicted classes with shape [batch_size] (or an iterable
  of predicted classes if as_iterable is True). Each predicted class is
  represented by its class index (i.e. integer from 0 to n_classes-1)."
  [self x input_fn batch_size  & {:keys [as_iterable]} ]
    (py/call-attr-kw self "predict_classes" [x input_fn batch_size] {:as_iterable as_iterable }))
(defn predict-proba 
  "Returns predicted probabilities for given features. (deprecated argument values)

Warning: SOME ARGUMENT VALUES ARE DEPRECATED: `(as_iterable=False)`. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

Args:
  x: features.
  input_fn: Input function. If set, x and y must be None.
  batch_size: Override default batch size.
  as_iterable: If True, return an iterable which keeps yielding predictions
    for each example until inputs are exhausted. Note: The inputs must
    terminate if you want the iterable to terminate (e.g. be sure to pass
    num_epochs=1 if you are using something like read_batch_features).

Returns:
  Numpy array of predicted probabilities with shape [batch_size, n_classes]
  (or an iterable of predicted probabilities if as_iterable is True)."
  [self x input_fn batch_size  & {:keys [as_iterable]} ]
    (py/call-attr-kw self "predict_proba" [x input_fn batch_size] {:as_iterable as_iterable }))

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
