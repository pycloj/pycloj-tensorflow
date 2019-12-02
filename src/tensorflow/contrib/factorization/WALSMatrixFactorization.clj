(ns tensorflow.contrib.factorization.WALSMatrixFactorization
  "An Estimator for Weighted Matrix Factorization, using the WALS method.

  WALS (Weighted Alternating Least Squares) is an algorithm for weighted matrix
  factorization. It computes a low-rank approximation of a given sparse (n x m)
  matrix `A`, by a product of two matrices, `U * V^T`, where `U` is a (n x k)
  matrix and `V` is a (m x k) matrix. Here k is the rank of the approximation,
  also called the embedding dimension. We refer to `U` as the row factors, and
  `V` as the column factors.
  See tensorflow/contrib/factorization/g3doc/wals.md for the precise problem
  formulation.

  The training proceeds in sweeps: during a row_sweep, we fix `V` and solve for
  `U`. During a column sweep, we fix `U` and solve for `V`. Each one of these
  problems is an unconstrained quadratic minimization problem and can be solved
  exactly (it can also be solved in mini-batches, since the solution decouples
  across rows of each matrix).
  The alternating between sweeps is achieved by using a hook during training,
  which is responsible for keeping track of the sweeps and running preparation
  ops at the beginning of each sweep. It also updates the global_step variable,
  which keeps track of the number of batches processed since the beginning of
  training.
  The current implementation assumes that the training is run on a single
  machine, and will fail if `config.num_worker_replicas` is not equal to one.
  Training is done by calling `self.fit(input_fn=input_fn)`, where `input_fn`
  provides two tensors: one for rows of the input matrix, and one for rows of
  the transposed input matrix (i.e. columns of the original matrix). Note that
  during a row sweep, only row batches are processed (ignoring column batches)
  and vice-versa.
  Also note that every row (respectively every column) of the input matrix
  must be processed at least once for the sweep to be considered complete. In
  particular, training will not make progress if some rows are not generated by
  the `input_fn`.

  For prediction, given a new set of input rows `A'`, we compute a corresponding
  set of row factors `U'`, such that `U' * V^T` is a good approximation of `A'`.
  We call this operation a row projection. A similar operation is defined for
  columns. Projection is done by calling
  `self.get_projections(input_fn=input_fn)`, where `input_fn` satisfies the
  constraints given below.

  The input functions must satisfy the following constraints: Calling `input_fn`
  must return a tuple `(features, labels)` where `labels` is None, and
  `features` is a dict containing the following keys:

  TRAIN:
    * `WALSMatrixFactorization.INPUT_ROWS`: float32 SparseTensor (matrix).
      Rows of the input matrix to process (or to project).
    * `WALSMatrixFactorization.INPUT_COLS`: float32 SparseTensor (matrix).
      Columns of the input matrix to process (or to project), transposed.

  INFER:
    * `WALSMatrixFactorization.INPUT_ROWS`: float32 SparseTensor (matrix).
      Rows to project.
    * `WALSMatrixFactorization.INPUT_COLS`: float32 SparseTensor (matrix).
      Columns to project.
    * `WALSMatrixFactorization.PROJECT_ROW`: Boolean Tensor. Whether to project
      the rows or columns.
    * `WALSMatrixFactorization.PROJECTION_WEIGHTS` (Optional): float32 Tensor
      (vector). The weights to use in the projection.

  EVAL:
    * `WALSMatrixFactorization.INPUT_ROWS`: float32 SparseTensor (matrix).
      Rows to project.
    * `WALSMatrixFactorization.INPUT_COLS`: float32 SparseTensor (matrix).
      Columns to project.
    * `WALSMatrixFactorization.PROJECT_ROW`: Boolean Tensor. Whether to project
      the rows or columns.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce factorization (import-module "tensorflow.contrib.factorization"))

(defn WALSMatrixFactorization 
  "An Estimator for Weighted Matrix Factorization, using the WALS method.

  WALS (Weighted Alternating Least Squares) is an algorithm for weighted matrix
  factorization. It computes a low-rank approximation of a given sparse (n x m)
  matrix `A`, by a product of two matrices, `U * V^T`, where `U` is a (n x k)
  matrix and `V` is a (m x k) matrix. Here k is the rank of the approximation,
  also called the embedding dimension. We refer to `U` as the row factors, and
  `V` as the column factors.
  See tensorflow/contrib/factorization/g3doc/wals.md for the precise problem
  formulation.

  The training proceeds in sweeps: during a row_sweep, we fix `V` and solve for
  `U`. During a column sweep, we fix `U` and solve for `V`. Each one of these
  problems is an unconstrained quadratic minimization problem and can be solved
  exactly (it can also be solved in mini-batches, since the solution decouples
  across rows of each matrix).
  The alternating between sweeps is achieved by using a hook during training,
  which is responsible for keeping track of the sweeps and running preparation
  ops at the beginning of each sweep. It also updates the global_step variable,
  which keeps track of the number of batches processed since the beginning of
  training.
  The current implementation assumes that the training is run on a single
  machine, and will fail if `config.num_worker_replicas` is not equal to one.
  Training is done by calling `self.fit(input_fn=input_fn)`, where `input_fn`
  provides two tensors: one for rows of the input matrix, and one for rows of
  the transposed input matrix (i.e. columns of the original matrix). Note that
  during a row sweep, only row batches are processed (ignoring column batches)
  and vice-versa.
  Also note that every row (respectively every column) of the input matrix
  must be processed at least once for the sweep to be considered complete. In
  particular, training will not make progress if some rows are not generated by
  the `input_fn`.

  For prediction, given a new set of input rows `A'`, we compute a corresponding
  set of row factors `U'`, such that `U' * V^T` is a good approximation of `A'`.
  We call this operation a row projection. A similar operation is defined for
  columns. Projection is done by calling
  `self.get_projections(input_fn=input_fn)`, where `input_fn` satisfies the
  constraints given below.

  The input functions must satisfy the following constraints: Calling `input_fn`
  must return a tuple `(features, labels)` where `labels` is None, and
  `features` is a dict containing the following keys:

  TRAIN:
    * `WALSMatrixFactorization.INPUT_ROWS`: float32 SparseTensor (matrix).
      Rows of the input matrix to process (or to project).
    * `WALSMatrixFactorization.INPUT_COLS`: float32 SparseTensor (matrix).
      Columns of the input matrix to process (or to project), transposed.

  INFER:
    * `WALSMatrixFactorization.INPUT_ROWS`: float32 SparseTensor (matrix).
      Rows to project.
    * `WALSMatrixFactorization.INPUT_COLS`: float32 SparseTensor (matrix).
      Columns to project.
    * `WALSMatrixFactorization.PROJECT_ROW`: Boolean Tensor. Whether to project
      the rows or columns.
    * `WALSMatrixFactorization.PROJECTION_WEIGHTS` (Optional): float32 Tensor
      (vector). The weights to use in the projection.

  EVAL:
    * `WALSMatrixFactorization.INPUT_ROWS`: float32 SparseTensor (matrix).
      Rows to project.
    * `WALSMatrixFactorization.INPUT_COLS`: float32 SparseTensor (matrix).
      Columns to project.
    * `WALSMatrixFactorization.PROJECT_ROW`: Boolean Tensor. Whether to project
      the rows or columns.
  "
  [num_rows num_cols embedding_dimension & {:keys [unobserved_weight regularization_coeff row_init col_init num_row_shards num_col_shards row_weights col_weights use_factors_weights_cache_for_training use_gramian_cache_for_training max_sweeps model_dir config]
                       :or {regularization_coeff None max_sweeps None model_dir None config None}} ]
    (py/call-attr-kw factorization "WALSMatrixFactorization" [num_rows num_cols embedding_dimension] {:unobserved_weight unobserved_weight :regularization_coeff regularization_coeff :row_init row_init :col_init col_init :num_row_shards num_row_shards :num_col_shards num_col_shards :row_weights row_weights :col_weights col_weights :use_factors_weights_cache_for_training use_factors_weights_cache_for_training :use_gramian_cache_for_training use_gramian_cache_for_training :max_sweeps max_sweeps :model_dir model_dir :config config }))

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
  "Exports inference graph into given dir. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2017-03-25.
Instructions for updating:
Please use Estimator.export_savedmodel() instead.

Args:
  export_dir: A string containing a directory to write the exported graph
    and checkpoints.
  input_fn: If `use_deprecated_input_fn` is true, then a function that given
    `Tensor` of `Example` strings, parses it into features that are then
    passed to the model. Otherwise, a function that takes no argument and
    returns a tuple of (features, labels), where features is a dict of
    string key to `Tensor` and labels is a `Tensor` that's currently not
    used (and so can be `None`).
  input_feature_key: Only used if `use_deprecated_input_fn` is false. String
    key into the features dict returned by `input_fn` that corresponds to a
    the raw `Example` strings `Tensor` that the exported model will take as
    input. Can only be `None` if you're using a custom `signature_fn` that
    does not use the first arg (examples).
  use_deprecated_input_fn: Determines the signature format of `input_fn`.
  signature_fn: Function that returns a default signature and a named
    signature map, given `Tensor` of `Example` strings, `dict` of `Tensor`s
    for features and `Tensor` or `dict` of `Tensor`s for predictions.
  prediction_key: The key for a tensor in the `predictions` dict (output
    from the `model_fn`) to use as the `predictions` input to the
    `signature_fn`. Optional. If `None`, predictions will pass to
    `signature_fn` without filtering.
  default_batch_size: Default batch size of the `Example` placeholder.
  exports_to_keep: Number of exports to keep.
  checkpoint_path: the checkpoint path of the model to be exported. If it is
      `None` (which is default), will use the latest checkpoint in
      export_dir.

Returns:
  The string path to the exported directory. NB: this functionality was
  added ca. 2016/09/25; clients that depend on the return value may need
  to handle the case where this function returns None because subclasses
  are not returning a value."
  [self export_dir & {:keys [input_fn input_feature_key use_deprecated_input_fn signature_fn prediction_key default_batch_size exports_to_keep checkpoint_path]
                       :or {input_feature_key None signature_fn None prediction_key None exports_to_keep None checkpoint_path None}} ]
    (py/call-attr-kw self "export" [export_dir] {:input_fn input_fn :input_feature_key input_feature_key :use_deprecated_input_fn use_deprecated_input_fn :signature_fn signature_fn :prediction_key prediction_key :default_batch_size default_batch_size :exports_to_keep exports_to_keep :checkpoint_path checkpoint_path }))

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

(defn get-col-factors 
  "Returns the column factors of the model, loading them from checkpoint.

    Should only be run after training.

    Returns:
      A list of the column factors of the model.
    "
  [ self  ]
  (py/call-attr self "get_col_factors"  self  ))
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

(defn get-projections 
  "Computes the projections of the rows or columns given in input_fn.

    Runs predict() with the given input_fn, and returns the results. Should only
    be run after training.

    Args:
      input_fn: Input function which specifies the rows or columns to project.
    Returns:
      A generator of the projected factors.
    "
  [ self input_fn ]
  (py/call-attr self "get_projections"  self input_fn ))

(defn get-row-factors 
  "Returns the row factors of the model, loading them from checkpoint.

    Should only be run after training.

    Returns:
      A list of the row factors of the model.
    "
  [ self  ]
  (py/call-attr self "get_row_factors"  self  ))

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
