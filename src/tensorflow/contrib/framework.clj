(ns tensorflow.contrib.framework
  "Framework utilities.

@@assert_same_float_dtype
@@assert_scalar
@@assert_scalar_int
@@convert_to_tensor_or_sparse_tensor
@@get_graph_from_inputs
@@is_numeric_tensor
@@is_non_decreasing
@@is_strictly_increasing
@@is_tensor
@@reduce_sum_n
@@remove_squeezable_dimensions
@@with_shape
@@with_same_shape

@@deprecated
@@deprecated_args
@@deprecated_arg_values

@@arg_scope
@@add_arg_scope
@@current_arg_scope
@@has_arg_scope
@@arg_scoped_arguments

@@prepend_name_scope
@@strip_name_scope

@@add_model_variable
@@assert_global_step
@@assert_or_get_global_step
@@assign_from_checkpoint
@@assign_from_checkpoint_fn
@@assign_from_values
@@assign_from_values_fn
@@create_global_step
@@filter_variables
@@fuse_op
@@get_global_step
@@get_or_create_global_step
@@get_local_variables
@@get_model_variables
@@get_name_scope
@@get_trainable_variables
@@get_unique_variable
@@get_variables_by_name
@@get_variables_by_suffix
@@get_variable_full_name
@@get_variables_to_restore
@@get_variables
@@global_variable
@@local_variable
@@model_variable
@@variable
@@VariableDeviceChooser
@@convolutional_delta_orthogonal
@@convolutional_orthogonal_1d
@@convolutional_orthogonal_2d
@@convolutional_orthogonal_3d
@@zero_initializer

@@load_checkpoint
@@list_variables
@@load_variable
@@init_from_checkpoint
@@load_and_remap_matrix_initializer
@@load_embedding_initializer
@@load_linear_multiclass_bias_initializer
@@load_variable_slot_initializer

@@argsort
@@py_func
@@sort

@@get_placeholders

@@smart_cond
@@smart_constant_value
@@smart_case

@@BoundedTensorSpec
@@TensorSpec

@@RecordInput
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce framework (import-module "tensorflow.contrib.framework"))

(defn add-arg-scope 
  "Decorates a function with args so it can be used within an arg_scope.

  Args:
    func: function to decorate.

  Returns:
    A tuple with the decorated function func_with_args().
  "
  [ func ]
  (py/call-attr framework "add_arg_scope"  func ))

(defn add-model-variable 
  "Adds a variable to the `GraphKeys.MODEL_VARIABLES` collection.

  Args:
    var: a variable.
  "
  [ var ]
  (py/call-attr framework "add_model_variable"  var ))

(defn arg-scope 
  "Stores the default arguments for the given set of list_ops.

  For usage, please see examples at top of the file.

  Args:
    list_ops_or_scope: List or tuple of operations to set argument scope for or
      a dictionary containing the current scope. When list_ops_or_scope is a
      dict, kwargs must be empty. When list_ops_or_scope is a list or tuple,
      then every op in it need to be decorated with @add_arg_scope to work.
    **kwargs: keyword=value that will define the defaults for each op in
              list_ops. All the ops need to accept the given set of arguments.

  Yields:
    the current_scope, which is a dictionary of {op: {arg: value}}
  Raises:
    TypeError: if list_ops is not a list or a tuple.
    ValueError: if any op in list_ops has not be decorated with @add_arg_scope.
  "
  [ list_ops_or_scope ]
  (py/call-attr framework "arg_scope"  list_ops_or_scope ))

(defn arg-scoped-arguments 
  "Returns the list kwargs that arg_scope can set for a func.

  Args:
    func: function which has been decorated with @add_arg_scope.

  Returns:
    a list of kwargs names.
  "
  [ func ]
  (py/call-attr framework "arg_scoped_arguments"  func ))

(defn argsort 
  "Returns the indices of a tensor that give its sorted order along an axis.

  For a 1D tensor, `tf.gather(values, tf.argsort(values))` is equivalent to
  `tf.sort(values)`. For higher dimensions, the output has the same shape as
  `values`, but along the given axis, values represent the index of the sorted
  element in that slice of the tensor at the given position.
  
  Usage:

  ```python
  import tensorflow as tf
  a = [1, 10, 26.9, 2.8, 166.32, 62.3]
  b = tf.argsort(a,axis=-1,direction='ASCENDING',stable=False,name=None)
  c = tf.keras.backend.eval(b)
  # Here, c = [0 3 1 2 5 4]
  ```

  Args:
    values: 1-D or higher numeric `Tensor`.
    axis: The axis along which to sort. The default is -1, which sorts the last
      axis.
    direction: The direction in which to sort the values (`'ASCENDING'` or
      `'DESCENDING'`).
    stable: If True, equal elements in the original tensor will not be
      re-ordered in the returned order. Unstable sort is not yet implemented,
      but will eventually be the default for performance reasons. If you require
      a stable order, pass `stable=True` for forwards compatibility.
    name: Optional name for the operation.

  Returns:
    An int32 `Tensor` with the same shape as `values`. The indices that would
        sort each slice of the given `values` along the given `axis`.

  Raises:
    ValueError: If axis is not a constant scalar, or the direction is invalid.
  "
  [values & {:keys [axis direction stable name]
                       :or {name None}} ]
    (py/call-attr-kw framework "argsort" [values] {:axis axis :direction direction :stable stable :name name }))

(defn assert-global-step 
  "DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please switch to tf.train.assert_global_step"
  [ global_step_tensor ]
  (py/call-attr framework "assert_global_step"  global_step_tensor ))

(defn assert-or-get-global-step 
  "Verifies that a global step tensor is valid or gets one if None is given.

  If `global_step_tensor` is not None, check that it is a valid global step
  tensor (using `assert_global_step`). Otherwise find a global step tensor using
  `get_global_step` and return it.

  Args:
    graph: The graph to find the global step tensor for.
    global_step_tensor: The tensor to check for suitability as a global step. If
      None is given (the default), find a global step tensor.

  Returns:
    A tensor suitable as a global step, or `None` if none was provided and none
    was found.
  "
  [ graph global_step_tensor ]
  (py/call-attr framework "assert_or_get_global_step"  graph global_step_tensor ))

(defn assert-same-float-dtype 
  "Validate and return float type based on `tensors` and `dtype`.

  For ops such as matrix multiplication, inputs and weights must be of the
  same float type. This function validates that all `tensors` are the same type,
  validates that type is `dtype` (if supplied), and returns the type. Type must
  be a floating point type. If neither `tensors` nor `dtype` is supplied,
  the function will return `dtypes.float32`.

  Args:
    tensors: Tensors of input values. Can include `None` elements, which will be
        ignored.
    dtype: Expected type.

  Returns:
    Validated type.

  Raises:
    ValueError: if neither `tensors` nor `dtype` is supplied, or result is not
        float, or the common type of the inputs is not a floating point type.
  "
  [ tensors dtype ]
  (py/call-attr framework "assert_same_float_dtype"  tensors dtype ))

(defn assert-scalar 
  "Asserts that the given `tensor` is a scalar (i.e. zero-dimensional).

  This function raises `ValueError` unless it can be certain that the given
  `tensor` is a scalar. `ValueError` is also raised if the shape of `tensor` is
  unknown.

  Args:
    tensor: A `Tensor`.
    name:  A name for this operation. Defaults to \"assert_scalar\"
    message: A string to prefix to the default message.

  Returns:
    The input tensor (potentially converted to a `Tensor`).

  Raises:
    ValueError: If the tensor is not scalar (rank 0), or if its shape is
      unknown.
  "
  [ tensor name message ]
  (py/call-attr framework "assert_scalar"  tensor name message ))

(defn assert-scalar-int 
  "Assert `tensor` is 0-D, of type `tf.int32` or `tf.int64`.

  Args:
    tensor: `Tensor` to test.
    name: Name of the op and of the new `Tensor` if one is created.
  Returns:
    `tensor`, for chaining.
  Raises:
    ValueError: if `tensor` is not 0-D, of integer type.
  "
  [ tensor name ]
  (py/call-attr framework "assert_scalar_int"  tensor name ))
(defn assign-from-checkpoint 
  "Creates an operation to assign specific variables from a checkpoint.

  Args:
    model_path: The full path to the model checkpoint. To get latest checkpoint
      use `model_path = tf.train.latest_checkpoint(checkpoint_dir)`
    var_list: A list of (possibly partitioned) `Variable` objects or a
      dictionary mapping names in the checkpoint to the corresponding variables
      or list of variables to initialize from that checkpoint value. For
      partitioned Variables, the name in the checkpoint must be the full
      variable, not the name of the partitioned variable, eg. \"my_var\" rather
      than \"my_var/part_4\". If empty, returns no_op(), {}.
    ignore_missing_vars: Boolean, if True ignore variables missing in the
      checkpoint with a warning instead of failing.

  Returns:
    the restore_op and the feed_dict that need to be run to restore var_list.

  Raises:
    ValueError: If `ignore_missing_vars` is False and the checkpoint specified
        at `model_path` is missing one of the variables in `var_list`.
  "
  [model_path var_list  & {:keys [ignore_missing_vars]} ]
    (py/call-attr-kw framework "assign_from_checkpoint" [model_path var_list] {:ignore_missing_vars ignore_missing_vars }))
(defn assign-from-checkpoint-fn 
  "Returns a function that assigns specific variables from a checkpoint.

  If ignore_missing_vars is True and no variables are found in the checkpoint
  it returns None.

  Args:
    model_path: The full path to the model checkpoint. To get latest checkpoint
      use `model_path = tf.train.latest_checkpoint(checkpoint_dir)`
    var_list: A list of `Variable` objects or a dictionary mapping names in the
      checkpoint to the corresponding variables to initialize. If empty or
      `None`, it would return `no_op(), None`.
    ignore_missing_vars: Boolean, if True it would ignore variables missing in
      the checkpoint with a warning instead of failing.
    reshape_variables: Boolean, if True it would automatically reshape variables
      which are of different shape then the ones stored in the checkpoint but
      which have the same number of elements.

  Returns:
    A function that takes a single argument, a `tf.compat.v1.Session`, that
    applies the
    assignment operation. If no matching variables were found in the checkpoint
    then `None` is returned.

  Raises:
    ValueError: If var_list is empty.
  "
  [model_path var_list  & {:keys [ignore_missing_vars reshape_variables]} ]
    (py/call-attr-kw framework "assign_from_checkpoint_fn" [model_path var_list] {:ignore_missing_vars ignore_missing_vars :reshape_variables reshape_variables }))

(defn assign-from-values 
  "Creates an assignment operation from a given mapping.

  This function provides a mechanism for performing assignment of variables
  to values in a way that does not fill the graph with large assignment values.

  Args:
    var_names_to_values: A map from variable names to values.

  Returns:
    assign_op: An `Operation` that assigns each of the given variables to the
      requested values.
    feed_dict: The feed dictionary to use when evaluating `assign_op`.

  Raises:
    ValueError: if any of the given variable names were not found.
  "
  [ var_names_to_values ]
  (py/call-attr framework "assign_from_values"  var_names_to_values ))

(defn assign-from-values-fn 
  "Returns a function that assigns specific variables from the given values.

  This function provides a mechanism for performing assignment of variables
  to values in a way that does not fill the graph with large assignment values.

  Args:
    var_names_to_values: A map from variable names to values.

  Returns:
    A function that takes a single argument, a `tf.compat.v1.Session`, that
    applies the
    assignment operation.

  Raises:
    ValueError: if any of the given variable names were not found.
  "
  [ var_names_to_values ]
  (py/call-attr framework "assign_from_values_fn"  var_names_to_values ))

(defn convert-to-tensor-or-sparse-tensor 
  "Converts value to a `SparseTensor` or `Tensor`.

  Args:
    value: A `SparseTensor`, `SparseTensorValue`, or an object whose type has a
      registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the type
      is inferred from the type of `value`.
    name: Optional name to use if a new `Tensor` is created.

  Returns:
    A `SparseTensor` or `Tensor` based on `value`.

  Raises:
    RuntimeError: If result type is incompatible with `dtype`.
  "
  [ value dtype name ]
  (py/call-attr framework "convert_to_tensor_or_sparse_tensor"  value dtype name ))

(defn create-global-step 
  "Create global step tensor in graph. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please switch to tf.train.create_global_step

This API is deprecated. Use core framework training version instead.

Args:
  graph: The graph in which to create the global step tensor. If missing, use
    default graph.

Returns:
  Global step tensor.

Raises:
  ValueError: if global step tensor is already defined."
  [ graph ]
  (py/call-attr framework "create_global_step"  graph ))

(defn current-arg-scope 
  ""
  [  ]
  (py/call-attr framework "current_arg_scope"  ))
(defn deprecated 
  "Decorator for marking functions or methods deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called. It has the following format:

    <function> (from <module>) is deprecated and will be removed after <date>.
    Instructions for updating:
    <instructions>

  If `date` is None, 'after <date>' is replaced with 'in a future version'.
  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated)' is appended
  to the first line of the docstring and a deprecation notice is prepended
  to the rest of the docstring.

  Args:
    date: String or None. The date the function is scheduled to be removed.
      Must be ISO 8601 (YYYY-MM-DD), or None.
    instructions: String. Instructions on how to update code using the
      deprecated function.
    warn_once: Boolean. Set to `True` to warn only the first time the decorated
      function is called. Otherwise, every call will log a warning.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not None or in ISO 8601 format, or instructions are
      empty.
  "
  [date instructions  & {:keys [warn_once]} ]
    (py/call-attr-kw framework "deprecated" [date instructions] {:warn_once warn_once }))
(defn deprecated-arg-values 
  "Decorator for marking specific function argument values as deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called with the deprecated argument values. It has the following format:

    Calling <function> (from <module>) with <arg>=<value> is deprecated and
    will be removed after <date>. Instructions for updating:
      <instructions>

  If `date` is None, 'after <date>' is replaced with 'in a future version'.
  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated arguments)' is
  appended to the first line of the docstring and a deprecation notice is
  prepended to the rest of the docstring.

  Args:
    date: String or None. The date the function is scheduled to be removed.
      Must be ISO 8601 (YYYY-MM-DD), or None
    instructions: String. Instructions on how to update code using the
      deprecated function.
    warn_once: If `True`, warn only the first time this function is called with
      deprecated argument values. Otherwise, every call (with a deprecated
      argument value) will log a warning.
    **deprecated_kwargs: The deprecated argument values.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not None or in ISO 8601 format, or instructions are
      empty.
  "
  [date instructions  & {:keys [warn_once]} ]
    (py/call-attr-kw framework "deprecated_arg_values" [date instructions] {:warn_once warn_once }))

(defn deprecated-args 
  "Decorator for marking specific function arguments as deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called with the deprecated argument. It has the following format:

    Calling <function> (from <module>) with <arg> is deprecated and will be
    removed after <date>. Instructions for updating:
      <instructions>

  If `date` is None, 'after <date>' is replaced with 'in a future version'.
  <function> includes the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated arguments)' is
  appended to the first line of the docstring and a deprecation notice is
  prepended to the rest of the docstring.

  Args:
    date: String or None. The date the function is scheduled to be removed.
      Must be ISO 8601 (YYYY-MM-DD), or None.
    instructions: String. Instructions on how to update code using the
      deprecated function.
    *deprecated_arg_names_or_tuples: String or 2-Tuple(String,
      [ok_vals]).  The string is the deprecated argument name.
      Optionally, an ok-value may be provided.  If the user provided
      argument equals this value, the warning is suppressed.
    **kwargs: If `warn_once=False` is passed, every call with a deprecated
      argument will log a warning. The default behavior is to only warn the
      first time the function is called with any given deprecated argument.
      All other kwargs raise `ValueError`.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not None or in ISO 8601 format, instructions are
      empty, the deprecated arguments are not present in the function
      signature, the second element of a deprecated_tuple is not a
      list, or if a kwarg other than `warn_once` is passed.
  "
  [ date instructions ]
  (py/call-attr framework "deprecated_args"  date instructions ))
(defn filter-variables 
  "Filter a list of variables using regular expressions.

  First includes variables according to the list of include_patterns.
  Afterwards, eliminates variables according to the list of exclude_patterns.

  For example, one can obtain a list of variables with the weights of all
  convolutional layers (depending on the network definition) by:

  ```python
  variables = tf.contrib.framework.get_model_variables()
  conv_weight_variables = tf.contrib.framework.filter_variables(
      variables,
      include_patterns=['Conv'],
      exclude_patterns=['biases', 'Logits'])
  ```

  Args:
    var_list: list of variables.
    include_patterns: list of regular expressions to include. Defaults to None,
      which means all variables are selected according to the include rules. A
      variable is included if it matches any of the include_patterns.
    exclude_patterns: list of regular expressions to exclude. Defaults to None,
      which means all variables are selected according to the exclude rules. A
      variable is excluded if it matches any of the exclude_patterns.
    reg_search: boolean. If True (default), performs re.search to find matches
      (i.e. pattern can match any substring of the variable name). If False,
      performs re.match (i.e. regexp should match from the beginning of the
      variable name).

  Returns:
    filtered list of variables.
  "
  [var_list include_patterns exclude_patterns  & {:keys [reg_search]} ]
    (py/call-attr-kw framework "filter_variables" [var_list include_patterns exclude_patterns] {:reg_search reg_search }))

(defn fuse-op 
  "Fuse subgraph between input_nodes and output_nodes into a single custom op.

  Args:
    graph_def: A graph_pb2.GraphDef proto.
    input_nodes: input nodes to the subgraph to be fused.
    output_nodes: output nodes to the subgraph to be fused.
    output_dtypes: A list of output datatypes for the custom op
    output_quantized: A boolean flag that indicates if output is quantized
    op_name: fused op name.
    op_type: fused op type.
  Returns:
    The GraphDef of the new graph.

  Raises:
    TypeError: If 'graph_def' is not a graph_pb2.GraphDef proto.
  "
  [ graph_def input_nodes output_nodes output_dtypes output_quantized op_name op_type ]
  (py/call-attr framework "fuse_op"  graph_def input_nodes output_nodes output_dtypes output_quantized op_name op_type ))

(defn get-global-step 
  "DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please switch to tf.train.get_global_step"
  [ graph ]
  (py/call-attr framework "get_global_step"  graph ))

(defn get-graph-from-inputs 
  "Returns the appropriate graph to use for the given inputs.

  1. If `graph` is provided, we validate that all inputs in `op_input_list` are
     from the same graph.
  2. Otherwise, we attempt to select a graph from the first Operation- or
     Tensor-valued input in `op_input_list`, and validate that all other
     such inputs are in the same graph.
  3. If the graph was not specified and it could not be inferred from
     `op_input_list`, we attempt to use the default graph.

  Args:
    op_input_list: A list of inputs to an operation, which may include `Tensor`,
      `Operation`, and other objects that may be converted to a graph element.
    graph: (Optional) The explicit graph to use.

  Raises:
    TypeError: If `op_input_list` is not a list or tuple, or if graph is not a
      Graph.
    ValueError: If a graph is explicitly passed and not all inputs are from it,
      or if the inputs are from multiple graphs, or we could not find a graph
      and there was no default graph.

  Returns:
    The appropriate graph to use for the given inputs.
  "
  [ op_input_list graph ]
  (py/call-attr framework "get_graph_from_inputs"  op_input_list graph ))

(defn get-local-variables 
  "Gets the list of local variables, filtered by scope and/or suffix.

  Args:
    scope: an optional scope for filtering the variables to return.
    suffix: an optional suffix for filtering the variables to return.

  Returns:
    a list of variables in collection with scope and suffix.
  "
  [ scope suffix ]
  (py/call-attr framework "get_local_variables"  scope suffix ))

(defn get-model-variables 
  "Gets the list of model variables, filtered by scope and/or suffix.

  Args:
    scope: an optional scope for filtering the variables to return.
    suffix: an optional suffix for filtering the variables to return.

  Returns:
    a list of variables in collection with scope and suffix.
  "
  [ scope suffix ]
  (py/call-attr framework "get_model_variables"  scope suffix ))

(defn get-name-scope 
  "Returns the current name scope of the default graph.

  For example:

    ```python
    with tf.name_scope('scope1'):
      with tf.name_scope('scope2'):
        print(tf.contrib.framework.get_name_scope())
    ```
    would print the string `scope1/scope2`.

  Returns:
    A string representing the current name scope.
  "
  [  ]
  (py/call-attr framework "get_name_scope"  ))

(defn get-or-create-global-step 
  "Returns and create (if necessary) the global step tensor. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please switch to tf.train.get_or_create_global_step

Args:
  graph: The graph in which to create the global step tensor. If missing, use
    default graph.

Returns:
  The global step tensor."
  [ graph ]
  (py/call-attr framework "get_or_create_global_step"  graph ))

(defn get-placeholders 
  "Get placeholders of a graph.

  For example:

  ```python
  a = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2, 2], name='a')
  a = tf.compat.v1.placeholder(dtype=tf.int32, shape=[3, 2], name='b')

  tf.contrib.framework.get_placeholders(tf.compat.v1.get_default_graph())
  # Returns:
  #  [<tf.Tensor 'a:0' shape=(2, 2) dtype=float32>,
  #   <tf.Tensor 'b:0' shape=(3, 2) dtype=int32>]
  ```

  Args:
    graph: A tf.Graph.
  Returns:
    A list contains all placeholders of given graph.

  Raises:
    TypeError: If `graph` is not a tensorflow graph.
  "
  [ graph ]
  (py/call-attr framework "get_placeholders"  graph ))

(defn get-trainable-variables 
  "Gets the list of trainable variables, filtered by scope and/or suffix.

  Args:
    scope: an optional scope for filtering the variables to return.
    suffix: an optional suffix for filtering the variables to return.

  Returns:
    a list of variables in the trainable collection with scope and suffix.
  "
  [ scope suffix ]
  (py/call-attr framework "get_trainable_variables"  scope suffix ))

(defn get-unique-variable 
  "Gets the variable uniquely identified by that var_op_name.

  Args:
    var_op_name: the full name of the variable op, including the scope.

  Returns:
    a tensorflow variable.

  Raises:
    ValueError: if no variable uniquely identified by the name exists.
  "
  [ var_op_name ]
  (py/call-attr framework "get_unique_variable"  var_op_name ))

(defn get-variable-full-name 
  "Returns the full name of a variable.

  For normal Variables, this is the same as the var.op.name.  For
  sliced or PartitionedVariables, this name is the same for all the
  slices/partitions. In both cases, this is normally the name used in
  a checkpoint file.

  Args:
    var: A `Variable` object.

  Returns:
    A string that is the full name.
  "
  [ var ]
  (py/call-attr framework "get_variable_full_name"  var ))
(defn get-variables 
  "Gets the list of variables, filtered by scope and/or suffix.

  Args:
    scope: an optional scope for filtering the variables to return. Can be a
      variable scope or a string.
    suffix: an optional suffix for filtering the variables to return.
    collection: in which collection search for. Defaults to
      `GraphKeys.GLOBAL_VARIABLES`.

  Returns:
    a list of variables in collection with scope and suffix.
  "
  [scope suffix  & {:keys [collection]} ]
    (py/call-attr-kw framework "get_variables" [scope suffix] {:collection collection }))

(defn get-variables-by-name 
  "Gets the list of variables that were given that name.

  Args:
    given_name: name given to the variable without any scope.
    scope: an optional scope for filtering the variables to return.

  Returns:
    a copied list of variables with the given name and scope.
  "
  [ given_name scope ]
  (py/call-attr framework "get_variables_by_name"  given_name scope ))

(defn get-variables-by-suffix 
  "Gets the list of variables that end with the given suffix.

  Args:
    suffix: suffix for filtering the variables to return.
    scope: an optional scope for filtering the variables to return.

  Returns:
    a copied list of variables with the given name and prefix.
  "
  [ suffix scope ]
  (py/call-attr framework "get_variables_by_suffix"  suffix scope ))

(defn get-variables-to-restore 
  "Gets the list of the variables to restore.

  Args:
    include: an optional list/tuple of scope strings for filtering which
      variables from the VARIABLES collection to include. None would include all
      the variables.
    exclude: an optional list/tuple of scope strings for filtering which
      variables from the VARIABLES collection to exclude. None it would not
      exclude any.

  Returns:
    a list of variables to restore.

  Raises:
    TypeError: include or exclude is provided but is not a list or a tuple.
  "
  [ include exclude ]
  (py/call-attr framework "get_variables_to_restore"  include exclude ))

(defn global-variable 
  "Create a variable with a value and add it to `GraphKeys.GLOBAL_VARIABLES`.

  Args:
    initial_value: See variables.Variable.__init__.
    validate_shape: See variables.Variable.__init__.
    name: See variables.Variable.__init__.
    use_resource: If `True` use a ResourceVariable instead of a Variable.

  Returns:
    New variable.
  "
  [initial_value & {:keys [validate_shape name use_resource]
                       :or {name None use_resource None}} ]
    (py/call-attr-kw framework "global_variable" [initial_value] {:validate_shape validate_shape :name name :use_resource use_resource }))

(defn has-arg-scope 
  "Checks whether a func has been decorated with @add_arg_scope or not.

  Args:
    func: function to check.

  Returns:
    a boolean.
  "
  [ func ]
  (py/call-attr framework "has_arg_scope"  func ))

(defn init-from-checkpoint 
  "Using assignment map initializes current variables with loaded tensors.

  Note: This overrides default initialization ops of specified variables and
  redefines dtype.

  Assignment map supports following syntax:

  * `'checkpoint_scope_name/': 'scope_name/'` - will load all variables in
    current `scope_name` from `checkpoint_scope_name` with matching variable
    names.
  * `'checkpoint_scope_name/some_other_variable': 'scope_name/variable_name'` -
    will initialize `scope_name/variable_name` variable
    from `checkpoint_scope_name/some_other_variable`.
  * `'scope_variable_name': variable` - will initialize given `tf.Variable`
    object with variable from the checkpoint.
  * `'scope_variable_name': list(variable)` - will initialize list of
    partitioned variables with variable from the checkpoint.
  * `'/': 'scope_name/'` - will load all variables in current `scope_name` from
    checkpoint's root (e.g. no scope).

  Supports loading into partitioned variables, which are represented as
  `'<variable>/part_<part #>'`.

  Example:

  ```python
    # Create variables.
    with tf.compat.v1.variable_scope('test'):
      m = tf.compat.v1.get_variable('my_var')
    with tf.compat.v1.variable_scope('test2'):
      var2 = tf.compat.v1.get_variable('my_var')
    var3 = tf.compat.v1.get_variable(name=\"my1\", shape=[100, 100],
                           partitioner=lambda shape, dtype: [5, 1])
    ...
    # Specify which variables to initialize from checkpoint.
    init_from_checkpoint(checkpoint_dir, {
      'some_var': 'test/my_var',
      'some_scope/': 'test2/'})
    ...
    # Or use `Variable` objects to identify what to initialize.
    init_from_checkpoint(checkpoint_dir, {
      'some_scope/var2': var2,
    })
    # Initialize partitioned variables
    init_from_checkpoint(checkpoint_dir, {
      'some_var_from_ckpt': 'part_var',
    })
    # Or specifying the list of `Variable` objects.
    init_from_checkpoint(checkpoint_dir, {
      'some_var_from_ckpt': var3._get_variable_list(),
    })
    ...
    # Initialize variables as usual.
    session.run(tf.get_all_variables())
  ```

  Args:
    checkpoint_dir: Directory with checkpoints file or path to checkpoint.
    assignment_map: Dict, where keys are names of the variables in the
      checkpoint and values are current variables or names of current variables
      (in default graph).

  Raises:
    tf.errors.OpError: If missing checkpoints or tensors in checkpoints.
    ValueError: If missing variables in current graph.
  "
  [ checkpoint_dir assignment_map ]
  (py/call-attr framework "init_from_checkpoint"  checkpoint_dir assignment_map ))

(defn is-tensor 
  "Checks whether `x` is a tensor or \"tensor-like\".

  If `is_tensor(x)` returns `True`, it is safe to assume that `x` is a tensor or
  can be converted to a tensor using `ops.convert_to_tensor(x)`.

  Args:
    x: A python object to check.

  Returns:
    `True` if `x` is a tensor or \"tensor-like\", `False` if not.
  "
  [ x ]
  (py/call-attr framework "is_tensor"  x ))

(defn list-variables 
  "Returns list of all variables in the latest checkpoint.

  Args:
    checkpoint_dir: Directory with checkpoints file or path to checkpoint.

  Returns:
    List of tuples `(name, shape)`.
  "
  [ checkpoint_dir ]
  (py/call-attr framework "list_variables"  checkpoint_dir ))

(defn load-and-remap-matrix-initializer 
  "Returns a var initializer for loading and remapping a 2-D (matrix) tensor.

  The returned initializer loads a 2-D (matrix) `Tensor` with name
  `old_tensor_name` from the checkpoint at `ckpt_path`. It will reorder the
  rows/columns according to the specified vocab files and append additional
  out-of-vocabulary rows/columns according to the number of OOV buckets.

  The format of the file at the `{old,new}_{row,col}_vocab_file` path should be
  a text file, with each line containing a single entity within the vocabulary.
  Let the function `line_of(f, \"x\")` return the 0-indexed line number of the
  entity \"x\" in file f, and the function `entity_at(f, i)` return the entity at
  line i of file f. Then, row i of the new output matrix will be taken from row
  `line_of(old_row_vocab_file, entity_at(new_row_vocab_file, i))` of the old
  matrix. If any entity in `new_row_vocab_file` is not found in
  `old_row_vocab_file`, that row is considered a \"missing\" row, and its values
  will be initialized using the `initializer` arg. The same logic also applies
  for the columns.

  For example, assuming that:

  * `old_row_vocab_file` contains \"mercury\nvenus\nmars\"
  * `new_row_vocab_file` contains \"venus\njupiter\nmercury\"
  * `old_col_vocab_file` contains \"good\nbetter\nbest\"
  * `new_col_vocab_file` contains \"good\nbest\nfantastic\"
  * `initializer` returns the natural numbers `[1, 2, 3, 4, ...]`
  * `w(i, j)` represents the value from row i, column j of the old matrix

  Then the new output matrix will look like:

  `[[w(1, 0), w(1, 2), 1],
    [2,       3,       4],
    [w(0, 0), w(0, 2), 5]]`

  If we further specify that:

  * `num_row_oov_buckets` == 2
  * `num_col_oov_buckets` == 1

  Then the new output matrix will look like:

  `[[w(1, 0), w(1, 2), 1,  12],
    [2,       3,       4,  13],
    [w(0, 0), w(0, 2), 5,  14],
    [6,       7,       8,  15],
    [9,       10,      11, 16]]`

  If `{old,new}_row_vocab_file` are None, we assume that the old and new row
  vocab files are the same, and no row remapping is done. If
  `{old,new}_col_vocab_file` are None, we assume that the old and new column
  vocab files are the same, and no column remapping is done.

  The returned initializer only supports div-partitioning along the row axis. It
  does not support partitioning along the column axis (as this is not common in
  practice) or mod-partitioning.

  NOTE: When this is used to warm-start variables, client code should use
  `tf.lookup.index_table_from_tensor()` like
  contrib/layers/python/layers/feature_column.py does, as opposed to
  `tf.feature_to_id()` - in order to ensure the underlying lookup tables are the
  same.

  Args:
    ckpt_path: Path to the TensorFlow checkpoint (version 2, `TensorBundle`)
      from which the old matrix `Tensor` will be loaded.
    old_tensor_name: Name of the 2-D `Tensor` to load from checkpoint.
    new_row_vocab_size: `int` specifying the number of entries in
      `new_row_vocab_file`. If no row remapping is needed (no row vocab
      provided), this should be equal to the number of rows to load from the old
      matrix (which can theoretically be smaller than the number of rows in the
      old matrix).
    new_col_vocab_size: `int` specifying the number of entries in
      `new_col_vocab_file`. If no column remapping is needed (no column vocab
      provided), this should be equal to the number of columns in the old
      matrix.
    old_row_vocab_size: The number of entries to consider in the old vocabulary.
      With the default value of -1, the entire old row vocabulary file will be
      used.  Otherwise, only the first `old_row_vocab_size` entries will be
      considered for remapping.Must be smaller than the length of
      `old_row_vocab_file`.  NOTE: we do not provide an equivalent
      `old_col_vocab_size` for classes.
    old_row_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the old row vocabulary file. Can be None, which represents no
      remapping on the row axis.
    new_row_vocab_file: A scalar `Tensor` of type `string` containing the path
      to the new row vocabulary file. Can be None, which represents no remapping
      on the row axis.
    old_col_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the old column vocabulary file. Can be None, which represents no
      remapping on the column axis.
    new_col_vocab_file: A scalar `Tensor` of type `string` containing the path
      to the new column vocabulary file. Can be None, which represents no
      remapping on the column axis.
    num_row_oov_buckets: `int` specifying the number of out-of-vocabulary rows
      to append. Must be >= 0.
    num_col_oov_buckets: `int` specifying the number of out-of-vocabulary
      columns to append. Must be >= 0.
    initializer: Initializer function to initialize missing values. Accepts a
      1-D tensor as the arg to specify the shape of the returned tensor. If
      `None`, defaults to using `zeros_initializer()`.
    max_rows_in_memory: `int` specifying the maximum number of rows to load from
      the checkpoint at once. If less than or equal to 0, the entire matrix will
      be loaded into memory. Setting this arg trades increased disk reads for
      lower memory usage.

  Returns:
    A variable initializer function that should be used to initialize a
    (potentially partitioned) `Variable` whose complete shape is
    `[new_row_vocab_size + num_row_oov_buckets, new_col_vocab_size +
    num_col_oov_buckets]`.

  Raises:
    TypeError: If `initializer` is specified but not callable.
  "
  [ckpt_path old_tensor_name new_row_vocab_size new_col_vocab_size & {:keys [old_row_vocab_size old_row_vocab_file new_row_vocab_file old_col_vocab_file new_col_vocab_file num_row_oov_buckets num_col_oov_buckets initializer max_rows_in_memory]
                       :or {old_row_vocab_file None new_row_vocab_file None old_col_vocab_file None new_col_vocab_file None initializer None}} ]
    (py/call-attr-kw framework "load_and_remap_matrix_initializer" [ckpt_path old_tensor_name new_row_vocab_size new_col_vocab_size] {:old_row_vocab_size old_row_vocab_size :old_row_vocab_file old_row_vocab_file :new_row_vocab_file new_row_vocab_file :old_col_vocab_file old_col_vocab_file :new_col_vocab_file new_col_vocab_file :num_row_oov_buckets num_row_oov_buckets :num_col_oov_buckets num_col_oov_buckets :initializer initializer :max_rows_in_memory max_rows_in_memory }))

(defn load-checkpoint 
  "Returns CheckpointReader for latest checkpoint.

  Args:
    filepattern: Directory with checkpoints file or path to checkpoint.

  Returns:
    `CheckpointReader` object.

  Raises:
    ValueError: if checkpoint_dir doesn't have 'checkpoint' file or checkpoints.
  "
  [ filepattern ]
  (py/call-attr framework "load_checkpoint"  filepattern ))

(defn load-embedding-initializer 
  "Returns a variable initializer for loading pre-trained embeddings.

  Wrapper around `load_and_remap_matrix_initializer()` specialized for loading
  embedding weights and remapping according to the provided vocab files. See
  docs for `load_and_remap_matrix_initializer()` for more details.

  NOTE: Only for use with div-partitioned variables / vocabularies.

  Args:
    ckpt_path: Path to the TensorFlow checkpoint (version 2, `TensorBundle`)
      from which the old matrix `Tensor` will be loaded.
    embedding_tensor_name: Name of the 2-D `Tensor` to load from checkpoint.
    new_vocab_size: Number of entries in the new vocab.
    embedding_dim: `int` specifying the dimension of the embedding vectors from
      the checkpoint. Must match the number of columns in the old embedding
      matrix.
    old_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the old vocabulary file.
    new_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the new vocabulary file.
    old_vocab_size: The number of entries to consider in the old vocabulary.
      With the default value of -1, the entire old row vocabulary file will be
      used.  Otherwise, only the first `old_vocab_size` entries will be
      considered for remapping.Must be smaller than the length of
      `old_row_vocab_file`.
    num_oov_buckets: `int` specifying the number of out-of-vocabulary
      buckets to use. Must be >= 0.
    initializer: Initializer function that accepts a 1-D tensor as the arg to
      specify the shape of the returned tensor. If `None`, defaults to using
      `truncated_normal_initializer()`.
    max_rows_in_memory: `int` specifying the maximum number of rows to load from
      the checkpoint at once. If less than or equal to 0, the entire matrix will
      be loaded into memory. Setting this arg trades increased disk reads for
      lower memory usage.

  Returns:
    A variable initializer function.
  "
  [ckpt_path embedding_tensor_name new_vocab_size embedding_dim old_vocab_file new_vocab_file & {:keys [old_vocab_size num_oov_buckets initializer max_rows_in_memory]
                       :or {initializer None}} ]
    (py/call-attr-kw framework "load_embedding_initializer" [ckpt_path embedding_tensor_name new_vocab_size embedding_dim old_vocab_file new_vocab_file] {:old_vocab_size old_vocab_size :num_oov_buckets num_oov_buckets :initializer initializer :max_rows_in_memory max_rows_in_memory }))

(defn load-linear-multiclass-bias-initializer 
  "Loads pre-trained multi-class biases for linear models from checkpoint.

  Wrapper around `load_and_remap_matrix_initializer()` specialized for loading
  multi-class bias and remapping according to the provided vocab files. See docs
  for `load_and_remap_matrix_initializer()` for more details. In this case, the
  provided row_vocab is the class vocabulary, and the expected shape is
  `[new_class_vocab_size, 1]`.

  Args:
    ckpt_path: Path to the TensorFlow checkpoint (version 2, `TensorBundle`)
      from which the old matrix `Tensor` will be loaded.
    bias_tensor_name: Tensor name to load from in the checkpoints.
    new_class_vocab_size: Number of entries in the new class vocab.
    old_class_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the old class vocabulary file.
    new_class_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the new class vocabulary file.
    num_class_oov_buckets: `int` specifying the number of out-of-vocabulary
      buckets to use for the classes. Must be >= 0.
    initializer: Initializer function that accepts a 1-D tensor as the arg to
      specify the shape of the returned tensor. If `None`, defaults to using
      `zeros_initializer()`.
    max_rows_in_memory: `int` specifying the maximum number of rows to load from
      the checkpoint at once. If less than or equal to 0, the entire matrix will
      be loaded into memory. Setting this arg trades increased disk reads for
      lower memory usage.

  Returns:
    A variable initializer function.
  "
  [ckpt_path bias_tensor_name new_class_vocab_size old_class_vocab_file new_class_vocab_file & {:keys [num_class_oov_buckets initializer max_rows_in_memory]
                       :or {initializer None}} ]
    (py/call-attr-kw framework "load_linear_multiclass_bias_initializer" [ckpt_path bias_tensor_name new_class_vocab_size old_class_vocab_file new_class_vocab_file] {:num_class_oov_buckets num_class_oov_buckets :initializer initializer :max_rows_in_memory max_rows_in_memory }))

(defn load-variable 
  "Returns a Tensor with the contents of the given variable in the checkpoint.

  Args:
    checkpoint_dir: Directory with checkpoints file or path to checkpoint.
    name: Name of the tensor to return.

  Returns:
    `Tensor` object.
  "
  [ checkpoint_dir name ]
  (py/call-attr framework "load_variable"  checkpoint_dir name ))

(defn load-variable-slot-initializer 
  "Loads pre-trained multi-class slots for linear models from checkpoint.

  Wrapper around `load_and_remap_matrix_initializer()` specialized for loading
  multi-class slots (such as optimizer accumulators) and remapping them
  according to the provided vocab files. See docs for
  `load_and_remap_matrix_initializer()` for more details.  Takes in a
  `variable_scope._PartitionInfo` representing the slot's primary `Variable`'s
  partitioning.  This is necessary since accumulator `Variable` creation ignores
  primary scoping and partitioning information.

  Args:
    ckpt_path: Path to the TensorFlow checkpoint (version 2, `TensorBundle`)
      from which the old matrix `Tensor` will be loaded.
    old_tensor_name: Name of the 2-D `Tensor` to load from checkpoint.
    primary_partition_info: A `variable_scope._PartitionInfo` containing this
      slot's primary `Variable`'s partitioning information.  This is used to
      calculate the offset and override the partition_info passed to the call to
      _initialize.
    new_row_vocab_size: `int` specifying the number of entries in
      `new_row_vocab_file`. If no row remapping is needed (no row vocab
      provided), this should be equal to the number of rows to load from the old
      matrix (which can theoretically be smaller than the number of rows in the
      old matrix).
    new_col_vocab_size: `int` specifying the number of entries in
      `new_col_vocab_file`. If no column remapping is needed (no column vocab
      provided), this should be equal to the number of columns in the old
      matrix.
    old_row_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the old row vocabulary file. Can be None, which represents no
      remapping on the row axis.
    new_row_vocab_file: A scalar `Tensor` of type `string` containing the path
      to the new row vocabulary file. Can be None, which represents no remapping
      on the row axis.
    old_col_vocab_file: A scalar `Tensor` of type `string` containing the
      path to the old column vocabulary file. Can be None, which represents no
      remapping on the column axis.
    new_col_vocab_file: A scalar `Tensor` of type `string` containing the path
      to the new column vocabulary file. Can be None, which represents no
      remapping on the column axis.
    num_row_oov_buckets: `int` specifying the number of out-of-vocabulary rows
      to append. Must be >= 0.
    num_col_oov_buckets: `int` specifying the number of out-of-vocabulary
      columns to append. Must be >= 0.
    initializer: Initializer function to initialize missing values. Accepts a
      1-D tensor as the arg to specify the shape of the returned tensor. If
      `None`, defaults to using `zeros_initializer()`.
    max_rows_in_memory: `int` specifying the maximum number of rows to load from
      the checkpoint at once. If less than or equal to 0, the entire matrix will
      be loaded into memory. Setting this arg trades increased disk reads for
      lower memory usage.

  Returns:
    A variable initializer function that should be used to initialize a
    (potentially partitioned) `Variable` whose complete shape is
    `[new_row_vocab_size + num_row_oov_buckets, new_col_vocab_size +
    num_col_oov_buckets]`.

  Raises:
    TypeError: If `initializer` is specified but not callable.
  "
  [ckpt_path old_tensor_name primary_partition_info new_row_vocab_size new_col_vocab_size old_row_vocab_file new_row_vocab_file old_col_vocab_file new_col_vocab_file & {:keys [num_row_oov_buckets num_col_oov_buckets initializer max_rows_in_memory]
                       :or {initializer None}} ]
    (py/call-attr-kw framework "load_variable_slot_initializer" [ckpt_path old_tensor_name primary_partition_info new_row_vocab_size new_col_vocab_size old_row_vocab_file new_row_vocab_file old_col_vocab_file new_col_vocab_file] {:num_row_oov_buckets num_row_oov_buckets :num_col_oov_buckets num_col_oov_buckets :initializer initializer :max_rows_in_memory max_rows_in_memory }))

(defn local-variable 
  "Create a variable with a value and add it to `GraphKeys.LOCAL_VARIABLES`.

  Args:
    initial_value: See variables.Variable.__init__.
    validate_shape: See variables.Variable.__init__.
    name: See variables.Variable.__init__.
    use_resource: If `True` use a ResourceVariable instead of a Variable.

  Returns:
    New variable.
  "
  [initial_value & {:keys [validate_shape name use_resource]
                       :or {name None use_resource None}} ]
    (py/call-attr-kw framework "local_variable" [initial_value] {:validate_shape validate_shape :name name :use_resource use_resource }))

(defn model-variable 
  "Gets an existing model variable with these parameters or creates a new one.

  Args:
    name: the name of the new or existing variable.
    shape: shape of the new or existing variable.
    dtype: type of the new or existing variable (defaults to `DT_FLOAT`).
    initializer: initializer for the variable if one is created.
    regularizer: a (Tensor -> Tensor or None) function; the result of applying
      it on a newly created variable will be added to the collection
      GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    collections: A list of collection names to which the Variable will be added.
      Note that the variable is always also added to the
      `GraphKeys.GLOBAL_VARIABLES` and `GraphKeys.MODEL_VARIABLES` collections.
    caching_device: Optional device string or function describing where the
      Variable should be cached for reading.  Defaults to the Variable's device.
    device: Optional device to place the variable. It can be an string or a
      function that is called to get the device for the variable.
    partitioner: Optional callable that accepts a fully defined `TensorShape`
      and dtype of the `Variable` to be created, and returns a list of
      partitions for each axis (currently only one axis can be partitioned).
    custom_getter: Callable that allows overwriting the internal get_variable
      method and has to have the same signature.
    use_resource: If `True` use a ResourceVariable instead of a Variable.
    synchronization: Indicates when a distributed a variable will be aggregated.
      Accepted values are constants defined in the class
      `tf.VariableSynchronization`. By default the synchronization is set to
      `AUTO` and the current `DistributionStrategy` chooses when to synchronize.
    aggregation: Indicates how a distributed variable will be aggregated.
      Accepted values are constants defined in the class
      `tf.VariableAggregation`.

  Returns:
    The created or existing variable.
  "
  [name shape & {:keys [dtype initializer regularizer trainable collections caching_device device partitioner custom_getter use_resource synchronization aggregation]
                       :or {initializer None regularizer None collections None caching_device None device None partitioner None custom_getter None use_resource None}} ]
    (py/call-attr-kw framework "model_variable" [name shape] {:dtype dtype :initializer initializer :regularizer regularizer :trainable trainable :collections collections :caching_device caching_device :device device :partitioner partitioner :custom_getter custom_getter :use_resource use_resource :synchronization synchronization :aggregation aggregation }))

(defn prepend-name-scope 
  "Prepends name scope to a name.

  Args:
    name: A `string` name.
    import_scope: Optional `string`. Name scope to add.

  Returns:
    Name with name scope added, or the original name if import_scope
    is None.
  "
  [ name import_scope ]
  (py/call-attr framework "prepend_name_scope"  name import_scope ))

(defn py-func 
  "Wraps a python function and uses it as a TensorFlow op.

  This function is a wrapper around `tf.compat.v1.py_func` and improve it with
  kwargs
  and output_shapes. Further it changed some argument names.

  Given a python function `func`, which takes numpy arrays as its
  inputs and returns numpy arrays as its outputs, wrap this function as an
  operation in a TensorFlow graph. The following snippet constructs a simple
  TensorFlow graph that invokes the `np.sinh()` NumPy function as a operation
  in the graph:

  ```python
  def my_func(x):
    # x will be a numpy array with the contents of the placeholder below
    return np.sinh(x)
  inp = tf.compat.v1.placeholder(tf.float32)
  y = tf.compat.v1.py_func(my_func, [inp], tf.float32)
  ```


  **N.B.** The `tf.compat.v1.py_func()` operation has the following known
  limitations:

  * The body of the function (i.e. `func`) will not be serialized in a
    `GraphDef`. Therefore, you should not use this function if you need to
    serialize your model and restore it in a different environment.

  * The operation must run in the same address space as the Python program
    that calls `tf.compat.v1.py_func()`. If you are using distributed
    TensorFlow, you
    must run a `tf.distribute.Server` in the same process as the program that
    calls
    `tf.compat.v1.py_func()` and you must pin the created operation to a device
    in that
    server (e.g. using `with tf.device():`).

  Args:
    func: A Python function, which accepts a list of NumPy `ndarray` objects
      having element types that match the corresponding `tf.Tensor` objects in
      `inp`, and returns a list of `ndarray` objects (or a single `ndarray`)
      having element types that match the corresponding values in `Tout`.
    args: A list of `Tensor` objects.
    kwargs: A dict with `Tensor` objects as values.
    output_types: A nested structure of tensorflow data types or a single
      tensorflow data type if there is only one, indicating what `func` returns.
    output_shapes: Same as output_types, except the types are replaces with
      shapes (optional).
    stateful: (Boolean.) If True, the function should be considered stateful. If
      a function is stateless, when given the same input it will return the same
      output and have no observable side effects. Optimizations such as common
      subexpression elimination are only performed on stateless operations.
    name: A name for the operation (optional).

  Returns:
    Tensorflow op that wraps the input python function.
  "
  [func & {:keys [args kwargs output_types output_shapes stateful name]
                       :or {kwargs None output_types None output_shapes None name None}} ]
    (py/call-attr-kw framework "py_func" [func] {:args args :kwargs kwargs :output_types output_types :output_shapes output_shapes :stateful stateful :name name }))

(defn reduce-sum-n 
  "Reduce tensors to a scalar sum.

  This reduces each tensor in `tensors` to a scalar via `tf.reduce_sum`, then
  adds them via `tf.add_n`.

  Args:
    tensors: List of tensors, all of the same numeric type.
    name: Tensor name, and scope for all other ops.

  Returns:
    Total loss tensor, or None if no losses have been configured.

  Raises:
    ValueError: if `losses` is missing or empty.
  "
  [ tensors name ]
  (py/call-attr framework "reduce_sum_n"  tensors name ))

(defn remove-squeezable-dimensions 
  "Squeeze last dim if ranks of `predictions` and `labels` differ by 1. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please switch to remove_squeezable_dimensions from tf.confusion_matrix. Note that the order of the inputs and outputs of labels and predictions have also been switched.

This will use static shape if available. Otherwise, it will add graph
operations, which could result in a performance hit.

Args:
  predictions: Predicted values, a `Tensor` of arbitrary dimensions.
  labels: Label values, a `Tensor` whose dimensions match `predictions`.
  name: Name of the op.

Returns:
  Tuple of `predictions` and `labels`, possibly with last dim squeezed."
  [ predictions labels name ]
  (py/call-attr framework "remove_squeezable_dimensions"  predictions labels name ))
(defn smart-case 
  "Like tf.case, except attempts to statically evaluate predicates.

  If any predicate in `pred_fn_pairs` is a bool or has a constant value, the
  associated callable will be called or omitted depending on its value.
  Otherwise this functions like tf.case.

  Args:
    pred_fn_pairs: Dict or list of pairs of a boolean scalar tensor and a
                   callable which returns a list of tensors.
    default: Optional callable that returns a list of tensors.
    exclusive: True iff at most one predicate is allowed to evaluate to `True`.
    name: A name for this operation (optional).

  Returns:
    The tensors returned by the first pair whose predicate evaluated to True, or
    those returned by `default` if none does.

  Raises:
    TypeError: If `pred_fn_pairs` is not a list/dictionary.
    TypeError: If `pred_fn_pairs` is a list but does not contain 2-tuples.
    TypeError: If `fns[i]` is not callable for any i, or `default` is not
               callable.
  "
  [pred_fn_pairs default  & {:keys [exclusive name]} ]
    (py/call-attr-kw framework "smart_case" [pred_fn_pairs default] {:exclusive exclusive :name name }))

(defn smart-cond 
  "Return either `true_fn()` if predicate `pred` is true else `false_fn()`.

  If `pred` is a bool or has a constant value, we return either `true_fn()`
  or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.

  Arguments:
    pred: A scalar determining whether to return the result of `true_fn` or
      `false_fn`.
    true_fn: The callable to be performed if pred is true.
    false_fn: The callable to be performed if pred is false.
    name: Optional name prefix when using `tf.cond`.

  Returns:
    Tensors returned by the call to either `true_fn` or `false_fn`.

  Raises:
    TypeError: If `true_fn` or `false_fn` is not callable.
  "
  [ pred true_fn false_fn name ]
  (py/call-attr framework "smart_cond"  pred true_fn false_fn name ))

(defn smart-constant-value 
  "Return the bool value for `pred`, or None if `pred` had a dynamic value.

  Arguments:
    pred: A scalar, either a Python bool or tensor.

  Returns:
    True or False if `pred` has a constant boolean value, None otherwise.

  Raises:
    TypeError: If `pred` is not a Tensor or bool.
  "
  [ pred ]
  (py/call-attr framework "smart_constant_value"  pred ))

(defn sort 
  "Sorts a tensor.
  
  Usage:

  ```python
  import tensorflow as tf
  a = [1, 10, 26.9, 2.8, 166.32, 62.3]
  b = tf.sort(a,axis=-1,direction='ASCENDING',name=None)
  c = tf.keras.backend.eval(b)
  # Here, c = [  1.     2.8   10.    26.9   62.3  166.32]
  ```
  
  Args:
    values: 1-D or higher numeric `Tensor`.
    axis: The axis along which to sort. The default is -1, which sorts the last
      axis.
    direction: The direction in which to sort the values (`'ASCENDING'` or
      `'DESCENDING'`).
    name: Optional name for the operation.

  Returns:
    A `Tensor` with the same dtype and shape as `values`, with the elements
        sorted along the given `axis`.

  Raises:
    ValueError: If axis is not a constant scalar, or the direction is invalid.
  "
  [values & {:keys [axis direction name]
                       :or {name None}} ]
    (py/call-attr-kw framework "sort" [values] {:axis axis :direction direction :name name }))

(defn strip-name-scope 
  "Removes name scope from a name.

  Args:
    name: A `string` name.
    export_scope: Optional `string`. Name scope to remove.

  Returns:
    Name with name scope removed, or the original name if export_scope
    is None.
  "
  [ name export_scope ]
  (py/call-attr framework "strip_name_scope"  name export_scope ))

(defn variable 
  "Gets an existing variable with these parameters or creates a new one.

  Args:
    name: the name of the new or existing variable.
    shape: shape of the new or existing variable.
    dtype: type of the new or existing variable (defaults to `DT_FLOAT`).
    initializer: initializer for the variable if one is created.
    regularizer: a (Tensor -> Tensor or None) function; the result of applying
      it on a newly created variable will be added to the collection
      GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    collections: A list of collection names to which the Variable will be added.
      If None it would default to `tf.GraphKeys.GLOBAL_VARIABLES`.
    caching_device: Optional device string or function describing where the
      Variable should be cached for reading.  Defaults to the Variable's device.
    device: Optional device to place the variable. It can be an string or a
      function that is called to get the device for the variable.
    partitioner: Optional callable that accepts a fully defined `TensorShape`
      and dtype of the `Variable` to be created, and returns a list of
      partitions for each axis (currently only one axis can be partitioned).
    custom_getter: Callable that allows overwriting the internal get_variable
      method and has to have the same signature.
    use_resource: If `True` use a ResourceVariable instead of a Variable.
    synchronization: Indicates when a distributed a variable will be aggregated.
      Accepted values are constants defined in the class
      `tf.VariableSynchronization`. By default the synchronization is set to
      `AUTO` and the current `DistributionStrategy` chooses when to synchronize.
    aggregation: Indicates how a distributed variable will be aggregated.
      Accepted values are constants defined in the class
      `tf.VariableAggregation`.

  Returns:
    The created or existing variable.
  "
  [name shape dtype initializer regularizer & {:keys [trainable collections caching_device device partitioner custom_getter use_resource synchronization aggregation]
                       :or {collections None caching_device None device None partitioner None custom_getter None use_resource None}} ]
    (py/call-attr-kw framework "variable" [name shape dtype initializer regularizer] {:trainable trainable :collections collections :caching_device caching_device :device device :partitioner partitioner :custom_getter custom_getter :use_resource use_resource :synchronization synchronization :aggregation aggregation }))

(defn with-same-shape 
  "Assert tensors are the same shape, from the same graph.

  Args:
    expected_tensor: Tensor with expected shape.
    tensor: Tensor of actual values.
  Returns:
    The original tensor argument, possibly with assert ops added.
  "
  [ expected_tensor tensor ]
  (py/call-attr framework "with_same_shape"  expected_tensor tensor ))

(defn with-shape 
  "Asserts tensor has expected shape.

  If tensor shape and expected_shape, are fully defined, assert they match.
  Otherwise, add assert op that will validate the shape when tensor is
  evaluated, and set shape on tensor.

  Args:
    expected_shape: Expected shape to assert, as a 1D array of ints, or tensor
        of same.
    tensor: Tensor whose shape we're validating.
  Returns:
    tensor, perhaps with a dependent assert operation.
  Raises:
    ValueError: if tensor has an invalid shape.
  "
  [ expected_shape tensor ]
  (py/call-attr framework "with_shape"  expected_shape tensor ))
(defn zero-initializer 
  "Initialize 'ref' with all zeros, ref tensor should be uninitialized.

  If already initialized, you will get ValueError. This op is intended to
  save memory during initialization.
  Args:
    ref: ref of the tensor need to be zero initialized.
    name: optional name for this operation.

  Returns:
    ref that initialized.
  Raises:
    ValueError: If ref tensor is initialized.
  "
  [ref  & {:keys [use_locking name]} ]
    (py/call-attr-kw framework "zero_initializer" [ref] {:use_locking use_locking :name name }))
