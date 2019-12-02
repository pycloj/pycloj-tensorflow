(ns tensorflow.contrib.opt.ElasticAverageOptimizer
  "Wrapper optimizer that implements the Elastic Average SGD algorithm.

  This is an async optimizer. During the training, Each worker will update
  the local variables and maintains its own local_step, which starts from 0
  and is incremented by 1 after each update of local variables. Whenever
  the communication period divides the local step, the worker requests
  the current global center variables and then computed the elastic difference
  between global center variables and local variables. The elastic difference
  then be used to update both local variables and global variables.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce opt (import-module "tensorflow.contrib.opt"))

(defn ElasticAverageOptimizer 
  "Wrapper optimizer that implements the Elastic Average SGD algorithm.

  This is an async optimizer. During the training, Each worker will update
  the local variables and maintains its own local_step, which starts from 0
  and is incremented by 1 after each update of local variables. Whenever
  the communication period divides the local step, the worker requests
  the current global center variables and then computed the elastic difference
  between global center variables and local variables. The elastic difference
  then be used to update both local variables and global variables.
  "
  [opt num_worker ea_custom_getter & {:keys [communication_period moving_rate rho use_locking synchronous name]
                       :or {moving_rate None rho None}} ]
    (py/call-attr-kw opt "ElasticAverageOptimizer" [opt num_worker ea_custom_getter] {:communication_period communication_period :moving_rate moving_rate :rho rho :use_locking use_locking :synchronous synchronous :name name }))

(defn apply-gradients 
  "Apply gradients to global variables.

    This is the second part of `minimize()`. It returns an `Operation` that
    applies gradients.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.

    Raises:
      TypeError: If `grads_and_vars` is malformed.
      ValueError: If none of the variables have gradients.
    "
  [ self grads_and_vars global_step name ]
  (py/call-attr self "apply_gradients"  self grads_and_vars global_step name ))

(defn compute-gradients 
  "Compute gradients of `loss` for the variables in `var_list`.

    Add rho*elastic_difference to loss to control the exploration
    This is the first part of `minimize()`.  It returns a list
    of (gradient, variable) pairs where \"gradient\" is the gradient
    for \"variable\".  Note that \"gradient\" can be a `Tensor`, an
    `IndexedSlices`, or `None` if there is no gradient for the
    given variable.

    Args:
      loss: A Tensor containing the value to minimize.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph under
        the key `GraphKey.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with the
        corresponding op.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.

    Returns:
      A list of (gradient, variable) pairs. Variable is always present, but
      gradient can be `None`.

    Raises:
      TypeError: If `var_list` contains anything else than `Variable` objects.
      ValueError: If some arguments are invalid.
    "
  [self loss var_list & {:keys [gate_gradients aggregation_method colocate_gradients_with_ops grad_loss]
                       :or {aggregation_method None grad_loss None}} ]
    (py/call-attr-kw self "compute_gradients" [loss var_list] {:gate_gradients gate_gradients :aggregation_method aggregation_method :colocate_gradients_with_ops colocate_gradients_with_ops :grad_loss grad_loss }))

(defn get-init-op 
  "Returns the op to let all the local variables and local center

    variables equal to the global center variables before the training begins
    "
  [ self task_index ]
  (py/call-attr self "get_init_op"  self task_index ))

(defn get-name 
  ""
  [ self  ]
  (py/call-attr self "get_name"  self  ))

(defn get-slot 
  "Return a slot named `name` created for `var` by the Optimizer.

    Some `Optimizer` subclasses use additional variables.  For example
    `Momentum` and `Adagrad` use variables to accumulate updates.  This method
    gives access to these `Variable` objects if for some reason you need them.

    Use `get_slot_names()` to get the list of slot names created by the
    `Optimizer`.

    Args:
      var: A variable passed to `minimize()` or `apply_gradients()`.
      name: A string.

    Returns:
      The `Variable` for the slot if it was created, `None` otherwise.
    "
  [ self var name ]
  (py/call-attr self "get_slot"  self var name ))

(defn get-slot-names 
  "Return a list of the names of slots created by the `Optimizer`.

    See `get_slot()`.

    Returns:
      A list of strings.
    "
  [ self  ]
  (py/call-attr self "get_slot_names"  self  ))

(defn make-session-run-hook 
  "Creates a hook to handle ElasticAverageOptimizerHook ops such as initialization."
  [ self is_chief task_index ]
  (py/call-attr self "make_session_run_hook"  self is_chief task_index ))

(defn minimize 
  "Add operations to minimize `loss` by updating `var_list`.

    This method simply combines calls `compute_gradients()` and
    `apply_gradients()`. If you want to process the gradient before applying
    them call `compute_gradients()` and `apply_gradients()` explicitly instead
    of using this function.

    Args:
      loss: A `Tensor` containing the value to minimize.
      global_step: Optional `Variable` to increment by one after the
        variables have been updated.
      var_list: Optional list or tuple of `Variable` objects to update to
        minimize `loss`.  Defaults to the list of variables collected in
        the graph under the key `GraphKeys.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.
      name: Optional name for the returned operation.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.

    Returns:
      An Operation that updates the variables in `var_list`.  If `global_step`
      was not `None`, that operation also increments `global_step`.

    Raises:
      ValueError: If some of the variables are not `Variable` objects.

    @compatibility(eager)
    When eager execution is enabled, `loss` should be a Python function that
    takes no arguments and computes the value to be minimized. Minimization (and
    gradient computation) is done with respect to the elements of `var_list` if
    not None, else with respect to any trainable variables created during the
    execution of the `loss` function. `gate_gradients`, `aggregation_method`,
    `colocate_gradients_with_ops` and `grad_loss` are ignored when eager
    execution is enabled.
    @end_compatibility
    "
  [self loss global_step var_list & {:keys [gate_gradients aggregation_method colocate_gradients_with_ops name grad_loss]
                       :or {aggregation_method None name None grad_loss None}} ]
    (py/call-attr-kw self "minimize" [loss global_step var_list] {:gate_gradients gate_gradients :aggregation_method aggregation_method :colocate_gradients_with_ops colocate_gradients_with_ops :name name :grad_loss grad_loss }))
(defn swapping-saver 
  "Create a saver copy global_center_variable to trainable variables

    Please call this function after all your variables created with
    ElasticAverageCustomGetter. For evaluations or inference, use this saver
    during training.  It will save the global_center_variable of the trained
    parameters under the original parameter names.
    Args:
      var_list: List of variables to save, as per `Saver()`. If set to None,
        save all the trainable_variables that have been created before this
        call.
      name: The name of the saver.
      **kwargs: Keyword arguments of `Saver()`.

    Returns:
      A `tf.compat.v1.train.Saver` object.
    Raises:
      RuntimeError: global_center_variable is empty, please make sure
                    this is called after model created and
                    ElasticAverageCustomGetter is used when declaring you model
    "
  [self var_list  & {:keys [name]} ]
    (py/call-attr-kw self "swapping_saver" [var_list] {:name name }))

(defn variables 
  "A list of variables which encode the current state of `Optimizer`.

    Includes slot variables and additional global variables created by the
    optimizer in the current default graph.

    Returns:
      A list of variables.
    "
  [ self  ]
  (py/call-attr self "variables"  self  ))
