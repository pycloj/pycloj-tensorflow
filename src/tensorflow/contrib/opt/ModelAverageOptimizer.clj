(ns tensorflow.contrib.opt.ModelAverageOptimizer
  "Wrapper optimizer that implements the Model Average algorithm.

  This is a sync optimizer. During the training, each worker will update
  the local variables and maintains its own local_step, which starts from 0
  and is incremented by 1 after each update of local variables. Whenever the
  interval_steps divides the local step, the local variables from all the
  workers will be averaged and assigned to global center variables. Then the
  local variables will be assigned by global center variables.
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
(defn ModelAverageOptimizer 
  "Wrapper optimizer that implements the Model Average algorithm.

  This is a sync optimizer. During the training, each worker will update
  the local variables and maintains its own local_step, which starts from 0
  and is incremented by 1 after each update of local variables. Whenever the
  interval_steps divides the local step, the local variables from all the
  workers will be averaged and assigned to global center variables. Then the
  local variables will be assigned by global center variables.
  "
  [opt num_worker is_chief ma_custom_getter  & {:keys [interval_steps use_locking name]} ]
    (py/call-attr-kw opt "ModelAverageOptimizer" [opt num_worker is_chief ma_custom_getter] {:interval_steps interval_steps :use_locking use_locking :name name }))

(defn apply-gradients 
  "Apply gradients to variables.

    This contains most of the synchronization implementation and also wraps the
    apply_gradients() from the real optimizer. The chief work updates global
    variables.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the variables
        have been updated.
      name: Optional name for the returned operation.  Default to the name
        passed to the Optimizer constructor.

    Returns:
      A conditional 'Operation' that update both local and global variables or
      just local variables

    Raises:
      ValueError: If the grads_and_vars is empty.
      ValueError: If global step is not provided, the staleness cannot be
        checked.
    "
  [ self grads_and_vars global_step name ]
  (py/call-attr self "apply_gradients"  self grads_and_vars global_step name ))

(defn compute-gradients 
  "Compute gradients of \"loss\" for the variables in \"var_list\".

    This simply wraps the compute_gradients() from the real optimizer.

    Args:
      *args: Arguments for compute_gradients().
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.
    "
  [ self  ]
  (py/call-attr self "compute_gradients"  self  ))

(defn get-init-op 
  "Returns the op.

    This method lets all the local variables equal to the global
    variables before the training begins.
    "
  [ self  ]
  (py/call-attr self "get_init_op"  self  ))

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
  "Creates a hook to handle ModelAverage ops such as initialization."
  [ self  ]
  (py/call-attr self "make_session_run_hook"  self  ))

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

(defn variables 
  "A list of variables which encode the current state of `Optimizer`.

    Includes slot variables and additional global variables created by the
    optimizer in the current default graph.

    Returns:
      A list of variables.
    "
  [ self  ]
  (py/call-attr self "variables"  self  ))
