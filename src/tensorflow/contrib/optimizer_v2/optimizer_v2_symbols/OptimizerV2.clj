(ns tensorflow.contrib.optimizer-v2.optimizer-v2-symbols.OptimizerV2
  "Updated base class for optimizers.

  This class defines the API to add Ops to train a model.  You never use this
  class directly, but instead instantiate one of its subclasses such as
  `GradientDescentOptimizer`, `AdagradOptimizer`, or `MomentumOptimizer`.

  ### Usage

  ```python
  # Create an optimizer with the desired parameters.
  opt = GradientDescentOptimizer(learning_rate=0.1)
  # Add Ops to the graph to minimize a cost by updating a list of variables.
  # \"cost\" is a Tensor, and the list of variables contains tf.Variable
  # objects.
  opt_op = opt.minimize(cost, var_list=<list of variables>)
  ```

  In the training program you will just have to run the returned Op.

  ```python
  # Execute opt_op to do one step of training:
  opt_op.run()
  ```

  ### Processing gradients before applying them.

  Calling `minimize()` takes care of both computing the gradients and
  applying them to the variables.  If you want to process the gradients
  before applying them you can instead use the optimizer in three steps:

  1.  Compute the gradients with `compute_gradients()`.
  2.  Process the gradients as you wish.
  3.  Apply the processed gradients with `apply_gradients()`.

  Example:

  ```python
  # Create an optimizer.
  opt = GradientDescentOptimizer(learning_rate=0.1)

  # Compute the gradients for a list of variables.
  grads_and_vars = opt.compute_gradients(loss, <list of variables>)

  # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
  # need to the 'gradient' part, for example cap them, etc.
  capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]

  # Ask the optimizer to apply the capped gradients.
  opt.apply_gradients(capped_grads_and_vars)
  ```

  ### Gating Gradients

  Both `minimize()` and `compute_gradients()` accept a `gate_gradients`
  argument that controls the degree of parallelism during the application of
  the gradients.

  The possible values are: `GATE_NONE`, `GATE_OP`, and `GATE_GRAPH`.

  <b>`GATE_NONE`</b>: Compute and apply gradients in parallel.  This provides
  the maximum parallelism in execution, at the cost of some non-reproducibility
  in the results.  For example the two gradients of `matmul` depend on the input
  values: With `GATE_NONE` one of the gradients could be applied to one of the
  inputs _before_ the other gradient is computed resulting in non-reproducible
  results.

  <b>`GATE_OP`</b>: For each Op, make sure all gradients are computed before
  they are used.  This prevents race conditions for Ops that generate gradients
  for multiple inputs where the gradients depend on the inputs.

  <b>`GATE_GRAPH`</b>: Make sure all gradients for all variables are computed
  before any one of them is used.  This provides the least parallelism but can
  be useful if you want to process all gradients before applying any of them.

  ### Slots

  Some optimizer subclasses, such as `MomentumOptimizer` and `AdagradOptimizer`
  allocate and manage additional variables associated with the variables to
  train.  These are called <i>Slots</i>.  Slots have names and you can ask the
  optimizer for the names of the slots that it uses.  Once you have a slot name
  you can ask the optimizer for the variable it created to hold the slot value.

  This can be useful if you want to log debug a training algorithm, report stats
  about the slots, etc.

  ### Non-slot variables

  Some optimizer subclasses, such as `AdamOptimizer` have variables that
  are not associated with the variables to train, just the step itself.

  ### Hyper parameters

  These are arguments passed to the optimizer subclass constructor
  (the `__init__` method), and then passed to `self._set_hyper()`.
  They can be either regular Python values (like 1.0), tensors, or
  callables. If they are callable, the callable will be called during
  `apply_gradients()` to get the value for the hyper parameter.

  ### State

  Internal methods are passed a `state` argument with the correct
  values to use for the slot and non-slot variables, and the hyper
  parameters.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce optimizer-v2-symbols (import-module "tensorflow.contrib.optimizer_v2.optimizer_v2_symbols"))

(defn OptimizerV2 
  "Updated base class for optimizers.

  This class defines the API to add Ops to train a model.  You never use this
  class directly, but instead instantiate one of its subclasses such as
  `GradientDescentOptimizer`, `AdagradOptimizer`, or `MomentumOptimizer`.

  ### Usage

  ```python
  # Create an optimizer with the desired parameters.
  opt = GradientDescentOptimizer(learning_rate=0.1)
  # Add Ops to the graph to minimize a cost by updating a list of variables.
  # \"cost\" is a Tensor, and the list of variables contains tf.Variable
  # objects.
  opt_op = opt.minimize(cost, var_list=<list of variables>)
  ```

  In the training program you will just have to run the returned Op.

  ```python
  # Execute opt_op to do one step of training:
  opt_op.run()
  ```

  ### Processing gradients before applying them.

  Calling `minimize()` takes care of both computing the gradients and
  applying them to the variables.  If you want to process the gradients
  before applying them you can instead use the optimizer in three steps:

  1.  Compute the gradients with `compute_gradients()`.
  2.  Process the gradients as you wish.
  3.  Apply the processed gradients with `apply_gradients()`.

  Example:

  ```python
  # Create an optimizer.
  opt = GradientDescentOptimizer(learning_rate=0.1)

  # Compute the gradients for a list of variables.
  grads_and_vars = opt.compute_gradients(loss, <list of variables>)

  # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
  # need to the 'gradient' part, for example cap them, etc.
  capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]

  # Ask the optimizer to apply the capped gradients.
  opt.apply_gradients(capped_grads_and_vars)
  ```

  ### Gating Gradients

  Both `minimize()` and `compute_gradients()` accept a `gate_gradients`
  argument that controls the degree of parallelism during the application of
  the gradients.

  The possible values are: `GATE_NONE`, `GATE_OP`, and `GATE_GRAPH`.

  <b>`GATE_NONE`</b>: Compute and apply gradients in parallel.  This provides
  the maximum parallelism in execution, at the cost of some non-reproducibility
  in the results.  For example the two gradients of `matmul` depend on the input
  values: With `GATE_NONE` one of the gradients could be applied to one of the
  inputs _before_ the other gradient is computed resulting in non-reproducible
  results.

  <b>`GATE_OP`</b>: For each Op, make sure all gradients are computed before
  they are used.  This prevents race conditions for Ops that generate gradients
  for multiple inputs where the gradients depend on the inputs.

  <b>`GATE_GRAPH`</b>: Make sure all gradients for all variables are computed
  before any one of them is used.  This provides the least parallelism but can
  be useful if you want to process all gradients before applying any of them.

  ### Slots

  Some optimizer subclasses, such as `MomentumOptimizer` and `AdagradOptimizer`
  allocate and manage additional variables associated with the variables to
  train.  These are called <i>Slots</i>.  Slots have names and you can ask the
  optimizer for the names of the slots that it uses.  Once you have a slot name
  you can ask the optimizer for the variable it created to hold the slot value.

  This can be useful if you want to log debug a training algorithm, report stats
  about the slots, etc.

  ### Non-slot variables

  Some optimizer subclasses, such as `AdamOptimizer` have variables that
  are not associated with the variables to train, just the step itself.

  ### Hyper parameters

  These are arguments passed to the optimizer subclass constructor
  (the `__init__` method), and then passed to `self._set_hyper()`.
  They can be either regular Python values (like 1.0), tensors, or
  callables. If they are callable, the callable will be called during
  `apply_gradients()` to get the value for the hyper parameter.

  ### State

  Internal methods are passed a `state` argument with the correct
  values to use for the slot and non-slot variables, and the hyper
  parameters.
  "
  [ use_locking name ]
  (py/call-attr optimizer-v2 "OptimizerV2"  use_locking name ))

(defn apply-gradients 
  "Apply gradients to variables.

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

    This is the first part of `minimize()`.  It returns a list
    of (gradient, variable) pairs where \"gradient\" is the gradient
    for \"variable\".  Note that \"gradient\" can be a `Tensor`, an
    `IndexedSlices`, or `None` if there is no gradient for the
    given variable.

    Args:
      loss: A Tensor containing the value to minimize or a callable taking no
        arguments which returns the value to minimize. When eager execution is
        enabled it must be a callable.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph under
        the key `GraphKeys.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
      stop_gradients: Optional. A Tensor or list of tensors not to differentiate
        through.
      scale_loss_by_num_replicas: Optional boolean. If true, scale the loss down
        by the number of replicas. DEPRECATED and generally no longer needed.

    Returns:
      A list of (gradient, variable) pairs. Variable is always present, but
      gradient can be `None`.

    Raises:
      TypeError: If `var_list` contains anything else than `Variable` objects.
      ValueError: If some arguments are invalid.
      RuntimeError: If called with eager execution enabled and `loss` is
        not callable.

    @compatibility(eager)
    When eager execution is enabled, `gate_gradients`, and `aggregation_method`
    are ignored.
    @end_compatibility
    "
  [self loss var_list & {:keys [gate_gradients aggregation_method grad_loss stop_gradients scale_loss_by_num_replicas]
                       :or {aggregation_method None grad_loss None stop_gradients None}} ]
    (py/call-attr-kw self "compute_gradients" [loss var_list] {:gate_gradients gate_gradients :aggregation_method aggregation_method :grad_loss grad_loss :stop_gradients stop_gradients :scale_loss_by_num_replicas scale_loss_by_num_replicas }))

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

(defn minimize 
  "Add operations to minimize `loss` by updating `var_list`.

    This method simply combines calls `compute_gradients()` and
    `apply_gradients()`. If you want to process the gradient before applying
    them call `compute_gradients()` and `apply_gradients()` explicitly instead
    of using this function.

    Args:
      loss: A `Tensor` containing the value to minimize.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      var_list: Optional list or tuple of `Variable` objects to update to
        minimize `loss`.  Defaults to the list of variables collected in the
        graph under the key `GraphKeys.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      name: Optional name for the returned operation.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
      stop_gradients: Optional. A Tensor or list of tensors not to differentiate
        through.
      scale_loss_by_num_replicas: Optional boolean. If true, scale the loss down
        by the number of replicas. DEPRECATED and generally no longer needed.

    Returns:
      An Operation that updates the variables in `var_list`.  If `global_step`
      was not `None`, that operation also increments `global_step`.

    Raises:
      ValueError: If some of the variables are not `Variable` objects.

    @compatibility(eager)
    When eager execution is enabled, `loss` should be a Python function that
    takes elements of `var_list` as arguments and computes the value to be
    minimized. If `var_list` is None, `loss` should take no arguments.
    Minimization (and gradient computation) is done with respect to the
    elements of `var_list` if not None, else with respect to any trainable
    variables created during the execution of the `loss` function.
    `gate_gradients`, `aggregation_method`, and `grad_loss` are ignored when
    eager execution is enabled.
    @end_compatibility
    "
  [self loss global_step var_list & {:keys [gate_gradients aggregation_method name grad_loss stop_gradients scale_loss_by_num_replicas]
                       :or {aggregation_method None name None grad_loss None stop_gradients None}} ]
    (py/call-attr-kw self "minimize" [loss global_step var_list] {:gate_gradients gate_gradients :aggregation_method aggregation_method :name name :grad_loss grad_loss :stop_gradients stop_gradients :scale_loss_by_num_replicas scale_loss_by_num_replicas }))

(defn variables 
  "A list of variables which encode the current state of `Optimizer`.

    Includes slot variables and additional global variables created by the
    optimizer in the current default graph.

    Returns:
      A list of variables.
    "
  [ self  ]
  (py/call-attr self "variables"  self  ))
