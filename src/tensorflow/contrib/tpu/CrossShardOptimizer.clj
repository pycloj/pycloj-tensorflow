(ns tensorflow.contrib.tpu.CrossShardOptimizer
  "An optimizer that averages gradients across TPU shards."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tpu (import-module "tensorflow.contrib.tpu"))

(defn CrossShardOptimizer 
  "An optimizer that averages gradients across TPU shards."
  [opt & {:keys [reduction name group_assignment]
                       :or {group_assignment None}} ]
    (py/call-attr-kw tpu "CrossShardOptimizer" [opt] {:reduction reduction :name name :group_assignment group_assignment }))

(defn apply-gradients 
  "Apply gradients to variables.

    Calls tpu_ops.cross_replica_sum() to sum gradient contributions across
    replicas, and then applies the real optimizer.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      An `Operation` that applies the gradients. If `global_step` was not None,
      that operation also increments `global_step`.

    Raises:
      ValueError: If the grads_and_vars is malformed.
    "
  [ self grads_and_vars global_step name ]
  (py/call-attr self "apply_gradients"  self grads_and_vars global_step name ))

(defn compute-gradients 
  "Compute gradients of \"loss\" for the variables in \"var_list\".

    This simply wraps the compute_gradients() from the real optimizer. The
    gradients will be aggregated in the apply_gradients() so that user can
    modify the gradients like clipping with per replica global norm if needed.
    The global norm with aggregated gradients can be bad as one replica's huge
    gradients can hurt the gradients from other replicas.

    Args:
      loss: A Tensor containing the value to minimize.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKey.TRAINABLE_VARIABLES`.
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.

    Raises:
      ValueError: If not within a tpu_shard_context or group_assignment is
        invalid.
    "
  [ self loss var_list ]
  (py/call-attr self "compute_gradients"  self loss var_list ))

(defn get-name 
  ""
  [ self  ]
  (py/call-attr self "get_name"  self  ))

(defn get-slot 
  "Return a slot named \"name\" created for \"var\" by the Optimizer.

    This simply wraps the get_slot() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      The `Variable` for the slot if it was created, `None` otherwise.
    "
  [ self  ]
  (py/call-attr self "get_slot"  self  ))

(defn get-slot-names 
  "Return a list of the names of slots created by the `Optimizer`.

    This simply wraps the get_slot_names() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

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
  "Forwarding the variables from the underlying optimizer."
  [ self  ]
  (py/call-attr self "variables"  self  ))
