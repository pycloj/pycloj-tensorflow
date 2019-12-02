(ns tensorflow.contrib.opt.VariableClippingOptimizer
  "Wrapper optimizer that clips the norm of specified variables after update.

  This optimizer delegates all aspects of gradient calculation and application
  to an underlying optimizer.  After applying gradients, this optimizer then
  clips the variable to have a maximum L2 norm along specified dimensions.
  NB: this is quite different from clipping the norm of the gradients.

  Multiple instances of `VariableClippingOptimizer` may be chained to specify
  different max norms for different subsets of variables.

  This is more efficient at serving-time than using normalization during
  embedding lookup, at the expense of more expensive training and fewer
  guarantees about the norms.

  @@__init__

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
(defn VariableClippingOptimizer 
  "Wrapper optimizer that clips the norm of specified variables after update.

  This optimizer delegates all aspects of gradient calculation and application
  to an underlying optimizer.  After applying gradients, this optimizer then
  clips the variable to have a maximum L2 norm along specified dimensions.
  NB: this is quite different from clipping the norm of the gradients.

  Multiple instances of `VariableClippingOptimizer` may be chained to specify
  different max norms for different subsets of variables.

  This is more efficient at serving-time than using normalization during
  embedding lookup, at the expense of more expensive training and fewer
  guarantees about the norms.

  @@__init__

  "
  [opt vars_to_clip_dims max_norm  & {:keys [use_locking colocate_clip_ops_with_vars name]} ]
    (py/call-attr-kw opt "VariableClippingOptimizer" [opt vars_to_clip_dims max_norm] {:use_locking use_locking :colocate_clip_ops_with_vars colocate_clip_ops_with_vars :name name }))

(defn apply-gradients 
  ""
  [ self grads_and_vars global_step name ]
  (py/call-attr self "apply_gradients"  self grads_and_vars global_step name ))

(defn compute-gradients 
  ""
  [ self  ]
  (py/call-attr self "compute_gradients"  self  ))

(defn get-name 
  ""
  [ self  ]
  (py/call-attr self "get_name"  self  ))

(defn get-slot 
  ""
  [ self  ]
  (py/call-attr self "get_slot"  self  ))

(defn get-slot-names 
  ""
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
  "A list of variables which encode the current state of `Optimizer`.

    Includes slot variables and additional global variables created by the
    optimizer in the current default graph.

    Returns:
      A list of variables.
    "
  [ self  ]
  (py/call-attr self "variables"  self  ))
