(ns tensorflow.contrib.opt.MomentumWOptimizer
  "Optimizer that implements the Momentum algorithm with weight_decay.

  This is an implementation of the SGDW optimizer described in \"Fixing
  Weight Decay Regularization in Adam\" by Loshchilov & Hutter
  (https://arxiv.org/abs/1711.05101)
  ([pdf])(https://arxiv.org/pdf/1711.05101.pdf).
  It computes the update step of `train.MomentumOptimizer` and additionally
  decays the variable. Note that this is different from adding
  L2 regularization on the variables to the loss. Decoupling the weight decay
  from other hyperparameters (in particular the learning rate) simplifies
  hyperparameter search.

  For further information see the documentation of the Momentum Optimizer.

  Note that this optimizer can also be instantiated as
  ```python
  extend_with_weight_decay(tf.compat.v1.train.MomentumOptimizer,
                           weight_decay=weight_decay)
  ```
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
(defn MomentumWOptimizer 
  "Optimizer that implements the Momentum algorithm with weight_decay.

  This is an implementation of the SGDW optimizer described in \"Fixing
  Weight Decay Regularization in Adam\" by Loshchilov & Hutter
  (https://arxiv.org/abs/1711.05101)
  ([pdf])(https://arxiv.org/pdf/1711.05101.pdf).
  It computes the update step of `train.MomentumOptimizer` and additionally
  decays the variable. Note that this is different from adding
  L2 regularization on the variables to the loss. Decoupling the weight decay
  from other hyperparameters (in particular the learning rate) simplifies
  hyperparameter search.

  For further information see the documentation of the Momentum Optimizer.

  Note that this optimizer can also be instantiated as
  ```python
  extend_with_weight_decay(tf.compat.v1.train.MomentumOptimizer,
                           weight_decay=weight_decay)
  ```
  "
  [weight_decay learning_rate momentum  & {:keys [use_locking name use_nesterov]} ]
    (py/call-attr-kw opt "MomentumWOptimizer" [weight_decay learning_rate momentum] {:use_locking use_locking :name name :use_nesterov use_nesterov }))

(defn apply-gradients 
  "Apply gradients to variables and decay the variables.

    This function is the same as Optimizer.apply_gradients except that it
    allows to specify the variables that should be decayed using
    decay_var_list. If decay_var_list is None, all variables in var_list
    are decayed.

    For more information see the documentation of Optimizer.apply_gradients.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.
      decay_var_list: Optional list of decay variables.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.
    "
  [ self grads_and_vars global_step name decay_var_list ]
  (py/call-attr self "apply_gradients"  self grads_and_vars global_step name decay_var_list ))

(defn compute-gradients 
  "Compute gradients of `loss` for the variables in `var_list`.

    This is the first part of `minimize()`.  It returns a list
    of (gradient, variable) pairs where \"gradient\" is the gradient
    for \"variable\".  Note that \"gradient\" can be a `Tensor`, an
    `IndexedSlices`, or `None` if there is no gradient for the
    given variable.

    Args:
      loss: A Tensor containing the value to minimize or a callable taking
        no arguments which returns the value to minimize. When eager execution
        is enabled it must be a callable.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKeys.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.

    Returns:
      A list of (gradient, variable) pairs. Variable is always present, but
      gradient can be `None`.

    Raises:
      TypeError: If `var_list` contains anything else than `Variable` objects.
      ValueError: If some arguments are invalid.
      RuntimeError: If called with eager execution enabled and `loss` is
        not callable.

    @compatibility(eager)
    When eager execution is enabled, `gate_gradients`, `aggregation_method`,
    and `colocate_gradients_with_ops` are ignored.
    @end_compatibility
    "
  [self loss var_list & {:keys [gate_gradients aggregation_method colocate_gradients_with_ops grad_loss]
                       :or {aggregation_method None grad_loss None}} ]
    (py/call-attr-kw self "compute_gradients" [loss var_list] {:gate_gradients gate_gradients :aggregation_method aggregation_method :colocate_gradients_with_ops colocate_gradients_with_ops :grad_loss grad_loss }))

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
  "Add operations to minimize `loss` by updating `var_list` with decay.

    This function is the same as Optimizer.minimize except that it allows to
    specify the variables that should be decayed using decay_var_list.
    If decay_var_list is None, all variables in var_list are decayed.

    For more information see the documentation of Optimizer.minimize.

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
      colocate_gradients_with_ops: If True, try colocating gradients with the
        corresponding op.
      name: Optional name for the returned operation.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
      decay_var_list: Optional list of decay variables.

    Returns:
      An Operation that updates the variables in `var_list`.  If `global_step`
      was not `None`, that operation also increments `global_step`.

    "
  [self loss global_step var_list & {:keys [gate_gradients aggregation_method colocate_gradients_with_ops name grad_loss decay_var_list]
                       :or {aggregation_method None name None grad_loss None decay_var_list None}} ]
    (py/call-attr-kw self "minimize" [loss global_step var_list] {:gate_gradients gate_gradients :aggregation_method aggregation_method :colocate_gradients_with_ops colocate_gradients_with_ops :name name :grad_loss grad_loss :decay_var_list decay_var_list }))

(defn variables 
  "A list of variables which encode the current state of `Optimizer`.

    Includes slot variables and additional global variables created by the
    optimizer in the current default graph.

    Returns:
      A list of variables.
    "
  [ self  ]
  (py/call-attr self "variables"  self  ))
