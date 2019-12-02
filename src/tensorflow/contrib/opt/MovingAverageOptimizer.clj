(ns tensorflow.contrib.opt.MovingAverageOptimizer
  "Optimizer that computes a moving average of the variables.

  Empirically it has been found that using the moving average of the trained
  parameters of a deep network is better than using its trained parameters
  directly. This optimizer allows you to compute this moving average and swap
  the variables at save time so that any code outside of the training loop will
  use by default the averaged values instead of the original ones.

  Example of usage:

  ```python

  // Encapsulate your favorite optimizer (here the momentum one)
  // inside the MovingAverageOptimizer.
  opt = tf.compat.v1.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
  opt = tf.contrib.opt.MovingAverageOptimizer(opt)
  // Then create your model and all its variables.
  model = build_model()
  // Add the training op that optimizes using opt.
  // This needs to be called before swapping_saver().
  opt.minimize(cost, var_list)
  // Then create your saver like this:
  saver = opt.swapping_saver()
  // Pass it to your training loop.
      slim.learning.train(
          model,
          ...
          saver=saver)
  ```

  Note that for evaluation, the normal saver should be used instead of
  swapping_saver().
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

(defn MovingAverageOptimizer 
  "Optimizer that computes a moving average of the variables.

  Empirically it has been found that using the moving average of the trained
  parameters of a deep network is better than using its trained parameters
  directly. This optimizer allows you to compute this moving average and swap
  the variables at save time so that any code outside of the training loop will
  use by default the averaged values instead of the original ones.

  Example of usage:

  ```python

  // Encapsulate your favorite optimizer (here the momentum one)
  // inside the MovingAverageOptimizer.
  opt = tf.compat.v1.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
  opt = tf.contrib.opt.MovingAverageOptimizer(opt)
  // Then create your model and all its variables.
  model = build_model()
  // Add the training op that optimizes using opt.
  // This needs to be called before swapping_saver().
  opt.minimize(cost, var_list)
  // Then create your saver like this:
  saver = opt.swapping_saver()
  // Pass it to your training loop.
      slim.learning.train(
          model,
          ...
          saver=saver)
  ```

  Note that for evaluation, the normal saver should be used instead of
  swapping_saver().
  "
  [opt & {:keys [average_decay num_updates sequential_update]
                       :or {num_updates None}} ]
    (py/call-attr-kw opt "MovingAverageOptimizer" [opt] {:average_decay average_decay :num_updates num_updates :sequential_update sequential_update }))

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
  "Create a saver swapping moving averages and variables.

    You should use this saver during training.  It will save the moving averages
    of the trained parameters under the original parameter names.  For
    evaluations or inference you should use a regular saver and it will
    automatically use the moving averages for the trained variable.

    You must call this function after all variables have been created and after
    you have called Optimizer.minimize().

    Args:
      var_list: List of variables to save, as per `Saver()`.
                If set to None, will save all the variables that have been
                created before this call.
      name: The name of the saver.
      **kwargs: Keyword arguments of `Saver()`.

    Returns:
      A `tf.compat.v1.train.Saver` object.

    Raises:
      RuntimeError: If apply_gradients or minimize has not been called before.
      ValueError: If var_list is provided and contains some variables but not
        their moving average counterpart.
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
