(ns tensorflow.contrib.mixed-precision.LossScaleOptimizer
  "An optimizer that applies loss scaling in backprop.

  This class is useful for \"mixed precision training\" on GPUs (or other
  potential accelerators), an approach to improve compute throughput without
  compromising model quality.

  The canonical way to perform mixed precision training is the following:
  * Model variables are kept in high precision (e.g. float32).
  * Computations are done in lower precision (e.g. float16), which enjoys
    performance speedup by virtue of hardware support. Variables are casted to
    lower precision before they're used.
  * Final gradients are casted back to high precision dtype, then used to update
    variables.

  The side-effect of performing computation in lower precision, is that it comes
  with smaller numerical range. During backproping, small gradients might
  underflow in the reduced numerical range, causing a model to converge at
  suboptimal level.

  To prevent underflow, this optimizer multiplies the loss by a factor before
  backprop starts. Consequently, the gradients are linearly scaled up by the
  same factor, thus not falling into the underflow zone. After that, to perserve
  the correctness of backprop, the gradients are down-scaled by the same factor,
  casted to the (higher) variable precision, then applied on the variables.

  See [Nvidia's manual on mixed precision training](
  https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)
  for more details.

  To use loss scale optimizer, one only needs choose a loss scale strategy and
  wrap a regular optimizer. See examples below.

  ```
  loss = loss_fn()
  opt = tf.AdamOptimizer(learning_rate=...)

  # Choose a loss scale manager which decides how to pick the right loss scale
  # throughout the training process.
  loss_scale_manager = tf.contrib.mixed_precision.FixedLossScaleManager(5000)

  # Wraps the original optimizer in a LossScaleOptimizer.
  loss_scale_optimizer =
      tf.contrib.mixed_precision.LossScaleOptimizer(opt, loss_scale_manager)

  # Call minimize() on the loss scale optimizer.
  train_op = loss_scale_optimizer.minimize(loss)
  ```

  If gradients clipping is applied, one can call
  `optimizer.compute_gradients()` and `optimizer.apply_gradients()`
  separately.

  Notice the following way of using LossScaleOptimizer is not intended. Always
  use `loss_scale_optimizer.compute_gradients()` to compute gradients instead of
  `tf.gradients()` if doing mixed precision training.

  ```
  # The following is a wrong way to use LossScaleOptimizer along with
  # tf.gradients().

  # Always use loss_scale_optimizer.compute_gradients() to compute grads, or
  # loss scale is not correctly applied.
  grads = tf.gradients(loss, ...)

  # Do some custom grad clipping.
  grads = clip_grads(grads, ...)

  loss_scale_optimizer.apply(grads_and_vars)
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
(defonce mixed-precision (import-module "tensorflow.contrib.mixed_precision"))

(defn LossScaleOptimizer 
  "An optimizer that applies loss scaling in backprop.

  This class is useful for \"mixed precision training\" on GPUs (or other
  potential accelerators), an approach to improve compute throughput without
  compromising model quality.

  The canonical way to perform mixed precision training is the following:
  * Model variables are kept in high precision (e.g. float32).
  * Computations are done in lower precision (e.g. float16), which enjoys
    performance speedup by virtue of hardware support. Variables are casted to
    lower precision before they're used.
  * Final gradients are casted back to high precision dtype, then used to update
    variables.

  The side-effect of performing computation in lower precision, is that it comes
  with smaller numerical range. During backproping, small gradients might
  underflow in the reduced numerical range, causing a model to converge at
  suboptimal level.

  To prevent underflow, this optimizer multiplies the loss by a factor before
  backprop starts. Consequently, the gradients are linearly scaled up by the
  same factor, thus not falling into the underflow zone. After that, to perserve
  the correctness of backprop, the gradients are down-scaled by the same factor,
  casted to the (higher) variable precision, then applied on the variables.

  See [Nvidia's manual on mixed precision training](
  https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)
  for more details.

  To use loss scale optimizer, one only needs choose a loss scale strategy and
  wrap a regular optimizer. See examples below.

  ```
  loss = loss_fn()
  opt = tf.AdamOptimizer(learning_rate=...)

  # Choose a loss scale manager which decides how to pick the right loss scale
  # throughout the training process.
  loss_scale_manager = tf.contrib.mixed_precision.FixedLossScaleManager(5000)

  # Wraps the original optimizer in a LossScaleOptimizer.
  loss_scale_optimizer =
      tf.contrib.mixed_precision.LossScaleOptimizer(opt, loss_scale_manager)

  # Call minimize() on the loss scale optimizer.
  train_op = loss_scale_optimizer.minimize(loss)
  ```

  If gradients clipping is applied, one can call
  `optimizer.compute_gradients()` and `optimizer.apply_gradients()`
  separately.

  Notice the following way of using LossScaleOptimizer is not intended. Always
  use `loss_scale_optimizer.compute_gradients()` to compute gradients instead of
  `tf.gradients()` if doing mixed precision training.

  ```
  # The following is a wrong way to use LossScaleOptimizer along with
  # tf.gradients().

  # Always use loss_scale_optimizer.compute_gradients() to compute grads, or
  # loss scale is not correctly applied.
  grads = tf.gradients(loss, ...)

  # Do some custom grad clipping.
  grads = clip_grads(grads, ...)

  loss_scale_optimizer.apply(grads_and_vars)
  ```
  "
  [ opt loss_scale_manager ]
  (py/call-attr mixed-precision "LossScaleOptimizer"  opt loss_scale_manager ))

(defn apply-gradients 
  "Apply gradients. See base class `tf.compat.v1.train.Optimizer`."
  [ self grads_and_vars global_step name ]
  (py/call-attr self "apply_gradients"  self grads_and_vars global_step name ))

(defn compute-gradients 
  "Compute gradients. See base class `tf.compat.v1.train.Optimizer`."
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
