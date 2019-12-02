(ns tensorflow.python.keras.api.-v1.keras.mixed-precision.experimental.LossScaleOptimizer
  "An optimizer that applies loss scaling.

  Loss scaling is a process that multiplies the loss by a multiplier called the
  loss scale, and divides each gradient by the same multiplier. The pseudocode
  for this process is:

  ```
  loss = ...
  loss *= loss_scale
  grads = gradients(loss, vars)
  grads /= loss_scale
  ```

  Mathematically, loss scaling has no effect, but can help avoid numerical
  underflow in intermediate gradients when float16 tensors are used. By
  multiplying the loss, each intermediate gradient will have the same multiplier
  applied.

  The loss scale can either be a fixed constant, chosen by the user, or be
  dynamically determined. Dynamically determining the loss scale is convenient
  as a loss scale does not have to be explicitly chosen. However it reduces
  performance.

  This optimizer wraps another optimizer and applies loss scaling to it via a
  `LossScale`. Loss scaling is applied whenever gradients are
  computed, either through `minimize()` or `get_gradients()`. The loss scale is
  updated via `LossScale.update()` whenever gradients are applied, either
  through `minimize()` or `apply_gradients()`. For example:

  ```python
  opt = tf.keras.optimizers.SGD(0.1)
  opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, \"dynamic\")
  # 'minimize' applies loss scaling to the loss and updates the loss sale.
  opt.minimize(loss_fn)
  ```

  If a `tf.GradientTape` is used to compute gradients instead of
  `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, the loss
  and gradients must be scaled manually. This can be done by calling
  `LossScaleOptimizer.get_scaled_loss` before passing the loss to
  `tf.GradientTape`, and `LossScaleOptimizer.get_unscaled_gradients` after
  computing the gradients with `tf.GradientTape`. For example:

  ```python
  opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(...)
  vars = ...
  with tf.GradientTape() as tape:
    loss = ...
    scaled_loss = opt.get_scaled_loss(loss)
  scaled_grads = tape.gradient(scaled_loss, vars)
  grads = opt.get_unscaled_gradients(scaled_grads)
  opt.apply_gradients(zip(grads, vars))  # Loss scale will be updated here
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
(defonce experimental (import-module "tensorflow.python.keras.api._v1.keras.mixed_precision.experimental"))

(defn LossScaleOptimizer 
  "An optimizer that applies loss scaling.

  Loss scaling is a process that multiplies the loss by a multiplier called the
  loss scale, and divides each gradient by the same multiplier. The pseudocode
  for this process is:

  ```
  loss = ...
  loss *= loss_scale
  grads = gradients(loss, vars)
  grads /= loss_scale
  ```

  Mathematically, loss scaling has no effect, but can help avoid numerical
  underflow in intermediate gradients when float16 tensors are used. By
  multiplying the loss, each intermediate gradient will have the same multiplier
  applied.

  The loss scale can either be a fixed constant, chosen by the user, or be
  dynamically determined. Dynamically determining the loss scale is convenient
  as a loss scale does not have to be explicitly chosen. However it reduces
  performance.

  This optimizer wraps another optimizer and applies loss scaling to it via a
  `LossScale`. Loss scaling is applied whenever gradients are
  computed, either through `minimize()` or `get_gradients()`. The loss scale is
  updated via `LossScale.update()` whenever gradients are applied, either
  through `minimize()` or `apply_gradients()`. For example:

  ```python
  opt = tf.keras.optimizers.SGD(0.1)
  opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, \"dynamic\")
  # 'minimize' applies loss scaling to the loss and updates the loss sale.
  opt.minimize(loss_fn)
  ```

  If a `tf.GradientTape` is used to compute gradients instead of
  `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, the loss
  and gradients must be scaled manually. This can be done by calling
  `LossScaleOptimizer.get_scaled_loss` before passing the loss to
  `tf.GradientTape`, and `LossScaleOptimizer.get_unscaled_gradients` after
  computing the gradients with `tf.GradientTape`. For example:

  ```python
  opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(...)
  vars = ...
  with tf.GradientTape() as tape:
    loss = ...
    scaled_loss = opt.get_scaled_loss(loss)
  scaled_grads = tape.gradient(scaled_loss, vars)
  grads = opt.get_unscaled_gradients(scaled_grads)
  opt.apply_gradients(zip(grads, vars))  # Loss scale will be updated here
  ```
  "
  [ opt loss_scale ]
  (py/call-attr experimental "LossScaleOptimizer"  opt loss_scale ))
(defn add-slot 
  "Add a new slot variable for `var`."
  [self var slot_name  & {:keys [initializer]} ]
    (py/call-attr-kw self "add_slot" [var slot_name] {:initializer initializer }))

(defn add-weight 
  ""
  [self name shape dtype & {:keys [initializer trainable synchronization aggregation]
                       :or {trainable None}} ]
    (py/call-attr-kw self "add_weight" [name shape dtype] {:initializer initializer :trainable trainable :synchronization synchronization :aggregation aggregation }))

(defn apply-gradients 
  ""
  [ self grads_and_vars name ]
  (py/call-attr self "apply_gradients"  self grads_and_vars name ))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))

(defn get-gradients 
  ""
  [ self loss params ]
  (py/call-attr self "get_gradients"  self loss params ))

(defn get-scaled-loss 
  "Scales the loss by the loss scale.

    This method is only needed if you compute gradients manually, e.g. with
    `tf.GradientTape`. In that case, call this method to scale the loss before
    passing the loss to `tf.GradientTape`. If you use
    `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, loss
    scaling is automatically applied and this method is unneeded.

    If this method is called, `get_unscaled_gradients` should also be called.
    See the `tf.keras.mixed_precision.experimental.LossScaleOptimizer` doc for
    an example.

    Args:
      loss: The loss, which will be multiplied by the loss scale. Can either be
        a tensor or a callable returning a tensor.

    Returns:
      `loss` multiplied by `LossScaleOptimizer.loss_scale()`.
    "
  [ self loss ]
  (py/call-attr self "get_scaled_loss"  self loss ))

(defn get-slot 
  ""
  [ self var slot_name ]
  (py/call-attr self "get_slot"  self var slot_name ))

(defn get-slot-names 
  "A list of names for this optimizer's slots."
  [ self  ]
  (py/call-attr self "get_slot_names"  self  ))

(defn get-unscaled-gradients 
  "Unscales the gradients by the loss scale.

    This method is only needed if you compute gradients manually, e.g. with
    `tf.GradientTape`. In that case, call this method to unscale the gradients
    after computing them with `tf.GradientTape`. If you use
    `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, loss
    scaling is automatically applied and this method is unneeded.

    If this method is called, `get_scaled_loss` should also be called. See
    the `tf.keras.mixed_precision.experimental.LossScaleOptimizer` doc for an
    example.

    Args:
      grads: A list of tensors, each which will be divided by the loss scale.
        Can have None values, which are ignored.

    Returns:
      A new list the same size as `grads`, where every non-None value in `grads`
      is divided by `LossScaleOptimizer.loss_scale()`.
    "
  [ self grads ]
  (py/call-attr self "get_unscaled_gradients"  self grads ))

(defn get-updates 
  ""
  [ self loss params ]
  (py/call-attr self "get_updates"  self loss params ))

(defn get-weights 
  ""
  [ self  ]
  (py/call-attr self "get_weights"  self  ))

(defn iterations 
  ""
  [ self ]
    (py/call-attr self "iterations"))

(defn learning-rate 
  ""
  [ self ]
    (py/call-attr self "learning_rate"))

(defn loss-scale 
  "The `LossScale` instance associated with this optimizer."
  [ self ]
    (py/call-attr self "loss_scale"))

(defn lr 
  ""
  [ self ]
    (py/call-attr self "lr"))

(defn minimize 
  "Minimize `loss` by updating `var_list`.

    This method simply computes gradient using `tf.GradientTape` and calls
    `apply_gradients()`. If you want to process the gradient before applying
    then call `tf.GradientTape` and `apply_gradients()` explicitly instead
    of using this function.

    Args:
      loss: A callable taking no arguments which returns the value to minimize.
      var_list: list or tuple of `Variable` objects to update to minimize
        `loss`, or a callable returning the list or tuple of `Variable` objects.
        Use callable when the variable list would otherwise be incomplete before
        `minimize` since the variables are created at the first time `loss` is
        called.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
      name: Optional name for the returned operation.

    Returns:
      An Operation that updates the variables in `var_list`.  If `global_step`
      was not `None`, that operation also increments `global_step`.

    Raises:
      ValueError: If some of the variables are not `Variable` objects.

    "
  [ self loss var_list grad_loss name ]
  (py/call-attr self "minimize"  self loss var_list grad_loss name ))

(defn set-weights 
  ""
  [ self weights ]
  (py/call-attr self "set_weights"  self weights ))

(defn variables 
  "Returns variables of this Optimizer based on the order created."
  [ self  ]
  (py/call-attr self "variables"  self  ))

(defn weights 
  "Returns variables of this Optimizer based on the order created."
  [ self ]
    (py/call-attr self "weights"))
