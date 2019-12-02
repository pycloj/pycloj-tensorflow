(ns tensorflow.python.keras.api.-v1.keras.optimizers.RMSprop
  "Optimizer that implements the RMSprop algorithm.

  A detailed description of rmsprop.

    - maintain a moving (discounted) average of the square of gradients
    - divide gradient by the root of this average

  $$mean_square_t = rho * mean_square{t-1} + (1-rho) * gradient ** 2$$
  $$mom_t = momentum * mom_{t-1} + learning_rate * gradient / \sqrt{ /
      mean_square_t + \epsilon}$$
  $$variable_t := variable_{t-1} - mom_t$$

  This implementation of RMSprop uses plain momentum, not Nesterov momentum.

  The centered version additionally maintains a moving average of the
  gradients, and uses that average to estimate the variance:

  $$mean_grad_t = rho * mean_grad_{t-1} + (1-rho) * gradient$$
  $$mean_square_t = rho * mean_square_{t-1} + (1-rho) * gradient ** 2$$
  $$mom_t = momentum * mom_{t-1} + learning_rate * gradient /
      sqrt(mean_square_t - mean_grad_t**2 + epsilon)$$
  $$variable_t := variable_{t-1} - mom_t$$

  References
    See ([pdf]
      http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce optimizers (import-module "tensorflow.python.keras.api._v1.keras.optimizers"))

(defn RMSprop 
  "Optimizer that implements the RMSprop algorithm.

  A detailed description of rmsprop.

    - maintain a moving (discounted) average of the square of gradients
    - divide gradient by the root of this average

  $$mean_square_t = rho * mean_square{t-1} + (1-rho) * gradient ** 2$$
  $$mom_t = momentum * mom_{t-1} + learning_rate * gradient / \sqrt{ /
      mean_square_t + \epsilon}$$
  $$variable_t := variable_{t-1} - mom_t$$

  This implementation of RMSprop uses plain momentum, not Nesterov momentum.

  The centered version additionally maintains a moving average of the
  gradients, and uses that average to estimate the variance:

  $$mean_grad_t = rho * mean_grad_{t-1} + (1-rho) * gradient$$
  $$mean_square_t = rho * mean_square_{t-1} + (1-rho) * gradient ** 2$$
  $$mom_t = momentum * mom_{t-1} + learning_rate * gradient /
      sqrt(mean_square_t - mean_grad_t**2 + epsilon)$$
  $$variable_t := variable_{t-1} - mom_t$$

  References
    See ([pdf]
      http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).
  "
  [ & {:keys [learning_rate rho momentum epsilon centered name]} ]
   (py/call-attr-kw optimizers "RMSprop" [] {:learning_rate learning_rate :rho rho :momentum momentum :epsilon epsilon :centered centered :name name }))
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
  "Apply gradients to variables.

    This is the second part of `minimize()`. It returns an `Operation` that
    applies gradients.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that applies the specified gradients. The `iterations`
        will be automatically increased by 1.

    Raises:
      TypeError: If `grads_and_vars` is malformed.
      ValueError: If none of the variables have gradients.
    "
  [ self grads_and_vars name ]
  (py/call-attr self "apply_gradients"  self grads_and_vars name ))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))

(defn get-gradients 
  "Returns gradients of `loss` with respect to `params`.

    Arguments:
      loss: Loss tensor.
      params: List of variables.

    Returns:
      List of gradient tensors.

    Raises:
      ValueError: In case any gradient cannot be computed (e.g. if gradient
        function not implemented).
    "
  [ self loss params ]
  (py/call-attr self "get_gradients"  self loss params ))

(defn get-slot 
  ""
  [ self var slot_name ]
  (py/call-attr self "get_slot"  self var slot_name ))

(defn get-slot-names 
  "A list of names for this optimizer's slots."
  [ self  ]
  (py/call-attr self "get_slot_names"  self  ))

(defn get-updates 
  ""
  [ self loss params ]
  (py/call-attr self "get_updates"  self loss params ))

(defn get-weights 
  ""
  [ self  ]
  (py/call-attr self "get_weights"  self  ))

(defn iterations 
  "Variable. The number of training steps this Optimizer has run."
  [ self ]
    (py/call-attr self "iterations"))

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
