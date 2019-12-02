(ns tensorflow.python.keras.api.-v1.keras.optimizers.Adadelta
  "Optimizer that implements the Adadelta algorithm.

  Adadelta optimization is a stochastic gradient descent method that is based on
  adaptive learning rate per dimension to address two drawbacks:
    1) the continual decay of learning rates throughout training
    2) the need for a manually selected global learning rate

  Two accumulation steps are required:
    1) the accumulation of gradients squared,
    2) the accumulation of updates squared.

  Initialization:

  $$E[g^2]_0 := 0 \text{(Initialize gradient 2nd order moment vector)}$$
  $$E[\Delta x^2]_0 := 0 \text{(Initialize 2nd order variable update)}$$

  $$t := t + 1$$
  $$E[g^2]_t := \rho * E[g^2]_{t-1} + (1 - \rho) * g^2$$
  $$\Delta x_t = -RMS[\Delta x]_{t-1} * g_t / RMS[g]_t$$
  $$E[\Delta x^2]_t := \rho * E[\Delta x^2]_{t-1} + (1 - \rho) * \Delta x_t^2$$
  $$x_t := x_{t-1} + \Delta x_{t}$$

  References
    See [M. D. Zeiler](http://arxiv.org/abs/1212.5701)
      ([pdf](http://arxiv.org/pdf/1212.5701v1.pdf))

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

(defn Adadelta 
  "Optimizer that implements the Adadelta algorithm.

  Adadelta optimization is a stochastic gradient descent method that is based on
  adaptive learning rate per dimension to address two drawbacks:
    1) the continual decay of learning rates throughout training
    2) the need for a manually selected global learning rate

  Two accumulation steps are required:
    1) the accumulation of gradients squared,
    2) the accumulation of updates squared.

  Initialization:

  $$E[g^2]_0 := 0 \text{(Initialize gradient 2nd order moment vector)}$$
  $$E[\Delta x^2]_0 := 0 \text{(Initialize 2nd order variable update)}$$

  $$t := t + 1$$
  $$E[g^2]_t := \rho * E[g^2]_{t-1} + (1 - \rho) * g^2$$
  $$\Delta x_t = -RMS[\Delta x]_{t-1} * g_t / RMS[g]_t$$
  $$E[\Delta x^2]_t := \rho * E[\Delta x^2]_{t-1} + (1 - \rho) * \Delta x_t^2$$
  $$x_t := x_{t-1} + \Delta x_{t}$$

  References
    See [M. D. Zeiler](http://arxiv.org/abs/1212.5701)
      ([pdf](http://arxiv.org/pdf/1212.5701v1.pdf))

  "
  [ & {:keys [learning_rate rho epsilon name]} ]
   (py/call-attr-kw optimizers "Adadelta" [] {:learning_rate learning_rate :rho rho :epsilon epsilon :name name }))
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
