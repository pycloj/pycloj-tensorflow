(ns tensorflow.contrib.keras.api.keras.optimizers.SGD
  "Stochastic gradient descent optimizer.

  Includes support for momentum,
  learning rate decay, and Nesterov momentum.

  Arguments:
      lr: float >= 0. Learning rate.
      momentum: float >= 0. Parameter that accelerates SGD in the relevant
        direction and dampens oscillations.
      decay: float >= 0. Learning rate decay over each update.
      nesterov: boolean. Whether to apply Nesterov momentum.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce optimizers (import-module "tensorflow.contrib.keras.api.keras.optimizers"))

(defn SGD 
  "Stochastic gradient descent optimizer.

  Includes support for momentum,
  learning rate decay, and Nesterov momentum.

  Arguments:
      lr: float >= 0. Learning rate.
      momentum: float >= 0. Parameter that accelerates SGD in the relevant
        direction and dampens oscillations.
      decay: float >= 0. Learning rate decay over each update.
      nesterov: boolean. Whether to apply Nesterov momentum.
  "
  [ & {:keys [lr momentum decay nesterov]} ]
   (py/call-attr-kw optimizers "SGD" [] {:lr lr :momentum momentum :decay decay :nesterov nesterov }))

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

(defn get-updates 
  ""
  [ self loss params ]
  (py/call-attr self "get_updates"  self loss params ))

(defn get-weights 
  "Returns the current value of the weights of the optimizer.

    Returns:
        A list of numpy arrays.
    "
  [ self  ]
  (py/call-attr self "get_weights"  self  ))

(defn set-weights 
  "Sets the weights of the optimizer, from Numpy arrays.

    Should only be called after computing the gradients
    (otherwise the optimizer has no weights).

    Arguments:
        weights: a list of Numpy arrays. The number of arrays and their shape
          must match number of the dimensions of the weights of the optimizer
          (i.e. it should match the output of `get_weights`).

    Raises:
        ValueError: in case of incompatible weight shapes.
    "
  [ self weights ]
  (py/call-attr self "set_weights"  self weights ))
