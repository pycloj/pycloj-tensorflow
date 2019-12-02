(ns tensorflow.contrib.keras.api.keras.optimizers.RMSprop
  "RMSProp optimizer.

  It is recommended to leave the parameters of this optimizer
  at their default values
  (except the learning rate, which can be freely tuned).

  Arguments:
      lr: float >= 0. Learning rate.
      rho: float >= 0.
      epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
      decay: float >= 0. Learning rate decay over each update.
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

(defn RMSprop 
  "RMSProp optimizer.

  It is recommended to leave the parameters of this optimizer
  at their default values
  (except the learning rate, which can be freely tuned).

  Arguments:
      lr: float >= 0. Learning rate.
      rho: float >= 0.
      epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
      decay: float >= 0. Learning rate decay over each update.
  "
  [ & {:keys [lr rho epsilon decay]
       :or {epsilon None}} ]
  
   (py/call-attr-kw optimizers "RMSprop" [] {:lr lr :rho rho :epsilon epsilon :decay decay }))

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
