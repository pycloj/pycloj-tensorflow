(ns tensorflow.contrib.keras.api.keras.optimizers.Optimizer
  "Abstract optimizer base class.

  Note: this is the parent class of all optimizers, not an actual optimizer
  that can be used for training models.

  All Keras optimizers support the following keyword arguments:

      clipnorm: float >= 0. Gradients will be clipped
          when their L2 norm exceeds this value.
      clipvalue: float >= 0. Gradients will be clipped
          when their absolute value exceeds this value.
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

(defn Optimizer 
  "Abstract optimizer base class.

  Note: this is the parent class of all optimizers, not an actual optimizer
  that can be used for training models.

  All Keras optimizers support the following keyword arguments:

      clipnorm: float >= 0. Gradients will be clipped
          when their L2 norm exceeds this value.
      clipvalue: float >= 0. Gradients will be clipped
          when their absolute value exceeds this value.
  "
  [  ]
  (py/call-attr optimizers "Optimizer"  ))

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
