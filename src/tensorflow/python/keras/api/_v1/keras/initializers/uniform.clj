(ns tensorflow.python.keras.api.-v1.keras.initializers.uniform
  "Initializer that generates tensors with a uniform distribution.

  Args:
    minval: A python scalar or a scalar tensor. Lower bound of the range of
      random values to generate. Defaults to -0.05.
    maxval: A python scalar or a scalar tensor. Upper bound of the range of
      random values to generate. Defaults to 0.05.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: The data type.
    
  Returns:
    A RandomUniform instance.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce initializers (import-module "tensorflow.python.keras.api._v1.keras.initializers"))

(defn uniform 
  "Initializer that generates tensors with a uniform distribution.

  Args:
    minval: A python scalar or a scalar tensor. Lower bound of the range of
      random values to generate. Defaults to -0.05.
    maxval: A python scalar or a scalar tensor. Upper bound of the range of
      random values to generate. Defaults to 0.05.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: The data type.
    
  Returns:
    A RandomUniform instance.
  "
  [ & {:keys [minval maxval seed dtype]
       :or {seed None}} ]
  
   (py/call-attr-kw initializers "uniform" [] {:minval minval :maxval maxval :seed seed :dtype dtype }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
