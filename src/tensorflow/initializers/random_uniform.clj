(ns tensorflow.initializers.random-uniform
  "Initializer that generates tensors with a uniform distribution.

  Args:
    minval: A python scalar or a scalar tensor. Lower bound of the range of
      random values to generate.
    maxval: A python scalar or a scalar tensor. Upper bound of the range of
      random values to generate.  Defaults to 1 for float types.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce initializers (import-module "tensorflow.initializers"))

(defn random-uniform 
  "Initializer that generates tensors with a uniform distribution.

  Args:
    minval: A python scalar or a scalar tensor. Lower bound of the range of
      random values to generate.
    maxval: A python scalar or a scalar tensor. Upper bound of the range of
      random values to generate.  Defaults to 1 for float types.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer.
  "
  [ & {:keys [minval maxval seed dtype]
       :or {maxval None seed None}} ]
  
   (py/call-attr-kw initializers "random_uniform" [] {:minval minval :maxval maxval :seed seed :dtype dtype }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
