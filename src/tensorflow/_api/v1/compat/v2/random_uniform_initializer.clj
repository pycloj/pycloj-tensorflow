(ns tensorflow.-api.v1.compat.v2.random-uniform-initializer
  "Initializer that generates tensors with a uniform distribution.

  Args:
    minval: A python scalar or a scalar tensor. Lower bound of the range
      of random values to generate.
    maxval: A python scalar or a scalar tensor. Upper bound of the range
      of random values to generate.  Defaults to 1 for float types.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed`
      for behavior.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v2 (import-module "tensorflow._api.v1.compat.v2"))

(defn random-uniform-initializer 
  "Initializer that generates tensors with a uniform distribution.

  Args:
    minval: A python scalar or a scalar tensor. Lower bound of the range
      of random values to generate.
    maxval: A python scalar or a scalar tensor. Upper bound of the range
      of random values to generate.  Defaults to 1 for float types.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed`
      for behavior.
  "
  [ & {:keys [minval maxval seed]
       :or {seed None}} ]
  
   (py/call-attr-kw v2 "random_uniform_initializer" [] {:minval minval :maxval maxval :seed seed }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
