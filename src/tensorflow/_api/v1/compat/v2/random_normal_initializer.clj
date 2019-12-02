(ns tensorflow.-api.v1.compat.v2.random-normal-initializer
  "Initializer that generates tensors with a normal distribution.

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values
      to generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the
      random values to generate.
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

(defn random-normal-initializer 
  "Initializer that generates tensors with a normal distribution.

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values
      to generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the
      random values to generate.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed`
      for behavior.
  "
  [ & {:keys [mean stddev seed]
       :or {seed None}} ]
  
   (py/call-attr-kw v2 "random_normal_initializer" [] {:mean mean :stddev stddev :seed seed }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
