(ns tensorflow.contrib.keras.api.keras.initializers.RandomNormal
  "Initializer that generates tensors with a normal distribution.

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values to
      generate. Defaults to 0.
    stddev: a python scalar or a scalar tensor. Standard deviation of the random
      values to generate. Defaults to 0.05.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: The data type. Only floating point types are supported.

  Returns:
      RandomNormal instance.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce initializers (import-module "tensorflow.contrib.keras.api.keras.initializers"))

(defn RandomNormal 
  "Initializer that generates tensors with a normal distribution.

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values to
      generate. Defaults to 0.
    stddev: a python scalar or a scalar tensor. Standard deviation of the random
      values to generate. Defaults to 0.05.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: The data type. Only floating point types are supported.

  Returns:
      RandomNormal instance.
  "
  [ & {:keys [mean stddev seed dtype]
       :or {seed None}} ]
  
   (py/call-attr-kw initializers "RandomNormal" [] {:mean mean :stddev stddev :seed seed :dtype dtype }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
