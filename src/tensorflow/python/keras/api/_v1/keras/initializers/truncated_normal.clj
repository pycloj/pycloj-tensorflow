(ns tensorflow.python.keras.api.-v1.keras.initializers.truncated-normal
  "Initializer that generates a truncated normal distribution.

  These values are similar to values from a `random_normal_initializer`
  except that values more than two standard deviations from the mean
  are discarded and re-drawn. This is the recommended initializer for
  neural network weights and filters.

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values to
      generate. Defaults to 0.
    stddev: a python scalar or a scalar tensor. Standard deviation of the random
      values to generate. Defaults to 0.05.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: The data type. Only floating point types are supported.
    
  Returns:
    A TruncatedNormal instance.
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

(defn truncated-normal 
  "Initializer that generates a truncated normal distribution.

  These values are similar to values from a `random_normal_initializer`
  except that values more than two standard deviations from the mean
  are discarded and re-drawn. This is the recommended initializer for
  neural network weights and filters.

  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values to
      generate. Defaults to 0.
    stddev: a python scalar or a scalar tensor. Standard deviation of the random
      values to generate. Defaults to 0.05.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: The data type. Only floating point types are supported.
    
  Returns:
    A TruncatedNormal instance.
  "
  [ & {:keys [mean stddev seed dtype]
       :or {seed None}} ]
  
   (py/call-attr-kw initializers "truncated_normal" [] {:mean mean :stddev stddev :seed seed :dtype dtype }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
