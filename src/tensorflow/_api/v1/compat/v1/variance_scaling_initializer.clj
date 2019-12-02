(ns tensorflow.-api.v1.compat.v1.variance-scaling-initializer
  "Initializer capable of adapting its scale to the shape of weights tensors.

  With `distribution=\"truncated_normal\" or \"untruncated_normal\"`,
  samples are drawn from a truncated/untruncated normal
  distribution with a mean of zero and a standard deviation (after truncation,
  if used) `stddev = sqrt(scale / n)`
  where n is:
    - number of input units in the weight tensor, if mode = \"fan_in\"
    - number of output units, if mode = \"fan_out\"
    - average of the numbers of input and output units, if mode = \"fan_avg\"

  With `distribution=\"uniform\"`, samples are drawn from a uniform distribution
  within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

  Args:
    scale: Scaling factor (positive float).
    mode: One of \"fan_in\", \"fan_out\", \"fan_avg\".
    distribution: Random distribution to use. One of \"normal\", \"uniform\".
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.

  Raises:
    ValueError: In case of an invalid value for the \"scale\", mode\" or
      \"distribution\" arguments.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v1 (import-module "tensorflow._api.v1.compat.v1"))

(defn variance-scaling-initializer 
  "Initializer capable of adapting its scale to the shape of weights tensors.

  With `distribution=\"truncated_normal\" or \"untruncated_normal\"`,
  samples are drawn from a truncated/untruncated normal
  distribution with a mean of zero and a standard deviation (after truncation,
  if used) `stddev = sqrt(scale / n)`
  where n is:
    - number of input units in the weight tensor, if mode = \"fan_in\"
    - number of output units, if mode = \"fan_out\"
    - average of the numbers of input and output units, if mode = \"fan_avg\"

  With `distribution=\"uniform\"`, samples are drawn from a uniform distribution
  within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

  Args:
    scale: Scaling factor (positive float).
    mode: One of \"fan_in\", \"fan_out\", \"fan_avg\".
    distribution: Random distribution to use. One of \"normal\", \"uniform\".
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.

  Raises:
    ValueError: In case of an invalid value for the \"scale\", mode\" or
      \"distribution\" arguments.
  "
  [ & {:keys [scale mode distribution seed dtype]
       :or {seed None}} ]
  
   (py/call-attr-kw v1 "variance_scaling_initializer" [] {:scale scale :mode mode :distribution distribution :seed seed :dtype dtype }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
