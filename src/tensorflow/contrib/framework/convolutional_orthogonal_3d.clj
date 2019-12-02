(ns tensorflow.contrib.framework.convolutional-orthogonal-3d
  "Initializer that generates a 3D orthogonal kernel for ConvNets.

  The shape of the tensor must have length 5. The number of input
  filters must not exceed the number of output filters.
  The orthogonality(==isometry) is exact when the inputs are circular padded.
  There are finite-width effects with non-circular padding (e.g. zero padding).
  See algorithm 1 (Xiao et al., 2018).

  Args:
    gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
      The 2-norm of an input is multiplied by a factor of `gain` after applying
      this convolution.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  References:
      [Xiao et al., 2018](http://proceedings.mlr.press/v80/xiao18a.html)
      ([pdf](http://proceedings.mlr.press/v80/xiao18a/xiao18a.pdf))
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce framework (import-module "tensorflow.contrib.framework"))

(defn convolutional-orthogonal-3d 
  "Initializer that generates a 3D orthogonal kernel for ConvNets.

  The shape of the tensor must have length 5. The number of input
  filters must not exceed the number of output filters.
  The orthogonality(==isometry) is exact when the inputs are circular padded.
  There are finite-width effects with non-circular padding (e.g. zero padding).
  See algorithm 1 (Xiao et al., 2018).

  Args:
    gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
      The 2-norm of an input is multiplied by a factor of `gain` after applying
      this convolution.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  References:
      [Xiao et al., 2018](http://proceedings.mlr.press/v80/xiao18a.html)
      ([pdf](http://proceedings.mlr.press/v80/xiao18a/xiao18a.pdf))
  "
  [ & {:keys [gain seed dtype]
       :or {seed None}} ]
  
   (py/call-attr-kw framework "convolutional_orthogonal_3d" [] {:gain gain :seed seed :dtype dtype }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
