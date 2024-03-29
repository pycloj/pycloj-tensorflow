(ns tensorflow.initializers.constant
  "Initializer that generates tensors with constant values.

  The resulting tensor is populated with values of type `dtype`, as
  specified by arguments `value` following the desired `shape` of the
  new tensor (see examples below).

  The argument `value` can be a constant value, or a list of values of type
  `dtype`. If `value` is a list, then the length of the list must be less
  than or equal to the number of elements implied by the desired shape of the
  tensor. In the case where the total number of elements in `value` is less
  than the number of elements required by the tensor shape, the last element
  in `value` will be used to fill the remaining entries. If the total number of
  elements in `value` is greater than the number of elements required by the
  tensor shape, the initializer will raise a `ValueError`.

  Args:
    value: A Python scalar, list or tuple of values, or a N-dimensional numpy
      array. All elements of the initialized variable will be set to the
      corresponding value in the `value` argument.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer.
    verify_shape: Boolean that enables verification of the shape of `value`. If
      `True`, the initializer will throw an error if the shape of `value` is not
      compatible with the shape of the initialized tensor.

  Raises:
    TypeError: If the input `value` is not one of the expected types.

  Examples:
    The following example can be rewritten using a numpy.ndarray instead
    of the `value` list, even reshaped, as shown in the two commented lines
    below the `value` list initialization.

  ```python
    >>> import numpy as np
    >>> import tensorflow as tf

    >>> value = [0, 1, 2, 3, 4, 5, 6, 7]
    >>> # value = np.array(value)
    >>> # value = value.reshape([2, 4])
    >>> init = tf.compat.v1.constant_initializer(value)

    >>> print('fitting shape:')
    >>> with tf.compat.v1.Session():
    >>>   x = tf.compat.v1.get_variable('x', shape=[2, 4], initializer=init)
    >>>   x.initializer.run()
    >>>   print(x.eval())

    fitting shape:
    [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]]

    >>> print('larger shape:')
    >>> with tf.compat.v1.Session():
    >>>   x = tf.compat.v1.get_variable('x', shape=[3, 4], initializer=init)
    >>>   x.initializer.run()
    >>>   print(x.eval())

    larger shape:
    [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]
     [ 7.  7.  7.  7.]]

    >>> print('smaller shape:')
    >>> with tf.compat.v1.Session():
    >>>   x = tf.compat.v1.get_variable('x', shape=[2, 3], initializer=init)

    ValueError: Too many elements provided. Needed at most 6, but received 8

    >>> print('shape verification:')
    >>> init_verify = tf.compat.v1.constant_initializer(value,
    verify_shape=True)
    >>> with tf.compat.v1.Session():
    >>>   x = tf.compat.v1.get_variable('x', shape=[3, 4],
    initializer=init_verify)

    TypeError: Expected Tensor's shape: (3, 4), got (8,).
  ```
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

(defn constant 
  "Initializer that generates tensors with constant values.

  The resulting tensor is populated with values of type `dtype`, as
  specified by arguments `value` following the desired `shape` of the
  new tensor (see examples below).

  The argument `value` can be a constant value, or a list of values of type
  `dtype`. If `value` is a list, then the length of the list must be less
  than or equal to the number of elements implied by the desired shape of the
  tensor. In the case where the total number of elements in `value` is less
  than the number of elements required by the tensor shape, the last element
  in `value` will be used to fill the remaining entries. If the total number of
  elements in `value` is greater than the number of elements required by the
  tensor shape, the initializer will raise a `ValueError`.

  Args:
    value: A Python scalar, list or tuple of values, or a N-dimensional numpy
      array. All elements of the initialized variable will be set to the
      corresponding value in the `value` argument.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer.
    verify_shape: Boolean that enables verification of the shape of `value`. If
      `True`, the initializer will throw an error if the shape of `value` is not
      compatible with the shape of the initialized tensor.

  Raises:
    TypeError: If the input `value` is not one of the expected types.

  Examples:
    The following example can be rewritten using a numpy.ndarray instead
    of the `value` list, even reshaped, as shown in the two commented lines
    below the `value` list initialization.

  ```python
    >>> import numpy as np
    >>> import tensorflow as tf

    >>> value = [0, 1, 2, 3, 4, 5, 6, 7]
    >>> # value = np.array(value)
    >>> # value = value.reshape([2, 4])
    >>> init = tf.compat.v1.constant_initializer(value)

    >>> print('fitting shape:')
    >>> with tf.compat.v1.Session():
    >>>   x = tf.compat.v1.get_variable('x', shape=[2, 4], initializer=init)
    >>>   x.initializer.run()
    >>>   print(x.eval())

    fitting shape:
    [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]]

    >>> print('larger shape:')
    >>> with tf.compat.v1.Session():
    >>>   x = tf.compat.v1.get_variable('x', shape=[3, 4], initializer=init)
    >>>   x.initializer.run()
    >>>   print(x.eval())

    larger shape:
    [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]
     [ 7.  7.  7.  7.]]

    >>> print('smaller shape:')
    >>> with tf.compat.v1.Session():
    >>>   x = tf.compat.v1.get_variable('x', shape=[2, 3], initializer=init)

    ValueError: Too many elements provided. Needed at most 6, but received 8

    >>> print('shape verification:')
    >>> init_verify = tf.compat.v1.constant_initializer(value,
    verify_shape=True)
    >>> with tf.compat.v1.Session():
    >>>   x = tf.compat.v1.get_variable('x', shape=[3, 4],
    initializer=init_verify)

    TypeError: Expected Tensor's shape: (3, 4), got (8,).
  ```
  "
  [ & {:keys [value dtype verify_shape]} ]
   (py/call-attr-kw initializers "constant" [] {:value value :dtype dtype :verify_shape verify_shape }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
