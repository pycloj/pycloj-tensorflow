(ns tensorflow.contrib.keras.api.keras.backend
  "Keras backend API."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce backend (import-module "tensorflow.contrib.keras.api.keras.backend"))

(defn abs 
  "Element-wise absolute value.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  "
  [ x ]
  (py/call-attr backend "abs"  x ))
(defn all 
  "Bitwise reduction (logical AND).

  Arguments:
      x: Tensor or variable.
      axis: axis along which to perform the reduction.
      keepdims: whether the drop or broadcast the reduction axes.

  Returns:
      A uint8 tensor (0s and 1s).
  "
  [x axis  & {:keys [keepdims]} ]
    (py/call-attr-kw backend "all" [x axis] {:keepdims keepdims }))
(defn any 
  "Bitwise reduction (logical OR).

  Arguments:
      x: Tensor or variable.
      axis: axis along which to perform the reduction.
      keepdims: whether the drop or broadcast the reduction axes.

  Returns:
      A uint8 tensor (0s and 1s).
  "
  [x axis  & {:keys [keepdims]} ]
    (py/call-attr-kw backend "any" [x axis] {:keepdims keepdims }))
(defn arange 
  "Creates a 1D tensor containing a sequence of integers.

  The function arguments use the same convention as
  Theano's arange: if only one argument is provided,
  it is in fact the \"stop\" argument and \"start\" is 0.

  The default type of the returned tensor is `'int32'` to
  match TensorFlow's default.

  Arguments:
      start: Start value.
      stop: Stop value.
      step: Difference between two successive values.
      dtype: Integer dtype to use.

  Returns:
      An integer tensor.

  Example:
      ```python
      >>> tf.keras.backend.arange(start=0, stop=10, step=1.5)
      <tf.Tensor: id=96, shape=(7,), dtype=float32,
          numpy=array([0. , 1.5, 3. , 4.5, 6. , 7.5, 9. ], dtype=float32)>

      ```

  "
  [start stop  & {:keys [step dtype]} ]
    (py/call-attr-kw backend "arange" [start stop] {:step step :dtype dtype }))
(defn argmax 
  "Returns the index of the maximum value along an axis.

  Arguments:
      x: Tensor or variable.
      axis: axis along which to perform the reduction.

  Returns:
      A tensor.
  "
  [x  & {:keys [axis]} ]
    (py/call-attr-kw backend "argmax" [x] {:axis axis }))
(defn argmin 
  "Returns the index of the minimum value along an axis.

  Arguments:
      x: Tensor or variable.
      axis: axis along which to perform the reduction.

  Returns:
      A tensor.
  "
  [x  & {:keys [axis]} ]
    (py/call-attr-kw backend "argmin" [x] {:axis axis }))

(defn backend 
  "Publicly accessible method for determining the current backend.

  Only exists for API compatibility with multi-backend Keras.

  Returns:
      The string \"tensorflow\".
  "
  [  ]
  (py/call-attr backend "backend"  ))

(defn batch-dot 
  "Batchwise dot product.

  `batch_dot` is used to compute dot product of `x` and `y` when
  `x` and `y` are data in batch, i.e. in a shape of
  `(batch_size, :)`.
  `batch_dot` results in a tensor or variable with less dimensions
  than the input. If the number of dimensions is reduced to 1,
  we use `expand_dims` to make sure that ndim is at least 2.

  Arguments:
      x: Keras tensor or variable with `ndim >= 2`.
      y: Keras tensor or variable with `ndim >= 2`.
      axes: list of (or single) int with target dimensions.
          The lengths of `axes[0]` and `axes[1]` should be the same.

  Returns:
      A tensor with shape equal to the concatenation of `x`'s shape
      (less the dimension that was summed over) and `y`'s shape
      (less the batch dimension and the dimension that was summed over).
      If the final rank is 1, we reshape it to `(batch_size, 1)`.

  Examples:
      Assume `x = [[1, 2], [3, 4]]` and `y = [[5, 6], [7, 8]]`
      `batch_dot(x, y, axes=1) = [[17, 53]]` which is the main diagonal
      of `x.dot(y.T)`, although we never have to calculate the off-diagonal
      elements.

      Shape inference:
      Let `x`'s shape be `(100, 20)` and `y`'s shape be `(100, 30, 20)`.
      If `axes` is (1, 2), to find the output shape of resultant tensor,
          loop through each dimension in `x`'s shape and `y`'s shape:

      * `x.shape[0]` : 100 : append to output shape
      * `x.shape[1]` : 20 : do not append to output shape,
          dimension 1 of `x` has been summed over. (`dot_axes[0]` = 1)
      * `y.shape[0]` : 100 : do not append to output shape,
          always ignore first dimension of `y`
      * `y.shape[1]` : 30 : append to output shape
      * `y.shape[2]` : 20 : do not append to output shape,
          dimension 2 of `y` has been summed over. (`dot_axes[1]` = 2)
      `output_shape` = `(100, 30)`

  ```python
  >>> x_batch = K.ones(shape=(32, 20, 1))
  >>> y_batch = K.ones(shape=(32, 30, 20))
  >>> xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1, 2])
  >>> K.int_shape(xy_batch_dot)
  (32, 1, 30)
  ```
  "
  [ x y axes ]
  (py/call-attr backend "batch_dot"  x y axes ))

(defn batch-flatten 
  "Turn a nD tensor into a 2D tensor with same 0th dimension.

  In other words, it flattens each data samples of a batch.

  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor.

  Examples:
    Flattening a 3D tensor to 2D by collapsing the last dimension.

  ```python
  >>> from tensorflow.keras import backend as K
  >>> x_batch = K.ones(shape=(2, 3, 4, 5))
  >>> x_batch_flatten = K.batch_flatten(x_batch)
  >>> K.int_shape(x_batch_flatten)
  (2, 60)
  ```
  "
  [ x ]
  (py/call-attr backend "batch_flatten"  x ))

(defn batch-get-value 
  "Returns the value of more than one tensor variable.

  Arguments:
      tensors: list of ops to run.

  Returns:
      A list of Numpy arrays.

  Raises:
      RuntimeError: If this method is called inside defun.
  "
  [ tensors ]
  (py/call-attr backend "batch_get_value"  tensors ))
(defn batch-normalization 
  "Applies batch normalization on x given mean, var, beta and gamma.

  I.e. returns:
  `output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta`

  Arguments:
      x: Input tensor or variable.
      mean: Mean of batch.
      var: Variance of batch.
      beta: Tensor with which to center the input.
      gamma: Tensor by which to scale the input.
      axis: Integer, the axis that should be normalized.
          (typically the features axis).
      epsilon: Fuzz factor.

  Returns:
      A tensor.
  "
  [x mean var beta gamma  & {:keys [axis epsilon]} ]
    (py/call-attr-kw backend "batch_normalization" [x mean var beta gamma] {:axis axis :epsilon epsilon }))

(defn batch-set-value 
  "Sets the values of many tensor variables at once.

  Arguments:
      tuples: a list of tuples `(tensor, value)`.
          `value` should be a Numpy array.
  "
  [ tuples ]
  (py/call-attr backend "batch_set_value"  tuples ))

(defn bias-add 
  "Adds a bias vector to a tensor.

  Arguments:
      x: Tensor or variable.
      bias: Bias tensor to add.
      data_format: string, `\"channels_last\"` or `\"channels_first\"`.

  Returns:
      Output tensor.

  Raises:
      ValueError: In one of the two cases below:
                  1. invalid `data_format` argument.
                  2. invalid bias shape.
                     the bias should be either a vector or
                     a tensor with ndim(x) - 1 dimension
  "
  [ x bias data_format ]
  (py/call-attr backend "bias_add"  x bias data_format ))
(defn binary-crossentropy 
  "Binary crossentropy between an output tensor and a target tensor.

  Arguments:
      target: A tensor with the same shape as `output`.
      output: A tensor.
      from_logits: Whether `output` is expected to be a logits tensor.
          By default, we consider that `output`
          encodes a probability distribution.

  Returns:
      A tensor.
  "
  [target output  & {:keys [from_logits]} ]
    (py/call-attr-kw backend "binary_crossentropy" [target output] {:from_logits from_logits }))

(defn cast 
  "Casts a tensor to a different dtype and returns it.

  You can cast a Keras variable but it still returns a Keras tensor.

  Arguments:
      x: Keras tensor (or variable).
      dtype: String, either (`'float16'`, `'float32'`, or `'float64'`).

  Returns:
      Keras tensor with dtype `dtype`.

  Examples:
      Cast a float32 variable to a float64 tensor

  ```python
  >>> import tensorflow as tf
  >>> from tensorflow.keras import backend as K
  >>> input = K.ones(shape=(1,3))
  >>> print(input)
  >>> cast_input = K.cast(input, dtype='float64')
  >>> print(cast_input)

  <tf.Variable 'Variable:0' shape=(1, 3) dtype=float32,
       numpy=array([[1., 1., 1.]], dtype=float32)>
  tf.Tensor([[1. 1. 1.]], shape=(1, 3), dtype=float64)
  ```
  "
  [ x dtype ]
  (py/call-attr backend "cast"  x dtype ))

(defn cast-to-floatx 
  "Cast a Numpy array to the default Keras float type.

  Arguments:
      x: Numpy array.

  Returns:
      The same Numpy array, cast to its new type.

  Example:
  ```python
      >>> from tensorflow.keras import backend as K
      >>> K.floatx()
      'float32'
      >>> arr = numpy.array([1.0, 2.0], dtype='float64')
      >>> arr.dtype
      dtype('float64')
      >>> new_arr = K.cast_to_floatx(arr)
      >>> new_arr
      array([ 1.,  2.], dtype=float32)
      >>> new_arr.dtype
      dtype('float32')
  ```
  "
  [ x ]
  (py/call-attr backend "cast_to_floatx"  x ))
(defn categorical-crossentropy 
  "Categorical crossentropy between an output tensor and a target tensor.

  Arguments:
      target: A tensor of the same shape as `output`.
      output: A tensor resulting from a softmax
          (unless `from_logits` is True, in which
          case `output` is expected to be the logits).
      from_logits: Boolean, whether `output` is the
          result of a softmax, or is a tensor of logits.
      axis: Int specifying the channels axis. `axis=-1` corresponds to data
          format `channels_last', and `axis=1` corresponds to data format
          `channels_first`.

  Returns:
      Output tensor.

  Raises:
      ValueError: if `axis` is neither -1 nor one of the axes of `output`.

  Example:
  ```python:
  import tensorflow as tf
  from tensorflow.keras import backend as K
  a = tf.constant([1., 0., 0., 0., 1., 0., 0., 0., 1.], shape=[3,3])
  print(\"a: \", a)
  b = tf.constant([.9, .05, .05, .5, .89, .6, .05, .01, .94], shape=[3,3])
  print(\"b: \", b)
  loss = K.categorical_crossentropy(a, b)
  print('Loss: ', loss) #Loss: tf.Tensor([0.10536055 0.8046684  0.06187541], shape=(3,), dtype=float32)
  loss = K.categorical_crossentropy(a, a)
  print('Loss: ', loss) #Loss:  tf.Tensor([1.1920929e-07 1.1920929e-07 1.1920929e-07], shape=(3,), dtype=float32)
  ```
  "
  [target output  & {:keys [from_logits axis]} ]
    (py/call-attr-kw backend "categorical_crossentropy" [target output] {:from_logits from_logits :axis axis }))

(defn clear-session 
  "Destroys the current TF graph and creates a new one.

  Useful to avoid clutter from old models / layers.
  "
  [  ]
  (py/call-attr backend "clear_session"  ))

(defn clip 
  "Element-wise value clipping.

  Arguments:
      x: Tensor or variable.
      min_value: Python float or integer.
      max_value: Python float or integer.

  Returns:
      A tensor.
  "
  [ x min_value max_value ]
  (py/call-attr backend "clip"  x min_value max_value ))
(defn concatenate 
  "Concatenates a list of tensors alongside the specified axis.

  Arguments:
      tensors: list of tensors to concatenate.
      axis: concatenation axis.

  Returns:
      A tensor.

  Example:
      ```python
      >>> a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      >>> b = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
      >>> tf.keras.backend.concatenate((a, b), axis=-1)
      <tf.Tensor: id=14, shape=(3, 6), dtype=int32, numpy=
      array([[ 1,  2,  3, 10, 20, 30],
             [ 4,  5,  6, 40, 50, 60],
             [ 7,  8,  9, 70, 80, 90]], dtype=int32)>
      ```
  "
  [tensors  & {:keys [axis]} ]
    (py/call-attr-kw backend "concatenate" [tensors] {:axis axis }))

(defn constant 
  "Creates a constant tensor.

  Arguments:
      value: A constant value (or list)
      dtype: The type of the elements of the resulting tensor.
      shape: Optional dimensions of resulting tensor.
      name: Optional name for the tensor.

  Returns:
      A Constant Tensor.
  "
  [ value dtype shape name ]
  (py/call-attr backend "constant"  value dtype shape name ))

(defn conv1d 
  "1D convolution.

  Arguments:
      x: Tensor or variable.
      kernel: kernel tensor.
      strides: stride integer.
      padding: string, `\"same\"`, `\"causal\"` or `\"valid\"`.
      data_format: string, one of \"channels_last\", \"channels_first\".
      dilation_rate: integer dilate rate.

  Returns:
      A tensor, result of 1D convolution.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
  "
  [x kernel & {:keys [strides padding data_format dilation_rate]
                       :or {data_format None}} ]
    (py/call-attr-kw backend "conv1d" [x kernel] {:strides strides :padding padding :data_format data_format :dilation_rate dilation_rate }))

(defn conv2d 
  "2D convolution.

  Arguments:
      x: Tensor or variable.
      kernel: kernel tensor.
      strides: strides tuple.
      padding: string, `\"same\"` or `\"valid\"`.
      data_format: `\"channels_last\"` or `\"channels_first\"`.
      dilation_rate: tuple of 2 integers.

  Returns:
      A tensor, result of 2D convolution.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
  "
  [x kernel & {:keys [strides padding data_format dilation_rate]
                       :or {data_format None}} ]
    (py/call-attr-kw backend "conv2d" [x kernel] {:strides strides :padding padding :data_format data_format :dilation_rate dilation_rate }))

(defn conv2d-transpose 
  "2D deconvolution (i.e.

  transposed convolution).

  Arguments:
      x: Tensor or variable.
      kernel: kernel tensor.
      output_shape: 1D int tensor for the output shape.
      strides: strides tuple.
      padding: string, `\"same\"` or `\"valid\"`.
      data_format: string, `\"channels_last\"` or `\"channels_first\"`.
      dilation_rate: Tuple of 2 integers.

  Returns:
      A tensor, result of transposed 2D convolution.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
  "
  [x kernel output_shape & {:keys [strides padding data_format dilation_rate]
                       :or {data_format None}} ]
    (py/call-attr-kw backend "conv2d_transpose" [x kernel output_shape] {:strides strides :padding padding :data_format data_format :dilation_rate dilation_rate }))

(defn conv3d 
  "3D convolution.

  Arguments:
      x: Tensor or variable.
      kernel: kernel tensor.
      strides: strides tuple.
      padding: string, `\"same\"` or `\"valid\"`.
      data_format: string, `\"channels_last\"` or `\"channels_first\"`.
      dilation_rate: tuple of 3 integers.

  Returns:
      A tensor, result of 3D convolution.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
  "
  [x kernel & {:keys [strides padding data_format dilation_rate]
                       :or {data_format None}} ]
    (py/call-attr-kw backend "conv3d" [x kernel] {:strides strides :padding padding :data_format data_format :dilation_rate dilation_rate }))

(defn cos 
  "Computes cos of x element-wise.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  "
  [ x ]
  (py/call-attr backend "cos"  x ))

(defn count-params 
  "Returns the static number of elements in a variable or tensor.

  Arguments:
      x: Variable or tensor.

  Returns:
      Integer, the number of scalars in `x`.

  Example:
  ```python
  >>> kvar = K.zeros((2,3))
  >>> K.count_params(kvar)
  6
  >>> K.eval(kvar)
  array([[ 0.,  0.,  0.],
         [ 0.,  0.,  0.]], dtype=float32)
  ```
  "
  [ x ]
  (py/call-attr backend "count_params"  x ))

(defn ctc-batch-cost 
  "Runs CTC loss algorithm on each batch element.

  Arguments:
      y_true: tensor `(samples, max_string_length)`
          containing the truth labels.
      y_pred: tensor `(samples, time_steps, num_categories)`
          containing the prediction, or output of the softmax.
      input_length: tensor `(samples, 1)` containing the sequence length for
          each batch item in `y_pred`.
      label_length: tensor `(samples, 1)` containing the sequence length for
          each batch item in `y_true`.

  Returns:
      Tensor with shape (samples,1) containing the
          CTC loss of each element.
  "
  [ y_true y_pred input_length label_length ]
  (py/call-attr backend "ctc_batch_cost"  y_true y_pred input_length label_length ))
(defn ctc-decode 
  "Decodes the output of a softmax.

  Can use either greedy search (also known as best path)
  or a constrained dictionary search.

  Arguments:
      y_pred: tensor `(samples, time_steps, num_categories)`
          containing the prediction, or output of the softmax.
      input_length: tensor `(samples, )` containing the sequence length for
          each batch item in `y_pred`.
      greedy: perform much faster best-path search if `true`.
          This does not use a dictionary.
      beam_width: if `greedy` is `false`: a beam search decoder will be used
          with a beam of this width.
      top_paths: if `greedy` is `false`,
          how many of the most probable paths will be returned.

  Returns:
      Tuple:
          List: if `greedy` is `true`, returns a list of one element that
              contains the decoded sequence.
              If `false`, returns the `top_paths` most probable
              decoded sequences.
              Important: blank labels are returned as `-1`.
          Tensor `(top_paths, )` that contains
              the log probability of each decoded sequence.
  "
  [y_pred input_length  & {:keys [greedy beam_width top_paths]} ]
    (py/call-attr-kw backend "ctc_decode" [y_pred input_length] {:greedy greedy :beam_width beam_width :top_paths top_paths }))

(defn ctc-label-dense-to-sparse 
  "Converts CTC labels from dense to sparse.

  Arguments:
      labels: dense CTC labels.
      label_lengths: length of the labels.

  Returns:
      A sparse tensor representation of the labels.
  "
  [ labels label_lengths ]
  (py/call-attr backend "ctc_label_dense_to_sparse"  labels label_lengths ))

(defn dot 
  "Multiplies 2 tensors (and/or variables) and returns a *tensor*.

  When attempting to multiply a nD tensor
  with a nD tensor, it reproduces the Theano behavior.
  (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A tensor, dot product of `x` and `y`.

  Examples:
  ```python
  # dot product between tensors
  >>> x = K.placeholder(shape=(2, 3))
  >>> y = K.placeholder(shape=(3, 4))
  >>> xy = K.dot(x, y)
  >>> xy
  <tf.Tensor 'MatMul_9:0' shape=(2, 4) dtype=float32>
  ```

  ```python
  # dot product between tensors
  >>> x = K.placeholder(shape=(32, 28, 3))
  >>> y = K.placeholder(shape=(3, 4))
  >>> xy = K.dot(x, y)
  >>> xy
  <tf.Tensor 'MatMul_9:0' shape=(32, 28, 4) dtype=float32>
  ```

  ```python
  # Theano-like behavior example
  >>> x = K.random_uniform_variable(shape=(2, 3), low=0, high=1)
  >>> y = K.ones((4, 3, 5))
  >>> xy = K.dot(x, y)
  >>> K.int_shape(xy)
  (2, 4, 5)
  ```
  "
  [ x y ]
  (py/call-attr backend "dot"  x y ))

(defn dropout 
  "Sets entries in `x` to zero at random, while scaling the entire tensor.

  Arguments:
      x: tensor
      level: fraction of the entries in the tensor
          that will be set to 0.
      noise_shape: shape for randomly generated keep/drop flags,
          must be broadcastable to the shape of `x`
      seed: random seed to ensure determinism.

  Returns:
      A tensor.
  "
  [ x level noise_shape seed ]
  (py/call-attr backend "dropout"  x level noise_shape seed ))

(defn dtype 
  "Returns the dtype of a Keras tensor or variable, as a string.

  Arguments:
      x: Tensor or variable.

  Returns:
      String, dtype of `x`.

  Examples:
  ```python
  >>> from keras import backend as K
  >>> K.dtype(K.placeholder(shape=(2,4,5)))
  'float32'
  >>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float32'))
  'float32'
  >>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float64'))
  'float64'
  # Keras variable
  >>> kvar = K.variable(np.array([[1, 2], [3, 4]]))
  >>> K.dtype(kvar)
  'float32'
  >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
  >>> K.dtype(kvar)
  'float32'
  ```
  "
  [ x ]
  (py/call-attr backend "dtype"  x ))
(defn elu 
  "Exponential linear unit.

  Arguments:
      x: A tensor or variable to compute the activation function for.
      alpha: A scalar, slope of negative section.

  Returns:
      A tensor.
  "
  [x  & {:keys [alpha]} ]
    (py/call-attr-kw backend "elu" [x] {:alpha alpha }))

(defn epsilon 
  "Returns the value of the fuzz factor used in numeric expressions.

  Returns:
      A float.

  Example:
  ```python
  keras.backend.epsilon() >>>1e-07
  ```
  "
  [  ]
  (py/call-attr backend "epsilon"  ))

(defn equal 
  "Element-wise equality between two tensors.

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  "
  [ x y ]
  (py/call-attr backend "equal"  x y ))

(defn eval 
  "Evaluates the value of a variable.

  Arguments:
      x: A variable.

  Returns:
      A Numpy array.

  Examples:
  ```python
  >>> from keras import backend as K
  >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
  >>> K.eval(kvar)
  array([[ 1.,  2.],
         [ 3.,  4.]], dtype=float32)
  ```
  "
  [ x ]
  (py/call-attr backend "eval"  x ))

(defn exp 
  "Element-wise exponential.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  "
  [ x ]
  (py/call-attr backend "exp"  x ))
(defn expand-dims 
  "Adds a 1-sized dimension at index \"axis\".

  Arguments:
      x: A tensor or variable.
      axis: Position where to add a new axis.

  Returns:
      A tensor with expanded dimensions.
  "
  [x  & {:keys [axis]} ]
    (py/call-attr-kw backend "expand_dims" [x] {:axis axis }))

(defn eye 
  "Instantiate an identity matrix and returns it.

  Arguments:
      size: Integer, number of rows/columns.
      dtype: String, data type of returned Keras variable.
      name: String, name of returned Keras variable.

  Returns:
      A Keras variable, an identity matrix.

  Example:
  ```python
  >>> from keras import backend as K
  >>> kvar = K.eye(3)
  >>> K.eval(kvar)
  array([[ 1.,  0.,  0.],
         [ 0.,  1.,  0.],
         [ 0.,  0.,  1.]], dtype=float32)
  ```

  "
  [ size dtype name ]
  (py/call-attr backend "eye"  size dtype name ))

(defn flatten 
  "Flatten a tensor.

  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor, reshaped into 1-D

  Example:
      ```python
      >>> b = tf.constant([[1, 2], [3, 4]])
      >>> b
      <tf.Tensor: id=102, shape=(2, 2), dtype=int32, numpy=
      array([[1, 2],
             [3, 4]], dtype=int32)>
      >>> tf.keras.backend.flatten(b)
      <tf.Tensor: id=105, shape=(4,), dtype=int32,
          numpy=array([1, 2, 3, 4], dtype=int32)>
      ```
  "
  [ x ]
  (py/call-attr backend "flatten"  x ))

(defn floatx 
  "Returns the default float type, as a string.

  E.g. 'float16', 'float32', 'float64'.

  Returns:
      String, the current default float type.

  Example:
  ```python
  keras.backend.floatx() >>> 'float32'
  ```
  "
  [  ]
  (py/call-attr backend "floatx"  ))

(defn foldl 
  "Reduce elems using fn to combine them from left to right.

  Arguments:
      fn: Callable that will be called upon each element in elems and an
          accumulator, for instance `lambda acc, x: acc + x`
      elems: tensor
      initializer: The first value used (`elems[0]` in case of None)
      name: A string name for the foldl node in the graph

  Returns:
      Tensor with same type and shape as `initializer`.
  "
  [ fn elems initializer name ]
  (py/call-attr backend "foldl"  fn elems initializer name ))

(defn foldr 
  "Reduce elems using fn to combine them from right to left.

  Arguments:
      fn: Callable that will be called upon each element in elems and an
          accumulator, for instance `lambda acc, x: acc + x`
      elems: tensor
      initializer: The first value used (`elems[-1]` in case of None)
      name: A string name for the foldr node in the graph

  Returns:
      Same type and shape as initializer
  "
  [ fn elems initializer name ]
  (py/call-attr backend "foldr"  fn elems initializer name ))

(defn function 
  "Instantiates a Keras function.

  Arguments:
      inputs: List of placeholder tensors.
      outputs: List of output tensors.
      updates: List of update ops.
      name: String, name of function.
      **kwargs: Passed to `tf.Session.run`.

  Returns:
      Output values as Numpy arrays.

  Raises:
      ValueError: if invalid kwargs are passed in or if in eager execution.
  "
  [ inputs outputs updates name ]
  (py/call-attr backend "function"  inputs outputs updates name ))

(defn gather 
  "Retrieves the elements of indices `indices` in the tensor `reference`.

  Arguments:
      reference: A tensor.
      indices: An integer tensor of indices.

  Returns:
      A tensor of same type as `reference`.
  "
  [ reference indices ]
  (py/call-attr backend "gather"  reference indices ))

(defn get-session 
  "Returns the TF session to be used by the backend.

  If a default TensorFlow session is available, we will return it.

  Else, we will return the global Keras session assuming it matches
  the current graph.

  If no global Keras session exists at this point:
  we will create a new global session.

  Note that you can manually set the global session
  via `K.set_session(sess)`.

  Arguments:
      op_input_list: An option sequence of tensors or ops, which will be used
        to determine the current graph. Otherwise the default graph will be
        used.

  Returns:
      A TensorFlow session.
  "
  [ & {:keys [op_input_list]} ]
   (py/call-attr-kw backend "get_session" [] {:op_input_list op_input_list }))

(defn get-uid 
  "Associates a string prefix with an integer counter in a TensorFlow graph.

  Arguments:
    prefix: String prefix to index.

  Returns:
    Unique integer ID.

  Example:

  ```
    >>> get_uid('dense')
    1
    >>> get_uid('dense')
    2
  ```
  "
  [ & {:keys [prefix]} ]
   (py/call-attr-kw backend "get_uid" [] {:prefix prefix }))

(defn get-value 
  "Returns the value of a variable.

  Arguments:
      x: input variable.

  Returns:
      A Numpy array.
  "
  [ x ]
  (py/call-attr backend "get_value"  x ))

(defn gradients 
  "Returns the gradients of `loss` w.r.t. `variables`.

  Arguments:
      loss: Scalar tensor to minimize.
      variables: List of variables.

  Returns:
      A gradients tensor.
  "
  [ loss variables ]
  (py/call-attr backend "gradients"  loss variables ))

(defn greater 
  "Element-wise truth value of (x > y).

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  "
  [ x y ]
  (py/call-attr backend "greater"  x y ))

(defn greater-equal 
  "Element-wise truth value of (x >= y).

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  "
  [ x y ]
  (py/call-attr backend "greater_equal"  x y ))

(defn hard-sigmoid 
  "Segment-wise linear approximation of sigmoid.

  Faster than sigmoid.
  Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
  In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.

  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor.
  "
  [ x ]
  (py/call-attr backend "hard_sigmoid"  x ))

(defn image-data-format 
  "Returns the default image data format convention.

  Returns:
      A string, either `'channels_first'` or `'channels_last'`

  Example:
  ```python
  keras.backend.image_data_format() >>> 'channels_first'
  ```
  "
  [  ]
  (py/call-attr backend "image_data_format"  ))

(defn in-test-phase 
  "Selects `x` in test phase, and `alt` otherwise.

  Note that `alt` should have the *same shape* as `x`.

  Arguments:
      x: What to return in test phase
          (tensor or callable that returns a tensor).
      alt: What to return otherwise
          (tensor or callable that returns a tensor).
      training: Optional scalar tensor
          (or Python boolean, or Python integer)
          specifying the learning phase.

  Returns:
      Either `x` or `alt` based on `K.learning_phase`.
  "
  [ x alt training ]
  (py/call-attr backend "in_test_phase"  x alt training ))

(defn in-top-k 
  "Returns whether the `targets` are in the top `k` `predictions`.

  Arguments:
      predictions: A tensor of shape `(batch_size, classes)` and type `float32`.
      targets: A 1D tensor of length `batch_size` and type `int32` or `int64`.
      k: An `int`, number of top elements to consider.

  Returns:
      A 1D tensor of length `batch_size` and type `bool`.
      `output[i]` is `True` if `predictions[i, targets[i]]` is within top-`k`
      values of `predictions[i]`.
  "
  [ predictions targets k ]
  (py/call-attr backend "in_top_k"  predictions targets k ))

(defn in-train-phase 
  "Selects `x` in train phase, and `alt` otherwise.

  Note that `alt` should have the *same shape* as `x`.

  Arguments:
      x: What to return in train phase
          (tensor or callable that returns a tensor).
      alt: What to return otherwise
          (tensor or callable that returns a tensor).
      training: Optional scalar tensor
          (or Python boolean, or Python integer)
          specifying the learning phase.

  Returns:
      Either `x` or `alt` based on the `training` flag.
      the `training` flag defaults to `K.learning_phase()`.
  "
  [ x alt training ]
  (py/call-attr backend "in_train_phase"  x alt training ))

(defn int-shape 
  "Returns the shape of tensor or variable as a tuple of int or None entries.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tuple of integers (or None entries).

  Examples:
  ```python
  >>> from keras import backend as K
  >>> input = K.placeholder(shape=(2, 4, 5))
  >>> K.int_shape(input)
  (2, 4, 5)
  >>> val = np.array([[1, 2], [3, 4]])
  >>> kvar = K.variable(value=val)
  >>> K.int_shape(kvar)
  (2, 2)
  ```
  "
  [ x ]
  (py/call-attr backend "int_shape"  x ))

(defn is-sparse 
  "Returns whether a tensor is a sparse tensor.

  Arguments:
      tensor: A tensor instance.

  Returns:
      A boolean.

  Example:
  ```python
  >>> from keras import backend as K
  >>> a = K.placeholder((2, 2), sparse=False)
  >>> print(K.is_sparse(a))
  False
  >>> b = K.placeholder((2, 2), sparse=True)
  >>> print(K.is_sparse(b))
  True
  ```
  "
  [ tensor ]
  (py/call-attr backend "is_sparse"  tensor ))

(defn l2-normalize 
  "Normalizes a tensor wrt the L2 norm alongside the specified axis.

  Arguments:
      x: Tensor or variable.
      axis: axis along which to perform normalization.

  Returns:
      A tensor.
  "
  [ x axis ]
  (py/call-attr backend "l2_normalize"  x axis ))

(defn learning-phase 
  "Returns the learning phase flag.

  The learning phase flag is a bool tensor (0 = test, 1 = train)
  to be passed as input to any Keras function
  that uses a different behavior at train time and test time.

  Returns:
      Learning phase (scalar integer tensor or Python integer).
  "
  [  ]
  (py/call-attr backend "learning_phase"  ))

(defn less 
  "Element-wise truth value of (x < y).

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  "
  [ x y ]
  (py/call-attr backend "less"  x y ))

(defn less-equal 
  "Element-wise truth value of (x <= y).

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  "
  [ x y ]
  (py/call-attr backend "less_equal"  x y ))

(defn log 
  "Element-wise log.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  "
  [ x ]
  (py/call-attr backend "log"  x ))

(defn manual-variable-initialization 
  "Sets the manual variable initialization flag.

  This boolean flag determines whether
  variables should be initialized
  as they are instantiated (default), or if
  the user should handle the initialization
  (e.g. via `tf.compat.v1.initialize_all_variables()`).

  Arguments:
      value: Python boolean.
  "
  [ value ]
  (py/call-attr backend "manual_variable_initialization"  value ))

(defn map-fn 
  "Map the function fn over the elements elems and return the outputs.

  Arguments:
      fn: Callable that will be called upon each element in elems
      elems: tensor
      name: A string name for the map node in the graph
      dtype: Output data type.

  Returns:
      Tensor with dtype `dtype`.
  "
  [ fn elems name dtype ]
  (py/call-attr backend "map_fn"  fn elems name dtype ))
(defn max 
  "Maximum value in a tensor.

  Arguments:
      x: A tensor or variable.
      axis: An integer, the axis to find maximum values.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`,
          the reduced dimension is retained with length 1.

  Returns:
      A tensor with maximum values of `x`.
  "
  [x axis  & {:keys [keepdims]} ]
    (py/call-attr-kw backend "max" [x axis] {:keepdims keepdims }))

(defn maximum 
  "Element-wise maximum of two tensors.

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A tensor with the element wise maximum value(s) of `x` and `y`.

  Examples:
  ```python
  # maximum of two tensors
  >>> x = tf.Variable([[1, 2], [3, 4]])
  >>> y = tf.Variable([[2, 1], [0, -1]])
  >>> m = tf.keras.backend.maximum(x, y)
  >>> m
  <tf.Tensor: id=42, shape=(2, 2), dtype=int32, numpy=
  array([[2, 2],
         [3, 4]], dtype=int32)>
  ```
  "
  [ x y ]
  (py/call-attr backend "maximum"  x y ))
(defn mean 
  "Mean of a tensor, alongside the specified axis.

  Arguments:
      x: A tensor or variable.
      axis: A list of integer. Axes to compute the mean.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1 for each entry in `axis`. If `keepdims` is `True`,
          the reduced dimensions are retained with length 1.

  Returns:
      A tensor with the mean of elements of `x`.
  "
  [x axis  & {:keys [keepdims]} ]
    (py/call-attr-kw backend "mean" [x axis] {:keepdims keepdims }))
(defn min 
  "Minimum value in a tensor.

  Arguments:
      x: A tensor or variable.
      axis: An integer, the axis to find minimum values.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`,
          the reduced dimension is retained with length 1.

  Returns:
      A tensor with minimum values of `x`.
  "
  [x axis  & {:keys [keepdims]} ]
    (py/call-attr-kw backend "min" [x axis] {:keepdims keepdims }))

(defn minimum 
  "Element-wise minimum of two tensors.

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A tensor.
  "
  [ x y ]
  (py/call-attr backend "minimum"  x y ))

(defn moving-average-update 
  "Compute the moving average of a variable.

  Arguments:
      x: A Variable.
      value: A tensor with the same shape as `variable`.
      momentum: The moving average momentum.

  Returns:
      An Operation to update the variable.
  "
  [ x value momentum ]
  (py/call-attr backend "moving_average_update"  x value momentum ))

(defn name-scope 
  "A context manager for use when defining a Python op.

  This context manager pushes a name scope, which will make the name of all
  operations added within it have a prefix.

  For example, to define a new Python op called `my_op`:

  ```python
  def my_op(a):
    with tf.name_scope(\"MyOp\") as scope:
      a = tf.convert_to_tensor(a, name=\"a\")
      # Define some computation that uses `a`.
      return foo_op(..., name=scope)
  ```

  When executed, the Tensor `a` will have the name `MyOp/a`.

  Args:
    name: The prefix to use on all names created within the name scope.

  Returns:
    Name scope context manager.
  "
  [ name ]
  (py/call-attr backend "name_scope"  name ))

(defn ndim 
  "Returns the number of axes in a tensor, as an integer.

  Arguments:
      x: Tensor or variable.

  Returns:
      Integer (scalar), number of axes.

  Examples:
  ```python
  >>> from keras import backend as K
  >>> input = K.placeholder(shape=(2, 4, 5))
  >>> val = np.array([[1, 2], [3, 4]])
  >>> kvar = K.variable(value=val)
  >>> K.ndim(input)
  3
  >>> K.ndim(kvar)
  2
  ```
  "
  [ x ]
  (py/call-attr backend "ndim"  x ))
(defn normalize-batch-in-training 
  "Computes mean and std for batch then apply batch_normalization on batch.

  Arguments:
      x: Input tensor or variable.
      gamma: Tensor by which to scale the input.
      beta: Tensor with which to center the input.
      reduction_axes: iterable of integers,
          axes over which to normalize.
      epsilon: Fuzz factor.

  Returns:
      A tuple length of 3, `(normalized_tensor, mean, variance)`.
  "
  [x gamma beta reduction_axes  & {:keys [epsilon]} ]
    (py/call-attr-kw backend "normalize_batch_in_training" [x gamma beta reduction_axes] {:epsilon epsilon }))

(defn not-equal 
  "Element-wise inequality between two tensors.

  Arguments:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  "
  [ x y ]
  (py/call-attr backend "not_equal"  x y ))

(defn one-hot 
  "Computes the one-hot representation of an integer tensor.

  Arguments:
      indices: nD integer tensor of shape
          `(batch_size, dim1, dim2, ... dim(n-1))`
      num_classes: Integer, number of classes to consider.

  Returns:
      (n + 1)D one hot representation of the input
      with shape `(batch_size, dim1, dim2, ... dim(n-1), num_classes)`

  Returns:
      The one-hot tensor.
  "
  [ indices num_classes ]
  (py/call-attr backend "one_hot"  indices num_classes ))

(defn ones 
  "Instantiates an all-ones variable and returns it.

  Arguments:
      shape: Tuple of integers, shape of returned Keras variable.
      dtype: String, data type of returned Keras variable.
      name: String, name of returned Keras variable.

  Returns:
      A Keras variable, filled with `1.0`.
      Note that if `shape` was symbolic, we cannot return a variable,
      and will return a dynamically-shaped tensor instead.

  Example:
  ```python
  >>> from keras import backend as K
  >>> kvar = K.ones((3,4))
  >>> K.eval(kvar)
  array([[ 1.,  1.,  1.,  1.],
         [ 1.,  1.,  1.,  1.],
         [ 1.,  1.,  1.,  1.]], dtype=float32)
  ```
  "
  [ shape dtype name ]
  (py/call-attr backend "ones"  shape dtype name ))

(defn ones-like 
  "Instantiates an all-ones variable of the same shape as another tensor.

  Arguments:
      x: Keras variable or tensor.
      dtype: String, dtype of returned Keras variable.
           None uses the dtype of x.
      name: String, name for the variable to create.

  Returns:
      A Keras variable with the shape of x filled with ones.

  Example:
  ```python
  >>> from keras import backend as K
  >>> kvar = K.variable(np.random.random((2,3)))
  >>> kvar_ones = K.ones_like(kvar)
  >>> K.eval(kvar_ones)
  array([[ 1.,  1.,  1.],
         [ 1.,  1.,  1.]], dtype=float32)
  ```
  "
  [ x dtype name ]
  (py/call-attr backend "ones_like"  x dtype name ))

(defn permute-dimensions 
  "Permutes axes in a tensor.

  Arguments:
      x: Tensor or variable.
      pattern: A tuple of
          dimension indices, e.g. `(0, 2, 1)`.

  Returns:
      A tensor.

  Example:
    ```python
    >>> a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    >>> a
    <tf.Tensor: id=49, shape=(4, 3), dtype=int32, numpy=
    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9],
           [10, 11, 12]], dtype=int32)>
    >>> tf.keras.backend.permute_dimensions(a, pattern=(1, 0))
    <tf.Tensor: id=52, shape=(3, 4), dtype=int32, numpy=
    array([[ 1,  4,  7, 10],
           [ 2,  5,  8, 11],
           [ 3,  6,  9, 12]], dtype=int32)>
    ```
  "
  [ x pattern ]
  (py/call-attr backend "permute_dimensions"  x pattern ))

(defn placeholder 
  "Instantiates a placeholder tensor and returns it.

  Arguments:
      shape: Shape of the placeholder
          (integer tuple, may include `None` entries).
      ndim: Number of axes of the tensor.
          At least one of {`shape`, `ndim`} must be specified.
          If both are specified, `shape` is used.
      dtype: Placeholder type.
      sparse: Boolean, whether the placeholder should have a sparse type.
      name: Optional name string for the placeholder.
      ragged: Boolean, whether the placeholder should have a ragged type.
          In this case, values of 'None' in the 'shape' argument represent
          ragged dimensions. For more information about RaggedTensors, see this
          [guide](https://www.tensorflow.org/guide/ragged_tensors).

  Raises:
      ValueError: If called with eager execution
      ValueError: If called with sparse = True and ragged = True.

  Returns:
      Tensor instance (with Keras metadata included).

  Examples:
  ```python
  >>> from keras import backend as K
  >>> input_ph = K.placeholder(shape=(2, 4, 5))
  >>> input_ph
  <tf.Tensor 'Placeholder_4:0' shape=(2, 4, 5) dtype=float32>
  ```
  "
  [shape ndim dtype & {:keys [sparse name ragged]
                       :or {name None}} ]
    (py/call-attr-kw backend "placeholder" [shape ndim dtype] {:sparse sparse :name name :ragged ragged }))

(defn pool2d 
  "2D Pooling.

  Arguments:
      x: Tensor or variable.
      pool_size: tuple of 2 integers.
      strides: tuple of 2 integers.
      padding: string, `\"same\"` or `\"valid\"`.
      data_format: string, `\"channels_last\"` or `\"channels_first\"`.
      pool_mode: string, `\"max\"` or `\"avg\"`.

  Returns:
      A tensor, result of 2D pooling.

  Raises:
      ValueError: if `data_format` is neither `\"channels_last\"` or
      `\"channels_first\"`.
      ValueError: if `pool_size` is not a tuple of 2 integers.
      ValueError: if `strides` is not a tuple of 2 integers.
      ValueError: if `pool_mode` is neither `\"max\"` or `\"avg\"`.
  "
  [x pool_size & {:keys [strides padding data_format pool_mode]
                       :or {data_format None}} ]
    (py/call-attr-kw backend "pool2d" [x pool_size] {:strides strides :padding padding :data_format data_format :pool_mode pool_mode }))

(defn pool3d 
  "3D Pooling.

  Arguments:
      x: Tensor or variable.
      pool_size: tuple of 3 integers.
      strides: tuple of 3 integers.
      padding: string, `\"same\"` or `\"valid\"`.
      data_format: string, `\"channels_last\"` or `\"channels_first\"`.
      pool_mode: string, `\"max\"` or `\"avg\"`.

  Returns:
      A tensor, result of 3D pooling.

  Raises:
      ValueError: if `data_format` is neither `\"channels_last\"` or
      `\"channels_first\"`.
      ValueError: if `pool_mode` is neither `\"max\"` or `\"avg\"`.
  "
  [x pool_size & {:keys [strides padding data_format pool_mode]
                       :or {data_format None}} ]
    (py/call-attr-kw backend "pool3d" [x pool_size] {:strides strides :padding padding :data_format data_format :pool_mode pool_mode }))

(defn pow 
  "Element-wise exponentiation.

  Arguments:
      x: Tensor or variable.
      a: Python integer.

  Returns:
      A tensor.
  "
  [ x a ]
  (py/call-attr backend "pow"  x a ))
(defn print-tensor 
  "Prints `message` and the tensor value when evaluated.

  Note that `print_tensor` returns a new tensor identical to `x`
  which should be used in the following code. Otherwise the
  print operation is not taken into account during evaluation.

  Example:

  ```python
  >>> x = K.print_tensor(x, message=\"x is: \")
  ```

  Arguments:
      x: Tensor to print.
      message: Message to print jointly with the tensor.

  Returns:
      The same tensor `x`, unchanged.
  "
  [x  & {:keys [message]} ]
    (py/call-attr-kw backend "print_tensor" [x] {:message message }))
(defn prod 
  "Multiplies the values in a tensor, alongside the specified axis.

  Arguments:
      x: A tensor or variable.
      axis: An integer, the axis to compute the product.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`,
          the reduced dimension is retained with length 1.

  Returns:
      A tensor with the product of elements of `x`.
  "
  [x axis  & {:keys [keepdims]} ]
    (py/call-attr-kw backend "prod" [x axis] {:keepdims keepdims }))

(defn random-binomial 
  "Returns a tensor with random binomial distribution of values.

  The binomial distribution with parameters `n` and `p` is the probability
  distribution of the number of successful Bernoulli process. Only supports
  `n` = 1 for now.

  Arguments:
      shape: A tuple of integers, the shape of tensor to create.
      p: A float, `0. <= p <= 1`, probability of binomial distribution.
      dtype: String, dtype of returned tensor.
      seed: Integer, random seed.

  Returns:
      A tensor.
  "
  [shape & {:keys [p dtype seed]
                       :or {dtype None seed None}} ]
    (py/call-attr-kw backend "random_binomial" [shape] {:p p :dtype dtype :seed seed }))

(defn random-normal 
  "Returns a tensor with normal distribution of values.

  Arguments:
      shape: A tuple of integers, the shape of tensor to create.
      mean: A float, mean of the normal distribution to draw samples.
      stddev: A float, standard deviation of the normal distribution
          to draw samples.
      dtype: String, dtype of returned tensor.
      seed: Integer, random seed.

  Returns:
      A tensor.
  "
  [shape & {:keys [mean stddev dtype seed]
                       :or {dtype None seed None}} ]
    (py/call-attr-kw backend "random_normal" [shape] {:mean mean :stddev stddev :dtype dtype :seed seed }))

(defn random-normal-variable 
  "Instantiates a variable with values drawn from a normal distribution.

  Arguments:
      shape: Tuple of integers, shape of returned Keras variable.
      mean: Float, mean of the normal distribution.
      scale: Float, standard deviation of the normal distribution.
      dtype: String, dtype of returned Keras variable.
      name: String, name of returned Keras variable.
      seed: Integer, random seed.

  Returns:
      A Keras variable, filled with drawn samples.

  Example:
  ```python
  # TensorFlow example
  >>> kvar = K.random_normal_variable((2,3), 0, 1)
  >>> kvar
  <tensorflow.python.ops.variables.Variable object at 0x10ab12dd0>
  >>> K.eval(kvar)
  array([[ 1.19591331,  0.68685907, -0.63814116],
         [ 0.92629528,  0.28055015,  1.70484698]], dtype=float32)
  ```
  "
  [ shape mean scale dtype name seed ]
  (py/call-attr backend "random_normal_variable"  shape mean scale dtype name seed ))

(defn random-uniform 
  "Returns a tensor with uniform distribution of values.

  Arguments:
      shape: A tuple of integers, the shape of tensor to create.
      minval: A float, lower boundary of the uniform distribution
          to draw samples.
      maxval: A float, upper boundary of the uniform distribution
          to draw samples.
      dtype: String, dtype of returned tensor.
      seed: Integer, random seed.

  Returns:
      A tensor.
  "
  [shape & {:keys [minval maxval dtype seed]
                       :or {dtype None seed None}} ]
    (py/call-attr-kw backend "random_uniform" [shape] {:minval minval :maxval maxval :dtype dtype :seed seed }))

(defn random-uniform-variable 
  "Instantiates a variable with values drawn from a uniform distribution.

  Arguments:
      shape: Tuple of integers, shape of returned Keras variable.
      low: Float, lower boundary of the output interval.
      high: Float, upper boundary of the output interval.
      dtype: String, dtype of returned Keras variable.
      name: String, name of returned Keras variable.
      seed: Integer, random seed.

  Returns:
      A Keras variable, filled with drawn samples.

  Example:
  ```python
  # TensorFlow example
  >>> kvar = K.random_uniform_variable((2,3), 0, 1)
  >>> kvar
  <tensorflow.python.ops.variables.Variable object at 0x10ab40b10>
  >>> K.eval(kvar)
  array([[ 0.10940075,  0.10047495,  0.476143  ],
         [ 0.66137183,  0.00869417,  0.89220798]], dtype=float32)
  ```
  "
  [ shape low high dtype name seed ]
  (py/call-attr backend "random_uniform_variable"  shape low high dtype name seed ))

(defn relu 
  "Rectified linear unit.

  With default values, it returns element-wise `max(x, 0)`.

  Otherwise, it follows:
  `f(x) = max_value` for `x >= max_value`,
  `f(x) = x` for `threshold <= x < max_value`,
  `f(x) = alpha * (x - threshold)` otherwise.

  Arguments:
      x: A tensor or variable.
      alpha: A scalar, slope of negative section (default=`0.`).
      max_value: float. Saturation threshold.
      threshold: float. Threshold value for thresholded activation.

  Returns:
      A tensor.
  "
  [x & {:keys [alpha max_value threshold]
                       :or {max_value None}} ]
    (py/call-attr-kw backend "relu" [x] {:alpha alpha :max_value max_value :threshold threshold }))

(defn repeat 
  "Repeats a 2D tensor.

  if `x` has shape (samples, dim) and `n` is `2`,
  the output will have shape `(samples, 2, dim)`.

  Arguments:
      x: Tensor or variable.
      n: Python integer, number of times to repeat.

  Returns:
      A tensor.

  Example:
      ```python
      >>> b = tf.constant([[1, 2], [3, 4]])
      >>> b
      <tf.Tensor: id=78, shape=(2, 2), dtype=int32, numpy=
      array([[1, 2],
             [3, 4]], dtype=int32)>
      >>> tf.keras.backend.repeat(b, n=2)
      <tf.Tensor: id=82, shape=(2, 2, 2), dtype=int32, numpy=
      array([[[1, 2],
              [1, 2]],
             [[3, 4],
              [3, 4]]], dtype=int32)>
      ```
  "
  [ x n ]
  (py/call-attr backend "repeat"  x n ))

(defn repeat-elements 
  "Repeats the elements of a tensor along an axis, like `np.repeat`.

  If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
  will have shape `(s1, s2 * rep, s3)`.

  Arguments:
      x: Tensor or variable.
      rep: Python integer, number of times to repeat.
      axis: Axis along which to repeat.

  Returns:
      A tensor.

  Example:
      ```python
      >>> b = tf.constant([1, 2, 3])
      >>> tf.keras.backend.repeat_elements(b, rep=2, axis=0)
      <tf.Tensor: id=70, shape=(6,), dtype=int32,
          numpy=array([1, 1, 2, 2, 3, 3], dtype=int32)>
      ```
  "
  [ x rep axis ]
  (py/call-attr backend "repeat_elements"  x rep axis ))

(defn reset-uids 
  "Resets graph identifiers.
  "
  [  ]
  (py/call-attr backend "reset_uids"  ))

(defn reshape 
  "Reshapes a tensor to the specified shape.

  Arguments:
      x: Tensor or variable.
      shape: Target shape tuple.

  Returns:
      A tensor.

  Example:
    ```python
    >>> a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    >>> a
    <tf.Tensor: id=32, shape=(4, 3), dtype=int32, numpy=
    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9],
           [10, 11, 12]], dtype=int32)>
    >>> tf.keras.backend.reshape(a, shape=(2, 6))
    <tf.Tensor: id=35, shape=(2, 6), dtype=int32, numpy=
    array([[ 1,  2,  3,  4,  5,  6],
           [ 7,  8,  9, 10, 11, 12]], dtype=int32)>
    ```
  "
  [ x shape ]
  (py/call-attr backend "reshape"  x shape ))
(defn resize-images 
  "Resizes the images contained in a 4D tensor.

  Arguments:
      x: Tensor or variable to resize.
      height_factor: Positive integer.
      width_factor: Positive integer.
      data_format: One of `\"channels_first\"`, `\"channels_last\"`.
      interpolation: A string, one of `nearest` or `bilinear`.

  Returns:
      A tensor.

  Raises:
      ValueError: in case of incorrect value for
        `data_format` or `interpolation`.
  "
  [x height_factor width_factor data_format  & {:keys [interpolation]} ]
    (py/call-attr-kw backend "resize_images" [x height_factor width_factor data_format] {:interpolation interpolation }))

(defn resize-volumes 
  "Resizes the volume contained in a 5D tensor.

  Arguments:
      x: Tensor or variable to resize.
      depth_factor: Positive integer.
      height_factor: Positive integer.
      width_factor: Positive integer.
      data_format: One of `\"channels_first\"`, `\"channels_last\"`.

  Returns:
      A tensor.

  Raises:
      ValueError: if `data_format` is neither
          `channels_last` or `channels_first`.
  "
  [ x depth_factor height_factor width_factor data_format ]
  (py/call-attr backend "resize_volumes"  x depth_factor height_factor width_factor data_format ))

(defn reverse 
  "Reverse a tensor along the specified axes.

  Arguments:
      x: Tensor to reverse.
      axes: Integer or iterable of integers.
          Axes to reverse.

  Returns:
      A tensor.
  "
  [ x axes ]
  (py/call-attr backend "reverse"  x axes ))

(defn rnn 
  "Iterates over the time dimension of a tensor.

  Arguments:
      step_function: RNN step function.
          Args;
              input; Tensor with shape `(samples, ...)` (no time dimension),
                  representing input for the batch of samples at a certain
                  time step.
              states; List of tensors.
          Returns;
              output; Tensor with shape `(samples, output_dim)`
                  (no time dimension).
              new_states; List of tensors, same length and shapes
                  as 'states'. The first state in the list must be the
                  output tensor at the previous timestep.
      inputs: Tensor of temporal data of shape `(samples, time, ...)`
          (at least 3D), or nested tensors, and each of which has shape
          `(samples, time, ...)`.
      initial_states: Tensor with shape `(samples, state_size)`
          (no time dimension), containing the initial values for the states used
          in the step function. In the case that state_size is in a nested
          shape, the shape of initial_states will also follow the nested
          structure.
      go_backwards: Boolean. If True, do the iteration over the time
          dimension in reverse order and return the reversed sequence.
      mask: Binary tensor with shape `(samples, time, 1)`,
          with a zero for every element that is masked.
      constants: List of constant values passed at each step.
      unroll: Whether to unroll the RNN or to use a symbolic `while_loop`.
      input_length: If specified, assume time dimension is of this length.
      time_major: Boolean. If true, the inputs and outputs will be in shape
          `(timesteps, batch, ...)`, whereas in the False case, it will be
          `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
          efficient because it avoids transposes at the beginning and end of the
          RNN calculation. However, most TensorFlow data is batch-major, so by
          default this function accepts input and emits output in batch-major
          form.
      zero_output_for_mask: Boolean. If True, the output for masked timestep
          will be zeros, whereas in the False case, output from previous
          timestep is returned.
  Returns:
      A tuple, `(last_output, outputs, new_states)`.
          last_output: the latest output of the rnn, of shape `(samples, ...)`
          outputs: tensor with shape `(samples, time, ...)` where each
              entry `outputs[s, t]` is the output of the step function
              at time `t` for sample `s`.
          new_states: list of tensors, latest states returned by
              the step function, of shape `(samples, ...)`.

  Raises:
      ValueError: if input dimension is less than 3.
      ValueError: if `unroll` is `True` but input timestep is not a fixed
      number.
      ValueError: if `mask` is provided (not `None`) but states is not provided
          (`len(states)` == 0).
  "
  [step_function inputs initial_states & {:keys [go_backwards mask constants unroll input_length time_major zero_output_for_mask]
                       :or {mask None constants None input_length None}} ]
    (py/call-attr-kw backend "rnn" [step_function inputs initial_states] {:go_backwards go_backwards :mask mask :constants constants :unroll unroll :input_length input_length :time_major time_major :zero_output_for_mask zero_output_for_mask }))

(defn round 
  "Element-wise rounding to the closest integer.

  In case of tie, the rounding mode used is \"half to even\".

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  "
  [ x ]
  (py/call-attr backend "round"  x ))

(defn separable-conv2d 
  "2D convolution with separable filters.

  Arguments:
      x: input tensor
      depthwise_kernel: convolution kernel for the depthwise convolution.
      pointwise_kernel: kernel for the 1x1 convolution.
      strides: strides tuple (length 2).
      padding: string, `\"same\"` or `\"valid\"`.
      data_format: string, `\"channels_last\"` or `\"channels_first\"`.
      dilation_rate: tuple of integers,
          dilation rates for the separable convolution.

  Returns:
      Output tensor.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
      ValueError: if `strides` is not a tuple of 2 integers.
  "
  [x depthwise_kernel pointwise_kernel & {:keys [strides padding data_format dilation_rate]
                       :or {data_format None}} ]
    (py/call-attr-kw backend "separable_conv2d" [x depthwise_kernel pointwise_kernel] {:strides strides :padding padding :data_format data_format :dilation_rate dilation_rate }))

(defn set-epsilon 
  "Sets the value of the fuzz factor used in numeric expressions.

  Arguments:
      value: float. New value of epsilon.
  Example: ```python from keras import backend as K K.epsilon() >>> 1e-07
    K.set_epsilon(1e-05) K.epsilon() >>> 1e-05 ```
  "
  [ value ]
  (py/call-attr backend "set_epsilon"  value ))

(defn set-floatx 
  "Sets the default float type.

  Arguments:
      value: String; 'float16', 'float32', or 'float64'.
  Example: ```python from keras import backend as K K.floatx() >>> 'float32'
    K.set_floatx('float16') K.floatx() >>> 'float16' ```

  Raises:
      ValueError: In case of invalid value.
  "
  [ value ]
  (py/call-attr backend "set_floatx"  value ))

(defn set-image-data-format 
  "Sets the value of the image data format convention.

  Arguments:
      data_format: string. `'channels_first'` or `'channels_last'`.
  Example: ```python from keras import backend as K K.image_data_format() >>>
    'channels_first' K.set_image_data_format('channels_last')
    K.image_data_format() >>> 'channels_last' ```

  Raises:
      ValueError: In case of invalid `data_format` value.
  "
  [ data_format ]
  (py/call-attr backend "set_image_data_format"  data_format ))

(defn set-learning-phase 
  "Sets the learning phase to a fixed value.

  Arguments:
      value: Learning phase value, either 0 or 1 (integers).

  Raises:
      ValueError: if `value` is neither `0` nor `1`.
  "
  [ value ]
  (py/call-attr backend "set_learning_phase"  value ))

(defn set-session 
  "Sets the global TensorFlow session.

  Arguments:
      session: A TF Session.
  "
  [ session ]
  (py/call-attr backend "set_session"  session ))

(defn set-value 
  "Sets the value of a variable, from a Numpy array.

  Arguments:
      x: Tensor to set to a new value.
      value: Value to set the tensor to, as a Numpy array
          (of the same shape).
  "
  [ x value ]
  (py/call-attr backend "set_value"  x value ))

(defn shape 
  "Returns the symbolic shape of a tensor or variable.

  Arguments:
      x: A tensor or variable.

  Returns:
      A symbolic shape (which is itself a tensor).

  Examples:

  ```python
  # TensorFlow example
  >>> from keras import backend as K
  >>> tf_session = K.get_session()
  >>> val = np.array([[1, 2], [3, 4]])
  >>> kvar = K.variable(value=val)
  >>> input = keras.backend.placeholder(shape=(2, 4, 5))
  >>> K.shape(kvar)
  <tf.Tensor 'Shape_8:0' shape=(2,) dtype=int32>
  >>> K.shape(input)
  <tf.Tensor 'Shape_9:0' shape=(3,) dtype=int32>
  # To get integer shape (Instead, you can use K.int_shape(x))
  >>> K.shape(kvar).eval(session=tf_session)
  array([2, 2], dtype=int32)
  >>> K.shape(input).eval(session=tf_session)
  array([2, 4, 5], dtype=int32)
  ```
  "
  [ x ]
  (py/call-attr backend "shape"  x ))

(defn sigmoid 
  "Element-wise sigmoid.

  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor.
  "
  [ x ]
  (py/call-attr backend "sigmoid"  x ))

(defn sign 
  "Element-wise sign.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  "
  [ x ]
  (py/call-attr backend "sign"  x ))

(defn sin 
  "Computes sin of x element-wise.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  "
  [ x ]
  (py/call-attr backend "sin"  x ))
(defn softmax 
  "Softmax of a tensor.

  Arguments:
      x: A tensor or variable.
      axis: The dimension softmax would be performed on.
          The default is -1 which indicates the last dimension.

  Returns:
      A tensor.
  "
  [x  & {:keys [axis]} ]
    (py/call-attr-kw backend "softmax" [x] {:axis axis }))

(defn softplus 
  "Softplus of a tensor.

  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor.
  "
  [ x ]
  (py/call-attr backend "softplus"  x ))

(defn softsign 
  "Softsign of a tensor.

  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor.
  "
  [ x ]
  (py/call-attr backend "softsign"  x ))
(defn sparse-categorical-crossentropy 
  "Categorical crossentropy with integer targets.

  Arguments:
      target: An integer tensor.
      output: A tensor resulting from a softmax
          (unless `from_logits` is True, in which
          case `output` is expected to be the logits).
      from_logits: Boolean, whether `output` is the
          result of a softmax, or is a tensor of logits.
      axis: Int specifying the channels axis. `axis=-1` corresponds to data
          format `channels_last', and `axis=1` corresponds to data format
          `channels_first`.

  Returns:
      Output tensor.

  Raises:
      ValueError: if `axis` is neither -1 nor one of the axes of `output`.
  "
  [target output  & {:keys [from_logits axis]} ]
    (py/call-attr-kw backend "sparse_categorical_crossentropy" [target output] {:from_logits from_logits :axis axis }))

(defn spatial-2d-padding 
  "Pads the 2nd and 3rd dimensions of a 4D tensor.

  Arguments:
      x: Tensor or variable.
      padding: Tuple of 2 tuples, padding pattern.
      data_format: One of `channels_last` or `channels_first`.

  Returns:
      A padded 4D tensor.

  Raises:
      ValueError: if `data_format` is neither
          `channels_last` or `channels_first`.
  "
  [x & {:keys [padding data_format]
                       :or {data_format None}} ]
    (py/call-attr-kw backend "spatial_2d_padding" [x] {:padding padding :data_format data_format }))

(defn spatial-3d-padding 
  "Pads 5D tensor with zeros along the depth, height, width dimensions.

  Pads these dimensions with respectively
  \"padding[0]\", \"padding[1]\" and \"padding[2]\" zeros left and right.

  For 'channels_last' data_format,
  the 2nd, 3rd and 4th dimension will be padded.
  For 'channels_first' data_format,
  the 3rd, 4th and 5th dimension will be padded.

  Arguments:
      x: Tensor or variable.
      padding: Tuple of 3 tuples, padding pattern.
      data_format: One of `channels_last` or `channels_first`.

  Returns:
      A padded 5D tensor.

  Raises:
      ValueError: if `data_format` is neither
          `channels_last` or `channels_first`.

  "
  [x & {:keys [padding data_format]
                       :or {data_format None}} ]
    (py/call-attr-kw backend "spatial_3d_padding" [x] {:padding padding :data_format data_format }))

(defn sqrt 
  "Element-wise square root.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  "
  [ x ]
  (py/call-attr backend "sqrt"  x ))

(defn square 
  "Element-wise square.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.
  "
  [ x ]
  (py/call-attr backend "square"  x ))

(defn squeeze 
  "Removes a 1-dimension from the tensor at index \"axis\".

  Arguments:
      x: A tensor or variable.
      axis: Axis to drop.

  Returns:
      A tensor with the same data as `x` but reduced dimensions.
  "
  [ x axis ]
  (py/call-attr backend "squeeze"  x axis ))
(defn stack 
  "Stacks a list of rank `R` tensors into a rank `R+1` tensor.

  Arguments:
      x: List of tensors.
      axis: Axis along which to perform stacking.

  Returns:
      A tensor.

  Example:
      ```python
      >>> a = tf.constant([[1, 2],[3, 4]])
      >>> b = tf.constant([[10, 20],[30, 40]])
      >>> tf.keras.backend.stack((a, b))
      <tf.Tensor: id=146, shape=(2, 2, 2), dtype=int32, numpy=
      array([[[ 1,  2],
              [ 3,  4]],
             [[10, 20],
              [30, 40]]], dtype=int32)>
      ```
  "
  [x  & {:keys [axis]} ]
    (py/call-attr-kw backend "stack" [x] {:axis axis }))
(defn std 
  "Standard deviation of a tensor, alongside the specified axis.

  Arguments:
      x: A tensor or variable.
      axis: An integer, the axis to compute the standard deviation.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`,
          the reduced dimension is retained with length 1.

  Returns:
      A tensor with the standard deviation of elements of `x`.
  "
  [x axis  & {:keys [keepdims]} ]
    (py/call-attr-kw backend "std" [x axis] {:keepdims keepdims }))

(defn stop-gradient 
  "Returns `variables` but with zero gradient w.r.t. every other variable.

  Arguments:
      variables: Tensor or list of tensors to consider constant with respect
        to any other variable.


  Returns:
      A single tensor or a list of tensors (depending on the passed argument)
      that has no gradient with respect to any other variable.
  "
  [ variables ]
  (py/call-attr backend "stop_gradient"  variables ))
(defn sum 
  "Sum of the values in a tensor, alongside the specified axis.

  Arguments:
      x: A tensor or variable.
      axis: An integer, the axis to sum over.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`,
          the reduced dimension is retained with length 1.

  Returns:
      A tensor with sum of `x`.
  "
  [x axis  & {:keys [keepdims]} ]
    (py/call-attr-kw backend "sum" [x axis] {:keepdims keepdims }))

(defn switch 
  "Switches between two operations depending on a scalar value.

  Note that both `then_expression` and `else_expression`
  should be symbolic tensors of the *same shape*.

  Arguments:
      condition: tensor (`int` or `bool`).
      then_expression: either a tensor, or a callable that returns a tensor.
      else_expression: either a tensor, or a callable that returns a tensor.

  Returns:
      The selected tensor.

  Raises:
      ValueError: If rank of `condition` is greater than rank of expressions.
  "
  [ condition then_expression else_expression ]
  (py/call-attr backend "switch"  condition then_expression else_expression ))

(defn tanh 
  "Element-wise tanh.

  Arguments:
      x: A tensor or variable.

  Returns:
      A tensor.
  "
  [ x ]
  (py/call-attr backend "tanh"  x ))
(defn temporal-padding 
  "Pads the middle dimension of a 3D tensor.

  Arguments:
      x: Tensor or variable.
      padding: Tuple of 2 integers, how many zeros to
          add at the start and end of dim 1.

  Returns:
      A padded 3D tensor.
  "
  [x  & {:keys [padding]} ]
    (py/call-attr-kw backend "temporal_padding" [x] {:padding padding }))

(defn to-dense 
  "Converts a sparse tensor into a dense tensor and returns it.

  Arguments:
      tensor: A tensor instance (potentially sparse).

  Returns:
      A dense tensor.

  Examples:
  ```python
  >>> from keras import backend as K
  >>> b = K.placeholder((2, 2), sparse=True)
  >>> print(K.is_sparse(b))
  True
  >>> c = K.to_dense(b)
  >>> print(K.is_sparse(c))
  False
  ```
  "
  [ tensor ]
  (py/call-attr backend "to_dense"  tensor ))

(defn transpose 
  "Transposes a tensor and returns it.

  Arguments:
      x: Tensor or variable.

  Returns:
      A tensor.

  Examples:
  ```python
  >>> var = K.variable([[1, 2, 3], [4, 5, 6]])
  >>> K.eval(var)
  array([[ 1.,  2.,  3.],
         [ 4.,  5.,  6.]], dtype=float32)
  >>> var_transposed = K.transpose(var)
  >>> K.eval(var_transposed)
  array([[ 1.,  4.],
         [ 2.,  5.],
         [ 3.,  6.]], dtype=float32)
  ```

  ```python
  >>> input = K.placeholder((2, 3))
  >>> input
  <tf.Tensor 'Placeholder_11:0' shape=(2, 3) dtype=float32>
  >>> input_transposed = K.transpose(input)
  >>> input_transposed
  <tf.Tensor 'transpose_4:0' shape=(3, 2) dtype=float32>

  ```
  "
  [ x ]
  (py/call-attr backend "transpose"  x ))

(defn truncated-normal 
  "Returns a tensor with truncated random normal distribution of values.

  The generated values follow a normal distribution
  with specified mean and standard deviation,
  except that values whose magnitude is more than
  two standard deviations from the mean are dropped and re-picked.

  Arguments:
      shape: A tuple of integers, the shape of tensor to create.
      mean: Mean of the values.
      stddev: Standard deviation of the values.
      dtype: String, dtype of returned tensor.
      seed: Integer, random seed.

  Returns:
      A tensor.
  "
  [shape & {:keys [mean stddev dtype seed]
                       :or {dtype None seed None}} ]
    (py/call-attr-kw backend "truncated_normal" [shape] {:mean mean :stddev stddev :dtype dtype :seed seed }))

(defn update 
  ""
  [ x new_x ]
  (py/call-attr backend "update"  x new_x ))

(defn update-add 
  "Update the value of `x` by adding `increment`.

  Arguments:
      x: A Variable.
      increment: A tensor of same shape as `x`.

  Returns:
      The variable `x` updated.
  "
  [ x increment ]
  (py/call-attr backend "update_add"  x increment ))

(defn update-sub 
  "Update the value of `x` by subtracting `decrement`.

  Arguments:
      x: A Variable.
      decrement: A tensor of same shape as `x`.

  Returns:
      The variable `x` updated.
  "
  [ x decrement ]
  (py/call-attr backend "update_sub"  x decrement ))
(defn var 
  "Variance of a tensor, alongside the specified axis.

  Arguments:
      x: A tensor or variable.
      axis: An integer, the axis to compute the variance.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`,
          the reduced dimension is retained with length 1.

  Returns:
      A tensor with the variance of elements of `x`.
  "
  [x axis  & {:keys [keepdims]} ]
    (py/call-attr-kw backend "var" [x axis] {:keepdims keepdims }))

(defn variable 
  "Instantiates a variable and returns it.

  Arguments:
      value: Numpy array, initial value of the tensor.
      dtype: Tensor type.
      name: Optional name string for the tensor.
      constraint: Optional projection function to be
          applied to the variable after an optimizer update.

  Returns:
      A variable instance (with Keras metadata included).

  Examples:
  ```python
  >>> import numpy as np
  >>> from keras import backend as K
  >>> val = np.array([[1, 2], [3, 4]])
  >>> kvar = K.variable(value=val, dtype='float64', name='example_var')
  >>> K.dtype(kvar)
  'float64'
  >>> print(kvar)
  example_var
  >>> kvar.eval()
  array([[ 1.,  2.],
         [ 3.,  4.]])
  ```
  "
  [ value dtype name constraint ]
  (py/call-attr backend "variable"  value dtype name constraint ))

(defn zeros 
  "Instantiates an all-zeros variable and returns it.

  Arguments:
      shape: Tuple or list of integers, shape of returned Keras variable
      dtype: data type of returned Keras variable
      name: name of returned Keras variable

  Returns:
      A variable (including Keras metadata), filled with `0.0`.
      Note that if `shape` was symbolic, we cannot return a variable,
      and will return a dynamically-shaped tensor instead.

  Example:

  ```python
  from tensorflow.keras import backend as K
  kvar = K.zeros((3,4))
  K.eval(kvar)
  # array([[ 0.,  0.,  0.,  0.], [ 0.,  0.,  0.,  0.],
  #       [ 0.,  0.,  0.,  0.]], dtype=float32)
  A = tf.constant([1,2,3])
  kvar2 = K.zeros(A.shape) # [0., 0., 0.] float32 by default
  kvar3 = K.zeros(A.shape,dtype=tf.int32) # [0, 0, 0] with int32 dtype
  kvar4 = K.zeros([2,3]) # [[0., 0., 0.], [0., 0., 0.]]
  ```

  "
  [ shape dtype name ]
  (py/call-attr backend "zeros"  shape dtype name ))

(defn zeros-like 
  "Instantiates an all-zeros variable of the same shape as another tensor.

  Arguments:
      x: Keras variable or Keras tensor.
      dtype: dtype of returned Keras variable.
             `None` uses the dtype of `x`.
      name: name for the variable to create.

  Returns:
      A Keras variable with the shape of `x` filled with zeros.

  Example:

  ```python
  from tensorflow.keras import backend as K
  kvar = K.variable(np.random.random((2,3)))
  kvar_zeros = K.zeros_like(kvar)
  K.eval(kvar_zeros)
  # array([[ 0.,  0.,  0.], [ 0.,  0.,  0.]], dtype=float32)
  ```

  "
  [ x dtype name ]
  (py/call-attr backend "zeros_like"  x dtype name ))
