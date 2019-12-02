(ns tensorflow.contrib.learn.python.learn.models
  "Various high level TF models (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce models (import-module "tensorflow.contrib.learn.python.learn.models"))

(defn bidirectional-rnn 
  "Creates a bidirectional recurrent neural network. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please consider `tf.nn.bidirectional_dynamic_rnn`.

Similar to the unidirectional case (rnn) but takes input and builds
independent forward and backward RNNs with the final forward and backward
outputs depth-concatenated, such that the output will have the format
[time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
forward and backward cell must match. The initial state for both directions
is zero by default (but can be set optionally) and no intermediate states
are ever returned -- the network is fully unrolled for the given (passed in)
length(s) of the sequence(s) or completely unrolled if length(s) is not
given.
Args:
  cell_fw: An instance of RNNCell, to be used for forward direction.
  cell_bw: An instance of RNNCell, to be used for backward direction.
  inputs: A length T list of inputs, each a tensor of shape
    [batch_size, cell.input_size].
  initial_state_fw: (optional) An initial state for the forward RNN.
    This must be a tensor of appropriate type and shape
    [batch_size x cell.state_size].
  initial_state_bw: (optional) Same as for initial_state_fw.
  dtype: (optional) The data type for the initial state.  Required if
    either of the initial states are not provided.
  sequence_length: (optional) An int64 vector (tensor) of size
    [batch_size],
    containing the actual lengths for each of the sequences.
  scope: VariableScope for the created subgraph; defaults to \"BiRNN\"

Returns:
  A pair (outputs, state) where:
    outputs is a length T list of outputs (one for each input), which
    are depth-concatenated forward and backward outputs
    state is the concatenated final state of the forward and backward RNN

Raises:
  TypeError: If \"cell_fw\" or \"cell_bw\" is not an instance of RNNCell.
  ValueError: If inputs is None or an empty list."
  [ cell_fw cell_bw inputs initial_state_fw initial_state_bw dtype sequence_length scope ]
  (py/call-attr models "bidirectional_rnn"  cell_fw cell_bw inputs initial_state_fw initial_state_bw dtype sequence_length scope ))
(defn deprecated 
  "Decorator for marking functions or methods deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called. It has the following format:

    <function> (from <module>) is deprecated and will be removed after <date>.
    Instructions for updating:
    <instructions>

  If `date` is None, 'after <date>' is replaced with 'in a future version'.
  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated)' is appended
  to the first line of the docstring and a deprecation notice is prepended
  to the rest of the docstring.

  Args:
    date: String or None. The date the function is scheduled to be removed.
      Must be ISO 8601 (YYYY-MM-DD), or None.
    instructions: String. Instructions on how to update code using the
      deprecated function.
    warn_once: Boolean. Set to `True` to warn only the first time the decorated
      function is called. Otherwise, every call will log a warning.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not None or in ISO 8601 format, or instructions are
      empty.
  "
  [date instructions  & {:keys [warn_once]} ]
    (py/call-attr-kw models "deprecated" [date instructions] {:warn_once warn_once }))

(defn get-rnn-model 
  "Returns a function that creates a RNN TensorFlow subgraph. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please consider tensorflow/tensor2tensor.

Args:
  rnn_size: The size for rnn cell, e.g. size of your word embeddings.
  cell_type: The type of rnn cell, including rnn, gru, and lstm.
  num_layers: The number of layers of the rnn model.
  input_op_fn: Function that will transform the input tensor, such as
               creating word embeddings, byte list, etc. This takes
               an argument `x` for input and returns transformed `x`.
  bidirectional: boolean, Whether this is a bidirectional rnn.
  target_predictor_fn: Function that will predict target from input
                       features. This can be logistic regression,
                       linear regression or any other model,
                       that takes `x`, `y` and returns predictions and loss
                       tensors.
  sequence_length: If sequence_length is provided, dynamic calculation is
    performed. This saves computational time when unrolling past max sequence
    length. Required for bidirectional RNNs.
  initial_state: An initial state for the RNN. This must be a tensor of
    appropriate type and shape [batch_size x cell.state_size].
  attn_length: integer, the size of attention vector attached to rnn cells.
  attn_size: integer, the size of an attention window attached to rnn cells.
  attn_vec_size: integer, the number of convolutional features calculated on
    attention state and the size of the hidden layer built from base cell
    state.

Returns:
  A function that creates the subgraph."
  [ rnn_size cell_type num_layers input_op_fn bidirectional target_predictor_fn sequence_length initial_state attn_length attn_size attn_vec_size ]
  (py/call-attr models "get_rnn_model"  rnn_size cell_type num_layers input_op_fn bidirectional target_predictor_fn sequence_length initial_state attn_length attn_size attn_vec_size ))
(defn linear-regression 
  "Creates linear regression TensorFlow subgraph. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Consider using a class from tf.estimator.

Args:
  x: tensor or placeholder for input features.
  y: tensor or placeholder for labels.
  init_mean: the mean value to use for initialization.
  init_stddev: the standard deviation to use for initialization.

Returns:
  Predictions and loss tensors.

Side effects:
  The variables linear_regression.weights and linear_regression.bias are
  initialized as follows.  If init_mean is not None, then initialization
  will be done using a random normal initializer with the given init_mean
  and init_stddv.  (These may be set to 0.0 each if a zero initialization
  is desirable for convex use cases.)  If init_mean is None, then the
  uniform_unit_scaling_initialzer will be used."
  [x y init_mean  & {:keys [init_stddev]} ]
    (py/call-attr-kw models "linear_regression" [x y init_mean] {:init_stddev init_stddev }))

(defn linear-regression-zero-init 
  "Linear regression subgraph with zero-value initial weights and bias. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Consider using a tf.estimator.LinearRegressor

Args:
  x: tensor or placeholder for input features.
  y: tensor or placeholder for labels.

Returns:
  Predictions and loss tensors."
  [ x y ]
  (py/call-attr models "linear_regression_zero_init"  x y ))
(defn logistic-regression 
  "Creates logistic regression TensorFlow subgraph. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Consider using a class from tf.estimator.

Args:
  x: tensor or placeholder for input features,
     shape should be [batch_size, n_features].
  y: tensor or placeholder for labels (one-hot),
     shape should be [batch_size, n_classes].
  class_weight: tensor, [n_classes], where for each class
                it has weight of the class. If not provided
                will check if graph contains tensor `class_weight:0`.
                If that is not provided either all ones are used.
  init_mean: the mean value to use for initialization.
  init_stddev: the standard deviation to use for initialization.

Returns:
  Predictions and loss tensors.

Side effects:
  The variables linear_regression.weights and linear_regression.bias are
  initialized as follows.  If init_mean is not None, then initialization
  will be done using a random normal initializer with the given init_mean
  and init_stddv.  (These may be set to 0.0 each if a zero initialization
  is desirable for convex use cases.)  If init_mean is None, then the
  uniform_unit_scaling_initialzer will be used."
  [x y class_weight init_mean  & {:keys [init_stddev]} ]
    (py/call-attr-kw models "logistic_regression" [x y class_weight init_mean] {:init_stddev init_stddev }))

(defn logistic-regression-zero-init 
  "Logistic regression subgraph with zero-value initial weights and bias. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Consider using a class from tf.estimator.LinearClassifier

Args:
  x: tensor or placeholder for input features.
  y: tensor or placeholder for labels.

Returns:
  Predictions and loss tensors."
  [ x y ]
  (py/call-attr models "logistic_regression_zero_init"  x y ))
