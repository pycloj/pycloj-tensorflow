(ns tensorflow.contrib.learn.python.learn.ops
  "Various TensorFlow Ops (deprecated).

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
(defonce ops (import-module "tensorflow.contrib.learn.python.learn.ops"))

(defn categorical-variable 
  "Creates an embedding for categorical variable with given number of classes. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-01.
Instructions for updating:
Use `tf.contrib.layers.embed_sequence` instead.

Args:
  tensor_in: Input tensor with class identifier (can be batch or
    N-dimensional).
  n_classes: Number of classes.
  embedding_size: Size of embedding vector to represent each class.
  name: Name of this categorical variable.
Returns:
  Tensor of input shape, with additional dimension for embedding.

Example:
  Calling categorical_variable([1, 2], 5, 10, \"my_cat\"), will return 2 x 10
  tensor, where each row is representation of the class."
  [ tensor_in n_classes embedding_size name ]
  (py/call-attr ops "categorical_variable"  tensor_in n_classes embedding_size name ))
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
    (py/call-attr-kw ops "deprecated" [date instructions] {:warn_once warn_once }))
(defn embedding-lookup 
  "Provides a N dimensional version of tf.embedding_lookup. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-01.
Instructions for updating:
Use `tf.embedding_lookup` instead.

Ids are flattened to a 1d tensor before being passed to embedding_lookup
then, they are unflattend to match the original ids shape plus an extra
leading dimension of the size of the embeddings.

Args:
  params: List of tensors of size D0 x D1 x ... x Dn-2 x Dn-1.
  ids: N-dimensional tensor of B0 x B1 x .. x Bn-2 x Bn-1.
    Must contain indexes into params.
  name: Optional name for the op.

Returns:
  A tensor of size B0 x B1 x .. x Bn-2 x Bn-1 x D1 x ... x Dn-2 x Dn-1
  containing the values from the params tensor(s) for indecies in ids.

Raises:
  ValueError: if some parameters are invalid."
  [params ids  & {:keys [name]} ]
    (py/call-attr-kw ops "embedding_lookup" [params ids] {:name name }))

(defn mean-squared-error-regressor 
  "Returns prediction and loss for mean squared error regression. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-01.
Instructions for updating:
Use `tf.losses.mean_squared_error` and explicit logits computation."
  [ tensor_in labels weights biases name ]
  (py/call-attr ops "mean_squared_error_regressor"  tensor_in labels weights biases name ))

(defn rnn-decoder 
  "RNN Decoder that creates training and sampling sub-graphs. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use tf.nn/tf.layers directly.

Args:
  decoder_inputs: Inputs for decoder, list of tensors.
    This is used only in training sub-graph.
  initial_state: Initial state for the decoder.
  cell: RNN cell to use for decoder.
  scope: Scope to use, if None new will be produced.

Returns:
  List of tensors for outputs and states for training and sampling sub-graphs."
  [ decoder_inputs initial_state cell scope ]
  (py/call-attr ops "rnn_decoder"  decoder_inputs initial_state cell scope ))

(defn rnn-seq2seq 
  "RNN Sequence to Sequence model. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use tf.nn/tf.layers directly.

Args:
  encoder_inputs: List of tensors, inputs for encoder.
  decoder_inputs: List of tensors, inputs for decoder.
  encoder_cell: RNN cell to use for encoder.
  decoder_cell: RNN cell to use for decoder, if None encoder_cell is used.
  dtype: Type to initialize encoder state with.
  scope: Scope to use, if None new will be produced.

Returns:
  List of tensors for outputs and states for training and sampling sub-graphs."
  [encoder_inputs decoder_inputs encoder_cell decoder_cell & {:keys [dtype scope]
                       :or {scope None}} ]
    (py/call-attr-kw ops "rnn_seq2seq" [encoder_inputs decoder_inputs encoder_cell decoder_cell] {:dtype dtype :scope scope }))

(defn seq2seq-inputs 
  "Processes inputs for Sequence to Sequence models. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use tf.nn/tf.layers directly.

Args:
  x: Input Tensor [batch_size, input_length, embed_dim].
  y: Output Tensor [batch_size, output_length, embed_dim].
  input_length: length of input x.
  output_length: length of output y.
  sentinel: optional first input to decoder and final output expected.
    If sentinel is not provided, zeros are used. Due to fact that y is not
    available in sampling time, shape of sentinel will be inferred from x.
  name: Operation name.

Returns:
  Encoder input from x, and decoder inputs and outputs from y."
  [ x y input_length output_length sentinel name ]
  (py/call-attr ops "seq2seq_inputs"  x y input_length output_length sentinel name ))

(defn sequence-classifier 
  "Returns predictions and loss for sequence of predictions. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use tf.nn/tf.layers directly.

Args:
  decoding: List of Tensors with predictions.
  labels: List of Tensors with labels.
  sampling_decoding: Optional, List of Tensor with predictions to be used
    in sampling. E.g. they shouldn't have dependncy on outputs.
    If not provided, decoding is used.
  name: Operation name.

Returns:
  Predictions and losses tensors."
  [ decoding labels sampling_decoding name ]
  (py/call-attr ops "sequence_classifier"  decoding labels sampling_decoding name ))

(defn softmax-classifier 
  "Returns prediction and loss for softmax classifier. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2016-12-01.
Instructions for updating:
Use `tf.losses.softmax_cross_entropy` and explicit logits computation.

This function returns \"probabilities\" and a cross entropy loss. To obtain
predictions, use `tf.argmax` on the returned probabilities.

This function requires labels to be passed in one-hot encoding.

Args:
  tensor_in: Input tensor, [batch_size, feature_size], features.
  labels: Tensor, [batch_size, n_classes], one-hot labels of the output
    classes.
  weights: Tensor, [batch_size, feature_size], linear transformation
    matrix.
  biases: Tensor, [batch_size], biases.
  class_weight: Tensor, optional, [n_classes], weight for each class.
    If not given, all classes are supposed to have weight one.
  name: Operation name.

Returns:
  `tuple` of softmax predictions and loss `Tensor`s."
  [ tensor_in labels weights biases class_weight name ]
  (py/call-attr ops "softmax_classifier"  tensor_in labels weights biases class_weight name ))
