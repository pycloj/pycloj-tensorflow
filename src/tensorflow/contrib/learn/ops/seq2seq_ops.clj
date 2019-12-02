(ns tensorflow.contrib.learn.python.learn.ops.seq2seq-ops
  "TensorFlow Ops for Sequence to Sequence models (deprecated).

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
(defonce seq2seq-ops (import-module "tensorflow.contrib.learn.python.learn.ops.seq2seq_ops"))
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
    (py/call-attr-kw seq2seq-ops "deprecated" [date instructions] {:warn_once warn_once }))

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
  (py/call-attr seq2seq-ops "rnn_decoder"  decoder_inputs initial_state cell scope ))

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
    (py/call-attr-kw seq2seq-ops "rnn_seq2seq" [encoder_inputs decoder_inputs encoder_cell decoder_cell] {:dtype dtype :scope scope }))

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
  (py/call-attr seq2seq-ops "seq2seq_inputs"  x y input_length output_length sentinel name ))

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
  (py/call-attr seq2seq-ops "sequence_classifier"  decoding labels sampling_decoding name ))
