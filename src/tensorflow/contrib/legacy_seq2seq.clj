(ns tensorflow.contrib.legacy-seq2seq
  "Deprecated library for creating sequence-to-sequence models in TensorFlow.

@@attention_decoder
@@basic_rnn_seq2seq
@@embedding_attention_decoder
@@embedding_attention_seq2seq
@@embedding_rnn_decoder
@@embedding_rnn_seq2seq
@@embedding_tied_rnn_seq2seq
@@model_with_buckets
@@one2many_rnn_seq2seq
@@rnn_decoder
@@sequence_loss
@@sequence_loss_by_example
@@tied_rnn_seq2seq
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce legacy-seq2seq (import-module "tensorflow.contrib.legacy_seq2seq"))

(defn attention-decoder 
  "RNN decoder with attention for the sequence-to-sequence model.

  In this context \"attention\" means that, during decoding, the RNN can look up
  information in the additional tensor attention_states, and it does this by
  focusing on a few entries from the tensor. This model has proven to yield
  especially good results in a number of sequence-to-sequence tasks. This
  implementation is based on http://arxiv.org/abs/1412.7449 (see below for
  details). It is recommended for complex sequence-to-sequence tasks.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: tf.compat.v1.nn.rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.
    num_heads: Number of attention heads that read from attention_states.
    loop_function: If not None, this function will be applied to i-th output in
      order to generate i+1-th input, and decoder_inputs will be ignored, except
      for the first element (\"GO\" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
        Signature -- loop_function(prev, i) = next * prev is a 2D Tensor of
        shape [batch_size x output_size], * i is an integer, the step number
        (when advanced control is needed), * next is a 2D Tensor of shape
        [batch_size x input_size].
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: \"attention_decoder\".
    initial_state_attention: If False (default), initial attentions are zero. If
      True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously stored
      decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either the i-th element
        of decoder_inputs or loop_function(output {i-1}, i)) as follows.
        First, we run the cell on a combination of the input and previous
        attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
          new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        and then we calculate the output:
          output = linear(cell_output, new_attn).
      state: The state of each decoder cell the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: when num_heads is not positive, there are no inputs, shapes
      of attention_states are not set, or input size cannot be inferred
      from the input.
  "
  [decoder_inputs initial_state attention_states cell output_size & {:keys [num_heads loop_function dtype scope initial_state_attention]
                       :or {loop_function None dtype None scope None}} ]
    (py/call-attr-kw legacy-seq2seq "attention_decoder" [decoder_inputs initial_state attention_states cell output_size] {:num_heads num_heads :loop_function loop_function :dtype dtype :scope scope :initial_state_attention initial_state_attention }))

(defn basic-rnn-seq2seq 
  "Basic RNN sequence-to-sequence model.

  This model first runs an RNN to encode encoder_inputs into a state vector,
  then runs decoder, initialized with the last encoder state, on decoder_inputs.
  Encoder and decoder use the same RNN cell type, but don't share parameters.

  Args:
    encoder_inputs: A list of 2D Tensors [batch_size x input_size].
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    cell: tf.compat.v1.nn.rnn_cell.RNNCell defining the cell function and size.
    dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
    scope: VariableScope for the created subgraph; default: \"basic_rnn_seq2seq\".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell in the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  "
  [encoder_inputs decoder_inputs cell & {:keys [dtype scope]
                       :or {scope None}} ]
    (py/call-attr-kw legacy-seq2seq "basic_rnn_seq2seq" [encoder_inputs decoder_inputs cell] {:dtype dtype :scope scope }))

(defn embedding-attention-decoder 
  "RNN decoder with embedding and attention and a pure-decoding option.

  Args:
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: tf.compat.v1.nn.rnn_cell.RNNCell defining the cell function.
    num_symbols: Integer, how many symbols come into the embedding.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_heads: Number of attention heads that read from attention_states.
    output_size: Size of the output vectors; if None, use output_size.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has shape
      [num_symbols]; if provided and feed_previous=True, each fed previous
      output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the \"GO\" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)), In effect,
        this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/abs/1506.03099. If False,
        decoder_inputs are used as given (the standard decoder case).
    update_embedding_for_previous: Boolean; if False and feed_previous=True,
      only the embedding for the first symbol of decoder_inputs (the \"GO\"
      symbol) will be updated by back propagation. Embeddings for the symbols
      generated from the decoder itself remain unchanged. This parameter has no
      effect if feed_previous=False.
    dtype: The dtype to use for the RNN initial states (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      \"embedding_attention_decoder\".
    initial_state_attention: If False (default), initial attentions are zero. If
      True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously stored
      decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  "
  [decoder_inputs initial_state attention_states cell num_symbols embedding_size & {:keys [num_heads output_size output_projection feed_previous update_embedding_for_previous dtype scope initial_state_attention]
                       :or {output_size None output_projection None dtype None scope None}} ]
    (py/call-attr-kw legacy-seq2seq "embedding_attention_decoder" [decoder_inputs initial_state attention_states cell num_symbols embedding_size] {:num_heads num_heads :output_size output_size :output_projection output_projection :feed_previous feed_previous :update_embedding_for_previous update_embedding_for_previous :dtype dtype :scope scope :initial_state_attention initial_state_attention }))

(defn embedding-attention-seq2seq 
  "Embedding sequence-to-sequence model with attention.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_encoder_symbols x input_size]). Then it runs an RNN to encode
  embedded encoder_inputs into a state vector. It keeps the outputs of this
  RNN at every step to use for attention later. Next, it embeds decoder_inputs
  by another newly created embedding (of shape [num_decoder_symbols x
  input_size]). Then it runs attention decoder, initialized with the last
  encoder state, on embedded decoder_inputs and attending to encoder outputs.

  Warning: when output_projection is None, the size of the attention vectors
  and variables will be made proportional to num_decoder_symbols, can be large.

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    cell: tf.compat.v1.nn.rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols: Integer; number of symbols on the decoder side.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_heads: Number of attention heads that read from attention_states.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_decoder_symbols] and B has shape
      [num_decoder_symbols]; if provided and feed_previous=True, each fed
      previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first of
      decoder_inputs will be used (the \"GO\" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial RNN state (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      \"embedding_attention_seq2seq\".
    initial_state_attention: If False (default), initial attentions are zero. If
      True, initialize the attentions from the initial state and attention
      states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x num_decoder_symbols] containing the generated
        outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  "
  [encoder_inputs decoder_inputs cell num_encoder_symbols num_decoder_symbols embedding_size & {:keys [num_heads output_projection feed_previous dtype scope initial_state_attention]
                       :or {output_projection None dtype None scope None}} ]
    (py/call-attr-kw legacy-seq2seq "embedding_attention_seq2seq" [encoder_inputs decoder_inputs cell num_encoder_symbols num_decoder_symbols embedding_size] {:num_heads num_heads :output_projection output_projection :feed_previous feed_previous :dtype dtype :scope scope :initial_state_attention initial_state_attention }))

(defn embedding-rnn-decoder 
  "RNN decoder with embedding and a pure-decoding option.

  Args:
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    cell: tf.compat.v1.nn.rnn_cell.RNNCell defining the cell function.
    num_symbols: Integer, how many symbols come into the embedding.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has shape
      [num_symbols]; if provided and feed_previous=True, each fed previous
      output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the \"GO\" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)), In effect,
        this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/abs/1506.03099. If False,
        decoder_inputs are used as given (the standard decoder case).
    update_embedding_for_previous: Boolean; if False and feed_previous=True,
      only the embedding for the first symbol of decoder_inputs (the \"GO\"
      symbol) will be updated by back propagation. Embeddings for the symbols
      generated from the decoder itself remain unchanged. This parameter has no
      effect if feed_previous=False.
    scope: VariableScope for the created subgraph; defaults to
      \"embedding_rnn_decoder\".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors. The
        output is of shape [batch_size x cell.output_size] when
        output_projection is not None (and represents the dense representation
        of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
        when output_projection is None.
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  "
  [decoder_inputs initial_state cell num_symbols embedding_size output_projection & {:keys [feed_previous update_embedding_for_previous scope]
                       :or {scope None}} ]
    (py/call-attr-kw legacy-seq2seq "embedding_rnn_decoder" [decoder_inputs initial_state cell num_symbols embedding_size output_projection] {:feed_previous feed_previous :update_embedding_for_previous update_embedding_for_previous :scope scope }))

(defn embedding-rnn-seq2seq 
  "Embedding RNN sequence-to-sequence model.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_encoder_symbols x input_size]). Then it runs an RNN to encode
  embedded encoder_inputs into a state vector. Next, it embeds decoder_inputs
  by another newly created embedding (of shape [num_decoder_symbols x
  input_size]). Then it runs RNN decoder, initialized with the last
  encoder state, on embedded decoder_inputs.

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    cell: tf.compat.v1.nn.rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols: Integer; number of symbols on the decoder side.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_decoder_symbols] and B has shape
      [num_decoder_symbols]; if provided and feed_previous=True, each fed
      previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first of
      decoder_inputs will be used (the \"GO\" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial state for both the encoder and encoder
      rnn cells (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      \"embedding_rnn_seq2seq\"

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors. The
        output is of shape [batch_size x cell.output_size] when
        output_projection is not None (and represents the dense representation
        of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
        when output_projection is None.
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  "
  [encoder_inputs decoder_inputs cell num_encoder_symbols num_decoder_symbols embedding_size output_projection & {:keys [feed_previous dtype scope]
                       :or {dtype None scope None}} ]
    (py/call-attr-kw legacy-seq2seq "embedding_rnn_seq2seq" [encoder_inputs decoder_inputs cell num_encoder_symbols num_decoder_symbols embedding_size output_projection] {:feed_previous feed_previous :dtype dtype :scope scope }))

(defn embedding-tied-rnn-seq2seq 
  "Embedding RNN sequence-to-sequence model with tied (shared) parameters.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_symbols x input_size]). Then it runs an RNN to encode embedded
  encoder_inputs into a state vector. Next, it embeds decoder_inputs using
  the same embedding. Then it runs RNN decoder, initialized with the last
  encoder state, on embedded decoder_inputs. The decoder output is over symbols
  from 0 to num_decoder_symbols - 1 if num_decoder_symbols is none; otherwise it
  is over 0 to num_symbols - 1.

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    cell: tf.compat.v1.nn.rnn_cell.RNNCell defining the cell function and size.
    num_symbols: Integer; number of symbols for both encoder and decoder.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_decoder_symbols: Integer; number of output symbols for decoder. If
      provided, the decoder output is over symbols 0 to num_decoder_symbols - 1.
      Otherwise, decoder output is over symbols 0 to num_symbols - 1. Note that
      this assumes that the vocabulary is set up such that the first
      num_decoder_symbols of num_symbols are part of decoding.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has shape
      [num_symbols]; if provided and feed_previous=True, each fed previous
      output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first of
      decoder_inputs will be used (the \"GO\" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype to use for the initial RNN states (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      \"embedding_tied_rnn_seq2seq\".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_symbols] containing the generated
        outputs where output_symbols = num_decoder_symbols if
        num_decoder_symbols is not None otherwise output_symbols = num_symbols.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  "
  [encoder_inputs decoder_inputs cell num_symbols embedding_size num_decoder_symbols output_projection & {:keys [feed_previous dtype scope]
                       :or {dtype None scope None}} ]
    (py/call-attr-kw legacy-seq2seq "embedding_tied_rnn_seq2seq" [encoder_inputs decoder_inputs cell num_symbols embedding_size num_decoder_symbols output_projection] {:feed_previous feed_previous :dtype dtype :scope scope }))

(defn model-with-buckets 
  "Create a sequence-to-sequence model with support for bucketing.

  The seq2seq argument is a function that defines a sequence-to-sequence model,
  e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(
      x, y, rnn_cell.GRUCell(24))

  Args:
    encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
    decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
    targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
    weights: List of 1D batch-sized float-Tensors to weight the targets.
    buckets: A list of pairs of (input size, output size) for each bucket.
    seq2seq: A sequence-to-sequence model function; it takes 2 input that agree
      with encoder_inputs and decoder_inputs, and returns a pair consisting of
      outputs and states (as, e.g., basic_rnn_seq2seq).
    softmax_loss_function: Function (labels, logits) -> loss-batch to be used
      instead of the standard softmax (the default if this is None). **Note that
      to avoid confusion, it is required for the function to accept named
      arguments.**
    per_example_loss: Boolean. If set, the returned loss will be a batch-sized
      tensor of losses for each sequence in the batch. If unset, it will be a
      scalar with the averaged loss from all examples.
    name: Optional name for this operation, defaults to \"model_with_buckets\".

  Returns:
    A tuple of the form (outputs, losses), where:
      outputs: The outputs for each bucket. Its j'th element consists of a list
        of 2D Tensors. The shape of output tensors can be either
        [batch_size x output_size] or [batch_size x num_decoder_symbols]
        depending on the seq2seq model used.
      losses: List of scalar Tensors, representing losses for each bucket, or,
        if per_example_loss is set, a list of 1D batch-sized float Tensors.

  Raises:
    ValueError: If length of encoder_inputs, targets, or weights is smaller
      than the largest (last) bucket.
  "
  [encoder_inputs decoder_inputs targets weights buckets seq2seq softmax_loss_function & {:keys [per_example_loss name]
                       :or {name None}} ]
    (py/call-attr-kw legacy-seq2seq "model_with_buckets" [encoder_inputs decoder_inputs targets weights buckets seq2seq softmax_loss_function] {:per_example_loss per_example_loss :name name }))

(defn one2many-rnn-seq2seq 
  "One-to-many RNN sequence-to-sequence model (multi-task).

  This is a multi-task sequence-to-sequence model with one encoder and multiple
  decoders. Reference to multi-task sequence-to-sequence learning can be found
  here: http://arxiv.org/abs/1511.06114

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs_dict: A dictionary mapping decoder name (string) to the
      corresponding decoder_inputs; each decoder_inputs is a list of 1D Tensors
      of shape [batch_size]; num_decoders is defined as
      len(decoder_inputs_dict).
    enc_cell: tf.compat.v1.nn.rnn_cell.RNNCell defining the encoder cell
      function and size.
    dec_cells_dict: A dictionary mapping encoder name (string) to an instance of
      tf.nn.rnn_cell.RNNCell.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols_dict: A dictionary mapping decoder name (string) to an
      integer specifying number of symbols for the corresponding decoder;
      len(num_decoder_symbols_dict) must be equal to num_decoders.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first of
      decoder_inputs will be used (the \"GO\" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial state for both the encoder and encoder
      rnn cells (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      \"one2many_rnn_seq2seq\"

  Returns:
    A tuple of the form (outputs_dict, state_dict), where:
      outputs_dict: A mapping from decoder name (string) to a list of the same
        length as decoder_inputs_dict[name]; each element in the list is a 2D
        Tensors with shape [batch_size x num_decoder_symbol_list[name]]
        containing the generated outputs.
      state_dict: A mapping from decoder name (string) to the final state of the
        corresponding decoder RNN; it is a 2D Tensor of shape
        [batch_size x cell.state_size].

  Raises:
    TypeError: if enc_cell or any of the dec_cells are not instances of RNNCell.
    ValueError: if len(dec_cells) != len(decoder_inputs_dict).
  "
  [encoder_inputs decoder_inputs_dict enc_cell dec_cells_dict num_encoder_symbols num_decoder_symbols_dict embedding_size & {:keys [feed_previous dtype scope]
                       :or {dtype None scope None}} ]
    (py/call-attr-kw legacy-seq2seq "one2many_rnn_seq2seq" [encoder_inputs decoder_inputs_dict enc_cell dec_cells_dict num_encoder_symbols num_decoder_symbols_dict embedding_size] {:feed_previous feed_previous :dtype dtype :scope scope }))

(defn rnn-decoder 
  "RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element (\"GO\" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
        Signature -- loop_function(prev, i) = next * prev is a 2D Tensor of
        shape [batch_size x output_size], * i is an integer, the step number
        (when advanced control is needed), * next is a 2D Tensor of shape
        [batch_size x input_size].
    scope: VariableScope for the created subgraph; defaults to \"rnn_decoder\".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      state: The state of each cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
         states can be the same. They are different for LSTM cells though.)
  "
  [ decoder_inputs initial_state cell loop_function scope ]
  (py/call-attr legacy-seq2seq "rnn_decoder"  decoder_inputs initial_state cell loop_function scope ))

(defn sequence-loss 
  "Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (labels, logits) -> loss-batch to be used
      instead of the standard softmax (the default if this is None). **Note that
      to avoid confusion, it is required for the function to accept named
      arguments.**
    name: Optional name for this operation, defaults to \"sequence_loss\".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  "
  [logits targets weights & {:keys [average_across_timesteps average_across_batch softmax_loss_function name]
                       :or {softmax_loss_function None name None}} ]
    (py/call-attr-kw legacy-seq2seq "sequence_loss" [logits targets weights] {:average_across_timesteps average_across_timesteps :average_across_batch average_across_batch :softmax_loss_function softmax_loss_function :name name }))

(defn sequence-loss-by-example 
  "Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (labels, logits) -> loss-batch to be used
      instead of the standard softmax (the default if this is None). **Note that
      to avoid confusion, it is required for the function to accept named
      arguments.**
    name: Optional name for this operation, default: \"sequence_loss_by_example\".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  "
  [logits targets weights & {:keys [average_across_timesteps softmax_loss_function name]
                       :or {softmax_loss_function None name None}} ]
    (py/call-attr-kw legacy-seq2seq "sequence_loss_by_example" [logits targets weights] {:average_across_timesteps average_across_timesteps :softmax_loss_function softmax_loss_function :name name }))

(defn tied-rnn-seq2seq 
  "RNN sequence-to-sequence model with tied encoder and decoder parameters.

  This model first runs an RNN to encode encoder_inputs into a state vector, and
  then runs decoder, initialized with the last encoder state, on decoder_inputs.
  Encoder and decoder use the same RNN cell and share parameters.

  Args:
    encoder_inputs: A list of 2D Tensors [batch_size x input_size].
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    cell: tf.compat.v1.nn.rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to i-th output in
      order to generate i+1-th input, and decoder_inputs will be ignored, except
      for the first element (\"GO\" symbol), see rnn_decoder for details.
    dtype: The dtype of the initial state of the rnn cell (default: tf.float32).
    scope: VariableScope for the created subgraph; default: \"tied_rnn_seq2seq\".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  "
  [encoder_inputs decoder_inputs cell loop_function & {:keys [dtype scope]
                       :or {scope None}} ]
    (py/call-attr-kw legacy-seq2seq "tied_rnn_seq2seq" [encoder_inputs decoder_inputs cell loop_function] {:dtype dtype :scope scope }))
