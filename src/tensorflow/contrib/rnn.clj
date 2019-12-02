(ns tensorflow.contrib.rnn
  "RNN Cells and additional RNN operations.

<!--From core-->
@@RNNCell
@@LayerRNNCell
@@BasicRNNCell
@@BasicLSTMCell
@@GRUCell
@@LSTMCell
@@LSTMStateTuple
@@DropoutWrapper
@@MultiRNNCell
@@DeviceWrapper
@@ResidualWrapper

<!--Used to be in core, but kept in contrib.-->
@@EmbeddingWrapper
@@InputProjectionWrapper
@@OutputProjectionWrapper

<!--Created in contrib, eventual plans to move to core.-->
@@LayerNormBasicLSTMCell
@@LSTMBlockWrapper
@@LSTMBlockCell
@@GRUBlockCell
@@GRUBlockCellV2
@@FusedRNNCell
@@FusedRNNCellAdaptor
@@TimeReversedFusedRNN
@@LSTMBlockFusedCell
@@CoupledInputForgetGateLSTMCell
@@TimeFreqLSTMCell
@@GridLSTMCell
@@BidirectionalGridLSTMCell
@@NASCell
@@UGRNNCell
@@IntersectionRNNCell
@@PhasedLSTMCell
@@ConvLSTMCell
@@Conv1DLSTMCell
@@Conv2DLSTMCell
@@Conv3DLSTMCell
@@HighwayWrapper
@@GLSTMCell
@@SRUCell
@@IndRNNCell
@@IndyGRUCell
@@IndyLSTMCell

<!--RNNCell wrappers-->
@@AttentionCellWrapper
@@CompiledWrapper

<!--RNN functions-->
@@static_rnn
@@static_state_saving_rnn
@@static_bidirectional_rnn
@@stack_bidirectional_dynamic_rnn
@@stack_bidirectional_rnn

<!--RNN utilities-->
@@transpose_batch_time
@@best_effort_input_batch_size
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce rnn (import-module "tensorflow.contrib.rnn"))

(defn best-effort-input-batch-size 
  "Get static input batch size if available, with fallback to the dynamic one.

  Args:
    flat_input: An iterable of time major input Tensors of shape `[max_time,
      batch_size, ...]`. All inputs should have compatible batch sizes.

  Returns:
    The batch size in Python integer if available, or a scalar Tensor otherwise.

  Raises:
    ValueError: if there is any input with an invalid shape.
  "
  [ flat_input ]
  (py/call-attr rnn "best_effort_input_batch_size"  flat_input ))

(defn stack-bidirectional-dynamic-rnn 
  "Creates a dynamic bidirectional recurrent neural network.

  Stacks several bidirectional rnn layers. The combined forward and backward
  layer outputs are used as input of the next layer. tf.bidirectional_rnn
  does not allow to share forward and backward information between layers.
  The input_size of the first forward and backward cells must match.
  The initial state for both directions is zero and no intermediate states
  are returned.

  Args:
    cells_fw: List of instances of RNNCell, one per layer,
      to be used for forward direction.
    cells_bw: List of instances of RNNCell, one per layer,
      to be used for backward direction.
    inputs: The RNN inputs. this must be a tensor of shape:
      `[batch_size, max_time, ...]`, or a nested tuple of such elements.
    initial_states_fw: (optional) A list of the initial states (one per layer)
      for the forward RNN.
      Each tensor must has an appropriate type and shape
      `[batch_size, cell_fw.state_size]`.
    initial_states_bw: (optional) Same as for `initial_states_fw`, but using
      the corresponding properties of `cells_bw`.
    dtype: (optional) The data type for the initial state.  Required if
      either of the initial states are not provided.
    sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
      containing the actual lengths for each of the sequences.
    parallel_iterations: (Default: 32).  The number of iterations to run in
      parallel.  Those operations which do not have any temporal dependency
      and can be run in parallel, will be.  This parameter trades off
      time for space.  Values >> 1 use more memory but take less time,
      while smaller values use less memory but computations take longer.
    time_major: The shape format of the inputs and outputs Tensors. If true,
      these Tensors must be shaped [max_time, batch_size, depth]. If false,
      these Tensors must be shaped [batch_size, max_time, depth]. Using
      time_major = True is a bit more efficient because it avoids transposes at
      the beginning and end of the RNN calculation. However, most TensorFlow
      data is batch-major, so by default this function accepts input and emits
      output in batch-major form.
    scope: VariableScope for the created subgraph; defaults to None.
    swap_memory: Transparently swap the tensors produced in forward inference
      but needed for back prop from GPU to CPU.  This allows training RNNs
      which would typically not fit on a single GPU, with very minimal (or no)
      performance penalty.

  Returns:
    A tuple (outputs, output_state_fw, output_state_bw) where:
      outputs: Output `Tensor` shaped:
        `[batch_size, max_time, layers_output]`. Where layers_output
        are depth-concatenated forward and backward outputs.
      output_states_fw is the final states, one tensor per layer,
        of the forward rnn.
      output_states_bw is the final states, one tensor per layer,
        of the backward rnn.

  Raises:
    TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
    ValueError: If inputs is `None`.
  "
  [cells_fw cells_bw inputs initial_states_fw initial_states_bw dtype sequence_length parallel_iterations & {:keys [time_major scope swap_memory]
                       :or {scope None}} ]
    (py/call-attr-kw rnn "stack_bidirectional_dynamic_rnn" [cells_fw cells_bw inputs initial_states_fw initial_states_bw dtype sequence_length parallel_iterations] {:time_major time_major :scope scope :swap_memory swap_memory }))

(defn stack-bidirectional-rnn 
  "Creates a bidirectional recurrent neural network.

  Stacks several bidirectional rnn layers. The combined forward and backward
  layer outputs are used as input of the next layer. tf.bidirectional_rnn
  does not allow to share forward and backward information between layers.
  The input_size of the first forward and backward cells must match.
  The initial state for both directions is zero and no intermediate states
  are returned.

  As described in https://arxiv.org/abs/1303.5778

  Args:
    cells_fw: List of instances of RNNCell, one per layer,
      to be used for forward direction.
    cells_bw: List of instances of RNNCell, one per layer,
      to be used for backward direction.
    inputs: A length T list of inputs, each a tensor of shape
      [batch_size, input_size], or a nested tuple of such elements.
    initial_states_fw: (optional) A list of the initial states (one per layer)
      for the forward RNN.
      Each tensor must has an appropriate type and shape
      `[batch_size, cell_fw.state_size]`.
    initial_states_bw: (optional) Same as for `initial_states_fw`, but using
      the corresponding properties of `cells_bw`.
    dtype: (optional) The data type for the initial state.  Required if
      either of the initial states are not provided.
    sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
      containing the actual lengths for each of the sequences.
    scope: VariableScope for the created subgraph; defaults to None.

  Returns:
    A tuple (outputs, output_state_fw, output_state_bw) where:
      outputs is a length `T` list of outputs (one for each input), which
        are depth-concatenated forward and backward outputs.
      output_states_fw is the final states, one tensor per layer,
        of the forward rnn.
      output_states_bw is the final states, one tensor per layer,
        of the backward rnn.

  Raises:
    TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
    ValueError: If inputs is None, not a list or an empty list.
  "
  [ cells_fw cells_bw inputs initial_states_fw initial_states_bw dtype sequence_length scope ]
  (py/call-attr rnn "stack_bidirectional_rnn"  cells_fw cells_bw inputs initial_states_fw initial_states_bw dtype sequence_length scope ))

(defn static-bidirectional-rnn 
  "Creates a bidirectional recurrent neural network. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use `keras.layers.Bidirectional(keras.layers.RNN(cell, unroll=True))`, which is equivalent to this API

Similar to the unidirectional case above (rnn) but takes input and builds
independent forward and backward RNNs with the final forward and backward
outputs depth-concatenated, such that the output will have the format
[time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
forward and backward cell must match. The initial state for both directions
is zero by default (but can be set optionally) and no intermediate states are
ever returned -- the network is fully unrolled for the given (passed in)
length(s) of the sequence(s) or completely unrolled if length(s) is not given.

Args:
  cell_fw: An instance of RNNCell, to be used for forward direction.
  cell_bw: An instance of RNNCell, to be used for backward direction.
  inputs: A length T list of inputs, each a tensor of shape [batch_size,
    input_size], or a nested tuple of such elements.
  initial_state_fw: (optional) An initial state for the forward RNN. This must
    be a tensor of appropriate type and shape `[batch_size,
    cell_fw.state_size]`. If `cell_fw.state_size` is a tuple, this should be a
    tuple of tensors having shapes `[batch_size, s] for s in
    cell_fw.state_size`.
  initial_state_bw: (optional) Same as for `initial_state_fw`, but using the
    corresponding properties of `cell_bw`.
  dtype: (optional) The data type for the initial state.  Required if either
    of the initial states are not provided.
  sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
    containing the actual lengths for each of the sequences.
  scope: VariableScope for the created subgraph; defaults to
    \"bidirectional_rnn\"

Returns:
  A tuple (outputs, output_state_fw, output_state_bw) where:
    outputs is a length `T` list of outputs (one for each input), which
      are depth-concatenated forward and backward outputs.
    output_state_fw is the final state of the forward rnn.
    output_state_bw is the final state of the backward rnn.

Raises:
  TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
  ValueError: If inputs is None or an empty list."
  [ cell_fw cell_bw inputs initial_state_fw initial_state_bw dtype sequence_length scope ]
  (py/call-attr rnn "static_bidirectional_rnn"  cell_fw cell_bw inputs initial_state_fw initial_state_bw dtype sequence_length scope ))

(defn static-rnn 
  "Creates a recurrent neural network specified by RNNCell `cell`. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use `keras.layers.RNN(cell, unroll=True)`, which is equivalent to this API

The simplest form of RNN network generated is:

```python
  state = cell.zero_state(...)
  outputs = []
  for input_ in inputs:
    output, state = cell(input_, state)
    outputs.append(output)
  return (outputs, state)
```
However, a few other options are available:

An initial state can be provided.
If the sequence_length vector is provided, dynamic calculation is performed.
This method of calculation does not compute the RNN steps past the maximum
sequence length of the minibatch (thus saving computational time),
and properly propagates the state at an example's sequence length
to the final state output.

The dynamic calculation performed is, at time `t` for batch row `b`,

```python
  (output, state)(b, t) =
    (t >= sequence_length(b))
      ? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
      : cell(input(b, t), state(b, t - 1))
```

Args:
  cell: An instance of RNNCell.
  inputs: A length T list of inputs, each a `Tensor` of shape `[batch_size,
    input_size]`, or a nested tuple of such elements.
  initial_state: (optional) An initial state for the RNN. If `cell.state_size`
    is an integer, this must be a `Tensor` of appropriate type and shape
    `[batch_size, cell.state_size]`. If `cell.state_size` is a tuple, this
    should be a tuple of tensors having shapes `[batch_size, s] for s in
    cell.state_size`.
  dtype: (optional) The data type for the initial state and expected output.
    Required if initial_state is not provided or RNN state has a heterogeneous
    dtype.
  sequence_length: Specifies the length of each sequence in inputs. An int32
    or int64 vector (tensor) size `[batch_size]`, values in `[0, T)`.
  scope: VariableScope for the created subgraph; defaults to \"rnn\".

Returns:
  A pair (outputs, state) where:

  - outputs is a length T list of outputs (one for each input), or a nested
    tuple of such elements.
  - state is the final state

Raises:
  TypeError: If `cell` is not an instance of RNNCell.
  ValueError: If `inputs` is `None` or an empty list, or if the input depth
    (column size) cannot be inferred from inputs via shape inference."
  [ cell inputs initial_state dtype sequence_length scope ]
  (py/call-attr rnn "static_rnn"  cell inputs initial_state dtype sequence_length scope ))

(defn static-state-saving-rnn 
  "RNN that accepts a state saver for time-truncated RNN calculation. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use `keras.layers.RNN(cell, stateful=True)`, which is equivalent to this API

Args:
  cell: An instance of `RNNCell`.
  inputs: A length T list of inputs, each a `Tensor` of shape `[batch_size,
    input_size]`.
  state_saver: A state saver object with methods `state` and `save_state`.
  state_name: Python string or tuple of strings.  The name to use with the
    state_saver. If the cell returns tuples of states (i.e., `cell.state_size`
    is a tuple) then `state_name` should be a tuple of strings having the same
    length as `cell.state_size`.  Otherwise it should be a single string.
  sequence_length: (optional) An int32/int64 vector size [batch_size]. See the
    documentation for rnn() for more details about sequence_length.
  scope: VariableScope for the created subgraph; defaults to \"rnn\".

Returns:
  A pair (outputs, state) where:
    outputs is a length T list of outputs (one for each input)
    states is the final state

Raises:
  TypeError: If `cell` is not an instance of RNNCell.
  ValueError: If `inputs` is `None` or an empty list, or if the arity and
   type of `state_name` does not match that of `cell.state_size`."
  [ cell inputs state_saver state_name sequence_length scope ]
  (py/call-attr rnn "static_state_saving_rnn"  cell inputs state_saver state_name sequence_length scope ))

(defn transpose-batch-time 
  "Transposes the batch and time dimensions of a Tensor.

  If the input tensor has rank < 2 it returns the original tensor. Retains as
  much of the static shape information as possible.

  Args:
    x: A Tensor.

  Returns:
    x transposed along the first two dimensions.
  "
  [ x ]
  (py/call-attr rnn "transpose_batch_time"  x ))
