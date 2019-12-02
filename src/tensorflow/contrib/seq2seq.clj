(ns tensorflow.contrib.seq2seq
  "Ops for building neural network seq2seq decoders and losses.

See the
[Contrib Seq2seq](https://tensorflow.org/api_guides/python/contrib.seq2seq)
guide.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce seq2seq (import-module "tensorflow.contrib.seq2seq"))

(defn dynamic-decode 
  "Perform dynamic decoding with `decoder`.

  Calls initialize() once and step() repeatedly on the Decoder object.

  Args:
    decoder: A `Decoder` instance.
    output_time_major: Python boolean.  Default: `False` (batch major).  If
      `True`, outputs are returned as time major tensors (this mode is faster).
      Otherwise, outputs are returned as batch major tensors (this adds extra
      time to the computation).
    impute_finished: Python boolean.  If `True`, then states for batch
      entries which are marked as finished get copied through and the
      corresponding outputs get zeroed out.  This causes some slowdown at
      each time step, but ensures that the final state and outputs have
      the correct values and that backprop ignores time steps that were
      marked as finished.
    maximum_iterations: `int32` scalar, maximum allowed number of decoding
       steps.  Default is `None` (decode until the decoder is fully done).
    parallel_iterations: Argument passed to `tf.while_loop`.
    swap_memory: Argument passed to `tf.while_loop`.
    scope: Optional variable scope to use.
    **kwargs: dict, other keyword arguments for dynamic_decode. It might contain
      arguments for `BaseDecoder` to initialize, which takes all tensor inputs
      during call().

  Returns:
    `(final_outputs, final_state, final_sequence_lengths)`.

  Raises:
    TypeError: if `decoder` is not an instance of `Decoder`.
    ValueError: if `maximum_iterations` is provided but is not a scalar.
  "
  [decoder & {:keys [output_time_major impute_finished maximum_iterations parallel_iterations swap_memory scope]
                       :or {maximum_iterations None scope None}} ]
    (py/call-attr-kw seq2seq "dynamic_decode" [decoder] {:output_time_major output_time_major :impute_finished impute_finished :maximum_iterations maximum_iterations :parallel_iterations parallel_iterations :swap_memory swap_memory :scope scope }))

(defn gather-tree 
  "Calculates the full beams from the per-step ids and parent beam ids.

  On CPU, if an out of bound parent id is found, an error is returned.
  On GPU, if an out of bound parent id is found, a -1 is stored in the
  corresponding output value and the execution for that beam returns early.

  For a given beam, past the time step containing the first decoded `end_token`
  all values are filled in with `end_token`.

  TODO(ebrevdo): fill in the remainder of this docstring.

  Args:
    step_ids: A `Tensor`. Must be one of the following types: `int32`.
      `[max_time, batch_size, beam_width]`.
    parent_ids: A `Tensor`. Must have the same type as `step_ids`.
      `[max_time, batch_size, beam_width]`.
    max_sequence_lengths: A `Tensor` of type `int32`. `[batch_size]`.
    end_token: A `Tensor`. Must have the same type as `step_ids`. `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `step_ids`.
    `[max_time, batch_size, beam_width]`.
  "
  [ step_ids parent_ids max_sequence_lengths end_token name ]
  (py/call-attr seq2seq "gather_tree"  step_ids parent_ids max_sequence_lengths end_token name ))

(defn hardmax 
  "Returns batched one-hot vectors.

  The depth index containing the `1` is that of the maximum logit value.

  Args:
    logits: A batch tensor of logit values.
    name: Name to use when creating ops.

  Returns:
    A batched one-hot tensor.
  "
  [ logits name ]
  (py/call-attr seq2seq "hardmax"  logits name ))

(defn monotonic-attention 
  "Compute monotonic attention distribution from choosing probabilities.

  Monotonic attention implies that the input sequence is processed in an
  explicitly left-to-right manner when generating the output sequence.  In
  addition, once an input sequence element is attended to at a given output
  timestep, elements occurring before it cannot be attended to at subsequent
  output timesteps.  This function generates attention distributions according
  to these assumptions.  For more information, see `Online and Linear-Time
  Attention by Enforcing Monotonic Alignments`.

  Args:
    p_choose_i: Probability of choosing input sequence/memory element i.  Should
      be of shape (batch_size, input_sequence_length), and should all be in the
      range [0, 1].
    previous_attention: The attention distribution from the previous output
      timestep.  Should be of shape (batch_size, input_sequence_length).  For
      the first output timestep, preevious_attention[n] should be [1, 0, 0, ...,
      0] for all n in [0, ... batch_size - 1].
    mode: How to compute the attention distribution.  Must be one of
      'recursive', 'parallel', or 'hard'. * 'recursive' uses tf.scan to
      recursively compute the distribution. This is slowest but is exact,
      general, and does not suffer from numerical instabilities. * 'parallel'
      uses parallelized cumulative-sum and cumulative-product operations to
      compute a closed-form solution to the recurrence relation defining the
      attention distribution.  This makes it more efficient than 'recursive',
      but it requires numerical checks which make the distribution non-exact.
      This can be a problem in particular when input_sequence_length is long
      and/or p_choose_i has entries very close to 0 or 1. * 'hard' requires that
      the probabilities in p_choose_i are all either 0 or 1, and subsequently
      uses a more efficient and exact solution.

  Returns:
    A tensor of shape (batch_size, input_sequence_length) representing the
    attention distributions for each sequence in the batch.

  Raises:
    ValueError: mode is not one of 'recursive', 'parallel', 'hard'.
  "
  [ p_choose_i previous_attention mode ]
  (py/call-attr seq2seq "monotonic_attention"  p_choose_i previous_attention mode ))

(defn safe-cumprod 
  "Computes cumprod of x in logspace using cumsum to avoid underflow.

  The cumprod function and its gradient can result in numerical instabilities
  when its argument has very small and/or zero values.  As long as the argument
  is all positive, we can instead compute the cumulative product as
  exp(cumsum(log(x))).  This function can be called identically to tf.cumprod.

  Args:
    x: Tensor to take the cumulative product of.
    *args: Passed on to cumsum; these are identical to those in cumprod.
    **kwargs: Passed on to cumsum; these are identical to those in cumprod.

  Returns:
    Cumulative product of x.
  "
  [ x ]
  (py/call-attr seq2seq "safe_cumprod"  x ))

(defn sequence-loss 
  "Weighted cross-entropy loss for a sequence of logits.

  Depending on the values of `average_across_timesteps` / `sum_over_timesteps`
  and `average_across_batch` / `sum_over_batch`, the return Tensor will have
  rank 0, 1, or 2 as these arguments reduce the cross-entropy at each target,
  which has shape `[batch_size, sequence_length]`, over their respective
  dimensions. For example, if `average_across_timesteps` is `True` and
  `average_across_batch` is `False`, then the return Tensor will have shape
  `[batch_size]`.

  Note that `average_across_timesteps` and `sum_over_timesteps` cannot be True
  at same time. Same for `average_across_batch` and `sum_over_batch`.

  The recommended loss reduction in tf 2.0 has been changed to sum_over, instead
  of weighted average. User are recommend to use `sum_over_timesteps` and
  `sum_over_batch` for reduction.

  Args:
    logits: A Tensor of shape
      `[batch_size, sequence_length, num_decoder_symbols]` and dtype float.
      The logits correspond to the prediction across all classes at each
      timestep.
    targets: A Tensor of shape `[batch_size, sequence_length]` and dtype
      int. The target represents the true class at each timestep.
    weights: A Tensor of shape `[batch_size, sequence_length]` and dtype
      float. `weights` constitutes the weighting of each prediction in the
      sequence. When using `weights` as masking, set all valid timesteps to 1
      and all padded timesteps to 0, e.g. a mask returned by `tf.sequence_mask`.
    average_across_timesteps: If set, sum the cost across the sequence
      dimension and divide the cost by the total label weight across timesteps.
    average_across_batch: If set, sum the cost across the batch dimension and
      divide the returned cost by the batch size.
    sum_over_timesteps: If set, sum the cost across the sequence dimension and
      divide the size of the sequence. Note that any element with 0 weights will
      be excluded from size calculation.
    sum_over_batch: if set, sum the cost across the batch dimension and divide
      the total cost by the batch size. Not that any element with 0 weights will
      be excluded from size calculation.
    softmax_loss_function: Function (labels, logits) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
      **Note that to avoid confusion, it is required for the function to accept
      named arguments.**
    name: Optional name for this operation, defaults to \"sequence_loss\".

  Returns:
    A float Tensor of rank 0, 1, or 2 depending on the
    `average_across_timesteps` and `average_across_batch` arguments. By default,
    it has rank 0 (scalar) and is the weighted average cross-entropy
    (log-perplexity) per symbol.

  Raises:
    ValueError: logits does not have 3 dimensions or targets does not have 2
                dimensions or weights does not have 2 dimensions.
  "
  [logits targets weights & {:keys [average_across_timesteps average_across_batch sum_over_timesteps sum_over_batch softmax_loss_function name]
                       :or {softmax_loss_function None name None}} ]
    (py/call-attr-kw seq2seq "sequence_loss" [logits targets weights] {:average_across_timesteps average_across_timesteps :average_across_batch average_across_batch :sum_over_timesteps sum_over_timesteps :sum_over_batch sum_over_batch :softmax_loss_function softmax_loss_function :name name }))

(defn tile-batch 
  "Tile the batch dimension of a (possibly nested structure of) tensor(s) t.

  For each tensor t in a (possibly nested structure) of tensors,
  this function takes a tensor t shaped `[batch_size, s0, s1, ...]` composed of
  minibatch entries `t[0], ..., t[batch_size - 1]` and tiles it to have a shape
  `[batch_size * multiplier, s0, s1, ...]` composed of minibatch entries
  `t[0], t[0], ..., t[1], t[1], ...` where each minibatch entry is repeated
  `multiplier` times.

  Args:
    t: `Tensor` shaped `[batch_size, ...]`.
    multiplier: Python int.
    name: Name scope for any created operations.

  Returns:
    A (possibly nested structure of) `Tensor` shaped
    `[batch_size * multiplier, ...]`.

  Raises:
    ValueError: if tensor(s) `t` do not have a statically known rank or
    the rank is < 1.
  "
  [ t multiplier name ]
  (py/call-attr seq2seq "tile_batch"  t multiplier name ))
