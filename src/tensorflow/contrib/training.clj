(ns tensorflow.contrib.training
  "Training and input utilities.

See
[Contrib Training](https://tensorflow.org/api_guides/python/contrib.training)
guide.

@@batch_sequences_with_states
@@NextQueuedSequenceBatch
@@SequenceQueueingStateSaver
@@rejection_sample
@@resample_at_rate
@@stratified_sample
@@weighted_resample
@@bucket
@@bucket_by_sequence_length
@@RandomStrategy
@@GreedyLoadBalancingStrategy
@@byte_size_load_fn
@@FailureTolerator
@@rejection_sample
@@stratified_sample
@@resample_at_rate
@@weighted_resample
@@HParams
@@HParamDef
@@parse_values
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce training (import-module "tensorflow.contrib.training"))

(defn add-gradients-summaries 
  "Add summaries to gradients.

  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).

  Returns:
    The list of created summaries.
  "
  [ grads_and_vars ]
  (py/call-attr training "add_gradients_summaries"  grads_and_vars ))

(defn batch-sequences-with-states 
  "Creates batches of segments of sequential input.

  This method creates a `SequenceQueueingStateSaver` (SQSS) and adds it to
  the queuerunners. It returns a `NextQueuedSequenceBatch`.

  It accepts one example at a time identified by a unique `input_key`.
  `input_sequence` is a dict with values that are tensors with time as first
  dimension. This time dimension must be the same across those tensors of an
  example. It can vary across examples. Although it always has to be a multiple
  of `num_unroll`. Hence, padding may be necessary and it is turned on by
  default by `pad=True`.

  `input_length` is a Tensor scalar or an int recording the time dimension prior
  to padding. It should be between 0 and the time dimension. One reason we want
  to keep track of it is so that we can take it into consideration when
  computing the loss. If `pad=True` then `input_length` can be `None` and will
  be inferred.

  This methods segments `input_sequence` into segments of length `num_unroll`.
  It batches input sequences from `batch_size` many examples. These mini-batches
  are available through the `sequence` property of the output. Moreover, for
  each entry in the batch we can access its original `input_key` in `key` and
  its input length in `total_length`. `length` records within this segment how
  many non-padded time steps there are.

  Static features of an example that do not vary across time can be part of the
  `input_context`, a dict with Tensor values. This method copies the context for
  each segment and makes it available in the `context` of the output.

  This method can maintain and update a state for each example. It accepts some
  initial_states as a dict with Tensor values. The first mini-batch an example
  is contained has initial_states as entry of the `state`. If save_state is
  called then the next segment will have the updated entry of the `state`.
  See `NextQueuedSequenceBatch` for a complete list of properties and methods.

  Example usage:

  ```python
  batch_size = 32
  num_unroll = 20
  num_enqueue_threads = 3
  lstm_size = 8
  cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size)

  key, sequences, context = my_parser(raw_data)
  initial_state_values = tf.zeros((state_size,), dtype=tf.float32)
  initial_states = {\"lstm_state\": initial_state_values}
  batch = tf.batch_sequences_with_states(
      input_key=key,
      input_sequences=sequences,
      input_context=context,
      input_length=tf.shape(sequences[\"input\"])[0],
      initial_states=initial_states,
      num_unroll=num_unroll,
      batch_size=batch_size,
      num_threads=num_enqueue_threads,
      capacity=batch_size * num_enqueue_threads * 2)

  inputs = batch.sequences[\"input\"]
  context_label = batch.context[\"label\"]

  inputs_by_time = tf.split(value=inputs, num_or_size_splits=num_unroll, axis=1)
  assert len(inputs_by_time) == num_unroll

  lstm_output, _ = tf.contrib.rnn.static_state_saving_rnn(
    cell,
    inputs_by_time,
    state_saver=batch,
    state_name=\"lstm_state\")

  # Start a prefetcher in the background
  sess = tf.compat.v1.Session()

  tf.compat.v1.train.start_queue_runners(sess=session)

  while True:
    # Step through batches, perform training or inference...
    session.run([lstm_output])
  ```

  Args:
    input_key: A string scalar `Tensor`, the **unique** key for the given
      input example.  This is used to keep track of the split minibatch elements
      of this input.  Batched keys of the current iteration are made
      accessible via the `key` property.  The shape of `input_key` (scalar) must
      be fully specified.  Consider setting `make_keys_unique` to True when
      iterating over the same input multiple times.

      **Note**: if `make_keys_unique=False` then `input_key`s must be unique.
    input_sequences: A dict mapping string names to `Tensor` values.  The values
      must all have matching first dimension, called `value_length`. They may
      vary from input to input. The remainder of the shape (other than the first
      dimension) must be fully specified.
      The `SequenceQueueingStateSaver` will split these tensors along
      this first dimension into minibatch elements of dimension `num_unrolled`.
      Batched and segmented sequences of the current iteration are made
      accessible via the `sequences` property.

      **Note**: if `pad=False`, then `value_length` must always be a multiple
        of `num_unroll`.
    input_context: A dict mapping string names to `Tensor` values.  The values
      are treated as \"global\" across all time splits of the given input example,
      and will be copied across for all minibatch elements accordingly.
      Batched and copied context of the current iteration are made
      accessible via the `context` property.

      **Note**: All input_context values must have fully defined shapes.
    input_length: None or an int32 scalar `Tensor`, the length of the sequence
      prior to padding. If `input_length=None` and `pad=True` then the length
      will be inferred and will be equal to `value_length`. If `pad=False` then
      `input_length` cannot be `None`: `input_length` must be specified. Its
      shape of `input_length` (scalar) must be fully specified. Its value may be
      at most `value_length` for any given input (see above for the definition
      of `value_length`). Batched and total lengths of the current iteration are
      made accessible via the `length` and `total_length` properties.
    initial_states: A dict mapping string state names to multi-dimensional
      values (e.g. constants or tensors).  This input defines the set of
      states that will be kept track of during computing iterations, and
      which can be accessed via the `state` and `save_state` methods.

      **Note**: All initial_state values must have fully defined shapes.
    num_unroll: Python integer, how many time steps to unroll at a time.
      The input sequences of length k are then split into k / num_unroll many
      segments.
    batch_size: int or int32 scalar `Tensor`, how large minibatches should
      be when accessing the `state()` method and `context`, `sequences`, etc,
      properties.
    num_threads: The int number of threads enqueuing input examples into a
      queue.
    capacity: The max capacity of the queue in number of examples. Needs to be
      at least `batch_size`. Defaults to 1000. When iterating over the same
      input example multiple times reusing their keys the `capacity` must be
      smaller than the number of examples.
    allow_small_batch: If true, the queue will return smaller batches when
      there aren't enough input examples to fill a whole batch and the end of
      the input has been reached.
    pad: If `True`, `input_sequences` will be padded to multiple of
      `num_unroll`. In that case `input_length` may be `None` and is assumed to
      be the length of first dimension of values in `input_sequences`
      (i.e. `value_length`).
    make_keys_unique: Whether to append a random integer to the `input_key` in
      an effort to make it unique. The seed can be set via
      `make_keys_unique_seed`.
    make_keys_unique_seed: If `make_keys_unique=True` this fixes the seed with
      which a random postfix is generated.
    name: An op name string (optional).

  Returns:
    A NextQueuedSequenceBatch with segmented and batched inputs and their
    states.

  Raises:
    TypeError: if any of the inputs is not an expected type.
    ValueError: if any of the input values is inconsistent, e.g. if
      not enough shape information is available from inputs to build
      the state saver.
  "
  [input_key input_sequences input_context input_length initial_states num_unroll batch_size & {:keys [num_threads capacity allow_small_batch pad make_keys_unique make_keys_unique_seed name]
                       :or {make_keys_unique_seed None name None}} ]
    (py/call-attr-kw training "batch_sequences_with_states" [input_key input_sequences input_context input_length initial_states num_unroll batch_size] {:num_threads num_threads :capacity capacity :allow_small_batch allow_small_batch :pad pad :make_keys_unique make_keys_unique :make_keys_unique_seed make_keys_unique_seed :name name }))

(defn bucket 
  "Lazy bucketing of input tensors according to `which_bucket`.

  The argument `tensors` can be a list or a dictionary of tensors.
  The value returned by the function will be of the same type
  as `tensors`.

  The tensors entering this function are put into the bucket given by
  `which_bucket`.  Each bucket has its own queue.  When a bucket contains
  `batch_size` elements, this minibatch is pushed onto a top queue.  The
  tensors returned from this function are a the result of dequeueing the
  next minibatch from this top queue.

  This function is implemented using several queues. A `QueueRunner` for the
  queues is added to the current `Graph`'s `QUEUE_RUNNER` collection.

  As the returned tensors are the result of a dequeue operation, evaluating
  them will throw a `tf.errors.OutOfRangeError` when the input queue is
  exhausted.  If these tensors are feeding another input queue, its queue runner
  will catch this exception, however, if they are used in your main thread
  you are responsible for catching this yourself.

  *N.B.:* If `dynamic_pad` is `False`, you must ensure that either
  (i) the `shapes` argument is passed, or (ii) all of the tensors in
  `tensors` must have fully-defined shapes. `ValueError` will be
  raised if neither of these conditions holds.

  If `dynamic_pad` is `True`, it is sufficient that the *rank* of the
  tensors is known, but individual dimensions may have shape `None`.
  In this case, for each enqueue the dimensions with value `None`
  may have a variable length; upon dequeue, the output tensors will be padded
  on the right to the maximum shape of the tensors in the current minibatch.
  For numbers, this padding takes value 0.  For strings, this padding is
  the empty string.  See `PaddingFIFOQueue` for more info.

  If `allow_smaller_final_batch` is `True`, a smaller batch value than
  `batch_size` is returned when the queues are closed and there are not enough
  elements to fill the batch, otherwise the pending elements are discarded.
  In addition, all output tensors' static shapes, as accessed via the
  `get_shape()` method will have a 0th `Dimension` value of `None`, and
  operations that depend on fixed batch_size would fail.

  Args:
    tensors: The list or dictionary of tensors, representing a single element,
      to bucket.  Nested lists are not supported.
    which_bucket: An `int32` scalar Tensor taking a value in `[0, num_buckets)`.
    batch_size: The new batch size pulled from the queue (all queues will have
      the same size).  If a list is passed in then each bucket will have a
      different batch_size.
      (python int, int32 scalar or iterable of integers of length num_buckets).
    num_buckets: A python integer, the number of buckets.
    num_threads: An integer.  The number of threads enqueuing `tensors`.
    capacity: An integer. The maximum number of minibatches in the top queue,
      and also (by default) the maximum number of elements within each bucket.
    bucket_capacities: (Optional) None or a list of integers, the capacities of
      each bucket. If None, capacity is used (default). If specified, it must
      be a list of integers of length num_buckets: the i-th element is used
      as capacity for the i-th bucket queue.
    shapes: (Optional) The shapes for each example.  Defaults to the
      inferred shapes for `tensors`.
    dynamic_pad: Boolean.  Allow variable dimensions in input shapes.
      The given dimensions are padded upon dequeue so that tensors within a
      batch have the same shapes.
    allow_smaller_final_batch: (Optional) Boolean. If `True`, allow the final
      batches to be smaller if there are insufficient items left in the queues.
    keep_input: A `bool` scalar Tensor.  If provided, this tensor controls
      whether the input is added to the queue or not.  If it evaluates `True`,
      then `tensors` are added to the bucket; otherwise they are dropped.  This
      tensor essentially acts as a filtering mechanism.
    shared_name: (Optional). If set, the queues will be shared under the given
      name across multiple sessions.
    name: (Optional) A name for the operations.

  Returns:
    A tuple `(bucket, outputs)` where `bucket` is
    a `int32` scalar tensor and `outputs` is a list or
    dictionary of batched outputs corresponding to elements of `tensors`.
    Every step will receive a new bucket of outputs.

  Raises:
    ValueError: If the `shapes` are not specified, and cannot be
      inferred from the elements of `tensors` or if batch_size is a sequence
      but its length != num_buckets. Also if bucket_capacities is not None but
      its length != num_buckets.
  "
  [tensors which_bucket batch_size num_buckets & {:keys [num_threads capacity bucket_capacities shapes dynamic_pad allow_smaller_final_batch keep_input shared_name name]
                       :or {bucket_capacities None shapes None shared_name None name None}} ]
    (py/call-attr-kw training "bucket" [tensors which_bucket batch_size num_buckets] {:num_threads num_threads :capacity capacity :bucket_capacities bucket_capacities :shapes shapes :dynamic_pad dynamic_pad :allow_smaller_final_batch allow_smaller_final_batch :keep_input keep_input :shared_name shared_name :name name }))

(defn bucket-by-sequence-length 
  "Lazy bucketing of inputs according to their length.

  This method calls `tf.contrib.training.bucket` under the hood, after first
  subdividing the bucket boundaries into separate buckets and identifying which
  bucket the given `input_length` belongs to.  See the documentation for
  `which_bucket` for details of the other arguments.

  Args:
    input_length: `int32` scalar `Tensor`, the sequence length of tensors.
    tensors: The list or dictionary of tensors, representing a single element,
      to bucket.  Nested lists are not supported.
    batch_size: The new batch size pulled from the queue (all queues will have
      the same size).  If a list is passed in then each bucket will have a
      different batch_size.
      (python int, int32 scalar or iterable of integers of length num_buckets).
    bucket_boundaries: int list, increasing non-negative numbers.
      The edges of the buckets to use when bucketing tensors.  Two extra buckets
      are created, one for `input_length < bucket_boundaries[0]` and
      one for `input_length >= bucket_boundaries[-1]`.
    num_threads: An integer.  The number of threads enqueuing `tensors`.
    capacity: An integer. The maximum number of minibatches in the top queue,
      and also the maximum number of elements within each bucket.
    bucket_capacities: (Optional) None or a list of integers, the capacities of
      each bucket. If None, capacity is used (default). If specified, it must
      be a list of integers of length one larger than bucket_boundaries.
      Its i-th element is used as capacity for the i-th bucket queue.
    shapes: (Optional) The shapes for each example.  Defaults to the
      inferred shapes for `tensors`.
    dynamic_pad: Boolean.  Allow variable dimensions in input shapes.
      The given dimensions are padded upon dequeue so that tensors within a
      batch have the same shapes.
    allow_smaller_final_batch: (Optional) Boolean. If `True`, allow the final
      batches to be smaller if there are insufficient items left in the queues.
    keep_input: A `bool` scalar Tensor.  If provided, this tensor controls
      whether the input is added to the queue or not.  If it evaluates `True`,
      then `tensors` are added to the bucket; otherwise they are dropped.  This
      tensor essentially acts as a filtering mechanism.
    shared_name: (Optional). If set, the queues will be shared under the given
      name across multiple sessions.
    name: (Optional) A name for the operations.

  Returns:
    A tuple `(sequence_length, outputs)` where `sequence_length` is
    a 1-D `Tensor` of size `batch_size` and `outputs` is a list or dictionary
    of batched, bucketed, outputs corresponding to elements of `tensors`.

  Raises:
    TypeError: if `bucket_boundaries` is not a list of python integers.
    ValueError: if `bucket_boundaries` is empty or contains non-increasing
      values or if batch_size is a list and it's length doesn't equal the number
      of buckets.
  "
  [input_length tensors batch_size bucket_boundaries & {:keys [num_threads capacity bucket_capacities shapes dynamic_pad allow_smaller_final_batch keep_input shared_name name]
                       :or {bucket_capacities None shapes None shared_name None name None}} ]
    (py/call-attr-kw training "bucket_by_sequence_length" [input_length tensors batch_size bucket_boundaries] {:num_threads num_threads :capacity capacity :bucket_capacities bucket_capacities :shapes shapes :dynamic_pad dynamic_pad :allow_smaller_final_batch allow_smaller_final_batch :keep_input keep_input :shared_name shared_name :name name }))

(defn byte-size-load-fn 
  "Load function that computes the byte size of a single-output `Operation`.

  This is intended to be used with `\"Variable\"` ops, which have a single
  `Tensor` output with the contents of the variable.  However, it can also be
  used for calculating the size of any op that has a single output.

  Intended to be used with `GreedyLoadBalancingStrategy`.

  Args:
    op: An `Operation` with a single output, typically a \"Variable\" op.

  Returns:
    The number of bytes in the output `Tensor`.

  Raises:
    ValueError: if `op` does not have a single output, or if the shape of the
      single output is not fully-defined.
  "
  [ op ]
  (py/call-attr training "byte_size_load_fn"  op ))

(defn checkpoints-iterator 
  "Continuously yield new checkpoint files as they appear.

  The iterator only checks for new checkpoints when control flow has been
  reverted to it. This means it can miss checkpoints if your code takes longer
  to run between iterations than `min_interval_secs` or the interval at which
  new checkpoints are written.

  The `timeout` argument is the maximum number of seconds to block waiting for
  a new checkpoint.  It is used in combination with the `timeout_fn` as
  follows:

  * If the timeout expires and no `timeout_fn` was specified, the iterator
    stops yielding.
  * If a `timeout_fn` was specified, that function is called and if it returns
    a true boolean value the iterator stops yielding.
  * If the function returns a false boolean value then the iterator resumes the
    wait for new checkpoints.  At this point the timeout logic applies again.

  This behavior gives control to callers on what to do if checkpoints do not
  come fast enough or stop being generated.  For example, if callers have a way
  to detect that the training has stopped and know that no new checkpoints
  will be generated, they can provide a `timeout_fn` that returns `True` when
  the training has stopped.  If they know that the training is still going on
  they return `False` instead.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    min_interval_secs: The minimum number of seconds between yielding
      checkpoints.
    timeout: The maximum number of seconds to wait between checkpoints. If left
      as `None`, then the process will wait indefinitely.
    timeout_fn: Optional function to call after a timeout.  If the function
      returns True, then it means that no new checkpoints will be generated and
      the iterator will exit.  The function is called with no arguments.

  Yields:
    String paths to latest checkpoint files as they arrive.
  "
  [checkpoint_dir & {:keys [min_interval_secs timeout timeout_fn]
                       :or {timeout None timeout_fn None}} ]
    (py/call-attr-kw training "checkpoints_iterator" [checkpoint_dir] {:min_interval_secs min_interval_secs :timeout timeout :timeout_fn timeout_fn }))

(defn clip-gradient-norms 
  "Clips the gradients by the given value.

  Args:
    gradients_to_variables: A list of gradient to variable pairs (tuples).
    max_norm: the maximum norm value.

  Returns:
    A list of clipped gradient to variable pairs.
  "
  [ gradients_to_variables max_norm ]
  (py/call-attr training "clip_gradient_norms"  gradients_to_variables max_norm ))

(defn clip-gradient-norms-fn 
  "Returns a `transform_grads_fn` function for gradient clipping."
  [ max_norm ]
  (py/call-attr training "clip_gradient_norms_fn"  max_norm ))

(defn create-train-op 
  "Creates an `Operation` that evaluates the gradients and returns the loss.

  Args:
    total_loss: A `Tensor` representing the total loss.
    optimizer: A tf.Optimizer to use for computing the gradients.
    global_step: A `Tensor` representing the global step variable. If left as
      `_USE_GLOBAL_STEP`, then tf.contrib.framework.global_step() is used.
    update_ops: An optional list of updates to execute. If `update_ops` is
      `None`, then the update ops are set to the contents of the
      `tf.GraphKeys.UPDATE_OPS` collection. If `update_ops` is not `None`, but
      it doesn't contain all of the update ops in `tf.GraphKeys.UPDATE_OPS`, a
      warning will be displayed.
    variables_to_train: an optional list of variables to train. If None, it will
      default to all tf.compat.v1.trainable_variables().
    transform_grads_fn: A function which takes a single argument, a list of
      gradient to variable pairs (tuples), performs any requested gradient
      updates, such as gradient clipping or multipliers, and returns the updated
      list.
    summarize_gradients: Whether or not add summaries for each gradient.
    gate_gradients: How to gate the computation of gradients. See tf.Optimizer.
    aggregation_method: Specifies the method used to combine gradient terms.
      Valid values are defined in the class `AggregationMethod`.
    colocate_gradients_with_ops: Whether or not to try colocating the gradients
      with the ops that generated them.
    check_numerics: Whether or not we apply check_numerics.

  Returns:
    A `Tensor` that when evaluated, computes the gradients and returns the total
      loss value.
  "
  [total_loss optimizer & {:keys [global_step update_ops variables_to_train transform_grads_fn summarize_gradients gate_gradients aggregation_method colocate_gradients_with_ops check_numerics]
                       :or {update_ops None variables_to_train None transform_grads_fn None aggregation_method None}} ]
    (py/call-attr-kw training "create_train_op" [total_loss optimizer] {:global_step global_step :update_ops update_ops :variables_to_train variables_to_train :transform_grads_fn transform_grads_fn :summarize_gradients summarize_gradients :gate_gradients gate_gradients :aggregation_method aggregation_method :colocate_gradients_with_ops colocate_gradients_with_ops :check_numerics check_numerics }))

(defn evaluate-once 
  "Evaluates the model at the given checkpoint path.

  During a single evaluation, the `eval_ops` is run until the session is
  interrupted or requested to finish. This is typically requested via a
  `tf.contrib.training.StopAfterNEvalsHook` which results in `eval_ops` running
  the requested number of times.

  Optionally, a user can pass in `final_ops`, a single `Tensor`, a list of
  `Tensors` or a dictionary from names to `Tensors`. The `final_ops` is
  evaluated a single time after `eval_ops` has finished running and the fetched
  values of `final_ops` are returned. If `final_ops` is left as `None`, then
  `None` is returned.

  One may also consider using a `tf.contrib.training.SummaryAtEndHook` to record
  summaries after the `eval_ops` have run. If `eval_ops` is `None`, the
  summaries run immediately after the model checkpoint has been restored.

  Note that `evaluate_once` creates a local variable used to track the number of
  evaluations run via `tf.contrib.training.get_or_create_eval_step`.
  Consequently, if a custom local init op is provided via a `scaffold`, the
  caller should ensure that the local init op also initializes the eval step.

  Args:
    checkpoint_path: The path to a checkpoint to use for evaluation.
    master: The BNS address of the TensorFlow master.
    scaffold: An tf.compat.v1.train.Scaffold instance for initializing variables
      and restoring variables. Note that `scaffold.init_fn` is used by the
      function to restore the checkpoint. If you supply a custom init_fn, then
      it must also take care of restoring the model from its checkpoint.
    eval_ops: A single `Tensor`, a list of `Tensors` or a dictionary of names to
      `Tensors`, which is run until the session is requested to stop, commonly
      done by a `tf.contrib.training.StopAfterNEvalsHook`.
    feed_dict: The feed dictionary to use when executing the `eval_ops`.
    final_ops: A single `Tensor`, a list of `Tensors` or a dictionary of names
      to `Tensors`.
    final_ops_feed_dict: A feed dictionary to use when evaluating `final_ops`.
    hooks: List of `tf.estimator.SessionRunHook` callbacks which are run inside
      the evaluation loop.
    config: An instance of `tf.compat.v1.ConfigProto` that will be used to
      configure the `Session`. If left as `None`, the default will be used.

  Returns:
    The fetched values of `final_ops` or `None` if `final_ops` is `None`.
  "
  [checkpoint_path & {:keys [master scaffold eval_ops feed_dict final_ops final_ops_feed_dict hooks config]
                       :or {scaffold None eval_ops None feed_dict None final_ops None final_ops_feed_dict None hooks None config None}} ]
    (py/call-attr-kw training "evaluate_once" [checkpoint_path] {:master master :scaffold scaffold :eval_ops eval_ops :feed_dict feed_dict :final_ops final_ops :final_ops_feed_dict final_ops_feed_dict :hooks hooks :config config }))

(defn evaluate-repeatedly 
  "Repeatedly searches for a checkpoint in `checkpoint_dir` and evaluates it.

  During a single evaluation, the `eval_ops` is run until the session is
  interrupted or requested to finish. This is typically requested via a
  `tf.contrib.training.StopAfterNEvalsHook` which results in `eval_ops` running
  the requested number of times.

  Optionally, a user can pass in `final_ops`, a single `Tensor`, a list of
  `Tensors` or a dictionary from names to `Tensors`. The `final_ops` is
  evaluated a single time after `eval_ops` has finished running and the fetched
  values of `final_ops` are returned. If `final_ops` is left as `None`, then
  `None` is returned.

  One may also consider using a `tf.contrib.training.SummaryAtEndHook` to record
  summaries after the `eval_ops` have run. If `eval_ops` is `None`, the
  summaries run immediately after the model checkpoint has been restored.

  Note that `evaluate_once` creates a local variable used to track the number of
  evaluations run via `tf.contrib.training.get_or_create_eval_step`.
  Consequently, if a custom local init op is provided via a `scaffold`, the
  caller should ensure that the local init op also initializes the eval step.

  Args:
    checkpoint_dir: The directory where checkpoints are stored.
    master: The address of the TensorFlow master.
    scaffold: An tf.compat.v1.train.Scaffold instance for initializing variables
      and restoring variables. Note that `scaffold.init_fn` is used by the
      function to restore the checkpoint. If you supply a custom init_fn, then
      it must also take care of restoring the model from its checkpoint.
    eval_ops: A single `Tensor`, a list of `Tensors` or a dictionary of names to
      `Tensors`, which is run until the session is requested to stop, commonly
      done by a `tf.contrib.training.StopAfterNEvalsHook`.
    feed_dict: The feed dictionary to use when executing the `eval_ops`.
    final_ops: A single `Tensor`, a list of `Tensors` or a dictionary of names
      to `Tensors`.
    final_ops_feed_dict: A feed dictionary to use when evaluating `final_ops`.
    eval_interval_secs: The minimum number of seconds between evaluations.
    hooks: List of `tf.estimator.SessionRunHook` callbacks which are run inside
      the evaluation loop.
    config: An instance of `tf.compat.v1.ConfigProto` that will be used to
      configure the `Session`. If left as `None`, the default will be used.
    max_number_of_evaluations: The maximum times to run the evaluation. If left
      as `None`, then evaluation runs indefinitely.
    timeout: The maximum number of seconds to wait between checkpoints. If left
      as `None`, then the process will wait indefinitely.
    timeout_fn: Optional function to call after a timeout.  If the function
      returns True, then it means that no new checkpoints will be generated and
      the iterator will exit.  The function is called with no arguments.

  Returns:
    The fetched values of `final_ops` or `None` if `final_ops` is `None`.
  "
  [checkpoint_dir & {:keys [master scaffold eval_ops feed_dict final_ops final_ops_feed_dict eval_interval_secs hooks config max_number_of_evaluations timeout timeout_fn]
                       :or {scaffold None eval_ops None feed_dict None final_ops None final_ops_feed_dict None hooks None config None max_number_of_evaluations None timeout None timeout_fn None}} ]
    (py/call-attr-kw training "evaluate_repeatedly" [checkpoint_dir] {:master master :scaffold scaffold :eval_ops eval_ops :feed_dict feed_dict :final_ops final_ops :final_ops_feed_dict final_ops_feed_dict :eval_interval_secs eval_interval_secs :hooks hooks :config config :max_number_of_evaluations max_number_of_evaluations :timeout timeout :timeout_fn timeout_fn }))

(defn get-or-create-eval-step 
  "Gets or creates the eval step `Tensor`.

  Returns:
    A `Tensor` representing a counter for the evaluation step.

  Raises:
    ValueError: If multiple `Tensors` have been added to the
      `tf.GraphKeys.EVAL_STEP` collection.
  "
  [  ]
  (py/call-attr training "get_or_create_eval_step"  ))

(defn multiply-gradients 
  "Multiply specified gradients.

  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).
    gradient_multipliers: A map from either `Variables` or `Variable` op names
      to the coefficient by which the associated gradient should be scaled.

  Returns:
    The updated list of gradient to variable pairs.

  Raises:
    ValueError: If `grads_and_vars` is not a list or if `gradient_multipliers`
    is empty or None or if `gradient_multipliers` is not a dictionary.
  "
  [ grads_and_vars gradient_multipliers ]
  (py/call-attr training "multiply_gradients"  grads_and_vars gradient_multipliers ))
(defn parse-values 
  "Parses hyperparameter values from a string into a python map.

  `values` is a string containing comma-separated `name=value` pairs.
  For each pair, the value of the hyperparameter named `name` is set to
  `value`.

  If a hyperparameter name appears multiple times in `values`, a ValueError
  is raised (e.g. 'a=1,a=2', 'a[1]=1,a[1]=2').

  If a hyperparameter name in both an index assignment and scalar assignment,
  a ValueError is raised.  (e.g. 'a=[1,2,3],a[0] = 1').

  The hyperparameter name may contain '.' symbols, which will result in an
  attribute name that is only accessible through the getattr and setattr
  functions.  (And must be first explicit added through add_hparam.)

  WARNING: Use of '.' in your variable names is allowed, but is not well
  supported and not recommended.

  The `value` in `name=value` must follows the syntax according to the
  type of the parameter:

  *  Scalar integer: A Python-parsable integer point value.  E.g.: 1,
     100, -12.
  *  Scalar float: A Python-parsable floating point value.  E.g.: 1.0,
     -.54e89.
  *  Boolean: Either true or false.
  *  Scalar string: A non-empty sequence of characters, excluding comma,
     spaces, and square brackets.  E.g.: foo, bar_1.
  *  List: A comma separated list of scalar values of the parameter type
     enclosed in square brackets.  E.g.: [1,2,3], [1.0,1e-12], [high,low].

  When index assignment is used, the corresponding type_map key should be the
  list name.  E.g. for \"arr[1]=0\" the type_map must have the key \"arr\" (not
  \"arr[1]\").

  Args:
    values: String.  Comma separated list of `name=value` pairs where
      'value' must follow the syntax described above.
    type_map: A dictionary mapping hyperparameter names to types.  Note every
      parameter name in values must be a key in type_map.  The values must
      conform to the types indicated, where a value V is said to conform to a
      type T if either V has type T, or V is a list of elements of type T.
      Hence, for a multidimensional parameter 'x' taking float values,
      'x=[0.1,0.2]' will parse successfully if type_map['x'] = float.
    ignore_unknown: Bool. Whether values that are missing a type in type_map
      should be ignored. If set to True, a ValueError will not be raised for
      unknown hyperparameter type.

  Returns:
    A python map mapping each name to either:
    * A scalar value.
    * A list of scalar values.
    * A dictionary mapping index numbers to scalar values.
    (e.g. \"x=5,L=[1,2],arr[1]=3\" results in {'x':5,'L':[1,2],'arr':{1:3}}\")

  Raises:
    ValueError: If there is a problem with input.
    * If `values` cannot be parsed.
    * If a list is assigned to a list index (e.g. 'a[1] = [1,2,3]').
    * If the same rvalue is assigned two different values (e.g. 'a=1,a=2',
      'a[1]=1,a[1]=2', or 'a=1,a=[1]')
  "
  [values type_map  & {:keys [ignore_unknown]} ]
    (py/call-attr-kw training "parse_values" [values type_map] {:ignore_unknown ignore_unknown }))

(defn rejection-sample 
  "Stochastically creates batches by rejection sampling.

  Each list of non-batched tensors is evaluated by `accept_prob_fn`, to produce
  a scalar tensor between 0 and 1. This tensor corresponds to the probability of
  being accepted. When `batch_size` tensor groups have been accepted, the batch
  queue will return a mini-batch.

  Args:
    tensors: List of tensors for data. All tensors are either one item or a
        batch, according to enqueue_many.
    accept_prob_fn: A python lambda that takes a non-batch tensor from each
        item in `tensors`, and produces a scalar tensor.
    batch_size: Size of batch to be returned.
    queue_threads: The number of threads for the queue that will hold the final
      batch.
    enqueue_many: Bool. If true, interpret input tensors as having a batch
        dimension.
    prebatch_capacity: Capacity for the large queue that is used to convert
      batched tensors to single examples.
    prebatch_threads: Number of threads for the large queue that is used to
      convert batched tensors to single examples.
    runtime_checks: Bool. If true, insert runtime checks on the output of
        `accept_prob_fn`. Using `True` might have a performance impact.
    name: Optional prefix for ops created by this function.
  Raises:
    ValueError: enqueue_many is True and labels doesn't have a batch
        dimension, or if enqueue_many is False and labels isn't a scalar.
    ValueError: enqueue_many is True, and batch dimension on data and labels
        don't match.
    ValueError: if a zero initial probability class has a nonzero target
        probability.
  Returns:
    A list of tensors of the same length as `tensors`, with batch dimension
    `batch_size`.

  Example:
    # Get tensor for a single data and label example.
    data, label = data_provider.Get(['data', 'label'])

    # Get stratified batch according to data tensor.
    accept_prob_fn = lambda x: (tf.tanh(x[0]) + 1) / 2
    data_batch = tf.contrib.training.rejection_sample(
        [data, label], accept_prob_fn, 16)

    # Run batch through network.
    ...
  "
  [tensors accept_prob_fn batch_size & {:keys [queue_threads enqueue_many prebatch_capacity prebatch_threads runtime_checks name]
                       :or {name None}} ]
    (py/call-attr-kw training "rejection_sample" [tensors accept_prob_fn batch_size] {:queue_threads queue_threads :enqueue_many enqueue_many :prebatch_capacity prebatch_capacity :prebatch_threads prebatch_threads :runtime_checks runtime_checks :name name }))
(defn resample-at-rate 
  "Given `inputs` tensors, stochastically resamples each at a given rate.

  For example, if the inputs are `[[a1, a2], [b1, b2]]` and the rates
  tensor contains `[3, 1]`, then the return value may look like `[[a1,
  a2, a1, a1], [b1, b2, b1, b1]]`. However, many other outputs are
  possible, since this is stochastic -- averaged over many repeated
  calls, each set of inputs should appear in the output `rate` times
  the number of invocations.

  Args:
    inputs: A list of tensors, each of which has a shape of `[batch_size, ...]`
    rates: A tensor of shape `[batch_size]` containing the resampling rates
       for each input.
    scope: Scope for the op.
    seed: Random seed to use.
    back_prop: Whether to allow back-propagation through this op.

  Returns:
    Selections from the input tensors.
  "
  [inputs rates scope seed  & {:keys [back_prop]} ]
    (py/call-attr-kw training "resample_at_rate" [inputs rates scope seed] {:back_prop back_prop }))

(defn stratified-sample 
  "Stochastically creates batches based on per-class probabilities.

  This method discards examples. Internally, it creates one queue to amortize
  the cost of disk reads, and one queue to hold the properly-proportioned
  batch.

  Args:
    tensors: List of tensors for data. All tensors are either one item or a
        batch, according to enqueue_many.
    labels: Tensor for label of data. Label is a single integer or a batch,
        depending on `enqueue_many`. It is not a one-hot vector.
    target_probs: Target class proportions in batch. An object whose type has a
        registered Tensor conversion function.
    batch_size: Size of batch to be returned.
    init_probs: Class proportions in the data. An object whose type has a
        registered Tensor conversion function, or `None` for estimating the
        initial distribution.
    enqueue_many: Bool. If true, interpret input tensors as having a batch
        dimension.
    queue_capacity: Capacity of the large queue that holds input examples.
    threads_per_queue: Number of threads for the large queue that holds input
        examples and for the final queue with the proper class proportions.
    name: Optional prefix for ops created by this function.
  Raises:
    ValueError: If `tensors` isn't iterable.
    ValueError: `enqueue_many` is True and labels doesn't have a batch
        dimension, or if `enqueue_many` is False and labels isn't a scalar.
    ValueError: `enqueue_many` is True, and batch dimension on data and labels
        don't match.
    ValueError: if probs don't sum to one.
    ValueError: if a zero initial probability class has a nonzero target
        probability.
    TFAssertion: if labels aren't integers in [0, num classes).
  Returns:
    (data_batch, label_batch), where data_batch is a list of tensors of the same
        length as `tensors`

  Example:
    # Get tensor for a single data and label example.
    data, label = data_provider.Get(['data', 'label'])

    # Get stratified batch according to per-class probabilities.
    target_probs = [...distribution you want...]
    [data_batch], labels = tf.contrib.training.stratified_sample(
        [data], label, target_probs)

    # Run batch through network.
    ...
  "
  [tensors labels target_probs batch_size init_probs & {:keys [enqueue_many queue_capacity threads_per_queue name]
                       :or {name None}} ]
    (py/call-attr-kw training "stratified_sample" [tensors labels target_probs batch_size init_probs] {:enqueue_many enqueue_many :queue_capacity queue_capacity :threads_per_queue threads_per_queue :name name }))

(defn train 
  "Runs the training loop.

  Args:
    train_op: A `Tensor` that, when executed, will apply the gradients and
      return the loss value.
    logdir: The directory where the graph and checkpoints are saved.
    master: The URL of the master.
    is_chief: Specifies whether or not the training is being run by the primary
      replica during replica training.
    scaffold: An tf.compat.v1.train.Scaffold instance.
    hooks: List of `tf.estimator.SessionRunHook` callbacks which are run inside
      the training loop.
    chief_only_hooks: List of `tf.estimator.SessionRunHook` instances which are
      run inside the training loop for the chief trainer only.
    save_checkpoint_secs: The frequency, in seconds, that a checkpoint is saved
      using a default checkpoint saver. If `save_checkpoint_secs` is set to
      `None`, then the default checkpoint saver isn't used.
    save_summaries_steps: The frequency, in number of global steps, that the
      summaries are written to disk using a default summary saver. If
      `save_summaries_steps` is set to `None`, then the default summary saver
      isn't used.
    config: An instance of `tf.compat.v1.ConfigProto`.
    max_wait_secs: Maximum time workers should wait for the session to become
      available. This should be kept relatively short to help detect incorrect
      code, but sometimes may need to be increased if the chief takes a while to
      start up.
    run_metadata: A [`RunMetadata`] protocol buffer.

  Returns:
    the value of the loss function after training.

  Raises:
    ValueError: if `logdir` is `None` and either `save_checkpoint_secs` or
    `save_summaries_steps` are `None.
  "
  [train_op logdir & {:keys [master is_chief scaffold hooks chief_only_hooks save_checkpoint_secs save_summaries_steps config max_wait_secs run_metadata]
                       :or {scaffold None hooks None chief_only_hooks None config None run_metadata None}} ]
    (py/call-attr-kw training "train" [train_op logdir] {:master master :is_chief is_chief :scaffold scaffold :hooks hooks :chief_only_hooks chief_only_hooks :save_checkpoint_secs save_checkpoint_secs :save_summaries_steps save_summaries_steps :config config :max_wait_secs max_wait_secs :run_metadata run_metadata }))

(defn wait-for-new-checkpoint 
  "Waits until a new checkpoint file is found.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    last_checkpoint: The last checkpoint path used or `None` if we're expecting
      a checkpoint for the first time.
    seconds_to_sleep: The number of seconds to sleep for before looking for a
      new checkpoint.
    timeout: The maximum number of seconds to wait. If left as `None`, then the
      process will wait indefinitely.

  Returns:
    a new checkpoint path, or None if the timeout was reached.
  "
  [checkpoint_dir last_checkpoint & {:keys [seconds_to_sleep timeout]
                       :or {timeout None}} ]
    (py/call-attr-kw training "wait_for_new_checkpoint" [checkpoint_dir last_checkpoint] {:seconds_to_sleep seconds_to_sleep :timeout timeout }))

(defn weighted-resample 
  "Performs an approximate weighted resampling of `inputs`.

  This method chooses elements from `inputs` where each item's rate of
  selection is proportional to its value in `weights`, and the average
  rate of selection across all inputs (and many invocations!) is
  `overall_rate`.

  Args:
    inputs: A list of tensors whose first dimension is `batch_size`.
    weights: A `[batch_size]`-shaped tensor with each batch member's weight.
    overall_rate: Desired overall rate of resampling.
    scope: Scope to use for the op.
    mean_decay: How quickly to decay the running estimate of the mean weight.
    seed: Random seed.

  Returns:
    A list of tensors exactly like `inputs`, but with an unknown (and
      possibly zero) first dimension.
    A tensor containing the effective resampling rate used for each output.
  "
  [inputs weights overall_rate scope & {:keys [mean_decay seed]
                       :or {seed None}} ]
    (py/call-attr-kw training "weighted_resample" [inputs weights overall_rate scope] {:mean_decay mean_decay :seed seed }))
