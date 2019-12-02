(ns tensorflow.-api.v1.compat.v1.train
  "Support for training models.

See the [Training](https://tensorflow.org/api_guides/python/train) guide.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce train (import-module "tensorflow._api.v1.compat.v1.train"))

(defn MonitoredTrainingSession 
  "Creates a `MonitoredSession` for training.

  For a chief, this utility sets proper session initializer/restorer. It also
  creates hooks related to checkpoint and summary saving. For workers, this
  utility sets proper session creator which waits for the chief to
  initialize/restore. Please check `tf.compat.v1.train.MonitoredSession` for
  more
  information.


  Args:
    master: `String` the TensorFlow master to use.
    is_chief: If `True`, it will take care of initialization and recovery the
      underlying TensorFlow session. If `False`, it will wait on a chief to
      initialize or recover the TensorFlow session.
    checkpoint_dir: A string.  Optional path to a directory where to restore
      variables.
    scaffold: A `Scaffold` used for gathering or building supportive ops. If not
      specified, a default one is created. It's used to finalize the graph.
    hooks: Optional list of `SessionRunHook` objects.
    chief_only_hooks: list of `SessionRunHook` objects. Activate these hooks if
      `is_chief==True`, ignore otherwise.
    save_checkpoint_secs: The frequency, in seconds, that a checkpoint is saved
      using a default checkpoint saver. If both `save_checkpoint_steps` and
      `save_checkpoint_secs` are set to `None`, then the default checkpoint
      saver isn't used. If both are provided, then only `save_checkpoint_secs`
      is used. Default 600.
    save_summaries_steps: The frequency, in number of global steps, that the
      summaries are written to disk using a default summary saver. If both
      `save_summaries_steps` and `save_summaries_secs` are set to `None`, then
      the default summary saver isn't used. Default 100.
    save_summaries_secs: The frequency, in secs, that the summaries are written
      to disk using a default summary saver.  If both `save_summaries_steps` and
      `save_summaries_secs` are set to `None`, then the default summary saver
      isn't used. Default not enabled.
    config: an instance of `tf.compat.v1.ConfigProto` proto used to configure
      the session. It's the `config` argument of constructor of
      `tf.compat.v1.Session`.
    stop_grace_period_secs: Number of seconds given to threads to stop after
      `close()` has been called.
    log_step_count_steps: The frequency, in number of global steps, that the
      global step/sec is logged.
    max_wait_secs: Maximum time workers should wait for the session to become
      available. This should be kept relatively short to help detect incorrect
      code, but sometimes may need to be increased if the chief takes a while to
      start up.
    save_checkpoint_steps: The frequency, in number of global steps, that a
      checkpoint is saved using a default checkpoint saver. If both
      `save_checkpoint_steps` and `save_checkpoint_secs` are set to `None`, then
      the default checkpoint saver isn't used. If both are provided, then only
      `save_checkpoint_secs` is used. Default not enabled.
    summary_dir: A string.  Optional path to a directory where to save
      summaries. If None, checkpoint_dir is used instead.

  Returns:
    A `MonitoredSession` object.
  "
  [ & {:keys [master is_chief checkpoint_dir scaffold hooks chief_only_hooks save_checkpoint_secs save_summaries_steps save_summaries_secs config stop_grace_period_secs log_step_count_steps max_wait_secs save_checkpoint_steps summary_dir]
       :or {checkpoint_dir None scaffold None hooks None chief_only_hooks None config None summary_dir None}} ]
  
   (py/call-attr-kw train "MonitoredTrainingSession" [] {:master master :is_chief is_chief :checkpoint_dir checkpoint_dir :scaffold scaffold :hooks hooks :chief_only_hooks chief_only_hooks :save_checkpoint_secs save_checkpoint_secs :save_summaries_steps save_summaries_steps :save_summaries_secs save_summaries_secs :config config :stop_grace_period_secs stop_grace_period_secs :log_step_count_steps log_step_count_steps :max_wait_secs max_wait_secs :save_checkpoint_steps save_checkpoint_steps :summary_dir summary_dir }))

(defn NewCheckpointReader 
  ""
  [ filepattern ]
  (py/call-attr train "NewCheckpointReader"  filepattern ))
(defn add-queue-runner 
  "Adds a `QueueRunner` to a collection in the graph. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
To construct input pipelines, use the `tf.data` module.

When building a complex model that uses many queues it is often difficult to
gather all the queue runners that need to be run.  This convenience function
allows you to add a queue runner to a well known collection in the graph.

The companion method `start_queue_runners()` can be used to start threads for
all the collected queue runners.

Args:
  qr: A `QueueRunner`.
  collection: A `GraphKey` specifying the graph collection to add
    the queue runner to.  Defaults to `GraphKeys.QUEUE_RUNNERS`."
  [qr  & {:keys [collection]} ]
    (py/call-attr-kw train "add_queue_runner" [qr] {:collection collection }))

(defn assert-global-step 
  "Asserts `global_step_tensor` is a scalar int `Variable` or `Tensor`.

  Args:
    global_step_tensor: `Tensor` to test.
  "
  [ global_step_tensor ]
  (py/call-attr train "assert_global_step"  global_step_tensor ))
(defn basic-train-loop 
  "Basic loop to train a model.

  Calls `train_step_fn` in a loop to train a model.  The function is called as:

  ```python
  train_step_fn(session, *args, **kwargs)
  ```

  It is passed a `tf.compat.v1.Session` in addition to `args` and `kwargs`.  The
  function
  typically runs one training step in the session.

  Args:
    supervisor: `tf.compat.v1.train.Supervisor` to run the training services.
    train_step_fn: Callable to execute one training step.  Called repeatedly as
      `train_step_fn(session, *args **kwargs)`.
    args: Optional positional arguments passed to `train_step_fn`.
    kwargs: Optional keyword arguments passed to `train_step_fn`.
    master: Master to use to create the training session.  Defaults to `\"\"`
      which causes the session to be created in the local process.
  "
  [supervisor train_step_fn args kwargs  & {:keys [master]} ]
    (py/call-attr-kw train "basic_train_loop" [supervisor train_step_fn args kwargs] {:master master }))

(defn batch 
  "Creates batches of tensors in `tensors`. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.batch(batch_size)` (or `padded_batch(...)` if `dynamic_pad=True`).

The argument `tensors` can be a list or a dictionary of tensors.
The value returned by the function will be of the same type
as `tensors`.

This function is implemented using a queue. A `QueueRunner` for the
queue is added to the current `Graph`'s `QUEUE_RUNNER` collection.

If `enqueue_many` is `False`, `tensors` is assumed to represent a single
example.  An input tensor with shape `[x, y, z]` will be output as a tensor
with shape `[batch_size, x, y, z]`.

If `enqueue_many` is `True`, `tensors` is assumed to represent a batch of
examples, where the first dimension is indexed by example, and all members of
`tensors` should have the same size in the first dimension.  If an input
tensor has shape `[*, x, y, z]`, the output will have shape `[batch_size, x,
y, z]`.  The `capacity` argument controls the how long the prefetching is
allowed to grow the queues.

The returned operation is a dequeue operation and will throw
`tf.errors.OutOfRangeError` if the input queue is exhausted. If this
operation is feeding another input queue, its queue runner will catch
this exception, however, if this operation is used in your main thread
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
`batch_size` is returned when the queue is closed and there are not enough
elements to fill the batch, otherwise the pending elements are discarded.
In addition, all output tensors' static shapes, as accessed via the
`shape` property will have a first `Dimension` value of `None`, and
operations that depend on fixed batch_size would fail.

Args:
  tensors: The list or dictionary of tensors to enqueue.
  batch_size: The new batch size pulled from the queue.
  num_threads: The number of threads enqueuing `tensors`.  The batching will
    be nondeterministic if `num_threads > 1`.
  capacity: An integer. The maximum number of elements in the queue.
  enqueue_many: Whether each tensor in `tensors` is a single example.
  shapes: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensors`.
  dynamic_pad: Boolean.  Allow variable dimensions in input shapes.
    The given dimensions are padded upon dequeue so that tensors within a
    batch have the same shapes.
  allow_smaller_final_batch: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
  shared_name: (Optional). If set, this queue will be shared under the given
    name across multiple sessions.
  name: (Optional) A name for the operations.

Returns:
  A list or dictionary of tensors with the same types as `tensors` (except if
  the input is a list of one element, then it returns a tensor, not a list).

Raises:
  ValueError: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensors`.

@compatibility(eager)
Input pipelines based on Queues are not supported when eager execution is
enabled. Please use the `tf.data` API to ingest data under eager execution.
@end_compatibility"
  [tensors batch_size & {:keys [num_threads capacity enqueue_many shapes dynamic_pad allow_smaller_final_batch shared_name name]
                       :or {shapes None shared_name None name None}} ]
    (py/call-attr-kw train "batch" [tensors batch_size] {:num_threads num_threads :capacity capacity :enqueue_many enqueue_many :shapes shapes :dynamic_pad dynamic_pad :allow_smaller_final_batch allow_smaller_final_batch :shared_name shared_name :name name }))

(defn batch-join 
  "Runs a list of tensors to fill a queue to create batches of examples. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.interleave(...).batch(batch_size)` (or `padded_batch(...)` if `dynamic_pad=True`).

The `tensors_list` argument is a list of tuples of tensors, or a list of
dictionaries of tensors.  Each element in the list is treated similarly
to the `tensors` argument of `tf.compat.v1.train.batch()`.

WARNING: This function is nondeterministic, since it starts a separate thread
for each tensor.

Enqueues a different list of tensors in different threads.
Implemented using a queue -- a `QueueRunner` for the queue
is added to the current `Graph`'s `QUEUE_RUNNER` collection.

`len(tensors_list)` threads will be started,
with thread `i` enqueuing the tensors from
`tensors_list[i]`. `tensors_list[i1][j]` must match
`tensors_list[i2][j]` in type and shape, except in the first
dimension if `enqueue_many` is true.

If `enqueue_many` is `False`, each `tensors_list[i]` is assumed
to represent a single example. An input tensor `x` will be output as a
tensor with shape `[batch_size] + x.shape`.

If `enqueue_many` is `True`, `tensors_list[i]` is assumed to
represent a batch of examples, where the first dimension is indexed
by example, and all members of `tensors_list[i]` should have the
same size in the first dimension.  The slices of any input tensor
`x` are treated as examples, and the output tensors will have shape
`[batch_size] + x.shape[1:]`.

The `capacity` argument controls the how long the prefetching is allowed to
grow the queues.

The returned operation is a dequeue operation and will throw
`tf.errors.OutOfRangeError` if the input queue is exhausted. If this
operation is feeding another input queue, its queue runner will catch
this exception, however, if this operation is used in your main thread
you are responsible for catching this yourself.

*N.B.:* If `dynamic_pad` is `False`, you must ensure that either
(i) the `shapes` argument is passed, or (ii) all of the tensors in
`tensors_list` must have fully-defined shapes. `ValueError` will be
raised if neither of these conditions holds.

If `dynamic_pad` is `True`, it is sufficient that the *rank* of the
tensors is known, but individual dimensions may have value `None`.
In this case, for each enqueue the dimensions with value `None`
may have a variable length; upon dequeue, the output tensors will be padded
on the right to the maximum shape of the tensors in the current minibatch.
For numbers, this padding takes value 0.  For strings, this padding is
the empty string.  See `PaddingFIFOQueue` for more info.

If `allow_smaller_final_batch` is `True`, a smaller batch value than
`batch_size` is returned when the queue is closed and there are not enough
elements to fill the batch, otherwise the pending elements are discarded.
In addition, all output tensors' static shapes, as accessed via the
`shape` property will have a first `Dimension` value of `None`, and
operations that depend on fixed batch_size would fail.

Args:
  tensors_list: A list of tuples or dictionaries of tensors to enqueue.
  batch_size: An integer. The new batch size pulled from the queue.
  capacity: An integer. The maximum number of elements in the queue.
  enqueue_many: Whether each tensor in `tensor_list_list` is a single
    example.
  shapes: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensor_list_list[i]`.
  dynamic_pad: Boolean.  Allow variable dimensions in input shapes.
    The given dimensions are padded upon dequeue so that tensors within a
    batch have the same shapes.
  allow_smaller_final_batch: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
  shared_name: (Optional) If set, this queue will be shared under the given
    name across multiple sessions.
  name: (Optional) A name for the operations.

Returns:
  A list or dictionary of tensors with the same number and types as
  `tensors_list[i]`.

Raises:
  ValueError: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensor_list_list`.

@compatibility(eager)
Input pipelines based on Queues are not supported when eager execution is
enabled. Please use the `tf.data` API to ingest data under eager execution.
@end_compatibility"
  [tensors_list batch_size & {:keys [capacity enqueue_many shapes dynamic_pad allow_smaller_final_batch shared_name name]
                       :or {shapes None shared_name None name None}} ]
    (py/call-attr-kw train "batch_join" [tensors_list batch_size] {:capacity capacity :enqueue_many enqueue_many :shapes shapes :dynamic_pad dynamic_pad :allow_smaller_final_batch allow_smaller_final_batch :shared_name shared_name :name name }))

(defn checkpoint-exists 
  "Checks whether a V1 or V2 checkpoint exists with the specified prefix. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use standard file APIs to check for files with this prefix.

This is the recommended way to check if a checkpoint exists, since it takes
into account the naming difference between V1 and V2 formats.

Args:
  checkpoint_prefix: the prefix of a V1 or V2 checkpoint, with V2 taking
    priority.  Typically the result of `Saver.save()` or that of
    `tf.train.latest_checkpoint()`, regardless of sharded/non-sharded or
    V1/V2.

Returns:
  A bool, true if a checkpoint referred to by `checkpoint_prefix` exists."
  [ checkpoint_prefix ]
  (py/call-attr train "checkpoint_exists"  checkpoint_prefix ))

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
    (py/call-attr-kw train "checkpoints_iterator" [checkpoint_dir] {:min_interval_secs min_interval_secs :timeout timeout :timeout_fn timeout_fn }))

(defn cosine-decay 
  "Applies cosine decay to the learning rate.

  See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent
  with Warm Restarts. https://arxiv.org/abs/1608.03983

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies a cosine decay function
  to a provided initial learning rate.  It requires a `global_step` value to
  compute the decayed learning rate.  You can just pass a TensorFlow variable
  that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:
  ```python
  global_step = min(global_step, decay_steps)
  cosine_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))
  decayed = (1 - alpha) * cosine_decay + alpha
  decayed_learning_rate = learning_rate * decayed
  ```

  Example usage:
  ```python
  decay_steps = 1000
  lr_decayed = cosine_decay(learning_rate, global_step, decay_steps)
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` Tensor or a Python number.
      The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number. Global
      step to use for the decay computation.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number. Number
      of steps to decay over.
    alpha: A scalar `float32` or `float64` Tensor or a Python number. Minimum
      learning rate value as a fraction of learning_rate.
    name: String. Optional name of the operation.  Defaults to 'CosineDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.
  Raises:
    ValueError: if `global_step` is not supplied.

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  "
  [learning_rate global_step decay_steps & {:keys [alpha name]
                       :or {name None}} ]
    (py/call-attr-kw train "cosine_decay" [learning_rate global_step decay_steps] {:alpha alpha :name name }))

(defn cosine-decay-restarts 
  "Applies cosine decay with restarts to the learning rate.

  See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent
  with Warm Restarts. https://arxiv.org/abs/1608.03983

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies a cosine decay function with
  restarts to a provided initial learning rate.  It requires a `global_step`
  value to compute the decayed learning rate.  You can just pass a TensorFlow
  variable that you increment at each training step.

  The function returns the decayed learning rate while taking into account
  possible warm restarts. The learning rate multiplier first decays
  from 1 to `alpha` for `first_decay_steps` steps. Then, a warm
  restart is performed. Each new warm restart runs for `t_mul` times more steps
  and with `m_mul` times smaller initial learning rate.

  Example usage:
  ```python
  first_decay_steps = 1000
  lr_decayed = cosine_decay_restarts(learning_rate, global_step,
                                     first_decay_steps)
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` Tensor or a Python number.
      The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number. Global
      step to use for the decay computation.
    first_decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
      Number of steps to decay over.
    t_mul: A scalar `float32` or `float64` `Tensor` or a Python number. Used to
      derive the number of iterations in the i-th period
    m_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
      Used to derive the initial learning rate of the i-th period:
    alpha: A scalar `float32` or `float64` Tensor or a Python number. Minimum
      learning rate value as a fraction of the learning_rate.
    name: String. Optional name of the operation.  Defaults to 'SGDRDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.
  Raises:
    ValueError: if `global_step` is not supplied.

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  "
  [learning_rate global_step first_decay_steps & {:keys [t_mul m_mul alpha name]
                       :or {name None}} ]
    (py/call-attr-kw train "cosine_decay_restarts" [learning_rate global_step first_decay_steps] {:t_mul t_mul :m_mul m_mul :alpha alpha :name name }))

(defn create-global-step 
  "Create global step tensor in graph.

  Args:
    graph: The graph in which to create the global step tensor. If missing, use
      default graph.

  Returns:
    Global step tensor.

  Raises:
    ValueError: if global step tensor is already defined.
  "
  [ graph ]
  (py/call-attr train "create_global_step"  graph ))

(defn do-quantize-training-on-graphdef 
  "A general quantization scheme is being developed in `tf.contrib.quantize`. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
GraphDef quantized training rewriter is deprecated in the long term

Consider using that instead, though since it is in the tf.contrib namespace,
it is not subject to backward compatibility guarantees."
  [ input_graph num_bits ]
  (py/call-attr train "do_quantize_training_on_graphdef"  input_graph num_bits ))

(defn exponential-decay 
  "Applies exponential decay to the learning rate.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies an exponential decay function
  to a provided initial learning rate.  It requires a `global_step` value to
  compute the decayed learning rate.  You can just pass a TensorFlow variable
  that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:

  ```python
  decayed_learning_rate = learning_rate *
                          decay_rate ^ (global_step / decay_steps)
  ```

  If the argument `staircase` is `True`, then `global_step / decay_steps` is an
  integer division and the decayed learning rate follows a staircase function.

  Example: decay every 100000 steps with a base of 0.96:

  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.1
  learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate,
  global_step,
                                             100000, 0.96, staircase=True)
  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
      tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
      .minimize(...my loss..., global_step=global_step)
  )
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a Python number.
      The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number. Global
      step to use for the decay computation.  Must not be negative.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number. Must
      be positive.  See the decay computation above.
    decay_rate: A scalar `float32` or `float64` `Tensor` or a Python number.
      The decay rate.
    staircase: Boolean.  If `True` decay the learning rate at discrete intervals
    name: String.  Optional name of the operation.  Defaults to
      'ExponentialDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.

  Raises:
    ValueError: if `global_step` is not supplied.

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  "
  [learning_rate global_step decay_steps decay_rate & {:keys [staircase name]
                       :or {name None}} ]
    (py/call-attr-kw train "exponential_decay" [learning_rate global_step decay_steps decay_rate] {:staircase staircase :name name }))

(defn export-meta-graph 
  "Returns `MetaGraphDef` proto.

  Optionally writes it to filename.

  This function exports the graph, saver, and collection objects into
  `MetaGraphDef` protocol buffer with the intention of it being imported
  at a later time or location to restart training, run inference, or be
  a subgraph.

  Args:
    filename: Optional filename including the path for writing the generated
      `MetaGraphDef` protocol buffer.
    meta_info_def: `MetaInfoDef` protocol buffer.
    graph_def: `GraphDef` protocol buffer.
    saver_def: `SaverDef` protocol buffer.
    collection_list: List of string keys to collect.
    as_text: If `True`, writes the `MetaGraphDef` as an ASCII proto.
    graph: The `Graph` to export. If `None`, use the default graph.
    export_scope: Optional `string`. Name scope under which to extract the
      subgraph. The scope name will be striped from the node definitions for
      easy import later into new name scopes. If `None`, the whole graph is
      exported. graph_def and export_scope cannot both be specified.
    clear_devices: Whether or not to clear the device field for an `Operation`
      or `Tensor` during export.
    clear_extraneous_savers: Remove any Saver-related information from the graph
      (both Save/Restore ops and SaverDefs) that are not associated with the
      provided SaverDef.
    strip_default_attrs: Boolean. If `True`, default-valued attributes will be
      removed from the NodeDefs. For a detailed guide, see
      [Stripping Default-Valued Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).
    save_debug_info: If `True`, save the GraphDebugInfo to a separate file,
      which in the same directory of filename and with `_debug` added before the
      file extend.
    **kwargs: Optional keyed arguments.

  Returns:
    A `MetaGraphDef` proto.

  Raises:
    ValueError: When the `GraphDef` is larger than 2GB.
    RuntimeError: If called with eager execution enabled.

  @compatibility(eager)
  Exporting/importing meta graphs is not supported unless both `graph_def` and
  `graph` are provided. No graph exists when eager execution is enabled.
  @end_compatibility
  "
  [filename meta_info_def graph_def saver_def collection_list & {:keys [as_text graph export_scope clear_devices clear_extraneous_savers strip_default_attrs save_debug_info]
                       :or {graph None export_scope None}} ]
    (py/call-attr-kw train "export_meta_graph" [filename meta_info_def graph_def saver_def collection_list] {:as_text as_text :graph graph :export_scope export_scope :clear_devices clear_devices :clear_extraneous_savers clear_extraneous_savers :strip_default_attrs strip_default_attrs :save_debug_info save_debug_info }))

(defn generate-checkpoint-state-proto 
  "Generates a checkpoint state proto.

  Args:
    save_dir: Directory where the model was saved.
    model_checkpoint_path: The checkpoint file.
    all_model_checkpoint_paths: List of strings.  Paths to all not-yet-deleted
      checkpoints, sorted from oldest to newest.  If this is a non-empty list,
      the last element must be equal to model_checkpoint_path.  These paths
      are also saved in the CheckpointState proto.
    all_model_checkpoint_timestamps: A list of floats, indicating the number of
      seconds since the Epoch when each checkpoint was generated.
    last_preserved_timestamp: A float, indicating the number of seconds since
      the Epoch when the last preserved checkpoint was written, e.g. due to a
      `keep_checkpoint_every_n_hours` parameter (see
      `tf.contrib.checkpoint.CheckpointManager` for an implementation).
  Returns:
    CheckpointState proto with model_checkpoint_path and
    all_model_checkpoint_paths updated to either absolute paths or
    relative paths to the current save_dir.

  Raises:
    ValueError: If `all_model_checkpoint_timestamps` was provided but its length
      does not match `all_model_checkpoint_paths`.
  "
  [ save_dir model_checkpoint_path all_model_checkpoint_paths all_model_checkpoint_timestamps last_preserved_timestamp ]
  (py/call-attr train "generate_checkpoint_state_proto"  save_dir model_checkpoint_path all_model_checkpoint_paths all_model_checkpoint_timestamps last_preserved_timestamp ))

(defn get-checkpoint-mtimes 
  "Returns the mtimes (modification timestamps) of the checkpoints. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use standard file utilities to get mtimes.

Globs for the checkpoints pointed to by `checkpoint_prefixes`.  If the files
exist, collect their mtime.  Both V2 and V1 checkpoints are considered, in
that priority.

This is the recommended way to get the mtimes, since it takes into account
the naming difference between V1 and V2 formats.

Note: If not all checkpoints exist, the length of the returned mtimes list
will be smaller than the length of `checkpoint_prefixes` list, so mapping
checkpoints to corresponding mtimes will not be possible.

Args:
  checkpoint_prefixes: a list of checkpoint paths, typically the results of
    `Saver.save()` or those of `tf.train.latest_checkpoint()`, regardless of
    sharded/non-sharded or V1/V2.
Returns:
  A list of mtimes (in microseconds) of the found checkpoints."
  [ checkpoint_prefixes ]
  (py/call-attr train "get_checkpoint_mtimes"  checkpoint_prefixes ))

(defn get-checkpoint-state 
  "Returns CheckpointState proto from the \"checkpoint\" file.

  If the \"checkpoint\" file contains a valid CheckpointState
  proto, returns it.

  Args:
    checkpoint_dir: The directory of checkpoints.
    latest_filename: Optional name of the checkpoint file.  Default to
      'checkpoint'.

  Returns:
    A CheckpointState if the state was available, None
    otherwise.

  Raises:
    ValueError: if the checkpoint read doesn't have model_checkpoint_path set.
  "
  [ checkpoint_dir latest_filename ]
  (py/call-attr train "get_checkpoint_state"  checkpoint_dir latest_filename ))

(defn get-global-step 
  "Get the global step tensor.

  The global step tensor must be an integer variable. We first try to find it
  in the collection `GLOBAL_STEP`, or by name `global_step:0`.

  Args:
    graph: The graph to find the global step in. If missing, use default graph.

  Returns:
    The global step variable, or `None` if none was found.

  Raises:
    TypeError: If the global step tensor has a non-integer type, or if it is not
      a `Variable`.
  "
  [ graph ]
  (py/call-attr train "get_global_step"  graph ))

(defn get-or-create-global-step 
  "Returns and create (if necessary) the global step tensor.

  Args:
    graph: The graph in which to create the global step tensor. If missing, use
      default graph.

  Returns:
    The global step tensor.
  "
  [ graph ]
  (py/call-attr train "get_or_create_global_step"  graph ))

(defn global-step 
  "Small helper to get the global step.

  ```python
  # Create a variable to hold the global_step.
  global_step_tensor = tf.Variable(10, trainable=False, name='global_step')
  # Create a session.
  sess = tf.compat.v1.Session()
  # Initialize the variable
  sess.run(global_step_tensor.initializer)
  # Get the variable value.
  print('global_step: %s' % tf.compat.v1.train.global_step(sess,
  global_step_tensor))

  global_step: 10
  ```

  Args:
    sess: A TensorFlow `Session` object.
    global_step_tensor:  `Tensor` or the `name` of the operation that contains
      the global step.

  Returns:
    The global step value.
  "
  [ sess global_step_tensor ]
  (py/call-attr train "global_step"  sess global_step_tensor ))

(defn import-meta-graph 
  "Recreates a Graph saved in a `MetaGraphDef` proto.

  This function takes a `MetaGraphDef` protocol buffer as input. If
  the argument is a file containing a `MetaGraphDef` protocol buffer ,
  it constructs a protocol buffer from the file content. The function
  then adds all the nodes from the `graph_def` field to the
  current graph, recreates all the collections, and returns a saver
  constructed from the `saver_def` field.

  In combination with `export_meta_graph()`, this function can be used to

  * Serialize a graph along with other Python objects such as `QueueRunner`,
    `Variable` into a `MetaGraphDef`.

  * Restart training from a saved graph and checkpoints.

  * Run inference from a saved graph and checkpoints.

  ```Python
  ...
  # Create a saver.
  saver = tf.compat.v1.train.Saver(...variables...)
  # Remember the training_op we want to run by adding it to a collection.
  tf.compat.v1.add_to_collection('train_op', train_op)
  sess = tf.compat.v1.Session()
  for step in xrange(1000000):
      sess.run(train_op)
      if step % 1000 == 0:
          # Saves checkpoint, which by default also exports a meta_graph
          # named 'my-model-global_step.meta'.
          saver.save(sess, 'my-model', global_step=step)
  ```

  Later we can continue training from this saved `meta_graph` without building
  the model from scratch.

  ```Python
  with tf.Session() as sess:
    new_saver =
    tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
    new_saver.restore(sess, 'my-save-dir/my-model-10000')
    # tf.get_collection() returns a list. In this example we only want
    # the first one.
    train_op = tf.get_collection('train_op')[0]
    for step in xrange(1000000):
      sess.run(train_op)
  ```

  NOTE: Restarting training from saved `meta_graph` only works if the
  device assignments have not changed.

  Example:
  Variables, placeholders, and independent operations can also be stored, as
  shown in the following example.

  ```Python
  # Saving contents and operations.
  v1 = tf.placeholder(tf.float32, name=\"v1\")
  v2 = tf.placeholder(tf.float32, name=\"v2\")
  v3 = tf.math.multiply(v1, v2)
  vx = tf.Variable(10.0, name=\"vx\")
  v4 = tf.add(v3, vx, name=\"v4\")
  saver = tf.train.Saver([vx])
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(vx.assign(tf.add(vx, vx)))
  result = sess.run(v4, feed_dict={v1:12.0, v2:3.3})
  print(result)
  saver.save(sess, \"./model_ex1\")
  ```

  Later this model can be restored and contents loaded.

  ```Python
  # Restoring variables and running operations.
  saver = tf.train.import_meta_graph(\"./model_ex1.meta\")
  sess = tf.Session()
  saver.restore(sess, \"./model_ex1\")
  result = sess.run(\"v4:0\", feed_dict={\"v1:0\": 12.0, \"v2:0\": 3.3})
  print(result)
  ```

  Args:
    meta_graph_or_file: `MetaGraphDef` protocol buffer or filename (including
      the path) containing a `MetaGraphDef`.
    clear_devices: Whether or not to clear the device field for an `Operation`
      or `Tensor` during import.
    import_scope: Optional `string`. Name scope to add. Only used when
      initializing from protocol buffer.
    **kwargs: Optional keyed arguments.

  Returns:
    A saver constructed from `saver_def` in `MetaGraphDef` or None.

    A None value is returned if no variables exist in the `MetaGraphDef`
    (i.e., there are no variables to restore).

  Raises:
    RuntimeError: If called with eager execution enabled.

  @compatibility(eager)
  Exporting/importing meta graphs is not supported. No graph exists when eager
  execution is enabled.
  @end_compatibility
  "
  [meta_graph_or_file & {:keys [clear_devices import_scope]
                       :or {import_scope None}} ]
    (py/call-attr-kw train "import_meta_graph" [meta_graph_or_file] {:clear_devices clear_devices :import_scope import_scope }))

(defn init-from-checkpoint 
  "Replaces `tf.Variable` initializers so they load from a checkpoint file.

  Values are not loaded immediately, but when the initializer is run
  (typically by running a `tf.compat.v1.global_variables_initializer` op).

  Note: This overrides default initialization ops of specified variables and
  redefines dtype.

  Assignment map supports following syntax:

  * `'checkpoint_scope_name/': 'scope_name/'` - will load all variables in
    current `scope_name` from `checkpoint_scope_name` with matching tensor
    names.
  * `'checkpoint_scope_name/some_other_variable': 'scope_name/variable_name'` -
    will initialize `scope_name/variable_name` variable
    from `checkpoint_scope_name/some_other_variable`.
  * `'scope_variable_name': variable` - will initialize given `tf.Variable`
    object with tensor 'scope_variable_name' from the checkpoint.
  * `'scope_variable_name': list(variable)` - will initialize list of
    partitioned variables with tensor 'scope_variable_name' from the checkpoint.
  * `'/': 'scope_name/'` - will load all variables in current `scope_name` from
    checkpoint's root (e.g. no scope).

  Supports loading into partitioned variables, which are represented as
  `'<variable>/part_<part #>'`.

  Example:

  ```python

  # Say, '/tmp/model.ckpt' has the following tensors:
  #  -- name='old_scope_1/var1', shape=[20, 2]
  #  -- name='old_scope_1/var2', shape=[50, 4]
  #  -- name='old_scope_2/var3', shape=[100, 100]

  # Create new model's variables
  with tf.compat.v1.variable_scope('new_scope_1'):
    var1 = tf.compat.v1.get_variable('var1', shape=[20, 2],
                           initializer=tf.compat.v1.zeros_initializer())
  with tf.compat.v1.variable_scope('new_scope_2'):
    var2 = tf.compat.v1.get_variable('var2', shape=[50, 4],
                           initializer=tf.compat.v1.zeros_initializer())
    # Partition into 5 variables along the first axis.
    var3 = tf.compat.v1.get_variable(name='var3', shape=[100, 100],
                           initializer=tf.compat.v1.zeros_initializer(),
                           partitioner=lambda shape, dtype: [5, 1])

  # Initialize all variables in `new_scope_1` from `old_scope_1`.
  init_from_checkpoint('/tmp/model.ckpt', {'old_scope_1/': 'new_scope_1'})

  # Use names to specify which variables to initialize from checkpoint.
  init_from_checkpoint('/tmp/model.ckpt',
                       {'old_scope_1/var1': 'new_scope_1/var1',
                        'old_scope_1/var2': 'new_scope_2/var2'})

  # Or use tf.Variable objects to identify what to initialize.
  init_from_checkpoint('/tmp/model.ckpt',
                       {'old_scope_1/var1': var1,
                        'old_scope_1/var2': var2})

  # Initialize partitioned variables using variable's name
  init_from_checkpoint('/tmp/model.ckpt',
                       {'old_scope_2/var3': 'new_scope_2/var3'})

  # Or specify the list of tf.Variable objects.
  init_from_checkpoint('/tmp/model.ckpt',
                       {'old_scope_2/var3': var3._get_variable_list()})

  ```

  Args:
    ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint.
    assignment_map: Dict, where keys are names of the variables in the
      checkpoint and values are current variables or names of current variables
      (in default graph).

  Raises:
    ValueError: If missing variables in current graph, or if missing
      checkpoints or tensors in checkpoints.
  "
  [ ckpt_dir_or_file assignment_map ]
  (py/call-attr train "init_from_checkpoint"  ckpt_dir_or_file assignment_map ))

(defn input-producer 
  "Output the rows of `input_tensor` to a queue for an input pipeline. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.from_tensor_slices(input_tensor).shuffle(tf.shape(input_tensor, out_type=tf.int64)[0]).repeat(num_epochs)`. If `shuffle=False`, omit the `.shuffle(...)`.

Note: if `num_epochs` is not `None`, this function creates local counter
`epochs`. Use `local_variables_initializer()` to initialize local variables.

Args:
  input_tensor: A tensor with the rows to produce. Must be at least
    one-dimensional. Must either have a fully-defined shape, or
    `element_shape` must be defined.
  element_shape: (Optional.) A `TensorShape` representing the shape of a
    row of `input_tensor`, if it cannot be inferred.
  num_epochs: (Optional.) An integer. If specified `input_producer` produces
    each row of `input_tensor` `num_epochs` times before generating an
    `OutOfRange` error. If not specified, `input_producer` can cycle through
    the rows of `input_tensor` an unlimited number of times.
  shuffle: (Optional.) A boolean. If true, the rows are randomly shuffled
    within each epoch.
  seed: (Optional.) An integer. The seed to use if `shuffle` is true.
  capacity: (Optional.) The capacity of the queue to be used for buffering
    the input.
  shared_name: (Optional.) If set, this queue will be shared under the given
    name across multiple sessions.
  summary_name: (Optional.) If set, a scalar summary for the current queue
    size will be generated, using this name as part of the tag.
  name: (Optional.) A name for queue.
  cancel_op: (Optional.) Cancel op for the queue

Returns:
  A queue with the output rows.  A `QueueRunner` for the queue is
  added to the current `QUEUE_RUNNER` collection of the current
  graph.

Raises:
  ValueError: If the shape of the input cannot be inferred from the arguments.
  RuntimeError: If called with eager execution enabled.

@compatibility(eager)
Input pipelines based on Queues are not supported when eager execution is
enabled. Please use the `tf.data` API to ingest data under eager execution.
@end_compatibility"
  [input_tensor element_shape num_epochs & {:keys [shuffle seed capacity shared_name summary_name name cancel_op]
                       :or {seed None shared_name None summary_name None name None cancel_op None}} ]
    (py/call-attr-kw train "input_producer" [input_tensor element_shape num_epochs] {:shuffle shuffle :seed seed :capacity capacity :shared_name shared_name :summary_name summary_name :name name :cancel_op cancel_op }))

(defn inverse-time-decay 
  "Applies inverse time decay to the initial learning rate.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies an inverse decay function
  to a provided initial learning rate.  It requires an `global_step` value to
  compute the decayed learning rate.  You can just pass a TensorFlow variable
  that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:

  ```python
  decayed_learning_rate = learning_rate / (1 + decay_rate * global_step /
  decay_step)
  ```

  or, if `staircase` is `True`, as:

  ```python
  decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step /
  decay_step))
  ```

  Example: decay 1/t with a rate of 0.5:

  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  learning_rate = 0.1
  decay_steps = 1.0
  decay_rate = 0.5
  learning_rate = tf.compat.v1.train.inverse_time_decay(learning_rate,
  global_step,
  decay_steps, decay_rate)

  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
      tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
      .minimize(...my loss..., global_step=global_step)
  )
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a Python number.
      The initial learning rate.
    global_step: A Python number. Global step to use for the decay computation.
      Must not be negative.
    decay_steps: How often to apply decay.
    decay_rate: A Python number.  The decay rate.
    staircase: Whether to apply decay in a discrete staircase, as opposed to
      continuous, fashion.
    name: String.  Optional name of the operation.  Defaults to
      'InverseTimeDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.

  Raises:
    ValueError: if `global_step` is not supplied.

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  "
  [learning_rate global_step decay_steps decay_rate & {:keys [staircase name]
                       :or {name None}} ]
    (py/call-attr-kw train "inverse_time_decay" [learning_rate global_step decay_steps decay_rate] {:staircase staircase :name name }))

(defn latest-checkpoint 
  "Finds the filename of latest saved checkpoint file.

  Args:
    checkpoint_dir: Directory where the variables were saved.
    latest_filename: Optional name for the protocol buffer file that
      contains the list of most recent checkpoint filenames.
      See the corresponding argument to `Saver.save()`.

  Returns:
    The full path to the latest checkpoint or `None` if no checkpoint was found.
  "
  [ checkpoint_dir latest_filename ]
  (py/call-attr train "latest_checkpoint"  checkpoint_dir latest_filename ))

(defn limit-epochs 
  "Returns tensor `num_epochs` times and then raises an `OutOfRange` error. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.from_tensors(tensor).repeat(num_epochs)`.

Note: creates local counter `epochs`. Use `local_variables_initializer()` to
initialize local variables.

Args:
  tensor: Any `Tensor`.
  num_epochs: A positive integer (optional).  If specified, limits the number
    of steps the output tensor may be evaluated.
  name: A name for the operations (optional).

Returns:
  tensor or `OutOfRange`.

Raises:
  ValueError: if `num_epochs` is invalid."
  [ tensor num_epochs name ]
  (py/call-attr train "limit_epochs"  tensor num_epochs name ))

(defn linear-cosine-decay 
  "Applies linear cosine decay to the learning rate.

  See [Bello et al., ICML2017] Neural Optimizer Search with RL.
  https://arxiv.org/abs/1709.07417

  For the idea of warm starts here controlled by `num_periods`,
  see [Loshchilov & Hutter, ICLR2016] SGDR: Stochastic Gradient Descent
  with Warm Restarts. https://arxiv.org/abs/1608.03983

  Note that linear cosine decay is more aggressive than cosine decay and
  larger initial learning rates can typically be used.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies a linear cosine decay function
  to a provided initial learning rate.  It requires a `global_step` value to
  compute the decayed learning rate.  You can just pass a TensorFlow variable
  that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:
  ```python
  global_step = min(global_step, decay_steps)
  linear_decay = (decay_steps - global_step) / decay_steps)
  cosine_decay = 0.5 * (
      1 + cos(pi * 2 * num_periods * global_step / decay_steps))
  decayed = (alpha + linear_decay) * cosine_decay + beta
  decayed_learning_rate = learning_rate * decayed
  ```

  Example usage:
  ```python
  decay_steps = 1000
  lr_decayed = linear_cosine_decay(learning_rate, global_step, decay_steps)
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` Tensor or a Python number.
      The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number. Global
      step to use for the decay computation.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number. Number
      of steps to decay over.
    num_periods: Number of periods in the cosine part of the decay. See
      computation above.
    alpha: See computation above.
    beta: See computation above.
    name: String.  Optional name of the operation.  Defaults to
      'LinearCosineDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.
  Raises:
    ValueError: if `global_step` is not supplied.

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  "
  [learning_rate global_step decay_steps & {:keys [num_periods alpha beta name]
                       :or {name None}} ]
    (py/call-attr-kw train "linear_cosine_decay" [learning_rate global_step decay_steps] {:num_periods num_periods :alpha alpha :beta beta :name name }))

(defn list-variables 
  "Returns list of all variables in the checkpoint.

  Args:
    ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint.

  Returns:
    List of tuples `(name, shape)`.
  "
  [ ckpt_dir_or_file ]
  (py/call-attr train "list_variables"  ckpt_dir_or_file ))

(defn load-checkpoint 
  "Returns `CheckpointReader` for checkpoint found in `ckpt_dir_or_file`.

  If `ckpt_dir_or_file` resolves to a directory with multiple checkpoints,
  reader for the latest checkpoint is returned.

  Args:
    ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint
      file.

  Returns:
    `CheckpointReader` object.

  Raises:
    ValueError: If `ckpt_dir_or_file` resolves to a directory with no
      checkpoints.
  "
  [ ckpt_dir_or_file ]
  (py/call-attr train "load_checkpoint"  ckpt_dir_or_file ))

(defn load-variable 
  "Returns the tensor value of the given variable in the checkpoint.

  Args:
    ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint.
    name: Name of the variable to return.

  Returns:
    A numpy `ndarray` with a copy of the value of this variable.
  "
  [ ckpt_dir_or_file name ]
  (py/call-attr train "load_variable"  ckpt_dir_or_file name ))

(defn match-filenames-once 
  "Save the list of files matching pattern, so it is only computed once.

  NOTE: The order of the files returned is deterministic.

  Args:
    pattern: A file pattern (glob), or 1D tensor of file patterns.
    name: A name for the operations (optional).

  Returns:
    A variable that is initialized to the list of files matching the pattern(s).
  "
  [ pattern name ]
  (py/call-attr train "match_filenames_once"  pattern name ))

(defn maybe-batch 
  "Conditionally creates batches of tensors based on `keep_input`. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.filter(...).batch(batch_size)` (or `padded_batch(...)` if `dynamic_pad=True`).

See docstring in `batch` for more details.

Args:
  tensors: The list or dictionary of tensors to enqueue.
  keep_input: A `bool` Tensor.  This tensor controls whether the input is
    added to the queue or not.  If it is a scalar and evaluates `True`, then
    `tensors` are all added to the queue. If it is a vector and `enqueue_many`
    is `True`, then each example is added to the queue only if the
    corresponding value in `keep_input` is `True`. This tensor essentially
    acts as a filtering mechanism.
  batch_size: The new batch size pulled from the queue.
  num_threads: The number of threads enqueuing `tensors`.  The batching will
    be nondeterministic if `num_threads > 1`.
  capacity: An integer. The maximum number of elements in the queue.
  enqueue_many: Whether each tensor in `tensors` is a single example.
  shapes: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensors`.
  dynamic_pad: Boolean.  Allow variable dimensions in input shapes.
    The given dimensions are padded upon dequeue so that tensors within a
    batch have the same shapes.
  allow_smaller_final_batch: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
  shared_name: (Optional). If set, this queue will be shared under the given
    name across multiple sessions.
  name: (Optional) A name for the operations.

Returns:
  A list or dictionary of tensors with the same types as `tensors`.

Raises:
  ValueError: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensors`."
  [tensors keep_input batch_size & {:keys [num_threads capacity enqueue_many shapes dynamic_pad allow_smaller_final_batch shared_name name]
                       :or {shapes None shared_name None name None}} ]
    (py/call-attr-kw train "maybe_batch" [tensors keep_input batch_size] {:num_threads num_threads :capacity capacity :enqueue_many enqueue_many :shapes shapes :dynamic_pad dynamic_pad :allow_smaller_final_batch allow_smaller_final_batch :shared_name shared_name :name name }))

(defn maybe-batch-join 
  "Runs a list of tensors to conditionally fill a queue to create batches. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.interleave(...).filter(...).batch(batch_size)` (or `padded_batch(...)` if `dynamic_pad=True`).

See docstring in `batch_join` for more details.

Args:
  tensors_list: A list of tuples or dictionaries of tensors to enqueue.
  keep_input: A `bool` Tensor.  This tensor controls whether the input is
    added to the queue or not.  If it is a scalar and evaluates `True`, then
    `tensors` are all added to the queue. If it is a vector and `enqueue_many`
    is `True`, then each example is added to the queue only if the
    corresponding value in `keep_input` is `True`. This tensor essentially
    acts as a filtering mechanism.
  batch_size: An integer. The new batch size pulled from the queue.
  capacity: An integer. The maximum number of elements in the queue.
  enqueue_many: Whether each tensor in `tensor_list_list` is a single
    example.
  shapes: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensor_list_list[i]`.
  dynamic_pad: Boolean.  Allow variable dimensions in input shapes.
    The given dimensions are padded upon dequeue so that tensors within a
    batch have the same shapes.
  allow_smaller_final_batch: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
  shared_name: (Optional) If set, this queue will be shared under the given
    name across multiple sessions.
  name: (Optional) A name for the operations.

Returns:
  A list or dictionary of tensors with the same number and types as
  `tensors_list[i]`.

Raises:
  ValueError: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensor_list_list`."
  [tensors_list keep_input batch_size & {:keys [capacity enqueue_many shapes dynamic_pad allow_smaller_final_batch shared_name name]
                       :or {shapes None shared_name None name None}} ]
    (py/call-attr-kw train "maybe_batch_join" [tensors_list keep_input batch_size] {:capacity capacity :enqueue_many enqueue_many :shapes shapes :dynamic_pad dynamic_pad :allow_smaller_final_batch allow_smaller_final_batch :shared_name shared_name :name name }))

(defn maybe-shuffle-batch 
  "Creates batches by randomly shuffling conditionally-enqueued tensors. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.filter(...).shuffle(min_after_dequeue).batch(batch_size)`.

See docstring in `shuffle_batch` for more details.

Args:
  tensors: The list or dictionary of tensors to enqueue.
  batch_size: The new batch size pulled from the queue.
  capacity: An integer. The maximum number of elements in the queue.
  min_after_dequeue: Minimum number elements in the queue after a
    dequeue, used to ensure a level of mixing of elements.
  keep_input: A `bool` Tensor.  This tensor controls whether the input is
    added to the queue or not.  If it is a scalar and evaluates `True`, then
    `tensors` are all added to the queue. If it is a vector and `enqueue_many`
    is `True`, then each example is added to the queue only if the
    corresponding value in `keep_input` is `True`. This tensor essentially
    acts as a filtering mechanism.
  num_threads: The number of threads enqueuing `tensor_list`.
  seed: Seed for the random shuffling within the queue.
  enqueue_many: Whether each tensor in `tensor_list` is a single example.
  shapes: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensor_list`.
  allow_smaller_final_batch: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
  shared_name: (Optional) If set, this queue will be shared under the given
    name across multiple sessions.
  name: (Optional) A name for the operations.

Returns:
  A list or dictionary of tensors with the types as `tensors`.

Raises:
  ValueError: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensors`.

@compatibility(eager)
Input pipelines based on Queues are not supported when eager execution is
enabled. Please use the `tf.data` API to ingest data under eager execution.
@end_compatibility"
  [tensors batch_size capacity min_after_dequeue keep_input & {:keys [num_threads seed enqueue_many shapes allow_smaller_final_batch shared_name name]
                       :or {seed None shapes None shared_name None name None}} ]
    (py/call-attr-kw train "maybe_shuffle_batch" [tensors batch_size capacity min_after_dequeue keep_input] {:num_threads num_threads :seed seed :enqueue_many enqueue_many :shapes shapes :allow_smaller_final_batch allow_smaller_final_batch :shared_name shared_name :name name }))

(defn maybe-shuffle-batch-join 
  "Create batches by randomly shuffling conditionally-enqueued tensors. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.interleave(...).filter(...).shuffle(min_after_dequeue).batch(batch_size)`.

See docstring in `shuffle_batch_join` for more details.

Args:
  tensors_list: A list of tuples or dictionaries of tensors to enqueue.
  batch_size: An integer. The new batch size pulled from the queue.
  capacity: An integer. The maximum number of elements in the queue.
  min_after_dequeue: Minimum number elements in the queue after a
    dequeue, used to ensure a level of mixing of elements.
  keep_input: A `bool` Tensor.  This tensor controls whether the input is
    added to the queue or not.  If it is a scalar and evaluates `True`, then
    `tensors` are all added to the queue. If it is a vector and `enqueue_many`
    is `True`, then each example is added to the queue only if the
    corresponding value in `keep_input` is `True`. This tensor essentially
    acts as a filtering mechanism.
  seed: Seed for the random shuffling within the queue.
  enqueue_many: Whether each tensor in `tensor_list_list` is a single
    example.
  shapes: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensors_list[i]`.
  allow_smaller_final_batch: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
  shared_name: (optional). If set, this queue will be shared under the given
    name across multiple sessions.
  name: (Optional) A name for the operations.

Returns:
  A list or dictionary of tensors with the same number and types as
  `tensors_list[i]`.

Raises:
  ValueError: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensors_list`.

@compatibility(eager)
Input pipelines based on Queues are not supported when eager execution is
enabled. Please use the `tf.data` API to ingest data under eager execution.
@end_compatibility"
  [tensors_list batch_size capacity min_after_dequeue keep_input seed & {:keys [enqueue_many shapes allow_smaller_final_batch shared_name name]
                       :or {shapes None shared_name None name None}} ]
    (py/call-attr-kw train "maybe_shuffle_batch_join" [tensors_list batch_size capacity min_after_dequeue keep_input seed] {:enqueue_many enqueue_many :shapes shapes :allow_smaller_final_batch allow_smaller_final_batch :shared_name shared_name :name name }))

(defn natural-exp-decay 
  "Applies natural exponential decay to the initial learning rate.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies an exponential decay function
  to a provided initial learning rate.  It requires an `global_step` value to
  compute the decayed learning rate.  You can just pass a TensorFlow variable
  that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:

  ```python
  decayed_learning_rate = learning_rate * exp(-decay_rate * global_step /
  decay_step)
  ```

  or, if `staircase` is `True`, as:

  ```python
  decayed_learning_rate = learning_rate * exp(-decay_rate * floor(global_step /
  decay_step))
  ```

  Example: decay exponentially with a base of 0.96:

  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  learning_rate = 0.1
  decay_steps = 5
  k = 0.5
  learning_rate = tf.compat.v1.train.natural_exp_decay(learning_rate,
  global_step,
                                             decay_steps, k)

  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
      tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
      .minimize(...my loss..., global_step=global_step)
  )
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a Python number.
      The initial learning rate.
    global_step: A Python number. Global step to use for the decay computation.
      Must not be negative.
    decay_steps: How often to apply decay.
    decay_rate: A Python number.  The decay rate.
    staircase: Whether to apply decay in a discrete staircase, as opposed to
      continuous, fashion.
    name: String.  Optional name of the operation.  Defaults to
      'ExponentialTimeDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.

  Raises:
    ValueError: if `global_step` is not supplied.

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  "
  [learning_rate global_step decay_steps decay_rate & {:keys [staircase name]
                       :or {name None}} ]
    (py/call-attr-kw train "natural_exp_decay" [learning_rate global_step decay_steps decay_rate] {:staircase staircase :name name }))

(defn noisy-linear-cosine-decay 
  "Applies noisy linear cosine decay to the learning rate.

  See [Bello et al., ICML2017] Neural Optimizer Search with RL.
  https://arxiv.org/abs/1709.07417

  For the idea of warm starts here controlled by `num_periods`,
  see [Loshchilov & Hutter, ICLR2016] SGDR: Stochastic Gradient Descent
  with Warm Restarts. https://arxiv.org/abs/1608.03983

  Note that linear cosine decay is more aggressive than cosine decay and
  larger initial learning rates can typically be used.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies a noisy linear
  cosine decay function to a provided initial learning rate.
  It requires a `global_step` value to compute the decayed learning rate.
  You can just pass a TensorFlow variable that you increment at each
  training step.

  The function returns the decayed learning rate.  It is computed as:
  ```python
  global_step = min(global_step, decay_steps)
  linear_decay = (decay_steps - global_step) / decay_steps)
  cosine_decay = 0.5 * (
      1 + cos(pi * 2 * num_periods * global_step / decay_steps))
  decayed = (alpha + linear_decay + eps_t) * cosine_decay + beta
  decayed_learning_rate = learning_rate * decayed
  ```
  where eps_t is 0-centered gaussian noise with variance
  initial_variance / (1 + global_step) ** variance_decay

  Example usage:
  ```python
  decay_steps = 1000
  lr_decayed = noisy_linear_cosine_decay(
    learning_rate, global_step, decay_steps)
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` Tensor or a Python number.
      The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number. Global
      step to use for the decay computation.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number. Number
      of steps to decay over.
    initial_variance: initial variance for the noise. See computation above.
    variance_decay: decay for the noise's variance. See computation above.
    num_periods: Number of periods in the cosine part of the decay. See
      computation above.
    alpha: See computation above.
    beta: See computation above.
    name: String.  Optional name of the operation.  Defaults to
      'NoisyLinearCosineDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.
  Raises:
    ValueError: if `global_step` is not supplied.

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  "
  [learning_rate global_step decay_steps & {:keys [initial_variance variance_decay num_periods alpha beta name]
                       :or {name None}} ]
    (py/call-attr-kw train "noisy_linear_cosine_decay" [learning_rate global_step decay_steps] {:initial_variance initial_variance :variance_decay variance_decay :num_periods num_periods :alpha alpha :beta beta :name name }))

(defn piecewise-constant 
  "Piecewise constant from boundaries and interval values.

  Example: use a learning rate that's 1.0 for the first 100001 steps, 0.5
    for the next 10000 steps, and 0.1 for any additional steps.

  ```python
  global_step = tf.Variable(0, trainable=False)
  boundaries = [100000, 110000]
  values = [1.0, 0.5, 0.1]
  learning_rate = tf.compat.v1.train.piecewise_constant(global_step, boundaries,
  values)

  # Later, whenever we perform an optimization step, we increment global_step.
  ```

  Args:
    x: A 0-D scalar `Tensor`. Must be one of the following types: `float32`,
      `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`.
    boundaries: A list of `Tensor`s or `int`s or `float`s with strictly
      increasing entries, and with all elements having the same type as `x`.
    values: A list of `Tensor`s or `float`s or `int`s that specifies the values
      for the intervals defined by `boundaries`. It should have one more element
      than `boundaries`, and all elements should have the same type.
    name: A string. Optional name of the operation. Defaults to
      'PiecewiseConstant'.

  Returns:
    A 0-D Tensor. Its value is `values[0]` when `x <= boundaries[0]`,
    `values[1]` when `x > boundaries[0]` and `x <= boundaries[1]`, ...,
    and values[-1] when `x > boundaries[-1]`.

  Raises:
    ValueError: if types of `x` and `boundaries` do not match, or types of all
        `values` do not match or
        the number of elements in the lists does not match.

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  "
  [ x boundaries values name ]
  (py/call-attr train "piecewise_constant"  x boundaries values name ))

(defn piecewise-constant-decay 
  "Piecewise constant from boundaries and interval values.

  Example: use a learning rate that's 1.0 for the first 100001 steps, 0.5
    for the next 10000 steps, and 0.1 for any additional steps.

  ```python
  global_step = tf.Variable(0, trainable=False)
  boundaries = [100000, 110000]
  values = [1.0, 0.5, 0.1]
  learning_rate = tf.compat.v1.train.piecewise_constant(global_step, boundaries,
  values)

  # Later, whenever we perform an optimization step, we increment global_step.
  ```

  Args:
    x: A 0-D scalar `Tensor`. Must be one of the following types: `float32`,
      `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`.
    boundaries: A list of `Tensor`s or `int`s or `float`s with strictly
      increasing entries, and with all elements having the same type as `x`.
    values: A list of `Tensor`s or `float`s or `int`s that specifies the values
      for the intervals defined by `boundaries`. It should have one more element
      than `boundaries`, and all elements should have the same type.
    name: A string. Optional name of the operation. Defaults to
      'PiecewiseConstant'.

  Returns:
    A 0-D Tensor. Its value is `values[0]` when `x <= boundaries[0]`,
    `values[1]` when `x > boundaries[0]` and `x <= boundaries[1]`, ...,
    and values[-1] when `x > boundaries[-1]`.

  Raises:
    ValueError: if types of `x` and `boundaries` do not match, or types of all
        `values` do not match or
        the number of elements in the lists does not match.

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  "
  [ x boundaries values name ]
  (py/call-attr train "piecewise_constant_decay"  x boundaries values name ))

(defn polynomial-decay 
  "Applies a polynomial decay to the learning rate.

  It is commonly observed that a monotonically decreasing learning rate, whose
  degree of change is carefully chosen, results in a better performing model.
  This function applies a polynomial decay function to a provided initial
  `learning_rate` to reach an `end_learning_rate` in the given `decay_steps`.

  It requires a `global_step` value to compute the decayed learning rate.  You
  can just pass a TensorFlow variable that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:

  ```python
  global_step = min(global_step, decay_steps)
  decayed_learning_rate = (learning_rate - end_learning_rate) *
                          (1 - global_step / decay_steps) ^ (power) +
                          end_learning_rate

  ```

  If `cycle` is True then a multiple of `decay_steps` is used, the first one
  that is bigger than `global_steps`.

  ```python
  decay_steps = decay_steps * ceil(global_step / decay_steps)
  decayed_learning_rate = (learning_rate - end_learning_rate) *
                          (1 - global_step / decay_steps) ^ (power) +
                          end_learning_rate

  ```

  Example: decay from 0.1 to 0.01 in 10000 steps using sqrt (i.e. power=0.5):

  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.1
  end_learning_rate = 0.01
  decay_steps = 10000
  learning_rate = tf.compat.v1.train.polynomial_decay(starter_learning_rate,
  global_step,
                                            decay_steps, end_learning_rate,
                                            power=0.5)
  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
      tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
      .minimize(...my loss..., global_step=global_step)
  )
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a Python number.
      The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number. Global
      step to use for the decay computation.  Must not be negative.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number. Must
      be positive.  See the decay computation above.
    end_learning_rate: A scalar `float32` or `float64` `Tensor` or a Python
      number.  The minimal end learning rate.
    power: A scalar `float32` or `float64` `Tensor` or a Python number.  The
      power of the polynomial. Defaults to linear, 1.0.
    cycle: A boolean, whether or not it should cycle beyond decay_steps.
    name: String.  Optional name of the operation. Defaults to
      'PolynomialDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.

  Raises:
    ValueError: if `global_step` is not supplied.

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  "
  [learning_rate global_step decay_steps & {:keys [end_learning_rate power cycle name]
                       :or {name None}} ]
    (py/call-attr-kw train "polynomial_decay" [learning_rate global_step decay_steps] {:end_learning_rate end_learning_rate :power power :cycle cycle :name name }))

(defn range-input-producer 
  "Produces the integers from 0 to limit-1 in a queue. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.range(limit).shuffle(limit).repeat(num_epochs)`. If `shuffle=False`, omit the `.shuffle(...)`.

Note: if `num_epochs` is not `None`, this function creates local counter
`epochs`. Use `local_variables_initializer()` to initialize local variables.

Args:
  limit: An int32 scalar tensor.
  num_epochs: An integer (optional). If specified, `range_input_producer`
    produces each integer `num_epochs` times before generating an
    OutOfRange error. If not specified, `range_input_producer` can cycle
    through the integers an unlimited number of times.
  shuffle: Boolean. If true, the integers are randomly shuffled within each
    epoch.
  seed: An integer (optional). Seed used if shuffle == True.
  capacity: An integer. Sets the queue capacity.
  shared_name: (optional). If set, this queue will be shared under the given
    name across multiple sessions.
  name: A name for the operations (optional).

Returns:
  A Queue with the output integers.  A `QueueRunner` for the Queue
  is added to the current `Graph`'s `QUEUE_RUNNER` collection.

@compatibility(eager)
Input pipelines based on Queues are not supported when eager execution is
enabled. Please use the `tf.data` API to ingest data under eager execution.
@end_compatibility"
  [limit num_epochs & {:keys [shuffle seed capacity shared_name name]
                       :or {seed None shared_name None name None}} ]
    (py/call-attr-kw train "range_input_producer" [limit num_epochs] {:shuffle shuffle :seed seed :capacity capacity :shared_name shared_name :name name }))
(defn remove-checkpoint 
  "Removes a checkpoint given by `checkpoint_prefix`. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use standard file APIs to delete files with this prefix.

Args:
  checkpoint_prefix: The prefix of a V1 or V2 checkpoint. Typically the result
    of `Saver.save()` or that of `tf.train.latest_checkpoint()`, regardless of
    sharded/non-sharded or V1/V2.
  checkpoint_format_version: `SaverDef.CheckpointFormatVersion`, defaults to
    `SaverDef.V2`.
  meta_graph_suffix: Suffix for `MetaGraphDef` file. Defaults to 'meta'."
  [checkpoint_prefix  & {:keys [checkpoint_format_version meta_graph_suffix]} ]
    (py/call-attr-kw train "remove_checkpoint" [checkpoint_prefix] {:checkpoint_format_version checkpoint_format_version :meta_graph_suffix meta_graph_suffix }))

(defn replica-device-setter 
  "Return a `device function` to use when building a Graph for replicas.

  Device Functions are used in `with tf.device(device_function):` statement to
  automatically assign devices to `Operation` objects as they are constructed,
  Device constraints are added from the inner-most context first, working
  outwards. The merging behavior adds constraints to fields that are yet unset
  by a more inner context. Currently the fields are (job, task, cpu/gpu).

  If `cluster` is `None`, and `ps_tasks` is 0, the returned function is a no-op.
  Otherwise, the value of `ps_tasks` is derived from `cluster`.

  By default, only Variable ops are placed on ps tasks, and the placement
  strategy is round-robin over all ps tasks. A custom `ps_strategy` may be used
  to do more intelligent placement, such as
  `tf.contrib.training.GreedyLoadBalancingStrategy`.

  For example,

  ```python
  # To build a cluster with two ps jobs on hosts ps0 and ps1, and 3 worker
  # jobs on hosts worker0, worker1 and worker2.
  cluster_spec = {
      \"ps\": [\"ps0:2222\", \"ps1:2222\"],
      \"worker\": [\"worker0:2222\", \"worker1:2222\", \"worker2:2222\"]}
  with
  tf.device(tf.compat.v1.train.replica_device_setter(cluster=cluster_spec)):
    # Build your graph
    v1 = tf.Variable(...)  # assigned to /job:ps/task:0
    v2 = tf.Variable(...)  # assigned to /job:ps/task:1
    v3 = tf.Variable(...)  # assigned to /job:ps/task:0
  # Run compute
  ```

  Args:
    ps_tasks: Number of tasks in the `ps` job.  Ignored if `cluster` is
      provided.
    ps_device: String.  Device of the `ps` job.  If empty no `ps` job is used.
      Defaults to `ps`.
    worker_device: String.  Device of the `worker` job.  If empty no `worker`
      job is used.
    merge_devices: `Boolean`. If `True`, merges or only sets a device if the
      device constraint is completely unset. merges device specification rather
      than overriding them.
    cluster: `ClusterDef` proto or `ClusterSpec`.
    ps_ops: List of strings representing `Operation` types that need to be
      placed on `ps` devices.  If `None`, defaults to `STANDARD_PS_OPS`.
    ps_strategy: A callable invoked for every ps `Operation` (i.e. matched by
      `ps_ops`), that takes the `Operation` and returns the ps task index to
      use.  If `None`, defaults to a round-robin strategy across all `ps`
      devices.

  Returns:
    A function to pass to `tf.device()`.

  Raises:
    TypeError if `cluster` is not a dictionary or `ClusterDef` protocol buffer,
    or if `ps_strategy` is provided but not a callable.
  "
  [ & {:keys [ps_tasks ps_device worker_device merge_devices cluster ps_ops ps_strategy]
       :or {cluster None ps_ops None ps_strategy None}} ]
  
   (py/call-attr-kw train "replica_device_setter" [] {:ps_tasks ps_tasks :ps_device ps_device :worker_device worker_device :merge_devices merge_devices :cluster cluster :ps_ops ps_ops :ps_strategy ps_strategy }))

(defn sdca-fprint 
  "Computes fingerprints of the input strings.

  Args:
    input: A `Tensor` of type `string`.
      vector of strings to compute fingerprints on.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  "
  [ input name ]
  (py/call-attr train "sdca_fprint"  input name ))

(defn sdca-optimizer 
  "Distributed version of Stochastic Dual Coordinate Ascent (SDCA) optimizer for

  linear models with L1 + L2 regularization. As global optimization objective is
  strongly-convex, the optimizer optimizes the dual objective at each step. The
  optimizer applies each update one example at a time. Examples are sampled
  uniformly, and the optimizer is learning rate free and enjoys linear convergence
  rate.

  [Proximal Stochastic Dual Coordinate Ascent](http://arxiv.org/pdf/1211.2717v1.pdf).<br>
  Shai Shalev-Shwartz, Tong Zhang. 2012

  $$Loss Objective = \sum f_{i} (wx_{i}) + (l2 / 2) * |w|^2 + l1 * |w|$$

  [Adding vs. Averaging in Distributed Primal-Dual Optimization](http://arxiv.org/abs/1502.03508).<br>
  Chenxin Ma, Virginia Smith, Martin Jaggi, Michael I. Jordan,
  Peter Richtarik, Martin Takac. 2015

  [Stochastic Dual Coordinate Ascent with Adaptive Probabilities](https://arxiv.org/abs/1502.08053).<br>
  Dominik Csiba, Zheng Qu, Peter Richtarik. 2015

  Args:
    sparse_example_indices: A list of `Tensor` objects with type `int64`.
      a list of vectors which contain example indices.
    sparse_feature_indices: A list with the same length as `sparse_example_indices` of `Tensor` objects with type `int64`.
      a list of vectors which contain feature indices.
    sparse_feature_values: A list of `Tensor` objects with type `float32`.
      a list of vectors which contains feature value
      associated with each feature group.
    dense_features: A list of `Tensor` objects with type `float32`.
      a list of matrices which contains the dense feature values.
    example_weights: A `Tensor` of type `float32`.
      a vector which contains the weight associated with each
      example.
    example_labels: A `Tensor` of type `float32`.
      a vector which contains the label/target associated with each
      example.
    sparse_indices: A list with the same length as `sparse_example_indices` of `Tensor` objects with type `int64`.
      a list of vectors where each value is the indices which has
      corresponding weights in sparse_weights. This field maybe omitted for the
      dense approach.
    sparse_weights: A list with the same length as `sparse_example_indices` of `Tensor` objects with type `float32`.
      a list of vectors where each value is the weight associated with
      a sparse feature group.
    dense_weights: A list with the same length as `dense_features` of `Tensor` objects with type `float32`.
      a list of vectors where the values are the weights associated
      with a dense feature group.
    example_state_data: A `Tensor` of type `float32`.
      a list of vectors containing the example state data.
    loss_type: A `string` from: `\"logistic_loss\", \"squared_loss\", \"hinge_loss\", \"smooth_hinge_loss\", \"poisson_loss\"`.
      Type of the primal loss. Currently SdcaSolver supports logistic,
      squared and hinge losses.
    l1: A `float`. Symmetric l1 regularization strength.
    l2: A `float`. Symmetric l2 regularization strength.
    num_loss_partitions: An `int` that is `>= 1`.
      Number of partitions of the global loss function.
    num_inner_iterations: An `int` that is `>= 1`.
      Number of iterations per mini-batch.
    adaptative: An optional `bool`. Defaults to `True`.
      Whether to use Adaptive SDCA for the inner loop.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (out_example_state_data, out_delta_sparse_weights, out_delta_dense_weights).

    out_example_state_data: A `Tensor` of type `float32`.
    out_delta_sparse_weights: A list with the same length as `sparse_example_indices` of `Tensor` objects with type `float32`.
    out_delta_dense_weights: A list with the same length as `dense_features` of `Tensor` objects with type `float32`.
  "
  [sparse_example_indices sparse_feature_indices sparse_feature_values dense_features example_weights example_labels sparse_indices sparse_weights dense_weights example_state_data loss_type l1 l2 num_loss_partitions num_inner_iterations & {:keys [adaptative name]
                       :or {name None}} ]
    (py/call-attr-kw train "sdca_optimizer" [sparse_example_indices sparse_feature_indices sparse_feature_values dense_features example_weights example_labels sparse_indices sparse_weights dense_weights example_state_data loss_type l1 l2 num_loss_partitions num_inner_iterations] {:adaptative adaptative :name name }))

(defn sdca-shrink-l1 
  "Applies L1 regularization shrink step on the parameters.

  Args:
    weights: A list of `Tensor` objects with type mutable `float32`.
      a list of vectors where each value is the weight associated with a
      feature group.
    l1: A `float`. Symmetric l1 regularization strength.
    l2: A `float`.
      Symmetric l2 regularization strength. Should be a positive float.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ weights l1 l2 name ]
  (py/call-attr train "sdca_shrink_l1"  weights l1 l2 name ))

(defn shuffle-batch 
  "Creates batches by randomly shuffling tensors. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.shuffle(min_after_dequeue).batch(batch_size)`.

This function adds the following to the current `Graph`:

* A shuffling queue into which tensors from `tensors` are enqueued.
* A `dequeue_many` operation to create batches from the queue.
* A `QueueRunner` to `QUEUE_RUNNER` collection, to enqueue the tensors
  from `tensors`.

If `enqueue_many` is `False`, `tensors` is assumed to represent a
single example.  An input tensor with shape `[x, y, z]` will be output
as a tensor with shape `[batch_size, x, y, z]`.

If `enqueue_many` is `True`, `tensors` is assumed to represent a
batch of examples, where the first dimension is indexed by example,
and all members of `tensors` should have the same size in the
first dimension.  If an input tensor has shape `[*, x, y, z]`, the
output will have shape `[batch_size, x, y, z]`.

The `capacity` argument controls the how long the prefetching is allowed to
grow the queues.

The returned operation is a dequeue operation and will throw
`tf.errors.OutOfRangeError` if the input queue is exhausted. If this
operation is feeding another input queue, its queue runner will catch
this exception, however, if this operation is used in your main thread
you are responsible for catching this yourself.

For example:

```python
# Creates batches of 32 images and 32 labels.
image_batch, label_batch = tf.compat.v1.train.shuffle_batch(
      [single_image, single_label],
      batch_size=32,
      num_threads=4,
      capacity=50000,
      min_after_dequeue=10000)
```

*N.B.:* You must ensure that either (i) the `shapes` argument is
passed, or (ii) all of the tensors in `tensors` must have
fully-defined shapes. `ValueError` will be raised if neither of
these conditions holds.

If `allow_smaller_final_batch` is `True`, a smaller batch value than
`batch_size` is returned when the queue is closed and there are not enough
elements to fill the batch, otherwise the pending elements are discarded.
In addition, all output tensors' static shapes, as accessed via the
`shape` property will have a first `Dimension` value of `None`, and
operations that depend on fixed batch_size would fail.

Args:
  tensors: The list or dictionary of tensors to enqueue.
  batch_size: The new batch size pulled from the queue.
  capacity: An integer. The maximum number of elements in the queue.
  min_after_dequeue: Minimum number elements in the queue after a
    dequeue, used to ensure a level of mixing of elements.
  num_threads: The number of threads enqueuing `tensor_list`.
  seed: Seed for the random shuffling within the queue.
  enqueue_many: Whether each tensor in `tensor_list` is a single example.
  shapes: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensor_list`.
  allow_smaller_final_batch: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
  shared_name: (Optional) If set, this queue will be shared under the given
    name across multiple sessions.
  name: (Optional) A name for the operations.

Returns:
  A list or dictionary of tensors with the types as `tensors`.

Raises:
  ValueError: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensors`.

@compatibility(eager)
Input pipelines based on Queues are not supported when eager execution is
enabled. Please use the `tf.data` API to ingest data under eager execution.
@end_compatibility"
  [tensors batch_size capacity min_after_dequeue & {:keys [num_threads seed enqueue_many shapes allow_smaller_final_batch shared_name name]
                       :or {seed None shapes None shared_name None name None}} ]
    (py/call-attr-kw train "shuffle_batch" [tensors batch_size capacity min_after_dequeue] {:num_threads num_threads :seed seed :enqueue_many enqueue_many :shapes shapes :allow_smaller_final_batch allow_smaller_final_batch :shared_name shared_name :name name }))

(defn shuffle-batch-join 
  "Create batches by randomly shuffling tensors. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.interleave(...).shuffle(min_after_dequeue).batch(batch_size)`.

The `tensors_list` argument is a list of tuples of tensors, or a list of
dictionaries of tensors.  Each element in the list is treated similarly
to the `tensors` argument of `tf.compat.v1.train.shuffle_batch()`.

This version enqueues a different list of tensors in different threads.
It adds the following to the current `Graph`:

* A shuffling queue into which tensors from `tensors_list` are enqueued.
* A `dequeue_many` operation to create batches from the queue.
* A `QueueRunner` to `QUEUE_RUNNER` collection, to enqueue the tensors
  from `tensors_list`.

`len(tensors_list)` threads will be started, with thread `i` enqueuing
the tensors from `tensors_list[i]`. `tensors_list[i1][j]` must match
`tensors_list[i2][j]` in type and shape, except in the first dimension if
`enqueue_many` is true.

If `enqueue_many` is `False`, each `tensors_list[i]` is assumed
to represent a single example.  An input tensor with shape `[x, y, z]`
will be output as a tensor with shape `[batch_size, x, y, z]`.

If `enqueue_many` is `True`, `tensors_list[i]` is assumed to
represent a batch of examples, where the first dimension is indexed
by example, and all members of `tensors_list[i]` should have the
same size in the first dimension.  If an input tensor has shape `[*, x,
y, z]`, the output will have shape `[batch_size, x, y, z]`.

The `capacity` argument controls the how long the prefetching is allowed to
grow the queues.

The returned operation is a dequeue operation and will throw
`tf.errors.OutOfRangeError` if the input queue is exhausted. If this
operation is feeding another input queue, its queue runner will catch
this exception, however, if this operation is used in your main thread
you are responsible for catching this yourself.

If `allow_smaller_final_batch` is `True`, a smaller batch value than
`batch_size` is returned when the queue is closed and there are not enough
elements to fill the batch, otherwise the pending elements are discarded.
In addition, all output tensors' static shapes, as accessed via the
`shape` property will have a first `Dimension` value of `None`, and
operations that depend on fixed batch_size would fail.

Args:
  tensors_list: A list of tuples or dictionaries of tensors to enqueue.
  batch_size: An integer. The new batch size pulled from the queue.
  capacity: An integer. The maximum number of elements in the queue.
  min_after_dequeue: Minimum number elements in the queue after a
    dequeue, used to ensure a level of mixing of elements.
  seed: Seed for the random shuffling within the queue.
  enqueue_many: Whether each tensor in `tensor_list_list` is a single
    example.
  shapes: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensors_list[i]`.
  allow_smaller_final_batch: (Optional) Boolean. If `True`, allow the final
    batch to be smaller if there are insufficient items left in the queue.
  shared_name: (optional). If set, this queue will be shared under the given
    name across multiple sessions.
  name: (Optional) A name for the operations.

Returns:
  A list or dictionary of tensors with the same number and types as
  `tensors_list[i]`.

Raises:
  ValueError: If the `shapes` are not specified, and cannot be
    inferred from the elements of `tensors_list`.

@compatibility(eager)
Input pipelines based on Queues are not supported when eager execution is
enabled. Please use the `tf.data` API to ingest data under eager execution.
@end_compatibility"
  [tensors_list batch_size capacity min_after_dequeue seed & {:keys [enqueue_many shapes allow_smaller_final_batch shared_name name]
                       :or {shapes None shared_name None name None}} ]
    (py/call-attr-kw train "shuffle_batch_join" [tensors_list batch_size capacity min_after_dequeue seed] {:enqueue_many enqueue_many :shapes shapes :allow_smaller_final_batch allow_smaller_final_batch :shared_name shared_name :name name }))

(defn slice-input-producer 
  "Produces a slice of each `Tensor` in `tensor_list`. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.from_tensor_slices(tuple(tensor_list)).shuffle(tf.shape(input_tensor, out_type=tf.int64)[0]).repeat(num_epochs)`. If `shuffle=False`, omit the `.shuffle(...)`.

Implemented using a Queue -- a `QueueRunner` for the Queue
is added to the current `Graph`'s `QUEUE_RUNNER` collection.

Args:
  tensor_list: A list of `Tensor` objects. Every `Tensor` in
    `tensor_list` must have the same size in the first dimension.
  num_epochs: An integer (optional). If specified, `slice_input_producer`
    produces each slice `num_epochs` times before generating
    an `OutOfRange` error. If not specified, `slice_input_producer` can cycle
    through the slices an unlimited number of times.
  shuffle: Boolean. If true, the integers are randomly shuffled within each
    epoch.
  seed: An integer (optional). Seed used if shuffle == True.
  capacity: An integer. Sets the queue capacity.
  shared_name: (optional). If set, this queue will be shared under the given
    name across multiple sessions.
  name: A name for the operations (optional).

Returns:
  A list of tensors, one for each element of `tensor_list`.  If the tensor
  in `tensor_list` has shape `[N, a, b, .., z]`, then the corresponding output
  tensor will have shape `[a, b, ..., z]`.

Raises:
  ValueError: if `slice_input_producer` produces nothing from `tensor_list`.

@compatibility(eager)
Input pipelines based on Queues are not supported when eager execution is
enabled. Please use the `tf.data` API to ingest data under eager execution.
@end_compatibility"
  [tensor_list num_epochs & {:keys [shuffle seed capacity shared_name name]
                       :or {seed None shared_name None name None}} ]
    (py/call-attr-kw train "slice_input_producer" [tensor_list num_epochs] {:shuffle shuffle :seed seed :capacity capacity :shared_name shared_name :name name }))
(defn start-queue-runners 
  "Starts all queue runners collected in the graph. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
To construct input pipelines, use the `tf.data` module.

This is a companion method to `add_queue_runner()`.  It just starts
threads for all queue runners collected in the graph.  It returns
the list of all threads.

Args:
  sess: `Session` used to run the queue ops.  Defaults to the
    default session.
  coord: Optional `Coordinator` for coordinating the started threads.
  daemon: Whether the threads should be marked as `daemons`, meaning
    they don't block program exit.
  start: Set to `False` to only create the threads, not start them.
  collection: A `GraphKey` specifying the graph collection to
    get the queue runners from.  Defaults to `GraphKeys.QUEUE_RUNNERS`.

Raises:
  ValueError: if `sess` is None and there isn't any default session.
  TypeError: if `sess` is not a `tf.compat.v1.Session` object.

Returns:
  A list of threads.

Raises:
  RuntimeError: If called with eager execution enabled.
  ValueError: If called without a default `tf.compat.v1.Session` registered.

@compatibility(eager)
Not compatible with eager execution. To ingest data under eager execution,
use the `tf.data` API instead.
@end_compatibility"
  [sess coord  & {:keys [daemon start collection]} ]
    (py/call-attr-kw train "start_queue_runners" [sess coord] {:daemon daemon :start start :collection collection }))

(defn string-input-producer 
  "Output strings (e.g. filenames) to a queue for an input pipeline. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.from_tensor_slices(string_tensor).shuffle(tf.shape(input_tensor, out_type=tf.int64)[0]).repeat(num_epochs)`. If `shuffle=False`, omit the `.shuffle(...)`.

Note: if `num_epochs` is not `None`, this function creates local counter
`epochs`. Use `local_variables_initializer()` to initialize local variables.

Args:
  string_tensor: A 1-D string tensor with the strings to produce.
  num_epochs: An integer (optional). If specified, `string_input_producer`
    produces each string from `string_tensor` `num_epochs` times before
    generating an `OutOfRange` error. If not specified,
    `string_input_producer` can cycle through the strings in `string_tensor`
    an unlimited number of times.
  shuffle: Boolean. If true, the strings are randomly shuffled within each
    epoch.
  seed: An integer (optional). Seed used if shuffle == True.
  capacity: An integer. Sets the queue capacity.
  shared_name: (optional). If set, this queue will be shared under the given
    name across multiple sessions. All sessions open to the device which has
    this queue will be able to access it via the shared_name. Using this in
    a distributed setting means each name will only be seen by one of the
    sessions which has access to this operation.
  name: A name for the operations (optional).
  cancel_op: Cancel op for the queue (optional).

Returns:
  A queue with the output strings.  A `QueueRunner` for the Queue
  is added to the current `Graph`'s `QUEUE_RUNNER` collection.

Raises:
  ValueError: If the string_tensor is a null Python list.  At runtime,
  will fail with an assertion if string_tensor becomes a null tensor.

@compatibility(eager)
Input pipelines based on Queues are not supported when eager execution is
enabled. Please use the `tf.data` API to ingest data under eager execution.
@end_compatibility"
  [string_tensor num_epochs & {:keys [shuffle seed capacity shared_name name cancel_op]
                       :or {seed None shared_name None name None cancel_op None}} ]
    (py/call-attr-kw train "string_input_producer" [string_tensor num_epochs] {:shuffle shuffle :seed seed :capacity capacity :shared_name shared_name :name name :cancel_op cancel_op }))

(defn summary-iterator 
  "An iterator for reading `Event` protocol buffers from an event file.

  You can use this function to read events written to an event file. It returns
  a Python iterator that yields `Event` protocol buffers.

  Example: Print the contents of an events file.

  ```python
  for e in tf.compat.v1.train.summary_iterator(path to events file):
      print(e)
  ```

  Example: Print selected summary values.

  ```python
  # This example supposes that the events file contains summaries with a
  # summary value tag 'loss'.  These could have been added by calling
  # `add_summary()`, passing the output of a scalar summary op created with
  # with: `tf.compat.v1.summary.scalar('loss', loss_tensor)`.
  for e in tf.compat.v1.train.summary_iterator(path to events file):
      for v in e.summary.value:
          if v.tag == 'loss':
              print(v.simple_value)
  ```

  See the protocol buffer definitions of
  [Event](https://www.tensorflow.org/code/tensorflow/core/util/event.proto)
  and
  [Summary](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
  for more information about their attributes.

  Args:
    path: The path to an event file created by a `SummaryWriter`.

  Yields:
    `Event` protocol buffers.
  "
  [ path ]
  (py/call-attr train "summary_iterator"  path ))

(defn update-checkpoint-state 
  "Updates the content of the 'checkpoint' file. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use `tf.train.CheckpointManager` to manage checkpoints rather than manually editing the Checkpoint proto.

This updates the checkpoint file containing a CheckpointState
proto.

Args:
  save_dir: Directory where the model was saved.
  model_checkpoint_path: The checkpoint file.
  all_model_checkpoint_paths: List of strings.  Paths to all not-yet-deleted
    checkpoints, sorted from oldest to newest.  If this is a non-empty list,
    the last element must be equal to model_checkpoint_path.  These paths
    are also saved in the CheckpointState proto.
  latest_filename: Optional name of the checkpoint file.  Default to
    'checkpoint'.
  all_model_checkpoint_timestamps: Optional list of timestamps (floats,
    seconds since the Epoch) indicating when the checkpoints in
    `all_model_checkpoint_paths` were created.
  last_preserved_timestamp: A float, indicating the number of seconds since
    the Epoch when the last preserved checkpoint was written, e.g. due to a
    `keep_checkpoint_every_n_hours` parameter (see
    `tf.contrib.checkpoint.CheckpointManager` for an implementation).
Raises:
  RuntimeError: If any of the model checkpoint paths conflict with the file
    containing CheckpointSate."
  [ save_dir model_checkpoint_path all_model_checkpoint_paths latest_filename all_model_checkpoint_timestamps last_preserved_timestamp ]
  (py/call-attr train "update_checkpoint_state"  save_dir model_checkpoint_path all_model_checkpoint_paths latest_filename all_model_checkpoint_timestamps last_preserved_timestamp ))

(defn warm-start 
  "Warm-starts a model using the given settings.

  If you are using a tf.estimator.Estimator, this will automatically be called
  during training.

  Args:
    ckpt_to_initialize_from: [Required] A string specifying the directory with
      checkpoint file(s) or path to checkpoint from which to warm-start the
      model parameters.
    vars_to_warm_start: [Optional] One of the following:

      - A regular expression (string) that captures which variables to
        warm-start (see tf.compat.v1.get_collection).  This expression will only
        consider variables in the TRAINABLE_VARIABLES collection -- if you need
        to warm-start non_TRAINABLE vars (such as optimizer accumulators or
        batch norm statistics), please use the below option.
      - A list of strings, each a regex scope provided to
        tf.compat.v1.get_collection with GLOBAL_VARIABLES (please see
        tf.compat.v1.get_collection).  For backwards compatibility reasons,
        this is separate from the single-string argument type.
      - A list of Variables to warm-start.  If you do not have access to the
        `Variable` objects at the call site, please use the above option.
      - `None`, in which case only TRAINABLE variables specified in
        `var_name_to_vocab_info` will be warm-started.

      Defaults to `'.*'`, which warm-starts all variables in the
      TRAINABLE_VARIABLES collection.  Note that this excludes variables such
      as accumulators and moving statistics from batch norm.
    var_name_to_vocab_info: [Optional] Dict of variable names (strings) to
      `tf.estimator.VocabInfo`. The variable names should be \"full\" variables,
      not the names of the partitions.  If not explicitly provided, the variable
      is assumed to have no (changes to) vocabulary.
    var_name_to_prev_var_name: [Optional] Dict of variable names (strings) to
      name of the previously-trained variable in `ckpt_to_initialize_from`. If
      not explicitly provided, the name of the variable is assumed to be same
      between previous checkpoint and current model.  Note that this has no
      effect on the set of variables that is warm-started, and only controls
      name mapping (use `vars_to_warm_start` for controlling what variables to
      warm-start).

  Raises:
    ValueError: If the WarmStartSettings contains prev_var_name or VocabInfo
      configuration for variable names that are not used.  This is to ensure
      a stronger check for variable configuration than relying on users to
      examine the logs.
  "
  [ckpt_to_initialize_from & {:keys [vars_to_warm_start var_name_to_vocab_info var_name_to_prev_var_name]
                       :or {var_name_to_vocab_info None var_name_to_prev_var_name None}} ]
    (py/call-attr-kw train "warm_start" [ckpt_to_initialize_from] {:vars_to_warm_start vars_to_warm_start :var_name_to_vocab_info var_name_to_vocab_info :var_name_to_prev_var_name var_name_to_prev_var_name }))
(defn write-graph 
  "Writes a graph proto to a file.

  The graph is written as a text proto unless `as_text` is `False`.

  ```python
  v = tf.Variable(0, name='my_variable')
  sess = tf.compat.v1.Session()
  tf.io.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')
  ```

  or

  ```python
  v = tf.Variable(0, name='my_variable')
  sess = tf.compat.v1.Session()
  tf.io.write_graph(sess.graph, '/tmp/my-model', 'train.pbtxt')
  ```

  Args:
    graph_or_graph_def: A `Graph` or a `GraphDef` protocol buffer.
    logdir: Directory where to write the graph. This can refer to remote
      filesystems, such as Google Cloud Storage (GCS).
    name: Filename for the graph.
    as_text: If `True`, writes the graph as an ASCII proto.

  Returns:
    The path of the output proto file.
  "
  [graph_or_graph_def logdir name  & {:keys [as_text]} ]
    (py/call-attr-kw train "write_graph" [graph_or_graph_def logdir name] {:as_text as_text }))
