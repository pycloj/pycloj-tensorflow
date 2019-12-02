(ns tensorflow.contrib.tpu
  "Ops related to Tensor Processing Units.

@@cross_replica_sum
@@infeed_dequeue
@@infeed_dequeue_tuple
@@infeed_enqueue
@@infeed_enqueue_tuple
@@outfeed_dequeue
@@outfeed_dequeue_tuple
@@outfeed_enqueue
@@outfeed_enqueue_tuple

@@initialize_system
@@shutdown_system
@@device_assignment
@@core
@@replicate
@@shard
@@batch_parallel
@@rewrite
@@outside_compilation

@@CrossShardOptimizer

@@InfeedQueue

@@DeviceAssignment
@@Topology

@@while_loop
@@repeat

@@TPUEstimator
@@TPUEstimatorSpec
@@export_estimator_savedmodel
@@RunConfig
@@InputPipelineConfig
@@TPUConfig
@@bfloat16_scope

@@TPUDistributionStrategy
@@keras_to_tpu_model

@@AsyncCheckpointSaverHook
@@TPUInMemoryEvalHook
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tpu (import-module "tensorflow.contrib.tpu"))

(defn batch-parallel 
  "Shards `computation` along the batch dimension for parallel execution.

  Convenience wrapper around shard().

  `inputs` must be a list of Tensors or None (equivalent to an empty list).
  Each input is split into `num_shards` pieces along the 0-th dimension, and
  computation is applied to each shard in parallel.

  Tensors are broadcast to all shards if they are lexically captured by
  `computation`. e.g.,

  x = tf.constant(7)
  def computation():
    return x + 3
  ... = shard(computation, ...)

  The outputs from all shards are concatenated back together along their 0-th
  dimension.

  Inputs and outputs of the computation must be at least rank-1 Tensors.

  Args:
    computation: A Python function that builds a computation to apply to each
      shard of the input.
    inputs: A list of input tensors or None (equivalent to an empty list). The
      0-th dimension of each Tensor must have size divisible by `num_shards`.
    num_shards: The number of shards.
    infeed_queue: If not `None`, the `InfeedQueue` from which to append a tuple
      of arguments as inputs to `computation`.
    device_assignment: If not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. Uses a default device assignment if `None`. The
      `DeviceAssignment` may be omitted if each shard of the computation uses
      only one core, and there is either only one shard, or the number of shards
      is equal to the number of cores in the TPU system.
    name: (Deprecated) Does nothing.
  Returns:
    A list of output tensors.
  Raises:
    ValueError: If `num_shards <= 0`
  "
  [computation inputs & {:keys [num_shards infeed_queue device_assignment name]
                       :or {infeed_queue None device_assignment None name None}} ]
    (py/call-attr-kw tpu "batch_parallel" [computation inputs] {:num_shards num_shards :infeed_queue infeed_queue :device_assignment device_assignment :name name }))

(defn bfloat16-scope 
  "Scope class for bfloat16 variables so that the model uses custom getter.

  This enables variables to be read as bfloat16 type when using get_variable.
  "
  [  ]
  (py/call-attr tpu "bfloat16_scope"  ))

(defn core 
  "Returns the device name for a core in a replicated TPU computation.

  Args:
    num: the virtual core number within each replica to which operators should
    be assigned.
  Returns:
    A device name, suitable for passing to `tf.device()`.
  "
  [ num ]
  (py/call-attr tpu "core"  num ))

(defn cross-replica-sum 
  "Sum the input tensor across replicas according to group_assignment.

  Args:
    x: The local tensor to the sum.
    group_assignment: Optional 2d int32 lists with shape [num_groups,
      num_replicas_per_group]. `group_assignment[i]` represents the replica
      ids in the ith subgroup.
    name: Optional op name.

  Returns:
    A `Tensor` which is summed across replicas.
  "
  [ x group_assignment name ]
  (py/call-attr tpu "cross_replica_sum"  x group_assignment name ))
(defn device-assignment 
  "Computes a device_assignment of a computation across a TPU topology.

  Attempts to choose a compact grid of cores for locality.

  Returns a `DeviceAssignment` that describes the cores in the topology assigned
  to each core of each replica.

  `computation_shape` and `computation_stride` values should be powers of 2 for
  optimal packing.

  Args:
    topology: A `Topology` object that describes the TPU cluster topology.
      To obtain a TPU topology, evaluate the `Tensor` returned by
      `initialize_system` using `Session.run`. Either a serialized
      `TopologyProto` or a `Topology` object may be passed. Note: you must
      evaluate the `Tensor` first; you cannot pass an unevaluated `Tensor` here.
    computation_shape: A rank 1 int32 numpy array with size equal to the
      topology rank, describing the shape of the computation's block of cores.
      If None, the `computation_shape` is `[1] * topology_rank`.
    computation_stride: A rank 1 int32 numpy array of size `topology_rank`,
      describing the inter-core spacing of the `computation_shape` cores in the
      TPU topology. If None, the `computation_stride` is `[1] * topology_rank`.
    num_replicas: The number of computation replicas to run. The replicas will
      be packed into the free spaces of the topology.

  Returns:
    A DeviceAssignment object, which describes the mapping between the logical
    cores in each computation replica and the physical cores in the TPU
    topology.

  Raises:
    ValueError: If `topology` is not a valid `Topology` object.
    ValueError: If `computation_shape` or `computation_stride` are not 1D int32
      numpy arrays with shape [3] where all values are positive.
    ValueError: If computation's replicas cannot fit into the TPU topology.
  "
  [topology computation_shape computation_stride  & {:keys [num_replicas]} ]
    (py/call-attr-kw tpu "device_assignment" [topology computation_shape computation_stride] {:num_replicas num_replicas }))

(defn export-estimator-savedmodel 
  "Export `Estimator` trained model for TPU inference.

  Args:
    estimator: `Estimator` with which model has been trained.
    export_dir_base: A string containing a directory in which to create
      timestamped subdirectories containing exported SavedModels.
    serving_input_receiver_fn: A function that takes no argument and returns a
      `ServingInputReceiver` or `TensorServingInputReceiver`.
    assets_extra: A dict specifying how to populate the assets.extra directory
      within the exported SavedModel, or `None` if no extra assets are needed.
    as_text: whether to write the SavedModel proto in text format.
    checkpoint_path: The checkpoint path to export.  If `None` (the default),
      the most recent checkpoint found within the model directory is chosen.

  Returns:
    The string path to the exported directory.
  "
  [estimator export_dir_base serving_input_receiver_fn assets_extra & {:keys [as_text checkpoint_path]
                       :or {checkpoint_path None}} ]
    (py/call-attr-kw tpu "export_estimator_savedmodel" [estimator export_dir_base serving_input_receiver_fn assets_extra] {:as_text as_text :checkpoint_path checkpoint_path }))

(defn infeed-dequeue 
  "A placeholder op for a value that will be fed into the computation.

  Args:
    dtype: A `tf.DType`. The type of elements in the tensor.
    shape: A `tf.TensorShape` or list of `ints`. The shape of the tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
    A tensor that will be provided using the infeed mechanism.

  Raises:
    TypeError: If 'dtype` is not a supported infeed type.
  "
  [ dtype shape name ]
  (py/call-attr tpu "infeed_dequeue"  dtype shape name ))

(defn infeed-dequeue-tuple 
  "A placeholder op for values fed into the TPU simultaneously as a tuple.

  Args:
    dtypes: A list of `tf.DType`s that has length `>= 1`.
      The element types of each element in `outputs`.
    shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      The shapes of each tensor in `outputs`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `dtypes`.
    A list of tensors that will be provided using the infeed mechanism.

  Raises:
    TypeError: If a type in 'dtypes` is not a supported infeed type.
  "
  [ dtypes shapes name ]
  (py/call-attr tpu "infeed_dequeue_tuple"  dtypes shapes name ))

(defn infeed-enqueue 
  "An op which feeds a single Tensor value into the computation.

  Args:
    input: A `Tensor`.
      A tensor that will be provided using the infeed mechanism.
    shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
      The shape of the tensor.
    layout: An optional list of `ints`. Defaults to `[]`.
      A vector holding the requested layout in minor-to-major sequence.
      If a layout attribute is passed, but its values are all -1, the layout will
      be computed by the infeed operation.
    device_ordinal: An optional `int`. Defaults to `-1`.
      The TPU device to use. This should be -1 when the Op
      is running on a TPU device, and >= 0 when the Op is running on the CPU
      device.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [input & {:keys [shape layout device_ordinal name]
                       :or {name None}} ]
    (py/call-attr-kw tpu "infeed_enqueue" [input] {:shape shape :layout layout :device_ordinal device_ordinal :name name }))

(defn infeed-enqueue-tuple 
  "Feeds multiple Tensor values into the computation as an XLA tuple.

  Args:
    inputs: A list of `Tensor` objects.
      A list of tensors that will be provided using the infeed mechanism.
    shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      The shapes of each tensor in `inputs`.
    layouts: An optional list of `ints`. Defaults to `[]`.
      A vector holding the requested layout in minor-to-major sequence for
      all the tuple shapes, in the order the shapes appear in the \"shapes\" input.
      The layout elements for a sub-shape can be set to -1, in which case the
      corresponding layout will be computed by the infeed operation.
    device_ordinal: An optional `int`. Defaults to `-1`.
      The TPU device to use. This should be -1 when the Op
      is running on a TPU device, and >= 0 when the Op is running on the CPU
      device.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [inputs shapes & {:keys [layouts device_ordinal name]
                       :or {name None}} ]
    (py/call-attr-kw tpu "infeed_enqueue_tuple" [inputs shapes] {:layouts layouts :device_ordinal device_ordinal :name name }))

(defn initialize-system 
  "Initializes a distributed TPU system for use with TensorFlow.

  Args:
    embedding_config: If not None, a `TPUEmbeddingConfiguration` proto
      describing the desired configuration of the hardware embedding lookup
      tables. If embedding_config is None, no hardware embeddings can be used.
    job: The job (the XXX in TensorFlow device specification /job:XXX) that
      contains the TPU devices that will be initialized. If job=None it is
      assumed there is only one job in the TensorFlow flock, and an error will
      be returned if this assumption does not hold.
  Returns:
    A serialized `TopologyProto` that describes the TPU system. Note:
      the topology must be evaluated using `Session.run` before it can be used.
  "
  [ embedding_config job ]
  (py/call-attr tpu "initialize_system"  embedding_config job ))

(defn keras-to-tpu-model 
  "Copy `model` along with weights to the TPU. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2019-02-20.
Instructions for updating:
Switch to tf.contrib.distribute.TPUStrategy. https://www.tensorflow.org/api_docs/python/tf/contrib/distribute/DistributionStrategy

Returns a TPU model.

Usage:
```
a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)

# If `num_cores_per_host` is greater than one, batch parallelism will be used
# to run on multiple TPU cores.
strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
model = keras_support.tpu_model(model, strategy)
model.compile(
    optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1.0),
    ...)
```

Args:
  model: A `tf.keras.Model` instance.
  strategy: `TPUDistributionStrategy`.  The strategy to use for replicating
    model across multiple TPU cores.

Returns:
  A new `KerasTPUModel` instance."
  [ model strategy ]
  (py/call-attr tpu "keras_to_tpu_model"  model strategy ))

(defn outfeed-dequeue 
  "Retrieves a single tensor from the computation outfeed.

  This operation will block indefinitely until data is available.

  Args:
    dtype: A `tf.DType`. The type of elements in the tensor.
    shape: A `tf.TensorShape` or list of `ints`. The shape of the tensor.
    device_ordinal: An optional `int`. Defaults to `-1`.
      The TPU device to use. This should be -1 when the Op
      is running on a TPU device, and >= 0 when the Op is running on the CPU
      device.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  "
  [dtype shape & {:keys [device_ordinal name]
                       :or {name None}} ]
    (py/call-attr-kw tpu "outfeed_dequeue" [dtype shape] {:device_ordinal device_ordinal :name name }))

(defn outfeed-dequeue-tuple 
  "Retrieve multiple values from the computation outfeed.

  This operation will block indefinitely until data is available. Output `i`
  corresponds to XLA tuple element `i`.

  Args:
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
      The element types of each element in `outputs`.
    shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      The shapes of each tensor in `outputs`.
    device_ordinal: An optional `int`. Defaults to `-1`.
      The TPU device to use. This should be -1 when the Op
      is running on a TPU device, and >= 0 when the Op is running on the CPU
      device.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `dtypes`.
  "
  [dtypes shapes & {:keys [device_ordinal name]
                       :or {name None}} ]
    (py/call-attr-kw tpu "outfeed_dequeue_tuple" [dtypes shapes] {:device_ordinal device_ordinal :name name }))

(defn outfeed-enqueue 
  "Enqueue a Tensor on the computation outfeed.

  Args:
    input: A `Tensor`. A tensor that will be inserted into the outfeed queue.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ input name ]
  (py/call-attr tpu "outfeed_enqueue"  input name ))

(defn outfeed-enqueue-tuple 
  "Enqueue multiple Tensor values on the computation outfeed.

  Args:
    inputs: A list of `Tensor` objects.
      A list of tensors that will be inserted into the outfeed queue as an
      XLA tuple.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  "
  [ inputs name ]
  (py/call-attr tpu "outfeed_enqueue_tuple"  inputs name ))

(defn outside-compilation 
  "Builds part of a computation outside any current TPU replicate scope.

  Args:
    computation: A Python function that builds the computation to
      place on the host.
    *args: the positional arguments for the computation.
    **kwargs: the keyword arguments for the computation.

  Returns:
    The Tensors returned by computation.
  "
  [ computation ]
  (py/call-attr tpu "outside_compilation"  computation ))

(defn repeat 
  "Builds a training loop that executes a fixed number of iterations.

  The set of loop-carried tensors correspond to `inputs`.
  `body` must be a function that takes and returns the values of the
  loop-carried tensors.

  Args:
    n: the number of loop iterations
    body: a Python function that builds the loop body.
    inputs: a list of initial values passed into the training loop or
      None (equivalent to an empty list).
    infeed_queue: if not None, the infeed queue from which to append a tuple
      of arguments as inputs to condition.
    name: (Deprecated) Does nothing.
  Returns:
    The final values of the loop-carried tensors.
  Raises:
    ValueError: if there is a type error.
  "
  [ n body inputs infeed_queue name ]
  (py/call-attr tpu "repeat"  n body inputs infeed_queue name ))

(defn replicate 
  "Builds a graph operator that runs a replicated TPU computation.

  Args:
    computation: A Python function that builds the computation to replicate.
    inputs: A list of lists of input tensors or `None` (equivalent to
      `[[]]`), indexed by `[replica_num][input_num]`. All replicas must
      have the same number of inputs. Each input can be a nested structure
      containing values that are convertible to tensors. Note that passing an
      N-dimension list of compatible values will result in a N-dimension list of
      scalar tensors rather than a single Rank-N tensors. If you need different
      behavior, convert part of inputs to tensors with `tf.convert_to_tensor`.
    infeed_queue: If not `None`, the `InfeedQueue` from which to append a tuple
      of arguments as inputs to computation.
    device_assignment: If not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. Uses a default device assignment if `None`. The
      `DeviceAssignment` may be omitted if each replica of the computation uses
      only one core, and there is either only one replica, or the number of
      replicas is equal to the number of cores in the TPU system.
    name: (Deprecated) Does nothing.
    maximum_shapes: A nested structure of tf.TensorShape representing the shape
      to which the respective component of each input element in each replica
      should be padded. Any unknown dimensions (e.g.
      tf.compat.v1.Dimension(None) in a tf.TensorShape or -1 in a tensor-like
      object) will be padded to the maximum size of that dimension over all
      replicas. The structure of `maximum_shapes` needs to be the same as
      `inputs[0]`.
  Returns:
    A list of outputs, indexed by `[replica_num]` each output can be a nested
    structure same as what computation() returns with a few exceptions.

    Exceptions include:
      1) None output: a NoOp would be returned which control-depends on
         computation.
      2) Single value output: A tuple containing the value would be returned.
      3) Operation-only outputs: a NoOp would be returned which
         control-depends on computation.
      TODO(b/121383831): Investigate into removing these special cases.

  Raises:
    ValueError: If all replicas do not have equal numbers of input tensors.
    ValueError: If the number of inputs per replica does not match
      the number of formal parameters to `computation`.
    ValueError: If the static `inputs` dimensions don't match with the values
      given in `maximum_shapes`.
    ValueError: If the structure of inputs per replica does not match
      the structure of `maximum_shapes`.
  "
  [ computation inputs infeed_queue device_assignment name maximum_shapes ]
  (py/call-attr tpu "replicate"  computation inputs infeed_queue device_assignment name maximum_shapes ))

(defn rewrite 
  "Rewrites `computation` for execution on a TPU system.

  Args:
    computation: A Python function that builds a computation to apply to the
      input. If the function takes n inputs, 'inputs' should be a list of n
      tensors.

      `computation` may return a list of operations and tensors. Tensors must
      come before operations in the returned list.  The return value of
      `rewrite` is a list of tensors corresponding to the tensors from the
      output of `computation`.

      All `Operation`s constructed during `computation` will be executed when
      evaluating any of the returned output tensors, not just the ones returned.
    inputs: A list of input tensors or `None` (equivalent to an empty list).
      Each input can be a nested structure containing values that are
      convertible to tensors. Note that passing an N-dimension list of
      compatible values will result in a N-dimention list of scalar tensors
      rather than a single Rank-N tensors. If you need different behavior,
      convert part of inputs to tensors with `tf.convert_to_tensor`.
    infeed_queue: If not `None`, the `InfeedQueue` from which to append a tuple
      of arguments as inputs to `computation`.
    device_assignment: if not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. May be omitted for a single-core computation, in which
      case the core attached to task 0, TPU device 0 is used.
    name: (Deprecated) Does nothing.
  Returns:
    Same data structure as if computation(*inputs) is called directly with some
    exceptions for correctness. Exceptions include:
      1) None output: a NoOp would be returned which control-depends on
         computation.
      2) Single value output: A tuple containing the value would be returned.
      3) Operation-only outputs: a NoOp would be returned which
         control-depends on computation.
      TODO(b/121383831): Investigate into removing these special cases.
  "
  [ computation inputs infeed_queue device_assignment name ]
  (py/call-attr tpu "rewrite"  computation inputs infeed_queue device_assignment name ))

(defn shard 
  "Shards `computation` for parallel execution.

  `inputs` must be a list of Tensors or None (equivalent to an empty list), each
  of which has a corresponding split axis (from `input_shard_axes`). Each input
  is split into `num_shards` pieces along the corresponding axis, and
  computation is applied to each shard in parallel.

  Tensors are broadcast to all shards if they are lexically captured by
  `computation`. e.g.,

  x = tf.constant(7)
  def computation():
    return x + 3
  ... = shard(computation, ...)

  TODO(phawkins): consider adding support for broadcasting Tensors passed
  as inputs.

  If `outputs_from_all_shards` is true, the outputs from all shards of
  `computation` are concatenated back together along their `output_shards_axes`.
  Otherwise, each output is taken from an arbitrary shard.

  Inputs and outputs of the computation must be at least rank-1 Tensors.

  Args:
    computation: A Python function that builds a computation to apply to each
      shard of the input.
    inputs: A list of input tensors or None (equivalent to an empty list). Each
      input tensor has a corresponding shard axes, given by `input_shard_axes`,
      which must have size divisible by `num_shards`.
    num_shards: The number of shards.
    input_shard_axes: A list of dimensions along which to shard `inputs`, or
      `None`. `None` means \"shard all inputs along dimension 0\". If not `None`,
      there must be one dimension per input.
    outputs_from_all_shards: Boolean or list of boolean. For each output, if
      `True`, outputs from all shards are concatenated along the corresponding
      `output_shard_axes` entry. Otherwise, each output is taken
      from an arbitrary shard. If the argument is a boolean, the argument's
      value is used for each output.
    output_shard_axes: A list of dimensions along which to concatenate the
      outputs of `computation`, or `None`. `None` means \"concatenate all outputs
      along dimension 0\". If not `None`, there must be one dimension per output.
      Ignored if `outputs_from_all_shards` is False.
    infeed_queue: If not `None`, the `InfeedQueue` to use to augment the inputs
      of `computation`.
    device_assignment: If not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. Uses a default device assignment if `None`. The
      `DeviceAssignment` may be omitted if each shard of the computation uses
      only one core, and there is either only one shard, or the number of shards
      is equal to the number of cores in the TPU system.
    name: (Deprecated) Does nothing.
  Returns:
    A list of output tensors.
  Raises:
    ValueError: If num_shards <= 0
    ValueError: If len(input_shard_axes) != len(inputs)
    ValueError: If len(output_shard_axes) != len(outputs from `computation`)
  "
  [computation inputs & {:keys [num_shards input_shard_axes outputs_from_all_shards output_shard_axes infeed_queue device_assignment name]
                       :or {input_shard_axes None output_shard_axes None infeed_queue None device_assignment None name None}} ]
    (py/call-attr-kw tpu "shard" [computation inputs] {:num_shards num_shards :input_shard_axes input_shard_axes :outputs_from_all_shards outputs_from_all_shards :output_shard_axes output_shard_axes :infeed_queue infeed_queue :device_assignment device_assignment :name name }))

(defn shutdown-system 
  "Shuts down a running a distributed TPU system.

  Args:
    job: The job (the XXX in TensorFlow device specification /job:XXX) that
      contains the TPU devices that will be shutdown. If job=None it is
      assumed there is only one job in the TensorFlow flock, and an error will
      be returned if this assumption does not hold.
  "
  [ job ]
  (py/call-attr tpu "shutdown_system"  job ))

(defn while-loop 
  "Builds a training loop for TPUs.

  The set of loop-carried tensors corresponds to `inputs`.  Both
  `condition` and `body` take the current value of the loop-carried
  tensors. 'body' additionally takes a tuple of infeed from
  infeed_queue if infeed_queue is not None. `condition` must return a
  single boolean value that determines whether iteration
  continues. `body` must return an updated list of values for the
  loop-carried tensors.

  Args:
    condition: a Python function that builds the loop condition.
    body: a Python function that builds the loop body.
    inputs: a list of initial values passed into the training loop, or
      None (equivalent to an empty list).
    infeed_queue: if not None, the infeed queue from which to append a tuple
      of arguments as inputs to condition.
    name: (Deprecated) Does nothing.

  Returns:
    The final values of the loop-carried tensors.

  Raises:
    TypeError: if body or condition has the wrong signature.
  "
  [ condition body inputs infeed_queue name ]
  (py/call-attr tpu "while_loop"  condition body inputs infeed_queue name ))
