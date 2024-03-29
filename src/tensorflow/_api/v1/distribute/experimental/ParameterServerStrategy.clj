(ns tensorflow.-api.v1.distribute.experimental.ParameterServerStrategy
  "An asynchronous multi-worker parameter server tf.distribute strategy.

  This strategy requires two jobs: workers and parameter servers. Variables and
  updates to those variables will be assigned to parameter servers and other
  operations are assigned to workers.

  When each worker has more than one GPU, operations will be replicated on all
  GPUs. Even though operations may be replicated, variables are not and each
  worker shares a common view for which parameter server a variable is assigned
  to.

  By default it uses `TFConfigClusterResolver` to detect configurations for
  multi-worker training. This requires a 'TF_CONFIG' environment variable and
  the 'TF_CONFIG' must have a cluster spec.

  This class assumes each worker is running the same code independently, but
  parameter servers are running a standard server. This means that while each
  worker will synchronously compute a single gradient update across all GPUs,
  updates between workers proceed asynchronously. Operations that occur only on
  the first replica (such as incrementing the global step), will occur on the
  first replica *of every worker*.

  It is expected to call `call_for_each_replica(fn, ...)` for any
  operations which potentially can be replicated across replicas (i.e. multiple
  GPUs) even if there is only CPU or one GPU. When defining the `fn`, extra
  caution needs to be taken:

  1) It is generally not recommended to open a device scope under the strategy's
  scope. A device scope (i.e. calling `tf.device`) will be merged with or
  override the device for operations but will not change the device for
  variables.

  2) It is also not recommended to open a colocation scope (i.e. calling
  `tf.compat.v1.colocate_with`) under the strategy's scope. For colocating
  variables, use `strategy.extended.colocate_vars_with` instead. Colocation of
  ops will possibly create device assignment conflicts.

  Note: This strategy only works with the Estimator API. Pass an instance of
  this strategy to the `experimental_distribute` argument when you create the
  `RunConfig`. This instance of `RunConfig` should then be passed to the
  `Estimator` instance on which `train_and_evaluate` is called.

  For Example:
  ```
  strategy = tf.distribute.experimental.ParameterServerStrategy()
  run_config = tf.estimator.RunConfig(
      experimental_distribute.train_distribute=strategy)
  estimator = tf.estimator.Estimator(config=run_config)
  tf.estimator.train_and_evaluate(estimator,...)
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.distribute.experimental"))

(defn ParameterServerStrategy 
  "An asynchronous multi-worker parameter server tf.distribute strategy.

  This strategy requires two jobs: workers and parameter servers. Variables and
  updates to those variables will be assigned to parameter servers and other
  operations are assigned to workers.

  When each worker has more than one GPU, operations will be replicated on all
  GPUs. Even though operations may be replicated, variables are not and each
  worker shares a common view for which parameter server a variable is assigned
  to.

  By default it uses `TFConfigClusterResolver` to detect configurations for
  multi-worker training. This requires a 'TF_CONFIG' environment variable and
  the 'TF_CONFIG' must have a cluster spec.

  This class assumes each worker is running the same code independently, but
  parameter servers are running a standard server. This means that while each
  worker will synchronously compute a single gradient update across all GPUs,
  updates between workers proceed asynchronously. Operations that occur only on
  the first replica (such as incrementing the global step), will occur on the
  first replica *of every worker*.

  It is expected to call `call_for_each_replica(fn, ...)` for any
  operations which potentially can be replicated across replicas (i.e. multiple
  GPUs) even if there is only CPU or one GPU. When defining the `fn`, extra
  caution needs to be taken:

  1) It is generally not recommended to open a device scope under the strategy's
  scope. A device scope (i.e. calling `tf.device`) will be merged with or
  override the device for operations but will not change the device for
  variables.

  2) It is also not recommended to open a colocation scope (i.e. calling
  `tf.compat.v1.colocate_with`) under the strategy's scope. For colocating
  variables, use `strategy.extended.colocate_vars_with` instead. Colocation of
  ops will possibly create device assignment conflicts.

  Note: This strategy only works with the Estimator API. Pass an instance of
  this strategy to the `experimental_distribute` argument when you create the
  `RunConfig`. This instance of `RunConfig` should then be passed to the
  `Estimator` instance on which `train_and_evaluate` is called.

  For Example:
  ```
  strategy = tf.distribute.experimental.ParameterServerStrategy()
  run_config = tf.estimator.RunConfig(
      experimental_distribute.train_distribute=strategy)
  estimator = tf.estimator.Estimator(config=run_config)
  tf.estimator.train_and_evaluate(estimator,...)
  "
  [ cluster_resolver ]
  (py/call-attr experimental "ParameterServerStrategy"  cluster_resolver ))

(defn colocate-vars-with 
  "DEPRECATED: use extended.colocate_vars_with() instead."
  [ self colocate_with_variable ]
  (py/call-attr self "colocate_vars_with"  self colocate_with_variable ))

(defn configure 
  "DEPRECATED: use `update_config_proto` instead.

    Configures the strategy class.

    DEPRECATED: This method's functionality has been split into the strategy
    constructor and `update_config_proto`. In the future, we will allow passing
    cluster and config_proto to the constructor to configure the strategy. And
    `update_config_proto` can be used to update the config_proto based on the
    specific strategy.
    "
  [ self session_config cluster_spec task_type task_id ]
  (py/call-attr self "configure"  self session_config cluster_spec task_type task_id ))

(defn experimental-distribute-dataset 
  "Distributes a tf.data.Dataset instance provided via `dataset`.

    The returned distributed dataset can be iterated over similar to how
    regular datasets can.
    NOTE: Currently, the user cannot add any more transformations to a
    distributed dataset.

    The following is an example:

    ```python
    strategy = tf.distribute.MirroredStrategy()

    # Create a dataset
    dataset = dataset_ops.Dataset.TFRecordDataset([
      \"/a/1.tfr\", \"/a/2.tfr\", \"/a/3.tfr\", \"/a/4.tfr\"])

    # Distribute that dataset
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    # Iterate over the distributed dataset
    for x in dist_dataset:
      # process dataset elements
      strategy.experimental_run_v2(train_step, args=(x,))
    ```

    We will assume that the input dataset is batched by the
    global batch size. With this assumption, we will make a best effort to
    divide each batch across all the replicas (one or more workers).

    In a multi-worker setting, we will first attempt to distribute the dataset
    by attempting to detect whether the dataset is being created out of
    ReaderDatasets (e.g. TFRecordDataset, TextLineDataset, etc.) and if so,
    attempting to shard the input files. Note that there has to be at least one
    input file per worker. If you have less than one input file per worker, we
    suggest that you should disable distributing your dataset using the method
    below.

    If that attempt is unsuccessful (e.g. the dataset is created from a
    Dataset.range), we will shard the dataset evenly at the end by appending a
    `.shard` operation to the end of the processing pipeline. This will cause
    the entire preprocessing pipeline for all the data to be run on every
    worker, and each worker will do redundant work. We will print a warning
    if this method of sharding is selected. In this case, consider using
    `experimental_distribute_datasets_from_function` instead.

    You can disable dataset sharding across workers using the `auto_shard`
    option in `tf.data.experimental.DistributeOptions`.

    Within each worker, we will also split the data among all the worker
    devices (if more than one a present), and this will happen even if
    multi-worker sharding is disabled using the method above.

    If the above batch splitting and dataset sharding logic is undesirable,
    please use `experimental_distribute_datasets_from_function` instead, which
    does not do any automatic splitting or sharding.

    Args:
      dataset: `tf.data.Dataset` that will be sharded across all replicas using
        the rules stated above.

    Returns:
      A \"distributed `Dataset`\", which acts like a `tf.data.Dataset` except
      it produces \"per-replica\" values.
    "
  [ self dataset ]
  (py/call-attr self "experimental_distribute_dataset"  self dataset ))

(defn experimental-distribute-datasets-from-function 
  "Distributes `tf.data.Dataset` instances created by calls to `dataset_fn`.

    `dataset_fn` will be called once for each worker in the strategy. Each
    replica on that worker will dequeue one batch of inputs from the local
    `Dataset` (i.e. if a worker has two replicas, two batches will be dequeued
    from the `Dataset` every step).

    This method can be used for several purposes. For example, where
    `experimental_distribute_dataset` is unable to shard the input files, this
    method might be used to manually shard the dataset (avoiding the slow
    fallback behavior in `experimental_distribute_dataset`). In cases where the
    dataset is infinite, this sharding can be done by creating dataset replicas
    that differ only in their random seed.
    `experimental_distribute_dataset` may also sometimes fail to split the
    batch across replicas on a worker. In that case, this method can be used
    where that limitation does not exist.

    The `dataset_fn` should take an `tf.distribute.InputContext` instance where
    information about batching and input replication can be accessed:

    ```
    def dataset_fn(input_context):
      batch_size = input_context.get_per_replica_batch_size(global_batch_size)
      d = tf.data.Dataset.from_tensors([[1.]]).repeat().batch(batch_size)
      return d.shard(
          input_context.num_input_pipelines, input_context.input_pipeline_id)

    inputs = strategy.experimental_distribute_datasets_from_function(dataset_fn)

    for batch in inputs:
      replica_results = strategy.experimental_run_v2(replica_fn, args=(batch,))
    ```

    IMPORTANT: The `tf.data.Dataset` returned by `dataset_fn` should have a
    per-replica batch size, unlike `experimental_distribute_dataset`, which uses
    the global batch size.  This may be computed using
    `input_context.get_per_replica_batch_size`.

    Args:
      dataset_fn: A function taking a `tf.distribute.InputContext` instance and
        returning a `tf.data.Dataset`.

    Returns:
      A \"distributed `Dataset`\", which acts like a `tf.data.Dataset` except
      it produces \"per-replica\" values.
    "
  [ self dataset_fn ]
  (py/call-attr self "experimental_distribute_datasets_from_function"  self dataset_fn ))

(defn experimental-local-results 
  "Returns the list of all local per-replica values contained in `value`.

    Note: This only returns values on the worker initiated by this client.
    When using a `tf.distribute.Strategy` like
    `tf.distribute.experimental.MultiWorkerMirroredStrategy`, each worker
    will be its own client, and this function will only return values
    computed on that worker.

    Args:
      value: A value returned by `experimental_run()`, `experimental_run_v2()`,
        `extended.call_for_each_replica()`, or a variable created in `scope`.

    Returns:
      A tuple of values contained in `value`. If `value` represents a single
      value, this returns `(value,).`
    "
  [ self value ]
  (py/call-attr self "experimental_local_results"  self value ))

(defn experimental-make-numpy-dataset 
  "Makes a tf.data.Dataset for input provided via a numpy array.

    This avoids adding `numpy_input` as a large constant in the graph,
    and copies the data to the machine or machines that will be processing
    the input.

    Note that you will likely need to use
    tf.distribute.Strategy.experimental_distribute_dataset
    with the returned dataset to further distribute it with the strategy.

    Example:
    ```
    numpy_input = np.ones([10], dtype=np.float32)
    dataset = strategy.experimental_make_numpy_dataset(numpy_input)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    ```

    Args:
      numpy_input: A nest of NumPy input arrays that will be converted into a
      dataset. Note that lists of Numpy arrays are stacked, as that is normal
      `tf.data.Dataset` behavior.
      session: (TensorFlow v1.x graph execution only) A session used for
        initialization.

    Returns:
      A `tf.data.Dataset` representing `numpy_input`.
    "
  [ self numpy_input session ]
  (py/call-attr self "experimental_make_numpy_dataset"  self numpy_input session ))

(defn experimental-run 
  "Runs ops in `fn` on each replica, with inputs from `input_iterator`.

    DEPRECATED: This method is not available in TF 2.x. Please switch
    to using `experimental_run_v2` instead.

    When eager execution is enabled, executes ops specified by `fn` on each
    replica. Otherwise, builds a graph to execute the ops on each replica.

    Each replica will take a single, different input from the inputs provided by
    one `get_next` call on the input iterator.

    `fn` may call `tf.distribute.get_replica_context()` to access members such
    as `replica_id_in_sync_group`.

    IMPORTANT: Depending on the `tf.distribute.Strategy` implementation being
    used, and whether eager execution is enabled, `fn` may be called one or more
    times (once for each replica).

    Args:
      fn: The function to run. The inputs to the function must match the outputs
        of `input_iterator.get_next()`. The output must be a `tf.nest` of
        `Tensor`s.
      input_iterator: (Optional) input iterator from which the inputs are taken.

    Returns:
      Merged return value of `fn` across replicas. The structure of the return
      value is the same as the return value from `fn`. Each element in the
      structure can either be `PerReplica` (if the values are unsynchronized),
      `Mirrored` (if the values are kept in sync), or `Tensor` (if running on a
      single replica).
    "
  [ self fn input_iterator ]
  (py/call-attr self "experimental_run"  self fn input_iterator ))

(defn experimental-run-v2 
  "Run `fn` on each replica, with the given arguments.

    Executes ops specified by `fn` on each replica. If `args` or `kwargs` have
    \"per-replica\" values, such as those produced by a \"distributed `Dataset`\",
    when `fn` is executed on a particular replica, it will be executed with the
    component of those \"per-replica\" values that correspond to that replica.

    `fn` may call `tf.distribute.get_replica_context()` to access members such
    as `all_reduce`.

    All arguments in `args` or `kwargs` should either be nest of tensors or
    per-replica objects containing tensors or composite tensors.

    IMPORTANT: Depending on the implementation of `tf.distribute.Strategy` and
    whether eager execution is enabled, `fn` may be called one or more times (
    once for each replica).

    Args:
      fn: The function to run. The output must be a `tf.nest` of `Tensor`s.
      args: (Optional) Positional arguments to `fn`.
      kwargs: (Optional) Keyword arguments to `fn`.

    Returns:
      Merged return value of `fn` across replicas. The structure of the return
      value is the same as the return value from `fn`. Each element in the
      structure can either be \"per-replica\" `Tensor` objects or `Tensor`s
      (for example, if running on a single replica).
    "
  [self fn & {:keys [args kwargs]
                       :or {kwargs None}} ]
    (py/call-attr-kw self "experimental_run_v2" [fn] {:args args :kwargs kwargs }))

(defn extended 
  "`tf.distribute.StrategyExtended` with additional methods."
  [ self ]
    (py/call-attr self "extended"))

(defn group 
  "Shortcut for `tf.group(self.experimental_local_results(value))`."
  [ self value name ]
  (py/call-attr self "group"  self value name ))

(defn make-dataset-iterator 
  "Makes an iterator for input provided via `dataset`.

    DEPRECATED: This method is not available in TF 2.x.

    Data from the given dataset will be distributed evenly across all the
    compute replicas. We will assume that the input dataset is batched by the
    global batch size. With this assumption, we will make a best effort to
    divide each batch across all the replicas (one or more workers).
    If this effort fails, an error will be thrown, and the user should instead
    use `make_input_fn_iterator` which provides more control to the user, and
    does not try to divide a batch across replicas.

    The user could also use `make_input_fn_iterator` if they want to
    customize which input is fed to which replica/worker etc.

    Args:
      dataset: `tf.data.Dataset` that will be distributed evenly across all
        replicas.

    Returns:
      An `tf.distribute.InputIterator` which returns inputs for each step of the
      computation.  User should call `initialize` on the returned iterator.
    "
  [ self dataset ]
  (py/call-attr self "make_dataset_iterator"  self dataset ))
(defn make-input-fn-iterator 
  "Returns an iterator split across replicas created from an input function.

    DEPRECATED: This method is not available in TF 2.x.

    The `input_fn` should take an `tf.distribute.InputContext` object where
    information about batching and input sharding can be accessed:

    ```
    def input_fn(input_context):
      batch_size = input_context.get_per_replica_batch_size(global_batch_size)
      d = tf.data.Dataset.from_tensors([[1.]]).repeat().batch(batch_size)
      return d.shard(input_context.num_input_pipelines,
                     input_context.input_pipeline_id)
    with strategy.scope():
      iterator = strategy.make_input_fn_iterator(input_fn)
      replica_results = strategy.experimental_run(replica_fn, iterator)
    ```

    The `tf.data.Dataset` returned by `input_fn` should have a per-replica
    batch size, which may be computed using
    `input_context.get_per_replica_batch_size`.

    Args:
      input_fn: A function taking a `tf.distribute.InputContext` object and
        returning a `tf.data.Dataset`.
      replication_mode: an enum value of `tf.distribute.InputReplicationMode`.
        Only `PER_WORKER` is supported currently, which means there will be
        a single call to `input_fn` per worker. Replicas will dequeue from the
        local `tf.data.Dataset` on their worker.

    Returns:
      An iterator object that should first be `.initialize()`-ed. It may then
      either be passed to `strategy.experimental_run()` or you can
      `iterator.get_next()` to get the next value to pass to
      `strategy.extended.call_for_each_replica()`.
    "
  [self input_fn  & {:keys [replication_mode]} ]
    (py/call-attr-kw self "make_input_fn_iterator" [input_fn] {:replication_mode replication_mode }))

(defn num-replicas-in-sync 
  "Returns number of replicas over which gradients are aggregated."
  [ self ]
    (py/call-attr self "num_replicas_in_sync"))

(defn reduce 
  "Reduce `value` across replicas.

    Given a per-replica value returned by `experimental_run_v2`, say a
    per-example loss, the batch will be divided across all the replicas.  This
    function allows you to aggregate across replicas and optionally also across
    batch elements.  For example, if you have a global batch size of 8 and 2
    replicas, values for examples `[0, 1, 2, 3]` will be on replica 0 and
    `[4, 5, 6, 7]` will be on replica 1. By default, `reduce` will just
    aggregate across replicas, returning `[0+4, 1+5, 2+6, 3+7]`. This is useful
    when each replica is computing a scalar or some other value that doesn't
    have a \"batch\" dimension (like a gradient). More often you will want to
    aggregate across the global batch, which you can get by specifying the batch
    dimension as the `axis`, typically `axis=0`. In this case it would return a
    scalar `0+1+2+3+4+5+6+7`.

    If there is a last partial batch, you will need to specify an axis so
    that the resulting shape is consistent across replicas. So if the last
    batch has size 6 and it is divided into [0, 1, 2, 3] and [4, 5], you
    would get a shape mismatch unless you specify `axis=0`. If you specify
    `tf.distribute.ReduceOp.MEAN`, using `axis=0` will use the correct
    denominator of 6. Contrast this with computing `reduce_mean` to get a
    scalar value on each replica and this function to average those means,
    which will weigh some values `1/8` and others `1/4`.

    Args:
      reduce_op: A `tf.distribute.ReduceOp` value specifying how values should
        be combined.
      value: A \"per replica\" value, e.g. returned by `experimental_run_v2` to
        be combined into a single tensor.
      axis: Specifies the dimension to reduce along within each
        replica's tensor. Should typically be set to the batch dimension, or
        `None` to only reduce across replicas (e.g. if the tensor has no batch
        dimension).

    Returns:
      A `Tensor`.
    "
  [ self reduce_op value axis ]
  (py/call-attr self "reduce"  self reduce_op value axis ))

(defn scope 
  "Returns a context manager selecting this Strategy as current.

    Inside a `with strategy.scope():` code block, this thread
    will use a variable creator set by `strategy`, and will
    enter its \"cross-replica context\".

    Returns:
      A context manager.
    "
  [ self  ]
  (py/call-attr self "scope"  self  ))

(defn unwrap 
  "Returns the list of all local per-replica values contained in `value`.

    DEPRECATED: Please use `experimental_local_results` instead.

    Note: This only returns values on the workers initiated by this client.
    When using a `tf.distribute.Strategy` like
    `tf.distribute.experimental.MultiWorkerMirroredStrategy`, each worker
    will be its own client, and this function will only return values
    computed on that worker.

    Args:
      value: A value returned by `experimental_run()`,
        `extended.call_for_each_replica()`, or a variable created in `scope`.

    Returns:
      A tuple of values contained in `value`. If `value` represents a single
      value, this returns `(value,).`
    "
  [ self value ]
  (py/call-attr self "unwrap"  self value ))

(defn update-config-proto 
  "Returns a copy of `config_proto` modified for use with this strategy.

    DEPRECATED: This method is not available in TF 2.x.

    The updated config has something needed to run a strategy, e.g.
    configuration to run collective ops, or device filters to improve
    distributed training performance.

    Args:
      config_proto: a `tf.ConfigProto` object.

    Returns:
      The updated copy of the `config_proto`.
    "
  [ self config_proto ]
  (py/call-attr self "update_config_proto"  self config_proto ))
