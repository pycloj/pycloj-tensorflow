(ns tensorflow.-api.v1.compat.v2.distribute.experimental.CentralStorageStrategy
  "A one-machine strategy that puts all variables on a single device.

  Variables are assigned to local CPU or the only GPU. If there is more
  than one GPU, compute operations (other than variable update operations)
  will be replicated across all GPUs.

  For Example:
  ```
  strategy = tf.distribute.experimental.CentralStorageStrategy()
  # Create a dataset
  ds = tf.data.Dataset.range(5).batch(2)
  # Distribute that dataset
  dist_dataset = strategy.experimental_distribute_dataset(ds)

  with strategy.scope():
    @tf.function
    def train_step(val):
      return val + 1

    # Iterate over the distributed dataset
    for x in dist_dataset:
      # process dataset elements
      strategy.experimental_run_v2(train_step, args=(x,))
  ```
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.compat.v2.distribute.experimental"))

(defn CentralStorageStrategy 
  "A one-machine strategy that puts all variables on a single device.

  Variables are assigned to local CPU or the only GPU. If there is more
  than one GPU, compute operations (other than variable update operations)
  will be replicated across all GPUs.

  For Example:
  ```
  strategy = tf.distribute.experimental.CentralStorageStrategy()
  # Create a dataset
  ds = tf.data.Dataset.range(5).batch(2)
  # Distribute that dataset
  dist_dataset = strategy.experimental_distribute_dataset(ds)

  with strategy.scope():
    @tf.function
    def train_step(val):
      return val + 1

    # Iterate over the distributed dataset
    for x in dist_dataset:
      # process dataset elements
      strategy.experimental_run_v2(train_step, args=(x,))
  ```
  "
  [ compute_devices parameter_device ]
  (py/call-attr experimental "CentralStorageStrategy"  compute_devices parameter_device ))

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
  "Distributes a tf.data.Dataset instance provided via dataset.

    The returned dataset is a wrapped strategy dataset which creates a
    multidevice iterator under the hood. It prefetches the input data to the
    specified devices on the worker. The returned distributed dataset can be
    iterated over similar to how regular datasets can.

    NOTE: Currently, the user cannot add any more transformations to a
    distributed dataset.

    For Example:
    ```
    strategy = tf.distribute.CentralStorageStrategy()  # with 1 CPU and 1 GPU
    dataset = tf.data.Dataset.range(10).batch(2)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    for x in dist_dataset:
      print(x)  # Prints PerReplica values [0, 1], [2, 3],...

    ```
    Args:
      dataset: `tf.data.Dataset` to be prefetched to device.

    Returns:
      A \"distributed `Dataset`\" that the caller can iterate over.
    "
  [ self dataset ]
  (py/call-attr self "experimental_distribute_dataset"  self dataset ))

(defn experimental-distribute-datasets-from-function 
  "Distributes `tf.data.Dataset` instances created by calls to `dataset_fn`.

    `dataset_fn` will be called once for each worker in the strategy. In this
    case, we only have one worker so `dataset_fn` is called once. Each replica
    on this worker will then dequeue a batch of elements from this local
    dataset.

    The `dataset_fn` should take an `tf.distribute.InputContext` instance where
    information about batching and input replication can be accessed.

    For Example:
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
      A \"distributed `Dataset`\", which the caller can iterate over like regular
      datasets.
    "
  [ self dataset_fn ]
  (py/call-attr self "experimental_distribute_datasets_from_function"  self dataset_fn ))

(defn experimental-local-results 
  "Returns the list of all local per-replica values contained in `value`.

    In `CentralStorageStrategy` there is a single worker so the value returned
    will be all the values on that worker.

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
  "Makes a `tf.data.Dataset` for input provided via a numpy array.

    This avoids adding `numpy_input` as a large constant in the graph,
    and copies the data to the machine or machines that will be processing
    the input.

    Note that you will likely need to use `experimental_distribute_dataset`
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

    Returns:
      A `tf.data.Dataset` representing `numpy_input`.
    "
  [ self numpy_input ]
  (py/call-attr self "experimental_make_numpy_dataset"  self numpy_input ))

(defn experimental-run 
  "DEPRECATED TF 1.x ONLY."
  [ self fn input_iterator ]
  (py/call-attr self "experimental_run"  self fn input_iterator ))

(defn experimental-run-v2 
  "Run `fn` on each replica, with the given arguments.

    In `CentralStorageStrategy`, `fn` is  called on each of the compute
    replicas, with the provided \"per replica\" arguments specific to that device.

    Args:
      fn: The function to run. The output must be a `tf.nest` of `Tensor`s.
      args: (Optional) Positional arguments to `fn`.
      kwargs: (Optional) Keyword arguments to `fn`.

    Returns:
      Return value from running `fn`.
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
  "DEPRECATED TF 1.x ONLY."
  [ self dataset ]
  (py/call-attr self "make_dataset_iterator"  self dataset ))
(defn make-input-fn-iterator 
  "DEPRECATED TF 1.x ONLY."
  [self input_fn  & {:keys [replication_mode]} ]
    (py/call-attr-kw self "make_input_fn_iterator" [input_fn] {:replication_mode replication_mode }))

(defn num-replicas-in-sync 
  "Returns number of replicas over which gradients are aggregated."
  [ self ]
    (py/call-attr self "num_replicas_in_sync"))

(defn reduce 
  "Reduce `value` across replicas.

    Given a per-replica value returned by `experimental_run_v2`, say a
    per-example loss, the batch will be divided across all the replicas. This
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

    For Example:
    ```
    strategy = tf.distribute.experimental.CentralStorageStrategy(
        compute_devices=['CPU:0', 'GPU:0'], parameter_device='CPU:0')
    ds = tf.data.Dataset.range(10)
    # Distribute that dataset
    dist_dataset = strategy.experimental_distribute_dataset(ds)

    with strategy.scope():
      @tf.function
      def train_step(val):
        # pass through
        return val

      # Iterate over the distributed dataset
      for x in dist_dataset:
        result = strategy.experimental_run_v2(train_step, args=(x,))

    result = strategy.reduce(tf.distribute.ReduceOp.SUM, result,
                             axis=None).numpy()
    # result: array([ 4,  6,  8, 10])

    result = strategy.reduce(tf.distribute.ReduceOp.SUM, result, axis=0).numpy()
    # result: 28
    ```

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
  "DEPRECATED TF 1.x ONLY."
  [ self config_proto ]
  (py/call-attr self "update_config_proto"  self config_proto ))
