(ns tensorflow.distribute.StrategyExtended
  "Additional APIs for algorithms that need to be distribution-aware.

  Note: For most usage of `tf.distribute.Strategy`, there should be no need to
  call these methods, since TensorFlow libraries (such as optimizers) already
  call these methods when needed on your behalf.

  Lower-level concepts:

  * Wrapped values: In order to represent values parallel across devices
    (either replicas or the devices associated with a particular value), we
    wrap them in a \"PerReplica\" or \"Mirrored\" object that contains a map
    from replica id to values. \"PerReplica\" is used when the value may be
    different across replicas, and \"Mirrored\" when the value are the same.
  * Unwrapping and merging: Consider calling a function `fn` on multiple
    replicas, like `experimental_run_v2(fn, args=[w])` with an
    argument `w` that is a wrapped value. This means `w` will have a map taking
    replica id `0` to `w0`, replica id `11` to `w1`, etc.
    `experimental_run_v2()` unwraps `w` before calling `fn`, so
    it calls `fn(w0)` on `d0`, `fn(w1)` on `d1`, etc.  It then merges the return
    values from `fn()`, which can possibly result in wrapped values. For
    example, let's say `fn()` returns a tuple with three components: `(x, a,
    v0)` from replica 0, `(x, b, v1)` on replica 1, etc. If the first component
    is the same object `x` from every replica, then the first component of the
    merged result will also be `x`. If the second component is different (`a`,
    `b`, ...)  from each replica, then the merged value will have a wrapped map
    from replica device to the different values. If the third component is the
    members of a mirrored variable (`v` maps `d0` to `v0`, `d1` to `v1`, etc.),
    then the merged result will be that mirrored variable (`v`).
  * Worker devices vs. parameter devices: Most replica computations will
    happen on worker devices. Since we don't yet support model
    parallelism, there will be one worker device per replica. When using
    parameter servers or central storage, the set of devices holding
    variables may be different, otherwise the parameter devices might
    match the worker devices.

  *Replica context vs. Cross-replica context*

  _replica context_ is when we are in some function that is being called once
  for each replica.  Otherwise we are in cross-replica context, which is
  useful for calling `tf.distribute.Strategy` methods which operate across the
  replicas (like `reduce_to()`). By default you start in a replica context
  (the \"default single replica context\") and then some methods can switch you
  back and forth. There is a third mode you can be in called _update context_
  used when updating variables.

  * `tf.distribute.Strategy.scope`: enters cross-replica context when
    no other strategy is in scope.
  * `tf.distribute.Strategy.experimental_run_v2`: calls a function in
    replica context.
  * `tf.distribute.ReplicaContext.merge_call`: transitions from replica
    context to cross-replica context.
  * `tf.distribute.StrategyExtended.update`: calls a function in an update
    context from a cross-replica context.

  In a replica context, you may freely read the values of variables, but
  you may only update their value if they specify a way to aggregate the
  update using the `aggregation` parameter in the variable's constructor.
  In a cross-replica context, you may read or write variables (writes may
  need to be broadcast to all copies of the variable if it is mirrored).

  *Sync on read variables*

  In some cases, such as a metric, we want to accumulate a bunch of updates on
  each replica independently and only aggregate when reading. This can be a big
  performance win when the value is read only rarely (maybe the value is only
  read at the end of an epoch or when checkpointing).  These are variables
  created by passing `synchronization=ON_READ` to the variable's constructor
  (and some value for `aggregation`).

  The strategy may choose to put the variable on multiple devices, like mirrored
  variables, but unlike mirrored variables we don't synchronize the updates to
  them to make sure they have the same value. Instead, the synchronization is
  performed when reading in cross-replica context.  In a replica context, reads
  and writes are performed on the local copy (we allow reads so you can write
  code like `v = 0.9*v + 0.1*update`).  We don't allow operations like
  `v.assign_add` in a cross-replica context for sync on read variables; right
  now we don't have a use case for such updates and depending on the aggregation
  mode such updates may not be sensible.

  *Locality*

  Depending on how a value is produced, it will have a type that will determine
  how it may be used.

  \"Per-replica\" values exist on the worker devices, with a different value for
  each replica. They are produced by iterating through a \"distributed `Dataset`\"
  returned by `tf.distribute.Strategy.experimental_distribute_dataset` and
  `tf.distribute.Strategy.experimental_distribute_datasets_from_function`.  They
  are also the typical result returned by
  `tf.distribute.Strategy.experimental_run_v2`. You typically can't use a
  per-replica value directly in a cross-replica context, without first resolving
  how to aggregate the values across replicas, for instance by using
  `tf.distribute.Strategy.reduce`.

  \"Mirrored\" values are like per-replica values, except we know that the value
  on all replicas are the same. We can safely read a mirrored value in a
  cross-replica context by using the value on any replica. You can convert
  a per-replica value into a mirrored value by using
  `tf.distribute.ReplicaContext.all_reduce`.

  Values can also have the same locality as a variable, which is a mirrored
  value but residing on the same devices as the variable (as opposed to the
  compute devices). Such values may be passed to a call to
  `tf.distribute.StrategyExtended.update` to update the value of a variable.
  You may use `tf.distribute.StrategyExtended.colocate_vars_with` to give a
  variable the same locality as another variable. This is useful, for example,
  for \"slot\" variables used by an optimizer for keeping track of statistics
  used to update a primary/model variable. You may convert a per-replica
  value to a variable's locality by using
  `tf.distribute.StrategyExtended.reduce_to` or
  `tf.distribute.StrategyExtended.batch_reduce_to`.

  In addition to slot variables which should be colocated with their primary
  variables, optimizers also define non-slot variables. These can be things like
  \"number of step updates performed\" or \"beta1^t\" and \"beta2^t\".  Each strategy
  has some policy for which devices those variables should be copied too, called
  the \"non-slot devices\" (some subset of the parameter devices). We require that
  all non-slot variables are allocated on the same device, or mirrored across
  the same set of devices. You can use
  `tf.distribute.StrategyExtended.non_slot_devices` to pick a consistent set of
  devices to pass to both `tf.distribute.StrategyExtended.colocate_vars_with`
  and `tf.distribute.StrategyExtended.update_non_slot`.

  *How to update a variable*

  The standard pattern for updating variables is to:

  1. In your function passed to `tf.distribute.Strategy.experimental_run_v2`,
     compute a list of (update, variable) pairs. For example, the update might
     be a the gradient of the loss with respect to the variable.
  2. Switch to cross-replica mode by calling
     `tf.distribute.get_replica_context().merge_call()` with the updates and
     variables as arguments.
  3. Call
     `tf.distribute.StrategyExtended.reduce_to(VariableAggregation.SUM, t, v)`
     (for one variable) or `tf.distribute.StrategyExtended.batch_reduce_to`
     (for a list of variables) to sum the updates.
     and broadcast the result to the variable's devices.
  4. Call `tf.distribute.StrategyExtended.update(v)` for each variable to update
     its value.

  Steps 2 through 4 are done automatically by class
  `tf.keras.optimizers.Optimizer` if you call its
  `tf.keras.optimizers.Optimizer.apply_gradients` method in a replica context.
  They are also done automatically if you call an `assign*` method on a (non
  sync-on-read) variable that was constructed with an aggregation method (which
  is used to determine the reduction used in step 3).

  *Distribute-aware layers*

  Layers are generally called in a replica context, except when defining a
  functional model. `tf.distribute.in_cross_replica_context` will let you
  determine which case you are in. If in a replica context,
  the `tf.distribute.get_replica_context` function will return a
  `tf.distribute.ReplicaContext` object. The `ReplicaContext` object has an
  `all_reduce` method for aggregating across all replicas. Alternatively, you
  can update variables following steps 2-4 above.

  Note: For new `tf.distribute.Strategy` implementations, please put all logic
  in a subclass of `tf.distribute.StrategyExtended`. The only code needed for
  the `tf.distribute.Strategy` subclass is for instantiating your subclass of
  `tf.distribute.StrategyExtended` in the `__init__` method.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distribute (import-module "tensorflow.distribute"))

(defn StrategyExtended 
  "Additional APIs for algorithms that need to be distribution-aware.

  Note: For most usage of `tf.distribute.Strategy`, there should be no need to
  call these methods, since TensorFlow libraries (such as optimizers) already
  call these methods when needed on your behalf.

  Lower-level concepts:

  * Wrapped values: In order to represent values parallel across devices
    (either replicas or the devices associated with a particular value), we
    wrap them in a \"PerReplica\" or \"Mirrored\" object that contains a map
    from replica id to values. \"PerReplica\" is used when the value may be
    different across replicas, and \"Mirrored\" when the value are the same.
  * Unwrapping and merging: Consider calling a function `fn` on multiple
    replicas, like `experimental_run_v2(fn, args=[w])` with an
    argument `w` that is a wrapped value. This means `w` will have a map taking
    replica id `0` to `w0`, replica id `11` to `w1`, etc.
    `experimental_run_v2()` unwraps `w` before calling `fn`, so
    it calls `fn(w0)` on `d0`, `fn(w1)` on `d1`, etc.  It then merges the return
    values from `fn()`, which can possibly result in wrapped values. For
    example, let's say `fn()` returns a tuple with three components: `(x, a,
    v0)` from replica 0, `(x, b, v1)` on replica 1, etc. If the first component
    is the same object `x` from every replica, then the first component of the
    merged result will also be `x`. If the second component is different (`a`,
    `b`, ...)  from each replica, then the merged value will have a wrapped map
    from replica device to the different values. If the third component is the
    members of a mirrored variable (`v` maps `d0` to `v0`, `d1` to `v1`, etc.),
    then the merged result will be that mirrored variable (`v`).
  * Worker devices vs. parameter devices: Most replica computations will
    happen on worker devices. Since we don't yet support model
    parallelism, there will be one worker device per replica. When using
    parameter servers or central storage, the set of devices holding
    variables may be different, otherwise the parameter devices might
    match the worker devices.

  *Replica context vs. Cross-replica context*

  _replica context_ is when we are in some function that is being called once
  for each replica.  Otherwise we are in cross-replica context, which is
  useful for calling `tf.distribute.Strategy` methods which operate across the
  replicas (like `reduce_to()`). By default you start in a replica context
  (the \"default single replica context\") and then some methods can switch you
  back and forth. There is a third mode you can be in called _update context_
  used when updating variables.

  * `tf.distribute.Strategy.scope`: enters cross-replica context when
    no other strategy is in scope.
  * `tf.distribute.Strategy.experimental_run_v2`: calls a function in
    replica context.
  * `tf.distribute.ReplicaContext.merge_call`: transitions from replica
    context to cross-replica context.
  * `tf.distribute.StrategyExtended.update`: calls a function in an update
    context from a cross-replica context.

  In a replica context, you may freely read the values of variables, but
  you may only update their value if they specify a way to aggregate the
  update using the `aggregation` parameter in the variable's constructor.
  In a cross-replica context, you may read or write variables (writes may
  need to be broadcast to all copies of the variable if it is mirrored).

  *Sync on read variables*

  In some cases, such as a metric, we want to accumulate a bunch of updates on
  each replica independently and only aggregate when reading. This can be a big
  performance win when the value is read only rarely (maybe the value is only
  read at the end of an epoch or when checkpointing).  These are variables
  created by passing `synchronization=ON_READ` to the variable's constructor
  (and some value for `aggregation`).

  The strategy may choose to put the variable on multiple devices, like mirrored
  variables, but unlike mirrored variables we don't synchronize the updates to
  them to make sure they have the same value. Instead, the synchronization is
  performed when reading in cross-replica context.  In a replica context, reads
  and writes are performed on the local copy (we allow reads so you can write
  code like `v = 0.9*v + 0.1*update`).  We don't allow operations like
  `v.assign_add` in a cross-replica context for sync on read variables; right
  now we don't have a use case for such updates and depending on the aggregation
  mode such updates may not be sensible.

  *Locality*

  Depending on how a value is produced, it will have a type that will determine
  how it may be used.

  \"Per-replica\" values exist on the worker devices, with a different value for
  each replica. They are produced by iterating through a \"distributed `Dataset`\"
  returned by `tf.distribute.Strategy.experimental_distribute_dataset` and
  `tf.distribute.Strategy.experimental_distribute_datasets_from_function`.  They
  are also the typical result returned by
  `tf.distribute.Strategy.experimental_run_v2`. You typically can't use a
  per-replica value directly in a cross-replica context, without first resolving
  how to aggregate the values across replicas, for instance by using
  `tf.distribute.Strategy.reduce`.

  \"Mirrored\" values are like per-replica values, except we know that the value
  on all replicas are the same. We can safely read a mirrored value in a
  cross-replica context by using the value on any replica. You can convert
  a per-replica value into a mirrored value by using
  `tf.distribute.ReplicaContext.all_reduce`.

  Values can also have the same locality as a variable, which is a mirrored
  value but residing on the same devices as the variable (as opposed to the
  compute devices). Such values may be passed to a call to
  `tf.distribute.StrategyExtended.update` to update the value of a variable.
  You may use `tf.distribute.StrategyExtended.colocate_vars_with` to give a
  variable the same locality as another variable. This is useful, for example,
  for \"slot\" variables used by an optimizer for keeping track of statistics
  used to update a primary/model variable. You may convert a per-replica
  value to a variable's locality by using
  `tf.distribute.StrategyExtended.reduce_to` or
  `tf.distribute.StrategyExtended.batch_reduce_to`.

  In addition to slot variables which should be colocated with their primary
  variables, optimizers also define non-slot variables. These can be things like
  \"number of step updates performed\" or \"beta1^t\" and \"beta2^t\".  Each strategy
  has some policy for which devices those variables should be copied too, called
  the \"non-slot devices\" (some subset of the parameter devices). We require that
  all non-slot variables are allocated on the same device, or mirrored across
  the same set of devices. You can use
  `tf.distribute.StrategyExtended.non_slot_devices` to pick a consistent set of
  devices to pass to both `tf.distribute.StrategyExtended.colocate_vars_with`
  and `tf.distribute.StrategyExtended.update_non_slot`.

  *How to update a variable*

  The standard pattern for updating variables is to:

  1. In your function passed to `tf.distribute.Strategy.experimental_run_v2`,
     compute a list of (update, variable) pairs. For example, the update might
     be a the gradient of the loss with respect to the variable.
  2. Switch to cross-replica mode by calling
     `tf.distribute.get_replica_context().merge_call()` with the updates and
     variables as arguments.
  3. Call
     `tf.distribute.StrategyExtended.reduce_to(VariableAggregation.SUM, t, v)`
     (for one variable) or `tf.distribute.StrategyExtended.batch_reduce_to`
     (for a list of variables) to sum the updates.
     and broadcast the result to the variable's devices.
  4. Call `tf.distribute.StrategyExtended.update(v)` for each variable to update
     its value.

  Steps 2 through 4 are done automatically by class
  `tf.keras.optimizers.Optimizer` if you call its
  `tf.keras.optimizers.Optimizer.apply_gradients` method in a replica context.
  They are also done automatically if you call an `assign*` method on a (non
  sync-on-read) variable that was constructed with an aggregation method (which
  is used to determine the reduction used in step 3).

  *Distribute-aware layers*

  Layers are generally called in a replica context, except when defining a
  functional model. `tf.distribute.in_cross_replica_context` will let you
  determine which case you are in. If in a replica context,
  the `tf.distribute.get_replica_context` function will return a
  `tf.distribute.ReplicaContext` object. The `ReplicaContext` object has an
  `all_reduce` method for aggregating across all replicas. Alternatively, you
  can update variables following steps 2-4 above.

  Note: For new `tf.distribute.Strategy` implementations, please put all logic
  in a subclass of `tf.distribute.StrategyExtended`. The only code needed for
  the `tf.distribute.Strategy` subclass is for instantiating your subclass of
  `tf.distribute.StrategyExtended` in the `__init__` method.
  "
  [ container_strategy ]
  (py/call-attr distribute "StrategyExtended"  container_strategy ))

(defn batch-reduce-to 
  "Combine multiple `reduce_to` calls into one for faster execution.

    Args:
      reduce_op: Reduction type, an instance of `tf.distribute.ReduceOp` enum.
      value_destination_pairs: A sequence of (value, destinations)
        pairs. See `reduce_to()` for a description.

    Returns:
      A list of mirrored values, one per pair in `value_destination_pairs`.
    "
  [ self reduce_op value_destination_pairs ]
  (py/call-attr self "batch_reduce_to"  self reduce_op value_destination_pairs ))

(defn broadcast-to 
  "Mirror a tensor on one device to all worker devices.

    Args:
      tensor: A Tensor value to broadcast.
      destinations: A mirrored variable or device string specifying the
        destination devices to copy `tensor` to.

    Returns:
      A value mirrored to `destinations` devices.
    "
  [ self tensor destinations ]
  (py/call-attr self "broadcast_to"  self tensor destinations ))

(defn call-for-each-replica 
  "Run `fn` once per replica.

    `fn` may call `tf.get_replica_context()` to access methods such as
    `replica_id_in_sync_group` and `merge_call()`.

    `merge_call()` is used to communicate between the replicas and
    re-enter the cross-replica context. All replicas pause their execution
    having encountered a `merge_call()` call. After that the
    `merge_fn`-function is executed. Its results are then unwrapped and
    given back to each replica call. After that execution resumes until
    `fn` is complete or encounters another `merge_call()`.  Example:

    ```python
    # Called once in \"cross-replica\" context.
    def merge_fn(distribution, three_plus_replica_id):
      # sum the values across replicas
      return sum(distribution.experimental_local_results(three_plus_replica_id))

    # Called once per replica in `distribution`, in a \"replica\" context.
    def fn(three):
      replica_ctx = tf.get_replica_context()
      v = three + replica_ctx.replica_id_in_sync_group
      # Computes the sum of the `v` values across all replicas.
      s = replica_ctx.merge_call(merge_fn, args=(v,))
      return s + v

    with distribution.scope():
      # in \"cross-replica\" context
      ...
      merged_results = distribution.experimental_run_v2(fn, args=[3])
      # merged_results has the values from every replica execution of `fn`.
      # This statement prints a list:
      print(distribution.experimental_local_results(merged_results))
    ```

    Args:
      fn: function to run (will be run once per replica).
      args: Tuple or list with positional arguments for `fn`.
      kwargs: Dict with keyword arguments for `fn`.

    Returns:
      Merged return value of `fn` across all replicas.
    "
  [self fn & {:keys [args kwargs]
                       :or {kwargs None}} ]
    (py/call-attr-kw self "call_for_each_replica" [fn] {:args args :kwargs kwargs }))

(defn colocate-vars-with 
  "Scope that controls which devices variables will be created on.

    No operations should be added to the graph inside this scope, it
    should only be used when creating variables (some implementations
    work by changing variable creation, others work by using a
    tf.compat.v1.colocate_with() scope).

    This may only be used inside `self.scope()`.

    Example usage:

    ```
    with strategy.scope():
      var1 = tf.Variable(...)
      with strategy.extended.colocate_vars_with(var1):
        # var2 and var3 will be created on the same device(s) as var1
        var2 = tf.Variable(...)
        var3 = tf.Variable(...)

      def fn(v1, v2, v3):
        # operates on v1 from var1, v2 from var2, and v3 from var3

      # `fn` runs on every device `var1` is on, `var2` and `var3` will be there
      # too.
      strategy.extended.update(var1, fn, args=(var2, var3))
    ```

    Args:
      colocate_with_variable: A variable created in this strategy's `scope()`.
        Variables created while in the returned context manager will be on the
        same set of devices as `colocate_with_variable`.

    Returns:
      A context manager.
    "
  [ self colocate_with_variable ]
  (py/call-attr self "colocate_vars_with"  self colocate_with_variable ))

(defn experimental-between-graph 
  "Whether the strategy uses between-graph replication or not.

      This is expected to return a constant value that will not be changed
      throughout its life cycle.
    "
  [ self ]
    (py/call-attr self "experimental_between_graph"))

(defn experimental-make-numpy-dataset 
  "Makes a dataset for input provided via a numpy array.

    This avoids adding `numpy_input` as a large constant in the graph,
    and copies the data to the machine or machines that will be processing
    the input.

    Args:
      numpy_input: A nest of NumPy input arrays that will be distributed evenly
        across all replicas. Note that lists of Numpy arrays are stacked, as
        that is normal `tf.data.Dataset` behavior.
      session: (TensorFlow v1.x graph execution only) A session used for
        initialization.

    Returns:
      A `tf.data.Dataset` representing `numpy_input`.
    "
  [ self numpy_input session ]
  (py/call-attr self "experimental_make_numpy_dataset"  self numpy_input session ))

(defn experimental-require-static-shapes 
  "Returns `True` if static shape is required; `False` otherwise."
  [ self ]
    (py/call-attr self "experimental_require_static_shapes"))

(defn experimental-run-steps-on-iterator 
  "DEPRECATED: please use `experimental_run_v2` instead.

    Run `fn` with input from `iterator` for `iterations` times.

    This method can be used to run a step function for training a number of
    times using input from a dataset.

    Args:
      fn: function to run using this distribution strategy. The function must
        have the following signature: `def fn(context, inputs)`. `context` is an
          instance of `MultiStepContext` that will be passed when `fn` is run.
          `context` can be used to specify the outputs to be returned from `fn`
          by calling `context.set_last_step_output`. It can also be used to
          capture non tensor outputs by `context.set_non_tensor_output`. See
          `MultiStepContext` documentation for more information. `inputs` will
          have same type/structure as `iterator.get_next()`. Typically, `fn`
          will use `call_for_each_replica` method of the strategy to distribute
          the computation over multiple replicas.
      iterator: Iterator of a dataset that represents the input for `fn`. The
        caller is responsible for initializing the iterator as needed.
      iterations: (Optional) Number of iterations that `fn` should be run.
        Defaults to 1.
      initial_loop_values: (Optional) Initial values to be passed into the
        loop that runs `fn`. Defaults to `None`. # TODO(priyag): Remove
          initial_loop_values argument when we have a mechanism to infer the
          outputs of `fn`.

    Returns:
      Returns the `MultiStepContext` object which has the following properties,
      among other things:
        - run_op: An op that runs `fn` `iterations` times.
        - last_step_outputs: A dictionary containing tensors set using
        `context.set_last_step_output`. Evaluating this returns the value of
        the tensors after the last iteration.
        - non_tensor_outputs: A dictionatry containing anything that was set by
          `fn` by calling `context.set_non_tensor_output`.
    "
  [self fn iterator & {:keys [iterations initial_loop_values]
                       :or {initial_loop_values None}} ]
    (py/call-attr-kw self "experimental_run_steps_on_iterator" [fn iterator] {:iterations iterations :initial_loop_values initial_loop_values }))

(defn experimental-should-init 
  "Whether initialization is needed."
  [ self ]
    (py/call-attr self "experimental_should_init"))

(defn non-slot-devices 
  "Device(s) for non-slot variables.

    Create variables on these devices in a
    `with colocate_vars_with(non_slot_devices(...)):` block.
    Update those using `update_non_slot()`.

    Args:
      var_list: The list of variables being optimized, needed with the
        default `tf.distribute.Strategy`.
    Returns:
      A sequence of devices for non-slot variables.
    "
  [ self var_list ]
  (py/call-attr self "non_slot_devices"  self var_list ))

(defn parameter-devices 
  "Returns the tuple of all devices used to place variables."
  [ self ]
    (py/call-attr self "parameter_devices"))

(defn read-var 
  "Reads the value of a variable.

    Returns the aggregate value of a replica-local variable, or the
    (read-only) value of any other variable.

    Args:
      v: A variable allocated within the scope of this `tf.distribute.Strategy`.

    Returns:
      A tensor representing the value of `v`, aggregated across replicas if
      necessary.
    "
  [ self v ]
  (py/call-attr self "read_var"  self v ))

(defn reduce-to 
  "Combine (via e.g. sum or mean) values across replicas.

    Args:
      reduce_op: Reduction type, an instance of `tf.distribute.ReduceOp` enum.
      value: A per-replica value with one value per replica.
      destinations: A mirrored variable, a per-replica tensor, or a device
        string. The return value will be copied to all destination devices (or
        all the devices where the `destinations` value resides). To perform an
        all-reduction, pass `value` to `destinations`.

    Returns:
      A tensor or value mirrored to `destinations`.
    "
  [ self reduce_op value destinations ]
  (py/call-attr self "reduce_to"  self reduce_op value destinations ))

(defn should-checkpoint 
  "Whether checkpointing is needed."
  [ self ]
    (py/call-attr self "should_checkpoint"))

(defn should-save-summary 
  "Whether saving summaries is needed."
  [ self ]
    (py/call-attr self "should_save_summary"))

(defn update 
  "Run `fn` to update `var` using inputs mirrored to the same devices.

    If `var` is mirrored across multiple devices, then this implements
    logic like:

    ```
    results = {}
    for device, v in var:
      with tf.device(device):
        # args and kwargs will be unwrapped if they are mirrored.
        results[device] = fn(v, *args, **kwargs)
    return merged(results)
    ```

    Otherwise this returns `fn(var, *args, **kwargs)` colocated with `var`.

    Neither `args` nor `kwargs` may contain per-replica values.
    If they contain mirrored values, they will be unwrapped before
    calling `fn`.

    Args:
      var: Variable, possibly mirrored to multiple devices, to operate on.
      fn: Function to call. Should take the variable as the first argument.
      args: Tuple or list. Additional positional arguments to pass to `fn()`.
      kwargs: Dict with keyword arguments to pass to `fn()`.
      group: Boolean. Defaults to True. If False, the return value will be
        unwrapped.

    Returns:
      By default, the merged return value of `fn` across all replicas.  The
      merged result has dependencies to make sure that if it is evaluated at
      all, the side effects (updates) will happen on every replica. If instead
      \"group=False\" is specified, this function will return a nest of lists
      where each list has an element per replica, and the caller is responsible
      for ensuring all elements are executed.
    "
  [self var fn & {:keys [args kwargs group]
                       :or {kwargs None}} ]
    (py/call-attr-kw self "update" [var fn] {:args args :kwargs kwargs :group group }))

(defn update-non-slot 
  "Runs `fn(*args, **kwargs)` on `colocate_with` devices.

    Args:
      colocate_with: The return value of `non_slot_devices()`.
      fn: Function to execute.
      args: Tuple or list. Positional arguments to pass to `fn()`.
      kwargs: Dict with keyword arguments to pass to `fn()`.
      group: Boolean. Defaults to True. If False, the return value will be
        unwrapped.

    Returns:
      Return value of `fn`, possibly merged across devices.
    "
  [self colocate_with fn & {:keys [args kwargs group]
                       :or {kwargs None}} ]
    (py/call-attr-kw self "update_non_slot" [colocate_with fn] {:args args :kwargs kwargs :group group }))

(defn value-container 
  "Returns the container that this per-replica `value` belongs to.

    Args:
      value: A value returned by `experimental_run_v2()` or a variable
        created in `scope()`.

    Returns:
      A container that `value` belongs to.
      If value does not belong to any container (including the case of
      container having been destroyed), returns the value itself.
      `value in experimental_local_results(value_container(value))` will
      always be true.
    "
  [ self value ]
  (py/call-attr self "value_container"  self value ))

(defn variable-created-in-scope 
  "Tests whether `v` was created while this strategy scope was active.

    Variables created inside the strategy scope are \"owned\" by it:

    >>> with strategy.scope():
    ...   v = tf.Variable(1.)
    >>> strategy.variable_created_in_scope(v)
    True

    Variables created outside the strategy are not owned by it:

    >>> v = tf.Variable(1.)
    >>> strategy.variable_created_in_scope(v)
    False

    Args:
      v: A `tf.Variable` instance.

    Returns:
      True if `v` was created inside the scope, False if not.
    "
  [ self v ]
  (py/call-attr self "variable_created_in_scope"  self v ))

(defn worker-devices 
  "Returns the tuple of all devices used to for compute replica execution.
    "
  [ self ]
    (py/call-attr self "worker_devices"))
