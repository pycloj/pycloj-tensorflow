(ns tensorflow.-api.v1.compat.v1.distribute
  "Library for running a computation across multiple devices.

See the guide for overview and examples:
[TensorFlow v1.x](https://www.tensorflow.org/guide/distribute_strategy),
[TensorFlow v2.x](https://www.tensorflow.org/alpha/guide/distribute_strategy).

The intent of this library is that you can write an algorithm in a stylized way
and it will be usable with a variety of different `tf.distribute.Strategy`
implementations. Each descendant will implement a different strategy for
distributing the algorithm across multiple devices/machines.  Furthermore, these
changes can be hidden inside the specific layers and other library classes that
need special treatment to run in a distributed setting, so that most users'
model definition code can run unchanged. The `tf.distribute.Strategy` API works
the same way with eager and graph execution.

*Glossary*

* _Data parallelism_ is where we run multiple copies of the model
  on different slices of the input data. This is in contrast to
  _model parallelism_ where we divide up a single copy of a model
  across multiple devices.
  Note: we only support data parallelism for now, but
  hope to add support for model parallelism in the future.
* A _device_ is a CPU or accelerator (e.g. GPUs, TPUs) on some machine that
  TensorFlow can run operations on (see e.g. `tf.device`). You may have multiple
  devices on a single machine, or be connected to devices on multiple
  machines. Devices used to run computations are called _worker devices_.
  Devices used to store variables are _parameter devices_. For some strategies,
  such as `tf.distribute.MirroredStrategy`, the worker and parameter devices
  will be the same (see mirrored variables below). For others they will be
  different.  For example, `tf.distribute.experimental.CentralStorageStrategy`
  puts the variables on a single device (which may be a worker device or may be
  the CPU), and `tf.distribute.experimental.ParameterServerStrategy` puts the
  variables on separate machines called parameter servers (see below).
* A _replica_ is one copy of the model, running on one slice of the
  input data. Right now each replica is executed on its own
  worker device, but once we add support for model parallelism
  a replica may span multiple worker devices.
* A _host_ is the CPU device on a machine with worker devices, typically
  used for running input pipelines.
* A _worker_ is defined to be the physical machine(s) containing the physical
  devices (e.g. GPUs, TPUs) on which the replicated computation is executed. A
  worker may contain one or more replicas, but contains at least one
  replica. Typically one worker will correspond to one machine, but in the case
  of very large models with model parallelism, one worker may span multiple
  machines. We typically run one input pipeline per worker, feeding all the
  replicas on that worker.
* _Synchronous_, or more commonly _sync_, training is where the updates from
  each replica are aggregated together before updating the model variables. This
  is in contrast to _asynchronous_, or _async_ training, where each replica
  updates the model variables independently. You may also have replicas
  partitioned into groups which are in sync within each group but async between
  groups.
* _Parameter servers_: These are machines that hold a single copy of
  parameters/variables, used by some strategies (right now just
  `tf.distribute.experimental.ParameterServerStrategy`). All replicas that want
  to operate on a variable retrieve it at the beginning of a step and send an
  update to be applied at the end of the step. These can in priniciple support
  either sync or async training, but right now we only have support for async
  training with parameter servers. Compare to
  `tf.distribute.experimental.CentralStorageStrategy`, which puts all variables
  on a single device on the same machine (and does sync training), and
  `tf.distribute.MirroredStrategy`, which mirrors variables to multiple devices
  (see below).
* _Mirrored variables_: These are variables that are copied to multiple
  devices, where we keep the copies in sync by applying the same
  updates to every copy. Normally would only be used with sync training.
* Reductions and all-reduce: A _reduction_ is some method of aggregating
  multiple values into one value, like \"sum\" or \"mean\". If a strategy is doing
  sync training, we will perform a reduction on the gradients to a parameter
  from all replicas before applying the update. _All-reduce_ is an algorithm for
  performing a reduction on values from multiple devices and making the result
  available on all of those devices.

Note that we provide a default version of `tf.distribute.Strategy` that is
used when no other strategy is in scope, that provides the same API with
reasonable default behavior.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distribute (import-module "tensorflow._api.v1.compat.v1.distribute"))

(defn experimental-set-strategy 
  "Set a `tf.distribute.Strategy` as current without `with strategy.scope()`.

  ```
  tf.distribute.experimental_set_strategy(strategy1)
  f()
  tf.distribute.experimental_set_strategy(strategy2)
  g()
  tf.distribute.experimental_set_strategy(None)
  h()
  ```

  is equivalent to:

  ```
  with strategy1.scope():
    f()
  with strategy2.scope():
    g()
  h()
  ```

  In general, you should use the `with strategy.scope():` API, but this
  alternative may be convenient in notebooks where you would have to put
  each cell in a `with strategy.scope():` block.

  Note: This should only be called outside of any TensorFlow scope to
  avoid improper nesting.

  Args:
    strategy: A `tf.distribute.Strategy` object or None.

  Raises:
    RuntimeError: If called inside a `with strategy.scope():`.
  "
  [ strategy ]
  (py/call-attr distribute "experimental_set_strategy"  strategy ))

(defn get-loss-reduction 
  "`tf.distribute.ReduceOp` corresponding to the last loss reduction.

  This is used to decide whether loss should be scaled in optimizer (used only
  for estimator + v1 optimizer use case).

  Returns:
    `tf.distribute.ReduceOp` corresponding to the last loss reduction for
    estimator and v1 optimizer use case. `tf.distribute.ReduceOp.SUM` otherwise.
  "
  [  ]
  (py/call-attr distribute "get_loss_reduction"  ))

(defn get-replica-context 
  "Returns the current `tf.distribute.ReplicaContext` or `None`.

  Returns `None` if in a cross-replica context.

  Note that execution:

  1. starts in the default (single-replica) replica context (this function
     will return the default `ReplicaContext` object);
  2. switches to cross-replica context (in which case this will return
     `None`) when entering a `with tf.distribute.Strategy.scope():` block;
  3. switches to a (non-default) replica context inside
     `strategy.experimental_run_v2(fn, ...)`;
  4. if `fn` calls `get_replica_context().merge_call(merge_fn, ...)`, then
     inside `merge_fn` you are back in the cross-replica context (and again
     this function will return `None`).

  Most `tf.distribute.Strategy` methods may only be executed in
  a cross-replica context, in a replica context you should use the
  API of the `tf.distribute.ReplicaContext` object returned by this
  method instead.

  ```
  assert tf.distribute.get_replica_context() is not None  # default
  with strategy.scope():
    assert tf.distribute.get_replica_context() is None

    def f():
      replica_context = tf.distribute.get_replica_context()  # for strategy
      assert replica_context is not None
      tf.print(\"Replica id: \", replica_context.replica_id_in_sync_group,
               \" of \", replica_context.num_replicas_in_sync)

    strategy.experimental_run_v2(f)
  ```

  Returns:
    The current `tf.distribute.ReplicaContext` object when in a replica context
    scope, else `None`.

    Within a particular block, exactly one of these two things will be true:

    * `get_replica_context()` returns non-`None`, or
    * `tf.distribute.is_cross_replica_context()` returns True.
  "
  [  ]
  (py/call-attr distribute "get_replica_context"  ))

(defn get-strategy 
  "Returns the current `tf.distribute.Strategy` object.

  Typically only used in a cross-replica context:

  ```
  if tf.distribute.in_cross_replica_context():
    strategy = tf.distribute.get_strategy()
    ...
  ```

  Returns:
    A `tf.distribute.Strategy` object. Inside a `with strategy.scope()` block,
    it returns `strategy`, otherwise it returns the default (single-replica)
    `tf.distribute.Strategy` object.
  "
  [  ]
  (py/call-attr distribute "get_strategy"  ))

(defn has-strategy 
  "Return if there is a current non-default `tf.distribute.Strategy`.

  ```
  assert not tf.distribute.has_strategy()
  with strategy.scope():
    assert tf.distribute.has_strategy()
  ```

  Returns:
    True if inside a `with strategy.scope():`.
  "
  [  ]
  (py/call-attr distribute "has_strategy"  ))

(defn in-cross-replica-context 
  "Returns `True` if in a cross-replica context.

  See `tf.distribute.get_replica_context` for details.

  ```
  assert not tf.distribute.in_cross_replica_context()
  with strategy.scope():
    assert tf.distribute.in_cross_replica_context()

    def f():
      assert not tf.distribute.in_cross_replica_context()

    strategy.experimental_run_v2(f)
  ```

  Returns:
    `True` if in a cross-replica context (`get_replica_context()` returns
    `None`), or `False` if in a replica context (`get_replica_context()` returns
    non-`None`).
  "
  [  ]
  (py/call-attr distribute "in_cross_replica_context"  ))
