(ns tensorflow.contrib.distribute
  "A distributed computation library for TF.

See [tensorflow/contrib/distribute/README.md](
https://www.tensorflow.org/code/tensorflow/contrib/distribute/README.md)
for overview and examples.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distribute (import-module "tensorflow.contrib.distribute"))

(defn get-cross-replica-context 
  "Returns the current tf.distribute.Strategy if in a cross-replica context.

  DEPRECATED: Please use `in_cross_replica_context()` and
  `get_strategy()` instead.

  Returns:
    Returns the current `tf.distribute.Strategy` object in a cross-replica
    context, or `None`.

    Exactly one of `get_replica_context()` and `get_cross_replica_context()`
    will return `None` in a particular block.
  "
  [  ]
  (py/call-attr distribute "get_cross_replica_context"  ))

(defn get-distribution-strategy 
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
  (py/call-attr distribute "get_distribution_strategy"  ))

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

(defn has-distribution-strategy 
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
  (py/call-attr distribute "has_distribution_strategy"  ))

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

(defn initialize-tpu-system 
  "Initialize the TPU devices.

  Args:
    cluster_resolver: A tf.distribute.cluster_resolver.TPUClusterResolver,
        which provides information about the TPU cluster.
  Returns:
    The tf.tpu.Topology object for the topology of the TPU cluster.

  Raises:
    RuntimeError: If no TPU devices found for eager execution.
  "
  [ cluster_resolver ]
  (py/call-attr distribute "initialize_tpu_system"  cluster_resolver ))

(defn require-replica-context 
  "Verify in `replica_ctx` replica context."
  [ replica_ctx ]
  (py/call-attr distribute "require_replica_context"  replica_ctx ))

(defn run-standard-tensorflow-server 
  "Starts a standard TensorFlow server.

  This method parses configurations from \"TF_CONFIG\" environment variable and
  starts a TensorFlow server. The \"TF_CONFIG\" is typically a json string and
  must have information of the cluster and the role of the server in the
  cluster. One example is:

  TF_CONFIG='{
      \"cluster\": {
          \"worker\": [\"host1:2222\", \"host2:2222\", \"host3:2222\"],
          \"ps\": [\"host4:2222\", \"host5:2222\"]
      },
      \"task\": {\"type\": \"worker\", \"index\": 1}
  }'

  This \"TF_CONFIG\" specifies there are 3 workers and 2 ps tasks in the cluster
  and the current role is worker 1.

  Valid task types are \"chief\", \"worker\", \"ps\" and \"evaluator\" and you can have
  at most one \"chief\" and at most one \"evaluator\".

  An optional key-value can be specified is \"rpc_layer\". The default value is
  \"grpc\".

  Args:
    session_config: an optional `tf.compat.v1.ConfigProto` object. Users can
      pass in the session config object to configure server-local devices.

  Returns:
    a `tf.distribute.Server` object which has already been started.

  Raises:
    ValueError: if the \"TF_CONFIG\" environment is not complete.
  "
  [ session_config ]
  (py/call-attr distribute "run_standard_tensorflow_server"  session_config ))
