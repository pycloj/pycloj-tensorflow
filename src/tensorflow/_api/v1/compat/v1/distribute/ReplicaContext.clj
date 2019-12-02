(ns tensorflow.-api.v1.compat.v1.distribute.ReplicaContext
  "`tf.distribute.Strategy` API when in a replica context.

  You can use `tf.distribute.get_replica_context` to get an instance of
  `ReplicaContext`. This should be inside your replicated step function, such
  as in a `tf.distribute.Strategy.experimental_run_v2` call.
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

(defn ReplicaContext 
  "`tf.distribute.Strategy` API when in a replica context.

  You can use `tf.distribute.get_replica_context` to get an instance of
  `ReplicaContext`. This should be inside your replicated step function, such
  as in a `tf.distribute.Strategy.experimental_run_v2` call.
  "
  [ strategy replica_id_in_sync_group ]
  (py/call-attr distribute "ReplicaContext"  strategy replica_id_in_sync_group ))

(defn all-reduce 
  "All-reduces the given `value Tensor` nest across replicas.

    If `all_reduce` is called in any replica, it must be called in all replicas.
    The nested structure and `Tensor` shapes must be identical in all replicas.

    IMPORTANT: The ordering of communications must be identical in all replicas.

    Example with two replicas:
      Replica 0 `value`: {'a': 1, 'b': [40, 1]}
      Replica 1 `value`: {'a': 3, 'b': [ 2, 98]}

      If `reduce_op` == `SUM`:
        Result (on all replicas): {'a': 4, 'b': [42, 99]}

      If `reduce_op` == `MEAN`:
        Result (on all replicas): {'a': 2, 'b': [21, 49.5]}

    Args:
      reduce_op: Reduction type, an instance of `tf.distribute.ReduceOp` enum.
      value: The nested structure of `Tensor`s to all-reduce. The structure must
        be compatible with `tf.nest`.

    Returns:
       A `Tensor` nest with the reduced `value`s from each replica.
    "
  [ self reduce_op value ]
  (py/call-attr self "all_reduce"  self reduce_op value ))

(defn devices 
  "The devices this replica is to be executed on, as a tuple of strings."
  [ self ]
    (py/call-attr self "devices"))

(defn merge-call 
  "Merge args across replicas and run `merge_fn` in a cross-replica context.

    This allows communication and coordination when there are multiple calls
    to the step_fn triggered by a call to
    `strategy.experimental_run_v2(step_fn, ...)`.

    See `tf.distribute.Strategy.experimental_run_v2` for an
    explanation.

    If not inside a distributed scope, this is equivalent to:

    ```
    strategy = tf.distribute.get_strategy()
    with cross-replica-context(strategy):
      return merge_fn(strategy, *args, **kwargs)
    ```

    Args:
      merge_fn: Function that joins arguments from threads that are given as
        PerReplica. It accepts `tf.distribute.Strategy` object as
        the first argument.
      args: List or tuple with positional per-thread arguments for `merge_fn`.
      kwargs: Dict with keyword per-thread arguments for `merge_fn`.

    Returns:
      The return value of `merge_fn`, except for `PerReplica` values which are
      unpacked.
    "
  [self merge_fn & {:keys [args kwargs]
                       :or {kwargs None}} ]
    (py/call-attr-kw self "merge_call" [merge_fn] {:args args :kwargs kwargs }))

(defn num-replicas-in-sync 
  "Returns number of replicas over which gradients are aggregated."
  [ self ]
    (py/call-attr self "num_replicas_in_sync"))

(defn replica-id-in-sync-group 
  "Returns the id of the replica being defined.

    This identifies the replica that is part of a sync group. Currently we
    assume that all sync groups contain the same number of replicas. The value
    of the replica id can range from 0 to `num_replica_in_sync` - 1.
    "
  [ self ]
    (py/call-attr self "replica_id_in_sync_group"))

(defn strategy 
  "The current `tf.distribute.Strategy` object."
  [ self ]
    (py/call-attr self "strategy"))
