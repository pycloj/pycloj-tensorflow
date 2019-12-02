(ns tensorflow.-api.v1.compat.v2.distribute.HierarchicalCopyAllReduce
  "Reduction using hierarchical copy all-reduce.

  It reduces to one GPU along edges in some hierarchy and broadcasts back to
  each GPU along the same path. Before performing all-reduce, tensors will be
  repacked or aggregated for more efficient cross-device transportation.

  This is a reduction created for Nvidia DGX-1 which assumes GPUs connects like
  that on DGX-1 machine. If you have different GPU inter-connections, it is
  likely that it would be slower than `tf.distribute.ReductionToOneDevice`.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distribute (import-module "tensorflow._api.v1.compat.v2.distribute"))

(defn HierarchicalCopyAllReduce 
  "Reduction using hierarchical copy all-reduce.

  It reduces to one GPU along edges in some hierarchy and broadcasts back to
  each GPU along the same path. Before performing all-reduce, tensors will be
  repacked or aggregated for more efficient cross-device transportation.

  This is a reduction created for Nvidia DGX-1 which assumes GPUs connects like
  that on DGX-1 machine. If you have different GPU inter-connections, it is
  likely that it would be slower than `tf.distribute.ReductionToOneDevice`.
  "
  [ & {:keys [num_packs]} ]
   (py/call-attr-kw distribute "HierarchicalCopyAllReduce" [] {:num_packs num_packs }))

(defn batch-reduce 
  "Reduce PerReplica objects in a batch.

    Reduce each first element in `value_destination_pairs` to each second
    element which indicates the destinations.

    Args:
      reduce_op: Indicates how per_replica_value will be reduced. Accepted
        values are `tf.distribute.ReduceOp.SUM`, `tf.distribute.ReduceOp.MEAN`.
      value_destination_pairs: a list or a tuple of tuples of PerReplica objects
        (or tensors with device set if there is one device) and destinations.

    Returns:
      a list of Mirrored objects.

    Raises:
      ValueError: if `value_destination_pairs` is not a list or a tuple of
        tuples of PerReplica objects and destinations
    "
  [ self reduce_op value_destination_pairs ]
  (py/call-attr self "batch_reduce"  self reduce_op value_destination_pairs ))

(defn batch-reduce-implementation 
  ""
  [ self reduce_op value_destination_pairs ]
  (py/call-attr self "batch_reduce_implementation"  self reduce_op value_destination_pairs ))

(defn broadcast 
  "Broadcast the `tensor` to destinations.

    Args:
      tensor: the tensor to broadcast.
      destinations: the broadcast destinations.

    Returns:
      a Mirrored object.
    "
  [ self tensor destinations ]
  (py/call-attr self "broadcast"  self tensor destinations ))

(defn broadcast-implementation 
  "Implementation of broadcast the `tensor` to destinations.

    Args:
      tensor: the tensor to broadcast.
      destinations: the broadcast destinations.

    Returns:
      a Mirrored object.
    "
  [ self tensor destinations ]
  (py/call-attr self "broadcast_implementation"  self tensor destinations ))

(defn reduce 
  "Reduce `per_replica_value` to `destinations`.

    It runs the reduction operation defined by `reduce_op` and put the
    result on `destinations`.

    Args:
      reduce_op: Indicates how per_replica_value will be reduced. Accepted
        values are `tf.distribute.ReduceOp.SUM`, `tf.distribute.ReduceOp.MEAN`.
      per_replica_value: a PerReplica object or a tensor with device set.
      destinations: the reduction destinations.

    Returns:
      a Mirrored object.

    Raises:
      ValueError: if per_replica_value can't be converted to a PerReplica
        object.
    "
  [ self reduce_op per_replica_value destinations ]
  (py/call-attr self "reduce"  self reduce_op per_replica_value destinations ))

(defn reduce-implementation 
  ""
  [ self reduce_op per_replica_value destinations ]
  (py/call-attr self "reduce_implementation"  self reduce_op per_replica_value destinations ))
