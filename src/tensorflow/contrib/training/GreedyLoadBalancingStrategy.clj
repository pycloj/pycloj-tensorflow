(ns tensorflow.contrib.training.GreedyLoadBalancingStrategy
  "Returns the least-loaded ps task for op placement.

  The load is calculated by a user-specified load function passed in at
  construction.  There are no units for load, and the load function is
  responsible for providing an internally consistent measure.

  Note that this strategy is very sensitive to the exact order in which
  ps ops (typically variables) are created, as it greedily places ops
  on the least-loaded ps at the point each op is processed.

  One reasonable heuristic is the `byte_size_load_fn`, which
  estimates load as the number of bytes that would be used to store and
  transmit the entire variable.  More advanced load functions
  could consider the difference in access patterns across ops, or trade
  off CPU-intensive ops with RAM-intensive ops with network bandwidth.

  This class is intended to be used as a `ps_strategy` in
  `tf.compat.v1.train.replica_device_setter`.
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

(defn GreedyLoadBalancingStrategy 
  "Returns the least-loaded ps task for op placement.

  The load is calculated by a user-specified load function passed in at
  construction.  There are no units for load, and the load function is
  responsible for providing an internally consistent measure.

  Note that this strategy is very sensitive to the exact order in which
  ps ops (typically variables) are created, as it greedily places ops
  on the least-loaded ps at the point each op is processed.

  One reasonable heuristic is the `byte_size_load_fn`, which
  estimates load as the number of bytes that would be used to store and
  transmit the entire variable.  More advanced load functions
  could consider the difference in access patterns across ops, or trade
  off CPU-intensive ops with RAM-intensive ops with network bandwidth.

  This class is intended to be used as a `ps_strategy` in
  `tf.compat.v1.train.replica_device_setter`.
  "
  [ num_tasks load_fn ]
  (py/call-attr training "GreedyLoadBalancingStrategy"  num_tasks load_fn ))
