(ns tensorflow-core.contrib.cluster-resolver.SimpleClusterResolver
  "Simple implementation of ClusterResolver that accepts a ClusterSpec."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cluster-resolver (import-module "tensorflow_core.contrib.cluster_resolver"))

(defn SimpleClusterResolver 
  "Simple implementation of ClusterResolver that accepts a ClusterSpec."
  [cluster_spec & {:keys [master task_type task_id environment num_accelerators rpc_layer]
                       :or {task_type None task_id None num_accelerators None rpc_layer None}} ]
    (py/call-attr-kw cluster-resolver "SimpleClusterResolver" [cluster_spec] {:master master :task_type task_type :task_id task_id :environment environment :num_accelerators num_accelerators :rpc_layer rpc_layer }))

(defn cluster-spec 
  "Returns the ClusterSpec passed into the constructor."
  [ self  ]
  (py/call-attr self "cluster_spec"  self  ))

(defn environment 
  ""
  [ self ]
    (py/call-attr self "environment"))

(defn master 
  "Returns the master address to use when creating a session.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the master.
      task_id: (Optional) The index of the TensorFlow task of the master.
      rpc_layer: (Optional) The RPC used by distributed TensorFlow.

    Returns:
      The name or URL of the session master.

    If a task_type and task_id is given, this will override the `master`
    string passed into the initialization function.
    "
  [ self task_type task_id rpc_layer ]
  (py/call-attr self "master"  self task_type task_id rpc_layer ))

(defn num-accelerators 
  "Returns the number of accelerator cores per worker.

    The SimpleClusterResolver does not do automatic detection of accelerators,
    so a TensorFlow session will never be created, and thus all arguments are
    unused and we simply assume that the type of accelerator is a GPU and return
    the value in provided to us in the constructor.

    Args:
      task_type: Unused.
      task_id: Unused.
      config_proto: Unused.
    "
  [ self task_type task_id config_proto ]
  (py/call-attr self "num_accelerators"  self task_type task_id config_proto ))

(defn rpc-layer 
  ""
  [ self ]
    (py/call-attr self "rpc_layer"))

(defn task-id 
  ""
  [ self ]
    (py/call-attr self "task_id"))

(defn task-type 
  ""
  [ self ]
    (py/call-attr self "task_type"))
