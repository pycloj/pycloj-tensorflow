(ns tensorflow-core.contrib.cluster-resolver.TFConfigClusterResolver
  "Implementation of a ClusterResolver which reads the TF_CONFIG EnvVar.

  This is an implementation of cluster resolvers when using TF_CONFIG to set
  information about the cluster. The cluster spec returned will be
  initialized from the TF_CONFIG environment variable.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cluster-resolver (import-module "tensorflow_core.contrib.cluster_resolver"))

(defn TFConfigClusterResolver 
  "Implementation of a ClusterResolver which reads the TF_CONFIG EnvVar.

  This is an implementation of cluster resolvers when using TF_CONFIG to set
  information about the cluster. The cluster spec returned will be
  initialized from the TF_CONFIG environment variable.
  "
  [ task_type task_id rpc_layer environment ]
  (py/call-attr cluster-resolver "TFConfigClusterResolver"  task_type task_id rpc_layer environment ))

(defn cluster-spec 
  "Returns a ClusterSpec based on the TF_CONFIG environment variable.

    Returns:
      A ClusterSpec with information from the TF_CONFIG environment variable.
    "
  [ self  ]
  (py/call-attr self "cluster_spec"  self  ))

(defn environment 
  ""
  [ self ]
    (py/call-attr self "environment"))

(defn master 
  "Returns the master address to use when creating a TensorFlow session.

    Args:
      task_type: (String, optional) Overrides and sets the task_type of the
        master.
      task_id: (Integer, optional) Overrides and sets the task id of the
        master.
      rpc_layer: (String, optional) Overrides and sets the protocol over which
        TensorFlow nodes communicate with each other.

    Returns:
      The address of the master.

    Raises:
      RuntimeError: If the task_type or task_id is not specified and the
        `TF_CONFIG` environment variable does not contain a task section.
    "
  [ self task_type task_id rpc_layer ]
  (py/call-attr self "master"  self task_type task_id rpc_layer ))

(defn num-accelerators 
  ""
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
