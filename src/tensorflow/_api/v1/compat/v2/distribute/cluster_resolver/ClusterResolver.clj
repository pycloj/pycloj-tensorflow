(ns tensorflow.-api.v1.compat.v2.distribute.cluster-resolver.ClusterResolver
  "Abstract class for all implementations of ClusterResolvers.

  This defines the skeleton for all implementations of ClusterResolvers.
  ClusterResolvers are a way for TensorFlow to communicate with various cluster
  management systems (e.g. GCE, AWS, etc...).

  By letting TensorFlow communicate with these systems, we will be able to
  automatically discover and resolve IP addresses for various TensorFlow
  workers. This will eventually allow us to automatically recover from
  underlying machine failures and scale TensorFlow worker clusters up and down.

  Note to Implementors: In addition to these abstract methods, you must also
  implement the task_type, task_id, and rpc_layer attributes. You may choose
  to implement them either as properties with getters or setters or directly
  set the attributes.

  - task_type is the name of the server's current named job (e.g. 'worker',
     'ps' in a distributed parameterized training job).
  - task_id is the ordinal index of the server within the task type.
  - rpc_layer is the protocol used by TensorFlow to communicate with other
      TensorFlow servers in a distributed environment.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cluster-resolver (import-module "tensorflow._api.v1.compat.v2.distribute.cluster_resolver"))

(defn ClusterResolver 
  "Abstract class for all implementations of ClusterResolvers.

  This defines the skeleton for all implementations of ClusterResolvers.
  ClusterResolvers are a way for TensorFlow to communicate with various cluster
  management systems (e.g. GCE, AWS, etc...).

  By letting TensorFlow communicate with these systems, we will be able to
  automatically discover and resolve IP addresses for various TensorFlow
  workers. This will eventually allow us to automatically recover from
  underlying machine failures and scale TensorFlow worker clusters up and down.

  Note to Implementors: In addition to these abstract methods, you must also
  implement the task_type, task_id, and rpc_layer attributes. You may choose
  to implement them either as properties with getters or setters or directly
  set the attributes.

  - task_type is the name of the server's current named job (e.g. 'worker',
     'ps' in a distributed parameterized training job).
  - task_id is the ordinal index of the server within the task type.
  - rpc_layer is the protocol used by TensorFlow to communicate with other
      TensorFlow servers in a distributed environment.
  "
  [  ]
  (py/call-attr cluster-resolver "ClusterResolver"  ))

(defn cluster-spec 
  "Retrieve the current state of the cluster and return a ClusterSpec.

    Returns:
      A ClusterSpec representing the state of the cluster at the moment this
      function is called.

    Implementors of this function must take care in ensuring that the
    ClusterSpec returned is up-to-date at the time of calling this function.
    This usually means retrieving the information from the underlying cluster
    management system every time this function is invoked and reconstructing
    a cluster_spec, rather than attempting to cache anything.
    "
  [ self  ]
  (py/call-attr self "cluster_spec"  self  ))

(defn environment 
  "Returns the current environment which TensorFlow is running in.

    There are two possible return values, \"google\" (when TensorFlow is running
    in a Google-internal environment) or an empty string (when TensorFlow is
    running elsewhere).

    If you are implementing a ClusterResolver that works in both the Google
    environment and the open-source world (for instance, a TPU ClusterResolver
    or similar), you will have to return the appropriate string depending on the
    environment, which you will have to detect.

    Otherwise, if you are implementing a ClusterResolver that will only work
    in open-source TensorFlow, you do not need to implement this property.
    "
  [ self ]
    (py/call-attr self "environment"))

(defn master 
  "Retrieves the name or URL of the session master.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the master.
      task_id: (Optional) The index of the TensorFlow task of the master.
      rpc_layer: (Optional) The RPC protocol for the given cluster.

    Returns:
      The name or URL of the session master.

    Implementors of this function must take care in ensuring that the master
    returned is up-to-date at the time to calling this function. This usually
    means retrieving the master every time this function is invoked.
    "
  [ self task_type task_id rpc_layer ]
  (py/call-attr self "master"  self task_type task_id rpc_layer ))

(defn num-accelerators 
  "Returns the number of accelerator cores per worker.

    This returns the number of accelerator cores (such as GPUs and TPUs)
    available per worker.

    Optionally, we allow callers to specify the task_type, and task_id, for
    if they want to target a specific TensorFlow process to query
    the number of accelerators. This is to support heterogenous environments,
    where the number of accelerators cores per host is different.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the machine we
        want to query.
      task_id: (Optional) The index of the TensorFlow task of the machine we
        want to query.
      config_proto: (Optional) Configuration for starting a new session to
        query how many accelerator cores it has.

    Returns:
      A map of accelerator types to number of cores.
    "
  [ self task_type task_id config_proto ]
  (py/call-attr self "num_accelerators"  self task_type task_id config_proto ))
