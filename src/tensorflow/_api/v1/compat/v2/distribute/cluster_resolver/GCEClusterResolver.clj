(ns tensorflow.-api.v1.compat.v2.distribute.cluster-resolver.GCEClusterResolver
  "ClusterResolver for Google Compute Engine.

  This is an implementation of cluster resolvers for the Google Compute Engine
  instance group platform. By specifying a project, zone, and instance group,
  this will retrieve the IP address of all the instances within the instance
  group and return a ClusterResolver object suitable for use for distributed
  TensorFlow.
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

(defn GCEClusterResolver 
  "ClusterResolver for Google Compute Engine.

  This is an implementation of cluster resolvers for the Google Compute Engine
  instance group platform. By specifying a project, zone, and instance group,
  this will retrieve the IP address of all the instances within the instance
  group and return a ClusterResolver object suitable for use for distributed
  TensorFlow.
  "
  [project zone instance_group port & {:keys [task_type task_id rpc_layer credentials service]
                       :or {service None}} ]
    (py/call-attr-kw cluster-resolver "GCEClusterResolver" [project zone instance_group port] {:task_type task_type :task_id task_id :rpc_layer rpc_layer :credentials credentials :service service }))

(defn cluster-spec 
  "Returns a ClusterSpec object based on the latest instance group info.

    This returns a ClusterSpec object for use based on information from the
    specified instance group. We will retrieve the information from the GCE APIs
    every time this method is called.

    Returns:
      A ClusterSpec containing host information retrieved from GCE.
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
  ""
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
