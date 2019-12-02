(ns tensorflow.-api.v1.distribute.cluster-resolver.KubernetesClusterResolver
  "ClusterResolver for Kubernetes.

  This is an implementation of cluster resolvers for Kubernetes. When given the
  the Kubernetes namespace and label selector for pods, we will retrieve the
  pod IP addresses of all running pods matching the selector, and return a
  ClusterSpec based on that information.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cluster-resolver (import-module "tensorflow._api.v1.distribute.cluster_resolver"))

(defn KubernetesClusterResolver 
  "ClusterResolver for Kubernetes.

  This is an implementation of cluster resolvers for Kubernetes. When given the
  the Kubernetes namespace and label selector for pods, we will retrieve the
  pod IP addresses of all running pods matching the selector, and return a
  ClusterSpec based on that information.
  "
  [job_to_label_mapping & {:keys [tf_server_port rpc_layer override_client]
                       :or {override_client None}} ]
    (py/call-attr-kw cluster-resolver "KubernetesClusterResolver" [job_to_label_mapping] {:tf_server_port tf_server_port :rpc_layer rpc_layer :override_client override_client }))

(defn cluster-spec 
  "Returns a ClusterSpec object based on the latest info from Kubernetes.

    We retrieve the information from the Kubernetes master every time this
    method is called.

    Returns:
      A ClusterSpec containing host information returned from Kubernetes.

    Raises:
      RuntimeError: If any of the pods returned by the master is not in the
        `Running` phase.
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
  "Returns the master address to use when creating a session.

    You must have set the task_type and task_id object properties before
    calling this function, or pass in the `task_type` and `task_id`
    parameters when using this function. If you do both, the function parameters
    will override the object properties.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the master.
      task_id: (Optional) The index of the TensorFlow task of the master.
      rpc_layer: (Optional) The RPC protocol for the given cluster.

    Returns:
      The name or URL of the session master.
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
