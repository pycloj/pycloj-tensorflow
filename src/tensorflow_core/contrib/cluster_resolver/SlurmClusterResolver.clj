(ns tensorflow-core.contrib.cluster-resolver.SlurmClusterResolver
  "ClusterResolver for system with Slurm workload manager.

  This is an implementation of cluster resolvers for Slurm clusters. This allows
  the specification of jobs and task counts, number of tasks per node, number of
  GPUs on each node and number of GPUs for each task. It retrieves system
  attributes by Slurm environment variables, resolves allocated computing node
  names, constructs a cluster and returns a ClusterResolver object which can be
  use for distributed TensorFlow.
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

(defn SlurmClusterResolver 
  "ClusterResolver for system with Slurm workload manager.

  This is an implementation of cluster resolvers for Slurm clusters. This allows
  the specification of jobs and task counts, number of tasks per node, number of
  GPUs on each node and number of GPUs for each task. It retrieves system
  attributes by Slurm environment variables, resolves allocated computing node
  names, constructs a cluster and returns a ClusterResolver object which can be
  use for distributed TensorFlow.
  "
  [jobs & {:keys [port_base gpus_per_node gpus_per_task tasks_per_node auto_set_gpu rpc_layer]
                       :or {tasks_per_node None}} ]
    (py/call-attr-kw cluster-resolver "SlurmClusterResolver" [jobs] {:port_base port_base :gpus_per_node gpus_per_node :gpus_per_task gpus_per_task :tasks_per_node tasks_per_node :auto_set_gpu auto_set_gpu :rpc_layer rpc_layer }))

(defn cluster-spec 
  "Returns a ClusterSpec object based on the latest instance group info.

    This returns a ClusterSpec object for use based on information from the
    specified initialization parameters and Slurm environment variables. The
    cluster specification is resolved each time this function is called. The
    resolver extract hostnames of nodes by scontrol and pack tasks in that
    order until a node a has number of tasks that is equal to specification.
    GPUs on nodes are allocated to tasks by specification through setting
    CUDA_VISIBLE_DEVICES environment variable.

    Returns:
      A ClusterSpec containing host information retrieved from Slurm's
        environment variables.
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

(defn get-task-info 
  "Returns job name and task_id for the process which calls this.

    This returns the job name and task index for the process which calls this
    function according to its rank and cluster specification. The job name and
    task index are set after a cluster is constructed by cluster_spec otherwise
    defaults to None.

    Returns:
      A string specifying job name the process belongs to and an integner
        specifying the task index the process belongs to in that job.
    "
  [ self  ]
  (py/call-attr self "get_task_info"  self  ))

(defn master 
  "Returns the master string for connecting to a TensorFlow master.

    Args:
      task_type: (Optional) Overrides the default auto-selected task type.
      task_id: (Optional) Overrides the default auto-slected task index.
      rpc_layer: (Optional) Overrides the default RPC protocol TensorFlow uses
        to communicate across nodes.

    Returns:
      A connection string for connecting to a TensorFlow master.
    "
  [ self task_type task_id rpc_layer ]
  (py/call-attr self "master"  self task_type task_id rpc_layer ))

(defn num-accelerators 
  ""
  [ self task_type task_id config_proto ]
  (py/call-attr self "num_accelerators"  self task_type task_id config_proto ))
