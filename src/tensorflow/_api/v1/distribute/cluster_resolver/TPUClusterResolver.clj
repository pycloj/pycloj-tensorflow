(ns tensorflow.-api.v1.distribute.cluster-resolver.TPUClusterResolver
  "Cluster Resolver for Google Cloud TPUs.

  This is an implementation of cluster resolvers for the Google Cloud TPU
  service. As Cloud TPUs are in alpha, you will need to specify a API definition
  file for this to consume, in addition to a list of Cloud TPUs in your Google
  Cloud Platform project.

  TPUClusterResolver supports the following distinct environments:
  Google Compute Engine
  Google Kubernetes Engine
  Google internal
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

(defn TPUClusterResolver 
  "Cluster Resolver for Google Cloud TPUs.

  This is an implementation of cluster resolvers for the Google Cloud TPU
  service. As Cloud TPUs are in alpha, you will need to specify a API definition
  file for this to consume, in addition to a list of Cloud TPUs in your Google
  Cloud Platform project.

  TPUClusterResolver supports the following distinct environments:
  Google Compute Engine
  Google Kubernetes Engine
  Google internal
  "
  [tpu zone project & {:keys [job_name coordinator_name coordinator_address credentials service discovery_url]
                       :or {coordinator_name None coordinator_address None service None discovery_url None}} ]
    (py/call-attr-kw cluster-resolver "TPUClusterResolver" [tpu zone project] {:job_name job_name :coordinator_name coordinator_name :coordinator_address coordinator_address :credentials credentials :service service :discovery_url discovery_url }))

(defn cluster-spec 
  "Returns a ClusterSpec object based on the latest TPU information.

    We retrieve the information from the GCE APIs every time this method is
    called.

    Returns:
      A ClusterSpec containing host information returned from Cloud TPUs,
      or None.

    Raises:
      RuntimeError: If the provided TPU is not healthy.
    "
  [ self  ]
  (py/call-attr self "cluster_spec"  self  ))

(defn environment 
  "Returns the current environment which TensorFlow is running in."
  [ self ]
    (py/call-attr self "environment"))

(defn get-job-name 
  ""
  [ self  ]
  (py/call-attr self "get_job_name"  self  ))

(defn get-master 
  ""
  [ self  ]
  (py/call-attr self "get_master"  self  ))

(defn master 
  "Get the Master string to be used for the session.

    In the normal case, this returns the grpc path (grpc://1.2.3.4:8470) of
    first instance in the ClusterSpec returned by the cluster_spec function.

    If a non-TPU name is used when constructing a TPUClusterResolver, that will
    be returned instead (e.g. If the tpus argument's value when constructing
    this TPUClusterResolver was 'grpc://10.240.1.2:8470',
    'grpc://10.240.1.2:8470' will be returned).

    Args:
      task_type: (Optional, string) The type of the TensorFlow task of the
        master.
      task_id: (Optional, integer) The index of the TensorFlow task of the
        master.
      rpc_layer: (Optional, string) The RPC protocol TensorFlow should use to
        communicate with TPUs.

    Returns:
      string, the connection string to use when creating a session.

    Raises:
      ValueError: If none of the TPUs specified exists.
    "
  [ self task_type task_id rpc_layer ]
  (py/call-attr self "master"  self task_type task_id rpc_layer ))

(defn num-accelerators 
  "Returns the number of TPU cores per worker.

    Connects to the master and list all the devices present in the master,
    and counts them up. Also verifies that the device counts per host in the
    cluster is the same before returning the number of TPU cores per host.

    Args:
      task_type: Unused.
      task_id: Unused.
      config_proto: Used to create a connection to a TPU master in order to
        retrieve the system metadata.

    Raises:
      RuntimeError: If we cannot talk to a TPU worker after retrying or if the
        number of TPU devices per host is different.
    "
  [ self task_type task_id config_proto ]
  (py/call-attr self "num_accelerators"  self task_type task_id config_proto ))
