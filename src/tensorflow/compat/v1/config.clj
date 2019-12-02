(ns tensorflow.-api.v1.compat.v1.config
  "Public API for tf.config namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce config (import-module "tensorflow._api.v1.compat.v1.config"))

(defn experimental-connect-to-cluster 
  "Connects to the given cluster.

  Will make devices on the cluster available to use. Note that calling this more
  than once will work, but will invalidate any tensor handles on the old remote
  devices.

  If the given local job name is not present in the cluster specification, it
  will be automatically added, using an unused port on the localhost.

  Args:
    cluster_spec_or_resolver: A `ClusterSpec` or `ClusterResolver` describing
      the cluster.
    job_name: The name of the local job.
    task_index: The local task index.
    protocol: The communication protocol, such as `\"grpc\"`. If unspecified, will
      use the default from `python/platform/remote_utils.py`.
  "
  [cluster_spec_or_resolver & {:keys [job_name task_index protocol]
                       :or {protocol None}} ]
    (py/call-attr-kw config "experimental_connect_to_cluster" [cluster_spec_or_resolver] {:job_name job_name :task_index task_index :protocol protocol }))
(defn experimental-connect-to-host 
  "Connects to a single machine to enable remote execution on it.

  Will make devices on the remote host available to use. Note that calling this
  more than once will work, but will invalidate any tensor handles on the old
  remote devices.

  Using the default job_name of worker, you can schedule ops to run remotely as
  follows:
  ```python
  # Enable eager execution, and connect to the remote host.
  tf.compat.v1.enable_eager_execution()
  tf.contrib.eager.connect_to_remote_host(\"exampleaddr.com:9876\")

  with ops.device(\"job:worker/replica:0/task:1/device:CPU:0\"):
    # The following tensors should be resident on the remote device, and the op
    # will also execute remotely.
    x1 = array_ops.ones([2, 2])
    x2 = array_ops.ones([2, 2])
    y = math_ops.matmul(x1, x2)
  ```

  Args:
    remote_host: a single or a list the remote server addr in host-port format.
    job_name: The job name under which the new server will be accessible.

  Raises:
    ValueError: if remote_host is None.
  "
  [remote_host  & {:keys [job_name]} ]
    (py/call-attr-kw config "experimental_connect_to_host" [remote_host] {:job_name job_name }))

(defn experimental-list-devices 
  "List the names of the available devices.

  Returns:
    Names of the available devices, as a `list`.
  "
  [  ]
  (py/call-attr config "experimental_list_devices"  ))

(defn experimental-run-functions-eagerly 
  "Enables / disables eager execution of `tf.function`s.

  After calling `tf.config.experimental_run_functions_eagerly(True)` all
  invocations of tf.function will run eagerly instead of running through a graph
  function.

  This can be useful for debugging or profiling.

  Similarly, calling `tf.config.experimental_run_functions_eagerly(False)` will
  revert the behavior of all functions to graph functions.

  Args:
    run_eagerly: Boolean. Whether to run functions eagerly.
  "
  [ run_eagerly ]
  (py/call-attr config "experimental_run_functions_eagerly"  run_eagerly ))

(defn get-soft-device-placement 
  "Get if soft device placement is enabled.

  If enabled, an op will be placed on CPU if any of the following are true
    1. there's no GPU implementation for the OP
    2. no GPU devices are known or registered
    3. need to co-locate with reftype input(s) which are from CPU

  Returns:
    If soft placement is enabled.
  "
  [  ]
  (py/call-attr config "get_soft_device_placement"  ))

(defn set-soft-device-placement 
  "Set if soft device placement is enabled.

  If enabled, an op will be placed on CPU if any of the following are true
    1. there's no GPU implementation for the OP
    2. no GPU devices are known or registered
    3. need to co-locate with reftype input(s) which are from CPU

  Args:
    enabled: Whether to enable soft placement.
  "
  [ enabled ]
  (py/call-attr config "set_soft_device_placement"  enabled ))
