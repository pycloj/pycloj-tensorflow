(ns tensorflow.distribute.Server
  "An in-process TensorFlow server, for use in distributed training.

  A `tf.distribute.Server` instance encapsulates a set of devices and a
  `tf.compat.v1.Session` target that
  can participate in distributed training. A server belongs to a
  cluster (specified by a `tf.train.ClusterSpec`), and
  corresponds to a particular task in a named job. The server can
  communicate with any other server in the same cluster.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distribute (import-module "tensorflow.distribute"))
(defn Server 
  "An in-process TensorFlow server, for use in distributed training.

  A `tf.distribute.Server` instance encapsulates a set of devices and a
  `tf.compat.v1.Session` target that
  can participate in distributed training. A server belongs to a
  cluster (specified by a `tf.train.ClusterSpec`), and
  corresponds to a particular task in a named job. The server can
  communicate with any other server in the same cluster.
  "
  [server_or_cluster_def job_name task_index protocol config  & {:keys [start]} ]
    (py/call-attr-kw distribute "Server" [server_or_cluster_def job_name task_index protocol config] {:start start }))
(defn create-local-server 
  "Creates a new single-process cluster running on the local host.

    This method is a convenience wrapper for creating a
    `tf.distribute.Server` with a `tf.train.ServerDef` that specifies a
    single-process cluster containing a single task in a job called
    `\"local\"`.

    Args:
      config: (Options.) A `tf.compat.v1.ConfigProto` that specifies default
        configuration options for all sessions that run on this server.
      start: (Optional.) Boolean, indicating whether to start the server after
        creating it. Defaults to `True`.

    Returns:
      A local `tf.distribute.Server`.
    "
  [self config  & {:keys [start]} ]
    (py/call-attr-kw self "create_local_server" [config] {:start start }))

(defn join 
  "Blocks until the server has shut down.

    This method currently blocks forever.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        joining the TensorFlow server.
    "
  [ self  ]
  (py/call-attr self "join"  self  ))

(defn server-def 
  "Returns the `tf.train.ServerDef` for this server.

    Returns:
      A `tf.train.ServerDef` protocol buffer that describes the configuration
      of this server.
    "
  [ self ]
    (py/call-attr self "server_def"))

(defn start 
  "Starts this server.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        starting the TensorFlow server.
    "
  [ self  ]
  (py/call-attr self "start"  self  ))

(defn target 
  "Returns the target for a `tf.compat.v1.Session` to connect to this server.

    To create a
    `tf.compat.v1.Session` that
    connects to this server, use the following snippet:

    ```python
    server = tf.distribute.Server(...)
    with tf.compat.v1.Session(server.target):
      # ...
    ```

    Returns:
      A string containing a session target for this server.
    "
  [ self ]
    (py/call-attr self "target"))
