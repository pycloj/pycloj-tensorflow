(ns tensorflow.-api.v1.compat.v1.train.ClusterSpec
  "Represents a cluster as a set of \"tasks\", organized into \"jobs\".

  A `tf.train.ClusterSpec` represents the set of processes that
  participate in a distributed TensorFlow computation. Every
  `tf.distribute.Server` is constructed in a particular cluster.

  To create a cluster with two jobs and five tasks, you specify the
  mapping from job names to lists of network addresses (typically
  hostname-port pairs).

  ```python
  cluster = tf.train.ClusterSpec({\"worker\": [\"worker0.example.com:2222\",
                                             \"worker1.example.com:2222\",
                                             \"worker2.example.com:2222\"],
                                  \"ps\": [\"ps0.example.com:2222\",
                                         \"ps1.example.com:2222\"]})
  ```

  Each job may also be specified as a sparse mapping from task indices
  to network addresses. This enables a server to be configured without
  needing to know the identity of (for example) all other worker
  tasks:

  ```python
  cluster = tf.train.ClusterSpec({\"worker\": {1: \"worker1.example.com:2222\"},
                                  \"ps\": [\"ps0.example.com:2222\",
                                         \"ps1.example.com:2222\"]})
  ```
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce train (import-module "tensorflow._api.v1.compat.v1.train"))

(defn ClusterSpec 
  "Represents a cluster as a set of \"tasks\", organized into \"jobs\".

  A `tf.train.ClusterSpec` represents the set of processes that
  participate in a distributed TensorFlow computation. Every
  `tf.distribute.Server` is constructed in a particular cluster.

  To create a cluster with two jobs and five tasks, you specify the
  mapping from job names to lists of network addresses (typically
  hostname-port pairs).

  ```python
  cluster = tf.train.ClusterSpec({\"worker\": [\"worker0.example.com:2222\",
                                             \"worker1.example.com:2222\",
                                             \"worker2.example.com:2222\"],
                                  \"ps\": [\"ps0.example.com:2222\",
                                         \"ps1.example.com:2222\"]})
  ```

  Each job may also be specified as a sparse mapping from task indices
  to network addresses. This enables a server to be configured without
  needing to know the identity of (for example) all other worker
  tasks:

  ```python
  cluster = tf.train.ClusterSpec({\"worker\": {1: \"worker1.example.com:2222\"},
                                  \"ps\": [\"ps0.example.com:2222\",
                                         \"ps1.example.com:2222\"]})
  ```
  "
  [ cluster ]
  (py/call-attr train "ClusterSpec"  cluster ))

(defn as-cluster-def 
  "Returns a `tf.train.ClusterDef` protocol buffer based on this cluster."
  [ self  ]
  (py/call-attr self "as_cluster_def"  self  ))

(defn as-dict 
  "Returns a dictionary from job names to their tasks.

    For each job, if the task index space is dense, the corresponding
    value will be a list of network addresses; otherwise it will be a
    dictionary mapping (sparse) task indices to the corresponding
    addresses.

    Returns:
      A dictionary mapping job names to lists or dictionaries
      describing the tasks in those jobs.
    "
  [ self  ]
  (py/call-attr self "as_dict"  self  ))

(defn job-tasks 
  "Returns a mapping from task ID to address in the given job.

    NOTE: For backwards compatibility, this method returns a list. If
    the given job was defined with a sparse set of task indices, the
    length of this list may not reflect the number of tasks defined in
    this job. Use the `tf.train.ClusterSpec.num_tasks` method
    to find the number of tasks defined in a particular job.

    Args:
      job_name: The string name of a job in this cluster.

    Returns:
      A list of task addresses, where the index in the list
      corresponds to the task index of each task. The list may contain
      `None` if the job was defined with a sparse set of task indices.

    Raises:
      ValueError: If `job_name` does not name a job in this cluster.
    "
  [ self job_name ]
  (py/call-attr self "job_tasks"  self job_name ))

(defn jobs 
  "Returns a list of job names in this cluster.

    Returns:
      A list of strings, corresponding to the names of jobs in this cluster.
    "
  [ self ]
    (py/call-attr self "jobs"))

(defn num-tasks 
  "Returns the number of tasks defined in the given job.

    Args:
      job_name: The string name of a job in this cluster.

    Returns:
      The number of tasks defined in the given job.

    Raises:
      ValueError: If `job_name` does not name a job in this cluster.
    "
  [ self job_name ]
  (py/call-attr self "num_tasks"  self job_name ))

(defn task-address 
  "Returns the address of the given task in the given job.

    Args:
      job_name: The string name of a job in this cluster.
      task_index: A non-negative integer.

    Returns:
      The address of the given task in the given job.

    Raises:
      ValueError: If `job_name` does not name a job in this cluster,
      or no task with index `task_index` is defined in that job.
    "
  [ self job_name task_index ]
  (py/call-attr self "task_address"  self job_name task_index ))

(defn task-indices 
  "Returns a list of valid task indices in the given job.

    Args:
      job_name: The string name of a job in this cluster.

    Returns:
      A list of valid task indices in the given job.

    Raises:
      ValueError: If `job_name` does not name a job in this cluster,
      or no task with index `task_index` is defined in that job.
    "
  [ self job_name ]
  (py/call-attr self "task_indices"  self job_name ))
