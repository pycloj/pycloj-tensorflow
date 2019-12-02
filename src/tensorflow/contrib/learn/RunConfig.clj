(ns tensorflow.contrib.learn.RunConfig
  "This class specifies the configurations for an `Estimator` run.

  This class is a deprecated implementation of `tf.estimator.RunConfig`
  interface.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce learn (import-module "tensorflow.contrib.learn"))

(defn RunConfig 
  "This class specifies the configurations for an `Estimator` run.

  This class is a deprecated implementation of `tf.estimator.RunConfig`
  interface.
  "
  [master & {:keys [num_cores log_device_placement gpu_memory_fraction tf_random_seed save_summary_steps save_checkpoints_secs save_checkpoints_steps keep_checkpoint_max keep_checkpoint_every_n_hours log_step_count_steps protocol evaluation_master model_dir session_config session_creation_timeout_secs]
                       :or {tf_random_seed None save_checkpoints_steps None protocol None model_dir None session_config None}} ]
    (py/call-attr-kw learn "RunConfig" [master] {:num_cores num_cores :log_device_placement log_device_placement :gpu_memory_fraction gpu_memory_fraction :tf_random_seed tf_random_seed :save_summary_steps save_summary_steps :save_checkpoints_secs save_checkpoints_secs :save_checkpoints_steps save_checkpoints_steps :keep_checkpoint_max keep_checkpoint_max :keep_checkpoint_every_n_hours keep_checkpoint_every_n_hours :log_step_count_steps log_step_count_steps :protocol protocol :evaluation_master evaluation_master :model_dir model_dir :session_config session_config :session_creation_timeout_secs session_creation_timeout_secs }))

(defn cluster-spec 
  ""
  [ self ]
    (py/call-attr self "cluster_spec"))

(defn device-fn 
  "Returns the device_fn.

    If device_fn is not `None`, it overrides the default
    device function used in `Estimator`.
    Otherwise the default one is used.
    "
  [ self ]
    (py/call-attr self "device_fn"))

(defn environment 
  ""
  [ self ]
    (py/call-attr self "environment"))

(defn eval-distribute 
  "Optional `tf.distribute.Strategy` for evaluation.
    "
  [ self ]
    (py/call-attr self "eval_distribute"))

(defn evaluation-master 
  ""
  [ self ]
    (py/call-attr self "evaluation_master"))

(defn experimental-max-worker-delay-secs 
  ""
  [ self ]
    (py/call-attr self "experimental_max_worker_delay_secs"))

(defn get-task-id 
  "Returns task index from `TF_CONFIG` environmental variable.

    If you have a ClusterConfig instance, you can just access its task_id
    property instead of calling this function and re-parsing the environmental
    variable.

    Returns:
      `TF_CONFIG['task']['index']`. Defaults to 0.
    "
  [ self  ]
  (py/call-attr self "get_task_id"  self  ))

(defn global-id-in-cluster 
  "The global id in the training cluster.

    All global ids in the training cluster are assigned from an increasing
    sequence of consecutive integers. The first id is 0.

    Note: Task id (the property field `task_id`) is tracking the index of the
    node among all nodes with the SAME task type. For example, given the cluster
    definition as follows:

    ```
      cluster = {'chief': ['host0:2222'],
                 'ps': ['host1:2222', 'host2:2222'],
                 'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
    ```

    Nodes with task type `worker` can have id 0, 1, 2.  Nodes with task type
    `ps` can have id, 0, 1. So, `task_id` is not unique, but the pair
    (`task_type`, `task_id`) can uniquely determine a node in the cluster.

    Global id, i.e., this field, is tracking the index of the node among ALL
    nodes in the cluster. It is uniquely assigned.  For example, for the cluster
    spec given above, the global ids are assigned as:
    ```
      task_type  | task_id  |  global_id
      --------------------------------
      chief      | 0        |  0
      worker     | 0        |  1
      worker     | 1        |  2
      worker     | 2        |  3
      ps         | 0        |  4
      ps         | 1        |  5
    ```

    Returns:
      An integer id.
    "
  [ self ]
    (py/call-attr self "global_id_in_cluster"))

(defn is-chief 
  ""
  [ self ]
    (py/call-attr self "is_chief"))

(defn keep-checkpoint-every-n-hours 
  ""
  [ self ]
    (py/call-attr self "keep_checkpoint_every_n_hours"))

(defn keep-checkpoint-max 
  ""
  [ self ]
    (py/call-attr self "keep_checkpoint_max"))

(defn log-step-count-steps 
  ""
  [ self ]
    (py/call-attr self "log_step_count_steps"))

(defn master 
  ""
  [ self ]
    (py/call-attr self "master"))

(defn model-dir 
  ""
  [ self ]
    (py/call-attr self "model_dir"))

(defn num-ps-replicas 
  ""
  [ self ]
    (py/call-attr self "num_ps_replicas"))

(defn num-worker-replicas 
  ""
  [ self ]
    (py/call-attr self "num_worker_replicas"))

(defn protocol 
  "Returns the optional protocol value."
  [ self ]
    (py/call-attr self "protocol"))

(defn replace 
  "Returns a new instance of `RunConfig` replacing specified properties.

    Only the properties in the following list are allowed to be replaced:

      - `model_dir`,
      - `tf_random_seed`,
      - `save_summary_steps`,
      - `save_checkpoints_steps`,
      - `save_checkpoints_secs`,
      - `session_config`,
      - `keep_checkpoint_max`,
      - `keep_checkpoint_every_n_hours`,
      - `log_step_count_steps`,
      - `train_distribute`,
      - `device_fn`,
      - `protocol`.
      - `eval_distribute`,
      - `experimental_distribute`,
      - `experimental_max_worker_delay_secs`,

    In addition, either `save_checkpoints_steps` or `save_checkpoints_secs`
    can be set (should not be both).

    Args:
      **kwargs: keyword named properties with new values.

    Raises:
      ValueError: If any property name in `kwargs` does not exist or is not
        allowed to be replaced, or both `save_checkpoints_steps` and
        `save_checkpoints_secs` are set.

    Returns:
      a new instance of `RunConfig`.
    "
  [ self  ]
  (py/call-attr self "replace"  self  ))

(defn save-checkpoints-secs 
  ""
  [ self ]
    (py/call-attr self "save_checkpoints_secs"))

(defn save-checkpoints-steps 
  ""
  [ self ]
    (py/call-attr self "save_checkpoints_steps"))

(defn save-summary-steps 
  ""
  [ self ]
    (py/call-attr self "save_summary_steps"))

(defn service 
  "Returns the platform defined (in TF_CONFIG) service dict."
  [ self ]
    (py/call-attr self "service"))

(defn session-config 
  ""
  [ self ]
    (py/call-attr self "session_config"))

(defn session-creation-timeout-secs 
  ""
  [ self ]
    (py/call-attr self "session_creation_timeout_secs"))

(defn task-id 
  ""
  [ self ]
    (py/call-attr self "task_id"))

(defn task-type 
  ""
  [ self ]
    (py/call-attr self "task_type"))

(defn tf-config 
  ""
  [ self ]
    (py/call-attr self "tf_config"))

(defn tf-random-seed 
  ""
  [ self ]
    (py/call-attr self "tf_random_seed"))

(defn train-distribute 
  "Optional `tf.distribute.Strategy` for training.
    "
  [ self ]
    (py/call-attr self "train_distribute"))

(defn uid 
  "Generates a 'Unique Identifier' based on all internal fields. (experimental)

Warning: THIS FUNCTION IS EXPERIMENTAL. It may change or be removed at any time, and without warning.

Caller should use the uid string to check `RunConfig` instance integrity
in one session use, but should not rely on the implementation details, which
is subject to change.

Args:
  whitelist: A list of the string names of the properties uid should not
    include. If `None`, defaults to `_DEFAULT_UID_WHITE_LIST`, which
    includes most properties user allowes to change.

Returns:
  A uid string."
  [ self whitelist ]
  (py/call-attr self "uid"  self whitelist ))
