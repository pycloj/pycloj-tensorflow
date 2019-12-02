(ns tensorflow-estimator.python.estimator.api.-v1.estimator.tpu.RunConfig
  "RunConfig with TPU support."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tpu (import-module "tensorflow_estimator.python.estimator.api._v1.estimator.tpu"))

(defn RunConfig 
  "RunConfig with TPU support."
  [ tpu_config evaluation_master master cluster ]
  (py/call-attr tpu "RunConfig"  tpu_config evaluation_master master cluster ))

(defn cluster 
  ""
  [ self ]
    (py/call-attr self "cluster"))

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
  ""
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

(defn tf-random-seed 
  ""
  [ self ]
    (py/call-attr self "tf_random_seed"))

(defn tpu-config 
  ""
  [ self ]
    (py/call-attr self "tpu_config"))

(defn train-distribute 
  "Optional `tf.distribute.Strategy` for training.
    "
  [ self ]
    (py/call-attr self "train_distribute"))
