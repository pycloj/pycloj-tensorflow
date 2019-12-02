(ns tensorflow-estimator.python.estimator.api.-v1.estimator.tpu.TPUConfig
  "TPU related configuration required by `TPUEstimator`.

  Args:
    iterations_per_loop: This is the number of train steps running in TPU
      system before returning to CPU host for each `Session.run`. This means
      global step is increased `iterations_per_loop` times in one `Session.run`.
      It is recommended to be set as number of global steps for next checkpoint.
      Note that in evaluation don't use this value, instead we run total eval
      `steps` on TPU for a single `Session.run`.
      [Experimental]: `iterations_per_loop` can be specified as a time interval.
      To specify N seconds in one `Session.run`, one can specify it as `Ns` and
      substitute the N with the N with the number of desired seconds.
      Alternatively, the unit of time can also be specified in minutes or hours,
      e.g. `3600s` or `60m` or `1h`.
    num_shards: (Deprecated, ignored by TPUEstimator).
      The number of model replicas in the system. For non-model-parallelism
      case, this number equals the total number of TPU cores. For
      model-parallelism, the total number of TPU cores equals
      num_cores_per_replica * num_shards.
    num_cores_per_replica: Defaults to `None`, which disables model parallelism.
      An integer which describes the number of TPU cores per model replica. This
      is required by model-parallelism which enables partitioning
      the model to multiple cores. Currently num_cores_per_replica must be
      1, 2, 4, or 8.
    per_host_input_for_training: If `True`, for `PER_HOST_V1`, the `input_fn` is
      invoked once on each host, and the number of hosts must be smaller or
      equal to the number of replicas. For PER_HOST_V2, the `input_fn` is
      invoked once for each host (if the number of hosts is less than the number
      of replicas) or replica (if the number of replicas is less than the number
      of hosts. With the per-core input pipeline configuration, it is invoked
      once for each core.
      With a global batch size `train_batch_size` in `TPUEstimator` constructor,
      the batch size for each shard is `train_batch_size` // #hosts in the
      `True` or `PER_HOST_V1` mode. In `PER_HOST_V2` mode, it is
      `train_batch_size` // #cores. In `BROADCAST` mode, `input_fn` is only
      invoked once on host 0 and the tensors are broadcasted to all other
      replicas. The batch size equals to `train_batch_size`. With the per-core
      input pipeline configuration, the shard batch size is also
      `train_batch_size` // #cores.
      Note: per_host_input_for_training==PER_SHARD_V1 only supports mode.TRAIN.
    tpu_job_name: The name of the TPU job. Typically, this name is auto-inferred
      within TPUEstimator, however when using ClusterSpec propagation in more
      esoteric cluster configurations, you may need to specify the job name as a
      string.
    initial_infeed_sleep_secs: The number of seconds the infeed thread should
      wait before enqueueing the first batch. This helps avoid timeouts for
      models that require a long compilation time.
    input_partition_dims: A nested list to describe the partition dims
      for all the tensors from input_fn(). The structure of
      input_partition_dims must match the structure of `features` and
      `labels` from input_fn(). The total number of partitions must match
      `num_cores_per_replica`. For example, if input_fn() returns two tensors:
      images with shape [N, H, W, C] and labels [N].
      input_partition_dims = [[1, 2, 2, 1], None] will split the images to 4
      pieces and feed into 4 TPU cores. labels tensor are directly broadcasted
      to all the TPU cores since the partition dims is `None`.
      Current limitations: This feature is only supported with the PER_HOST_V2
      input mode.
    eval_training_input_configuration: If `SLICED`, `input_fn` is only
      invoked once on host 0 and the tensors are broadcasted to all other
      replicas. Unlike per_host_input_for_training=BROADCAST, each replica will
      only get a slice of the data instead of a whole copy. If `PER_HOST_V1`,
      the behaviour is determined by per_host_input_for_training.
    experimental_host_call_every_n_steps: Within a training loop, this argument
      sets how often host calls are performed during training. Host calls will
      be evaluated every n steps within a training loop where n is the value of
      this argument.

  Raises:
      ValueError: If `num_cores_per_replica` is not 1, 2, 4, 8, ..., 128.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tpu (import-module "tensorflow_estimator.python.estimator.api._v1.estimator.tpu"))

(defn TPUConfig 
  "TPU related configuration required by `TPUEstimator`.

  Args:
    iterations_per_loop: This is the number of train steps running in TPU
      system before returning to CPU host for each `Session.run`. This means
      global step is increased `iterations_per_loop` times in one `Session.run`.
      It is recommended to be set as number of global steps for next checkpoint.
      Note that in evaluation don't use this value, instead we run total eval
      `steps` on TPU for a single `Session.run`.
      [Experimental]: `iterations_per_loop` can be specified as a time interval.
      To specify N seconds in one `Session.run`, one can specify it as `Ns` and
      substitute the N with the N with the number of desired seconds.
      Alternatively, the unit of time can also be specified in minutes or hours,
      e.g. `3600s` or `60m` or `1h`.
    num_shards: (Deprecated, ignored by TPUEstimator).
      The number of model replicas in the system. For non-model-parallelism
      case, this number equals the total number of TPU cores. For
      model-parallelism, the total number of TPU cores equals
      num_cores_per_replica * num_shards.
    num_cores_per_replica: Defaults to `None`, which disables model parallelism.
      An integer which describes the number of TPU cores per model replica. This
      is required by model-parallelism which enables partitioning
      the model to multiple cores. Currently num_cores_per_replica must be
      1, 2, 4, or 8.
    per_host_input_for_training: If `True`, for `PER_HOST_V1`, the `input_fn` is
      invoked once on each host, and the number of hosts must be smaller or
      equal to the number of replicas. For PER_HOST_V2, the `input_fn` is
      invoked once for each host (if the number of hosts is less than the number
      of replicas) or replica (if the number of replicas is less than the number
      of hosts. With the per-core input pipeline configuration, it is invoked
      once for each core.
      With a global batch size `train_batch_size` in `TPUEstimator` constructor,
      the batch size for each shard is `train_batch_size` // #hosts in the
      `True` or `PER_HOST_V1` mode. In `PER_HOST_V2` mode, it is
      `train_batch_size` // #cores. In `BROADCAST` mode, `input_fn` is only
      invoked once on host 0 and the tensors are broadcasted to all other
      replicas. The batch size equals to `train_batch_size`. With the per-core
      input pipeline configuration, the shard batch size is also
      `train_batch_size` // #cores.
      Note: per_host_input_for_training==PER_SHARD_V1 only supports mode.TRAIN.
    tpu_job_name: The name of the TPU job. Typically, this name is auto-inferred
      within TPUEstimator, however when using ClusterSpec propagation in more
      esoteric cluster configurations, you may need to specify the job name as a
      string.
    initial_infeed_sleep_secs: The number of seconds the infeed thread should
      wait before enqueueing the first batch. This helps avoid timeouts for
      models that require a long compilation time.
    input_partition_dims: A nested list to describe the partition dims
      for all the tensors from input_fn(). The structure of
      input_partition_dims must match the structure of `features` and
      `labels` from input_fn(). The total number of partitions must match
      `num_cores_per_replica`. For example, if input_fn() returns two tensors:
      images with shape [N, H, W, C] and labels [N].
      input_partition_dims = [[1, 2, 2, 1], None] will split the images to 4
      pieces and feed into 4 TPU cores. labels tensor are directly broadcasted
      to all the TPU cores since the partition dims is `None`.
      Current limitations: This feature is only supported with the PER_HOST_V2
      input mode.
    eval_training_input_configuration: If `SLICED`, `input_fn` is only
      invoked once on host 0 and the tensors are broadcasted to all other
      replicas. Unlike per_host_input_for_training=BROADCAST, each replica will
      only get a slice of the data instead of a whole copy. If `PER_HOST_V1`,
      the behaviour is determined by per_host_input_for_training.
    experimental_host_call_every_n_steps: Within a training loop, this argument
      sets how often host calls are performed during training. Host calls will
      be evaluated every n steps within a training loop where n is the value of
      this argument.

  Raises:
      ValueError: If `num_cores_per_replica` is not 1, 2, 4, 8, ..., 128.
  "
  [ & {:keys [iterations_per_loop num_shards num_cores_per_replica per_host_input_for_training tpu_job_name initial_infeed_sleep_secs input_partition_dims eval_training_input_configuration experimental_host_call_every_n_steps]
       :or {num_shards None num_cores_per_replica None tpu_job_name None initial_infeed_sleep_secs None input_partition_dims None}} ]
  
   (py/call-attr-kw tpu "TPUConfig" [] {:iterations_per_loop iterations_per_loop :num_shards num_shards :num_cores_per_replica num_cores_per_replica :per_host_input_for_training per_host_input_for_training :tpu_job_name tpu_job_name :initial_infeed_sleep_secs initial_infeed_sleep_secs :input_partition_dims input_partition_dims :eval_training_input_configuration eval_training_input_configuration :experimental_host_call_every_n_steps experimental_host_call_every_n_steps }))

(defn eval-training-input-configuration 
  "Alias for field number 7"
  [ self ]
    (py/call-attr self "eval_training_input_configuration"))

(defn experimental-host-call-every-n-steps 
  "Alias for field number 8"
  [ self ]
    (py/call-attr self "experimental_host_call_every_n_steps"))

(defn initial-infeed-sleep-secs 
  "Alias for field number 5"
  [ self ]
    (py/call-attr self "initial_infeed_sleep_secs"))

(defn input-partition-dims 
  "Alias for field number 6"
  [ self ]
    (py/call-attr self "input_partition_dims"))

(defn iterations-per-loop 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "iterations_per_loop"))

(defn num-cores-per-replica 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "num_cores_per_replica"))

(defn num-shards 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "num_shards"))

(defn per-host-input-for-training 
  "Alias for field number 3"
  [ self ]
    (py/call-attr self "per_host_input_for_training"))

(defn tpu-job-name 
  "Alias for field number 4"
  [ self ]
    (py/call-attr self "tpu_job_name"))
