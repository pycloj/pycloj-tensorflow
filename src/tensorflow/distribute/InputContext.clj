(ns tensorflow.distribute.InputContext
  "A class wrapping information needed by an input function.

  This is a context class that is passed to the user's input function and
  contains information about the compute replicas and input pipelines. The
  number of compute replicas (in sync training) helps compute the local batch
  size from the desired global batch size for each replica. The input pipeline
  information can be used to return a different subset of the input in each
  replica (for e.g. shard the input pipeline, use a different input
  source etc).
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

(defn InputContext 
  "A class wrapping information needed by an input function.

  This is a context class that is passed to the user's input function and
  contains information about the compute replicas and input pipelines. The
  number of compute replicas (in sync training) helps compute the local batch
  size from the desired global batch size for each replica. The input pipeline
  information can be used to return a different subset of the input in each
  replica (for e.g. shard the input pipeline, use a different input
  source etc).
  "
  [ & {:keys [num_input_pipelines input_pipeline_id num_replicas_in_sync]} ]
   (py/call-attr-kw distribute "InputContext" [] {:num_input_pipelines num_input_pipelines :input_pipeline_id input_pipeline_id :num_replicas_in_sync num_replicas_in_sync }))

(defn get-per-replica-batch-size 
  "Returns the per-replica batch size.

    Args:
      global_batch_size: the global batch size which should be divisible by
        `num_replicas_in_sync`.

    Returns:
      the per-replica batch size.

    Raises:
      ValueError: if `global_batch_size` not divisible by
        `num_replicas_in_sync`.
    "
  [ self global_batch_size ]
  (py/call-attr self "get_per_replica_batch_size"  self global_batch_size ))

(defn input-pipeline-id 
  "Returns the input pipeline ID."
  [ self ]
    (py/call-attr self "input_pipeline_id"))

(defn num-input-pipelines 
  "Returns the number of input pipelines."
  [ self ]
    (py/call-attr self "num_input_pipelines"))

(defn num-replicas-in-sync 
  "Returns the number of compute replicas in sync."
  [ self ]
    (py/call-attr self "num_replicas_in_sync"))
