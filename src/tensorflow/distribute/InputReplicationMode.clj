(ns tensorflow.distribute.InputReplicationMode
  "Replication mode for input function.

  * `PER_WORKER`: The input function will be called on each worker
    independently, creating as many input pipelines as number of workers.
    Replicas will dequeue from the local Dataset on their worker.
    `tf.distribute.Strategy` doesn't manage any state sharing between such
    separate input pipelines.
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
(defn InputReplicationMode 
  "Replication mode for input function.

  * `PER_WORKER`: The input function will be called on each worker
    independently, creating as many input pipelines as number of workers.
    Replicas will dequeue from the local Dataset on their worker.
    `tf.distribute.Strategy` doesn't manage any state sharing between such
    separate input pipelines.
  "
  [value names module qualname type  & {:keys [start]} ]
    (py/call-attr-kw distribute "InputReplicationMode" [value names module qualname type] {:start start }))
