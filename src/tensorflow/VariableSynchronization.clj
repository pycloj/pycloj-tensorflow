(ns tensorflow.VariableSynchronization
  "Indicates when a distributed variable will be synced.

  * `AUTO`: Indicates that the synchronization will be determined by the current
    `DistributionStrategy` (eg. With `MirroredStrategy` this would be
    `ON_WRITE`).
  * `NONE`: Indicates that there will only be one copy of the variable, so
    there is no need to sync.
  * `ON_WRITE`: Indicates that the variable will be updated across devices
    every time it is written.
  * `ON_READ`: Indicates that the variable will be aggregated across devices
    when it is read (eg. when checkpointing or when evaluating an op that uses
    the variable).
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tensorflow (import-module "tensorflow"))
(defn VariableSynchronization 
  "Indicates when a distributed variable will be synced.

  * `AUTO`: Indicates that the synchronization will be determined by the current
    `DistributionStrategy` (eg. With `MirroredStrategy` this would be
    `ON_WRITE`).
  * `NONE`: Indicates that there will only be one copy of the variable, so
    there is no need to sync.
  * `ON_WRITE`: Indicates that the variable will be updated across devices
    every time it is written.
  * `ON_READ`: Indicates that the variable will be aggregated across devices
    when it is read (eg. when checkpointing or when evaluating an op that uses
    the variable).
  "
  [value names module qualname type  & {:keys [start]} ]
    (py/call-attr-kw tensorflow "VariableSynchronization" [value names module qualname type] {:start start }))
