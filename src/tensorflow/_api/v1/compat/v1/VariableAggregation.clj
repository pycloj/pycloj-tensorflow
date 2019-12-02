(ns tensorflow.-api.v1.compat.v1.VariableAggregation
  "Indicates how a distributed variable will be aggregated.

  `tf.distribute.Strategy` distributes a model by making multiple copies
  (called \"replicas\") acting data-parallel on different elements of the input
  batch. When performing some variable-update operation, say
  `var.assign_add(x)`, in a model, we need to resolve how to combine the
  different values for `x` computed in the different replicas.

  * `NONE`: This is the default, giving an error if you use a
    variable-update operation with multiple replicas.
  * `SUM`: Add the updates across replicas.
  * `MEAN`: Take the arithmetic mean (\"average\") of the updates across replicas.
  * `ONLY_FIRST_REPLICA`: This is for when every replica is performing the same
    update, but we only want to perform the update once. Used, e.g., for the
    global step counter.
  * `ONLY_FIRST_TOWER`: Deprecated alias for `ONLY_FIRST_REPLICA`.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v1 (import-module "tensorflow._api.v1.compat.v1"))
(defn VariableAggregation 
  "Indicates how a distributed variable will be aggregated.

  `tf.distribute.Strategy` distributes a model by making multiple copies
  (called \"replicas\") acting data-parallel on different elements of the input
  batch. When performing some variable-update operation, say
  `var.assign_add(x)`, in a model, we need to resolve how to combine the
  different values for `x` computed in the different replicas.

  * `NONE`: This is the default, giving an error if you use a
    variable-update operation with multiple replicas.
  * `SUM`: Add the updates across replicas.
  * `MEAN`: Take the arithmetic mean (\"average\") of the updates across replicas.
  * `ONLY_FIRST_REPLICA`: This is for when every replica is performing the same
    update, but we only want to perform the update once. Used, e.g., for the
    global step counter.
  * `ONLY_FIRST_TOWER`: Deprecated alias for `ONLY_FIRST_REPLICA`.
  "
  [value names module qualname type  & {:keys [start]} ]
    (py/call-attr-kw v1 "VariableAggregation" [value names module qualname type] {:start start }))
