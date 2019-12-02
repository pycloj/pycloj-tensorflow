(ns tensorflow.-api.v1.compat.v2.data.Options
  "Represents options for tf.data.Dataset.

  An `Options` object can be, for instance, used to control which static
  optimizations to apply or whether to use performance modeling to dynamically
  tune the parallelism of operations such as `tf.data.Dataset.map` or
  `tf.data.Dataset.interleave`.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce data (import-module "tensorflow._api.v1.compat.v2.data"))

(defn Options 
  "Represents options for tf.data.Dataset.

  An `Options` object can be, for instance, used to control which static
  optimizations to apply or whether to use performance modeling to dynamically
  tune the parallelism of operations such as `tf.data.Dataset.map` or
  `tf.data.Dataset.interleave`.
  "
  [  ]
  (py/call-attr data "Options"  ))

(defn experimental-deterministic 
  "Whether the outputs need to be produced in deterministic order. If None, defaults to True."
  [ self ]
    (py/call-attr self "experimental_deterministic"))

(defn experimental-distribute 
  "The distribution strategy options associated with the dataset. See `tf.data.experimental.DistributeOptions` for more details."
  [ self ]
    (py/call-attr self "experimental_distribute"))

(defn experimental-optimization 
  "The optimization options associated with the dataset. See `tf.data.experimental.OptimizationOptions` for more details."
  [ self ]
    (py/call-attr self "experimental_optimization"))

(defn experimental-slack 
  "Whether to introduce 'slack' in the last `prefetch` of the input pipeline, if it exists. This may reduce CPU contention with accelerator host-side activity at the start of a step. The slack frequency is determined by the number of devices attached to this input pipeline. If None, defaults to False."
  [ self ]
    (py/call-attr self "experimental_slack"))

(defn experimental-stateful-whitelist 
  "By default, tf.data will refuse to serialize a dataset or checkpoint its iterator if the dataset contains a stateful op as the serialization / checkpointing won't be able to capture its state. Users can -- at their own risk -- override this restriction by explicitly whitelisting stateful ops by specifying them in this list."
  [ self ]
    (py/call-attr self "experimental_stateful_whitelist"))

(defn experimental-stats 
  "The statistics options associated with the dataset. See `tf.data.experimental.StatsOptions` for more details."
  [ self ]
    (py/call-attr self "experimental_stats"))

(defn experimental-threading 
  "The threading options associated with the dataset. See `tf.data.experimental.ThreadingOptions` for more details."
  [ self ]
    (py/call-attr self "experimental_threading"))

(defn merge 
  "Merges itself with the given `tf.data.Options`.

    The given `tf.data.Options` can be merged as long as there does not exist an
    attribute that is set to different values in `self` and `options`.

    Args:
      options: a `tf.data.Options` to merge with

    Raises:
      ValueError: if the given `tf.data.Options` cannot be merged

    Returns:
      New `tf.data.Options()` object which is the result of merging self with
      the input `tf.data.Options`.
    "
  [ self options ]
  (py/call-attr self "merge"  self options ))
