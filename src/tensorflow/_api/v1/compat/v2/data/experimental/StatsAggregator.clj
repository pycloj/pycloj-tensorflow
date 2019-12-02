(ns tensorflow.-api.v1.compat.v2.data.experimental.StatsAggregator
  "A stateful resource that aggregates statistics from one or more iterators.

  To record statistics, use one of the custom transformation functions defined
  in this module when defining your `tf.data.Dataset`. All statistics will be
  aggregated by the `StatsAggregator` that is associated with a particular
  iterator (see below). For example, to record the latency of producing each
  element by iterating over a dataset:

  ```python
  dataset = ...
  dataset = dataset.apply(tf.data.experimental.latency_stats(\"total_bytes\"))
  ```

  To associate a `StatsAggregator` with a `tf.data.Dataset` object, use
  the following pattern:

  ```python
  aggregator = tf.data.experimental.StatsAggregator()
  dataset = ...

  # Apply `StatsOptions` to associate `dataset` with `aggregator`.
  options = tf.data.Options()
  options.experimental_stats.aggregator = aggregator
  dataset = dataset.with_options(options)
  ```

  Note: This interface is experimental and expected to change. In particular,
  we expect to add other implementations of `StatsAggregator` that provide
  different ways of exporting statistics, and add more types of statistics.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.compat.v2.data.experimental"))

(defn StatsAggregator 
  "A stateful resource that aggregates statistics from one or more iterators.

  To record statistics, use one of the custom transformation functions defined
  in this module when defining your `tf.data.Dataset`. All statistics will be
  aggregated by the `StatsAggregator` that is associated with a particular
  iterator (see below). For example, to record the latency of producing each
  element by iterating over a dataset:

  ```python
  dataset = ...
  dataset = dataset.apply(tf.data.experimental.latency_stats(\"total_bytes\"))
  ```

  To associate a `StatsAggregator` with a `tf.data.Dataset` object, use
  the following pattern:

  ```python
  aggregator = tf.data.experimental.StatsAggregator()
  dataset = ...

  # Apply `StatsOptions` to associate `dataset` with `aggregator`.
  options = tf.data.Options()
  options.experimental_stats.aggregator = aggregator
  dataset = dataset.with_options(options)
  ```

  Note: This interface is experimental and expected to change. In particular,
  we expect to add other implementations of `StatsAggregator` that provide
  different ways of exporting statistics, and add more types of statistics.
  "
  [  ]
  (py/call-attr experimental "StatsAggregator"  ))
