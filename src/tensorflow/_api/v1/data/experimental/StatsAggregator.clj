(ns tensorflow.-api.v1.data.experimental.StatsAggregator
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

  To get a protocol buffer summary of the currently aggregated statistics,
  use the `StatsAggregator.get_summary()` tensor. The easiest way to do this
  is to add the returned tensor to the `tf.GraphKeys.SUMMARIES` collection,
  so that the summaries will be included with any existing summaries.

  ```python
  aggregator = tf.data.experimental.StatsAggregator()
  # ...
  stats_summary = aggregator.get_summary()
  tf.compat.v1.add_to_collection(tf.GraphKeys.SUMMARIES, stats_summary)
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
(defonce experimental (import-module "tensorflow._api.v1.data.experimental"))

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

  To get a protocol buffer summary of the currently aggregated statistics,
  use the `StatsAggregator.get_summary()` tensor. The easiest way to do this
  is to add the returned tensor to the `tf.GraphKeys.SUMMARIES` collection,
  so that the summaries will be included with any existing summaries.

  ```python
  aggregator = tf.data.experimental.StatsAggregator()
  # ...
  stats_summary = aggregator.get_summary()
  tf.compat.v1.add_to_collection(tf.GraphKeys.SUMMARIES, stats_summary)
  ```

  Note: This interface is experimental and expected to change. In particular,
  we expect to add other implementations of `StatsAggregator` that provide
  different ways of exporting statistics, and add more types of statistics.
  "
  [  ]
  (py/call-attr experimental "StatsAggregator"  ))

(defn get-summary 
  "Returns a string `tf.Tensor` that summarizes the aggregated statistics.

    The returned tensor will contain a serialized `tf.compat.v1.summary.Summary`
    protocol
    buffer, which can be used with the standard TensorBoard logging facilities.

    Returns:
      A scalar string `tf.Tensor` that summarizes the aggregated statistics.
    "
  [ self  ]
  (py/call-attr self "get_summary"  self  ))
