(ns tensorflow.-api.v1.compat.v2.data.experimental.StatsOptions
  "Represents options for collecting dataset stats using `StatsAggregator`.

  You can set the stats options of a dataset through the `experimental_stats`
  property of `tf.data.Options`; the property is an instance of
  `tf.data.experimental.StatsOptions`. For example, to collect latency stats
  on all dataset edges, use the following pattern:

  ```python
  aggregator = tf.data.experimental.StatsAggregator()

  options = tf.data.Options()
  options.experimental_stats.aggregator = aggregator
  options.experimental_stats.latency_all_edges = True
  dataset = dataset.with_options(options)
  ```
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

(defn StatsOptions 
  "Represents options for collecting dataset stats using `StatsAggregator`.

  You can set the stats options of a dataset through the `experimental_stats`
  property of `tf.data.Options`; the property is an instance of
  `tf.data.experimental.StatsOptions`. For example, to collect latency stats
  on all dataset edges, use the following pattern:

  ```python
  aggregator = tf.data.experimental.StatsAggregator()

  options = tf.data.Options()
  options.experimental_stats.aggregator = aggregator
  options.experimental_stats.latency_all_edges = True
  dataset = dataset.with_options(options)
  ```
  "
  [  ]
  (py/call-attr experimental "StatsOptions"  ))

(defn aggregator 
  "Associates the given statistics aggregator with the dataset pipeline."
  [ self ]
    (py/call-attr self "aggregator"))

(defn counter-prefix 
  "Prefix for the statistics recorded as counter."
  [ self ]
    (py/call-attr self "counter_prefix"))

(defn latency-all-edges 
  "Whether to add latency measurements on all edges. Defaults to False."
  [ self ]
    (py/call-attr self "latency_all_edges"))

(defn prefix 
  "Prefix to prepend all statistics recorded for the input `dataset` with."
  [ self ]
    (py/call-attr self "prefix"))
