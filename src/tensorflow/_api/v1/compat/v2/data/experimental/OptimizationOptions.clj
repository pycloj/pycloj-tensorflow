(ns tensorflow.-api.v1.compat.v2.data.experimental.OptimizationOptions
  "Represents options for dataset optimizations.

  You can set the optimization options of a dataset through the
  `experimental_optimization` property of `tf.data.Options`; the property is
  an instance of `tf.data.experimental.OptimizationOptions`.

  ```python
  options = tf.data.Options()
  options.experimental_optimization.noop_elimination = True
  options.experimental_optimization.map_vectorization.enabled = True
  options.experimental_optimization.apply_default_optimizations = False
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

(defn OptimizationOptions 
  "Represents options for dataset optimizations.

  You can set the optimization options of a dataset through the
  `experimental_optimization` property of `tf.data.Options`; the property is
  an instance of `tf.data.experimental.OptimizationOptions`.

  ```python
  options = tf.data.Options()
  options.experimental_optimization.noop_elimination = True
  options.experimental_optimization.map_vectorization.enabled = True
  options.experimental_optimization.apply_default_optimizations = False
  dataset = dataset.with_options(options)
  ```
  "
  [  ]
  (py/call-attr experimental "OptimizationOptions"  ))

(defn apply-default-optimizations 
  "Whether to apply default static optimizations. If False, only static optimizations that have been explicitly enabled will be applied."
  [ self ]
    (py/call-attr self "apply_default_optimizations"))

(defn autotune 
  "Whether to automatically tune performance knobs. If None, defaults to True."
  [ self ]
    (py/call-attr self "autotune"))

(defn autotune-algorithm 
  "When autotuning is enabled (through `autotune`), identifies the algorithm to use for the autotuning optimization."
  [ self ]
    (py/call-attr self "autotune_algorithm"))

(defn autotune-buffers 
  "When autotuning is enabled (through `autotune`), determines whether to also autotune buffer sizes for datasets with parallelism. If None, defaults to False."
  [ self ]
    (py/call-attr self "autotune_buffers"))

(defn autotune-cpu-budget 
  "When autotuning is enabled (through `autotune`), determines the CPU budget to use. Values greater than the number of schedulable CPU cores are allowed but may result in CPU contention. If None, defaults to the number of schedulable CPU cores."
  [ self ]
    (py/call-attr self "autotune_cpu_budget"))

(defn filter-fusion 
  "Whether to fuse filter transformations. If None, defaults to False."
  [ self ]
    (py/call-attr self "filter_fusion"))

(defn filter-with-random-uniform-fusion 
  "Whether to fuse filter dataset that predicts random_uniform < rate into a sampling dataset. If None, defaults to False."
  [ self ]
    (py/call-attr self "filter_with_random_uniform_fusion"))

(defn hoist-random-uniform 
  "Whether to hoist `tf.random_uniform()` ops out of map transformations. If None, defaults to False."
  [ self ]
    (py/call-attr self "hoist_random_uniform"))

(defn map-and-batch-fusion 
  "Whether to fuse map and batch transformations. If None, defaults to True."
  [ self ]
    (py/call-attr self "map_and_batch_fusion"))

(defn map-and-filter-fusion 
  "Whether to fuse map and filter transformations. If None, defaults to False."
  [ self ]
    (py/call-attr self "map_and_filter_fusion"))

(defn map-fusion 
  "Whether to fuse map transformations. If None, defaults to False."
  [ self ]
    (py/call-attr self "map_fusion"))

(defn map-parallelization 
  "Whether to parallelize stateless map transformations. If None, defaults to False."
  [ self ]
    (py/call-attr self "map_parallelization"))

(defn map-vectorization 
  "The map vectorization options associated with the dataset. See `tf.data.experimental.MapVectorizationOptions` for more details."
  [ self ]
    (py/call-attr self "map_vectorization"))

(defn noop-elimination 
  "Whether to eliminate no-op transformations. If None, defaults to True."
  [ self ]
    (py/call-attr self "noop_elimination"))

(defn parallel-batch 
  "Whether to parallelize copying of batch elements. If None, defaults to False."
  [ self ]
    (py/call-attr self "parallel_batch"))

(defn shuffle-and-repeat-fusion 
  "Whether to fuse shuffle and repeat transformations. If None, defaults to True."
  [ self ]
    (py/call-attr self "shuffle_and_repeat_fusion"))
