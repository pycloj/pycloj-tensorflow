(ns tensorflow.test.Benchmark
  "Abstract class that provides helpers for TensorFlow benchmarks."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce test (import-module "tensorflow.test"))

(defn Benchmark 
  "Abstract class that provides helpers for TensorFlow benchmarks."
  [  ]
  (py/call-attr test "Benchmark"  ))

(defn evaluate 
  "Evaluates tensors and returns numpy values.

    Args:
      tensors: A Tensor or a nested list/tuple of Tensors.

    Returns:
      tensors numpy values.
    "
  [ self tensors ]
  (py/call-attr self "evaluate"  self tensors ))

(defn report-benchmark 
  "Report a benchmark.

    Args:
      iters: (optional) How many iterations were run
      cpu_time: (optional) Median or mean cpu time in seconds.
      wall_time: (optional) Median or mean wall time in seconds.
      throughput: (optional) Throughput (in MB/s)
      extras: (optional) Dict mapping string keys to additional benchmark info.
        Values may be either floats or values that are convertible to strings.
      name: (optional) Override the BenchmarkEntry name with `name`.
        Otherwise it is inferred from the top-level method name.
      metrics: (optional) A list of dict, where each dict has the keys below
        name (required), string, metric name
        value (required), double, metric value
        min_value (optional), double, minimum acceptable metric value
        max_value (optional), double, maximum acceptable metric value
    "
  [ self iters cpu_time wall_time throughput extras name metrics ]
  (py/call-attr self "report_benchmark"  self iters cpu_time wall_time throughput extras name metrics ))

(defn run-op-benchmark 
  "Run an op or tensor in the given session.  Report the results.

    Args:
      sess: `Session` object to use for timing.
      op_or_tensor: `Operation` or `Tensor` to benchmark.
      feed_dict: A `dict` of values to feed for each op iteration (see the
        `feed_dict` parameter of `Session.run`).
      burn_iters: Number of burn-in iterations to run.
      min_iters: Minimum number of iterations to use for timing.
      store_trace: Boolean, whether to run an extra untimed iteration and
        store the trace of iteration in returned extras.
        The trace will be stored as a string in Google Chrome trace format
        in the extras field \"full_trace_chrome_format\". Note that trace
        will not be stored in test_log_pb2.TestResults proto.
      store_memory_usage: Boolean, whether to run an extra untimed iteration,
        calculate memory usage, and store that in extras fields.
      name: (optional) Override the BenchmarkEntry name with `name`.
        Otherwise it is inferred from the top-level method name.
      extras: (optional) Dict mapping string keys to additional benchmark info.
        Values may be either floats or values that are convertible to strings.
      mbs: (optional) The number of megabytes moved by this op, used to
        calculate the ops throughput.

    Returns:
      A `dict` containing the key-value pairs that were passed to
      `report_benchmark`. If `store_trace` option is used, then
      `full_chrome_trace_format` will be included in return dictionary even
      though it is not passed to `report_benchmark` with `extras`.
    "
  [self sess op_or_tensor feed_dict & {:keys [burn_iters min_iters store_trace store_memory_usage name extras mbs]
                       :or {name None extras None}} ]
    (py/call-attr-kw self "run_op_benchmark" [sess op_or_tensor feed_dict] {:burn_iters burn_iters :min_iters min_iters :store_trace store_trace :store_memory_usage store_memory_usage :name name :extras extras :mbs mbs }))
