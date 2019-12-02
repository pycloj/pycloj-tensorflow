(ns tensorflow.AggregationMethod
  "A class listing aggregation methods used to combine gradients.

  Computing partial derivatives can require aggregating gradient
  contributions. This class lists the various methods that can
  be used to combine gradients in the graph.

  The following aggregation methods are part of the stable API for
  aggregating gradients:

  *  `ADD_N`: All of the gradient terms are summed as part of one
     operation using the \"AddN\" op (see `tf.add_n`). This
     method has the property that all gradients must be ready and
     buffered separately in memory before any aggregation is performed.
  *  `DEFAULT`: The system-chosen default aggregation method.

  The following aggregation methods are experimental and may not
  be supported in future releases:

  * `EXPERIMENTAL_TREE`: Gradient terms are summed in pairs using
    using the \"AddN\" op. This method of summing gradients may reduce
    performance, but it can improve memory utilization because the
    gradients can be released earlier.

  * `EXPERIMENTAL_ACCUMULATE_N`: Gradient terms are summed using the
    \"AccumulateN\" op (see `tf.accumulate_n`), which accumulates the
    overall sum in a single buffer that is shared across threads.
    This method of summing gradients can result in a lower memory footprint
    and lower latency at the expense of higher CPU/GPU utilization.
    For gradients of types that \"AccumulateN\" does not support, this
    summation method falls back on the behavior of `EXPERIMENTAL_TREE`
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

(defn AggregationMethod 
  "A class listing aggregation methods used to combine gradients.

  Computing partial derivatives can require aggregating gradient
  contributions. This class lists the various methods that can
  be used to combine gradients in the graph.

  The following aggregation methods are part of the stable API for
  aggregating gradients:

  *  `ADD_N`: All of the gradient terms are summed as part of one
     operation using the \"AddN\" op (see `tf.add_n`). This
     method has the property that all gradients must be ready and
     buffered separately in memory before any aggregation is performed.
  *  `DEFAULT`: The system-chosen default aggregation method.

  The following aggregation methods are experimental and may not
  be supported in future releases:

  * `EXPERIMENTAL_TREE`: Gradient terms are summed in pairs using
    using the \"AddN\" op. This method of summing gradients may reduce
    performance, but it can improve memory utilization because the
    gradients can be released earlier.

  * `EXPERIMENTAL_ACCUMULATE_N`: Gradient terms are summed using the
    \"AccumulateN\" op (see `tf.accumulate_n`), which accumulates the
    overall sum in a single buffer that is shared across threads.
    This method of summing gradients can result in a lower memory footprint
    and lower latency at the expense of higher CPU/GPU utilization.
    For gradients of types that \"AccumulateN\" does not support, this
    summation method falls back on the behavior of `EXPERIMENTAL_TREE`
  "
  [  ]
  (py/call-attr tensorflow "AggregationMethod"  ))
