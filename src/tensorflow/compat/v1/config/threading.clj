(ns tensorflow.-api.v1.compat.v1.config.threading
  "Public API for tf.config.threading namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce threading (import-module "tensorflow._api.v1.compat.v1.config.threading"))

(defn get-inter-op-parallelism-threads 
  "Get number of threads used for parallelism between independent operations.

  Determines the number of threads used by independent non-blocking operations.
  0 means the system picks an appropriate number.

  Returns:
    Number of parallel threads
  "
  [  ]
  (py/call-attr threading "get_inter_op_parallelism_threads"  ))

(defn get-intra-op-parallelism-threads 
  "Get number of threads used within an individual op for parallelism.

  Certain operations like matrix multiplication and reductions can utilize
  parallel threads for speed ups. A value of 0 means the system picks an
  appropriate number.

  Returns:
    Number of parallel threads
  "
  [  ]
  (py/call-attr threading "get_intra_op_parallelism_threads"  ))

(defn set-inter-op-parallelism-threads 
  "Set number of threads used for parallelism between independent operations.

  Determines the number of threads used by independent non-blocking operations.
  0 means the system picks an appropriate number.

  Args:
    num_threads: Number of parallel threads
  "
  [ num_threads ]
  (py/call-attr threading "set_inter_op_parallelism_threads"  num_threads ))

(defn set-intra-op-parallelism-threads 
  "Set number of threads used within an individual op for parallelism.

  Certain operations like matrix multiplication and reductions can utilize
  parallel threads for speed ups. A value of 0 means the system picks an
  appropriate number.

  Args:
    num_threads: Number of parallel threads
  "
  [ num_threads ]
  (py/call-attr threading "set_intra_op_parallelism_threads"  num_threads ))
