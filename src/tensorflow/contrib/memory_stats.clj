(ns tensorflow.contrib.memory-stats
  "Ops for memory statistics.

@@BytesInUse
@@BytesLimit
@@MaxBytesInUse
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce memory-stats (import-module "tensorflow.contrib.memory_stats"))

(defn BytesInUse 
  "Generates an op that computes the current memory of a device."
  [  ]
  (py/call-attr memory-stats "BytesInUse"  ))

(defn BytesLimit 
  "Generates an op that measures the total memory (in bytes) of a device."
  [  ]
  (py/call-attr memory-stats "BytesLimit"  ))

(defn MaxBytesInUse 
  "Generates an op that computes the peak memory of a device."
  [  ]
  (py/call-attr memory-stats "MaxBytesInUse"  ))
