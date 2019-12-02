(ns tensorflow.-api.v1.compat.v1.distribute.experimental.CollectiveCommunication
  "Communication choices for CollectiveOps.

  * `AUTO`: Default to runtime's automatic choices.
  * `RING`: TensorFlow's ring algorithms for all-reduce and
    all-gather.
  * `NCCL`: Use ncclAllReduce for all-reduce, and ring algorithms for
    all-gather.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.compat.v1.distribute.experimental"))
(defn CollectiveCommunication 
  "Communication choices for CollectiveOps.

  * `AUTO`: Default to runtime's automatic choices.
  * `RING`: TensorFlow's ring algorithms for all-reduce and
    all-gather.
  * `NCCL`: Use ncclAllReduce for all-reduce, and ring algorithms for
    all-gather.
  "
  [value names module qualname type  & {:keys [start]} ]
    (py/call-attr-kw experimental "CollectiveCommunication" [value names module qualname type] {:start start }))
