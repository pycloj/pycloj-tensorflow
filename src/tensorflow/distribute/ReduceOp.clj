(ns tensorflow.distribute.ReduceOp
  "Indicates how a set of values should be reduced.

  * `SUM`: Add all the values.
  * `MEAN`: Take the arithmetic mean (\"average\") of the values.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distribute (import-module "tensorflow.distribute"))
(defn ReduceOp 
  "Indicates how a set of values should be reduced.

  * `SUM`: Add all the values.
  * `MEAN`: Take the arithmetic mean (\"average\") of the values.
  "
  [value names module qualname type  & {:keys [start]} ]
    (py/call-attr-kw distribute "ReduceOp" [value names module qualname type] {:start start }))
