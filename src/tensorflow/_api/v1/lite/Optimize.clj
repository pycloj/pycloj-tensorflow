(ns tensorflow.-api.v1.lite.Optimize
  "Enum defining the optimizations to apply when generating tflite graphs.

  Some optimizations may come at the cost of accuracy.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce lite (import-module "tensorflow._api.v1.lite"))
(defn Optimize 
  "Enum defining the optimizations to apply when generating tflite graphs.

  Some optimizations may come at the cost of accuracy.
  "
  [value names module qualname type  & {:keys [start]} ]
    (py/call-attr-kw lite "Optimize" [value names module qualname type] {:start start }))
