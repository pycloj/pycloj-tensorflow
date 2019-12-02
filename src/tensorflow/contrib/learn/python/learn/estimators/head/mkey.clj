(ns tensorflow.contrib.learn.python.learn.estimators.head.mkey
  "Metric key strings (deprecated)."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce head (import-module "tensorflow.contrib.learn.python.learn.estimators.head"))

(defn mkey 
  "Metric key strings (deprecated)."
  [  ]
  (py/call-attr head "mkey"  ))
