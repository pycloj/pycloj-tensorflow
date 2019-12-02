(ns tensorflow.contrib.learn.TaskType
  "DEPRECATED CLASS."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce learn (import-module "tensorflow.contrib.learn"))

(defn TaskType 
  "DEPRECATED CLASS."
  [  ]
  (py/call-attr learn "TaskType"  ))
