(ns tensorflow.contrib.slim.python.slim.queues.NestedQueueRunnerError
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce queues (import-module "tensorflow.contrib.slim.python.slim.queues"))
