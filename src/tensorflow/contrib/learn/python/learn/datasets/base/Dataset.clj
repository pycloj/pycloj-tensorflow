(ns tensorflow.contrib.learn.python.learn.datasets.base.Dataset
  "Dataset(data, target)"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce base (import-module "tensorflow.contrib.learn.python.learn.datasets.base"))

(defn Dataset 
  "Dataset(data, target)"
  [ data target ]
  (py/call-attr base "Dataset"  data target ))

(defn data 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "data"))

(defn target 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "target"))
