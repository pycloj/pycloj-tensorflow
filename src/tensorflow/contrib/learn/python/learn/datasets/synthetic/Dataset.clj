(ns tensorflow.contrib.learn.python.learn.datasets.synthetic.Dataset
  "Dataset(data, target)"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce synthetic (import-module "tensorflow.contrib.learn.python.learn.datasets.synthetic"))

(defn Dataset 
  "Dataset(data, target)"
  [ data target ]
  (py/call-attr synthetic "Dataset"  data target ))

(defn data 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "data"))

(defn target 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "target"))
