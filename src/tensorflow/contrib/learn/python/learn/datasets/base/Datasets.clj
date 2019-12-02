(ns tensorflow.contrib.learn.python.learn.datasets.base.Datasets
  "Datasets(train, validation, test)"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce base (import-module "tensorflow.contrib.learn.python.learn.datasets.base"))

(defn Datasets 
  "Datasets(train, validation, test)"
  [ train validation test ]
  (py/call-attr base "Datasets"  train validation test ))

(defn test 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "test"))

(defn train 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "train"))

(defn validation 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "validation"))
