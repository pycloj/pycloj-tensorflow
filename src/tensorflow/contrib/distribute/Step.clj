(ns tensorflow.contrib.distribute.Step
  "Interface for performing each step of a training algorithm."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distribute (import-module "tensorflow.contrib.distribute"))

(defn Step 
  "Interface for performing each step of a training algorithm."
  [ distribution ]
  (py/call-attr distribute "Step"  distribution ))

(defn distribution 
  ""
  [ self ]
    (py/call-attr self "distribution"))

(defn initialize 
  ""
  [ self  ]
  (py/call-attr self "initialize"  self  ))
