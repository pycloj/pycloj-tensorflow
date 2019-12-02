(ns tensorflow.contrib.distribute.StandardInputStep
  "Step with a standard implementation of input handling.

  Args:
    dataset_fn: a function that returns a tf.data Dataset that produces the
      input for the model.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce distribute (import-module "tensorflow.contrib.distribute"))

(defn StandardInputStep 
  "Step with a standard implementation of input handling.

  Args:
    dataset_fn: a function that returns a tf.data Dataset that produces the
      input for the model.
  "
  [ dataset_fn distribution ]
  (py/call-attr distribute "StandardInputStep"  dataset_fn distribution ))

(defn distribution 
  ""
  [ self ]
    (py/call-attr self "distribution"))

(defn initialize 
  ""
  [ self  ]
  (py/call-attr self "initialize"  self  ))
