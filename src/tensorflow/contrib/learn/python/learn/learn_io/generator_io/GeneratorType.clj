(ns tensorflow.contrib.learn.python.learn.learn-io.generator-io.GeneratorType
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce generator-io (import-module "tensorflow.contrib.learn.python.learn.learn_io.generator_io"))

(defn GeneratorType 
  ""
  [  ]
  (py/call-attr generator-io "GeneratorType"  ))
