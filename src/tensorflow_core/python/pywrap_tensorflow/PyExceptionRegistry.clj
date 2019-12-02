(ns tensorflow-core.python.pywrap-tensorflow.PyExceptionRegistry
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce pywrap-tensorflow (import-module "tensorflow_core.python.pywrap_tensorflow"))

(defn PyExceptionRegistry 
  ""
  [  ]
  (py/call-attr pywrap-tensorflow "PyExceptionRegistry"  ))
