(ns tensorflow-core.python.pywrap-tensorflow.PythonTraceMe
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

(defn PythonTraceMe 
  ""
  [ name ]
  (py/call-attr pywrap-tensorflow "PythonTraceMe"  name ))

(defn Enter 
  ""
  [ self  ]
  (py/call-attr self "Enter"  self  ))

(defn Exit 
  ""
  [ self  ]
  (py/call-attr self "Exit"  self  ))
