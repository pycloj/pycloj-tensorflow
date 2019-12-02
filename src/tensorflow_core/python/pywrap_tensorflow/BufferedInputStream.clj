(ns tensorflow-core.python.pywrap-tensorflow.BufferedInputStream
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

(defn BufferedInputStream 
  ""
  [  ]
  (py/call-attr pywrap-tensorflow "BufferedInputStream"  ))

(defn ReadLineAsString 
  ""
  [ self  ]
  (py/call-attr self "ReadLineAsString"  self  ))

(defn Seek 
  ""
  [ self position ]
  (py/call-attr self "Seek"  self position ))

(defn Tell 
  ""
  [ self  ]
  (py/call-attr self "Tell"  self  ))
