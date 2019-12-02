(ns tensorflow-core.python.pywrap-tensorflow.TF-Buffer
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

(defn TF-Buffer 
  ""
  [  ]
  (py/call-attr pywrap-tensorflow "TF_Buffer"  ))

(defn data 
  ""
  [ self ]
    (py/call-attr self "data"))

(defn data-deallocator 
  ""
  [ self ]
    (py/call-attr self "data_deallocator"))

(defn length 
  ""
  [ self ]
    (py/call-attr self "length"))
