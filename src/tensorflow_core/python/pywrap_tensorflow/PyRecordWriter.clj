(ns tensorflow-core.python.pywrap-tensorflow.PyRecordWriter
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

(defn PyRecordWriter 
  ""
  [  ]
  (py/call-attr pywrap-tensorflow "PyRecordWriter"  ))

(defn Close 
  ""
  [ self out_status ]
  (py/call-attr self "Close"  self out_status ))

(defn Flush 
  ""
  [ self out_status ]
  (py/call-attr self "Flush"  self out_status ))

(defn WriteRecord 
  ""
  [ self record out_status ]
  (py/call-attr self "WriteRecord"  self record out_status ))
