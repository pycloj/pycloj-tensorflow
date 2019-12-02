(ns tensorflow-core.python.pywrap-tensorflow.PyRecordReader
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

(defn PyRecordReader 
  ""
  [  ]
  (py/call-attr pywrap-tensorflow "PyRecordReader"  ))

(defn Close 
  ""
  [ self  ]
  (py/call-attr self "Close"  self  ))

(defn GetNext 
  ""
  [ self  ]
  (py/call-attr self "GetNext"  self  ))

(defn offset 
  ""
  [ self  ]
  (py/call-attr self "offset"  self  ))

(defn record 
  ""
  [ self  ]
  (py/call-attr self "record"  self  ))
