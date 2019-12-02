(ns tensorflow-core.python.pywrap-tensorflow.EventsWriter
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

(defn EventsWriter 
  ""
  [ file_prefix ]
  (py/call-attr pywrap-tensorflow "EventsWriter"  file_prefix ))

(defn Close 
  ""
  [ self  ]
  (py/call-attr self "Close"  self  ))

(defn FileName 
  ""
  [ self  ]
  (py/call-attr self "FileName"  self  ))

(defn Flush 
  ""
  [ self  ]
  (py/call-attr self "Flush"  self  ))

(defn InitWithSuffix 
  ""
  [ self suffix ]
  (py/call-attr self "InitWithSuffix"  self suffix ))

(defn WriteEvent 
  ""
  [ self event ]
  (py/call-attr self "WriteEvent"  self event ))
