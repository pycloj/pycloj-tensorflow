(ns tensorflow-core.python.pywrap-tensorflow.Status
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

(defn Status 
  ""
  [  ]
  (py/call-attr pywrap-tensorflow "Status"  ))

(defn IgnoreError 
  ""
  [ self  ]
  (py/call-attr self "IgnoreError"  self  ))

(defn ToString 
  ""
  [ self  ]
  (py/call-attr self "ToString"  self  ))

(defn Update 
  ""
  [ self new_status ]
  (py/call-attr self "Update"  self new_status ))

(defn code 
  ""
  [ self  ]
  (py/call-attr self "code"  self  ))

(defn error-message 
  ""
  [ self  ]
  (py/call-attr self "error_message"  self  ))

(defn ok 
  ""
  [ self  ]
  (py/call-attr self "ok"  self  ))
