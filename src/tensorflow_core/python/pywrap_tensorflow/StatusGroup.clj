(ns tensorflow-core.python.pywrap-tensorflow.StatusGroup
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

(defn StatusGroup 
  ""
  [  ]
  (py/call-attr pywrap-tensorflow "StatusGroup"  ))

(defn AttachLogMessages 
  ""
  [ self  ]
  (py/call-attr self "AttachLogMessages"  self  ))

(defn HasLogMessages 
  ""
  [ self  ]
  (py/call-attr self "HasLogMessages"  self  ))

(defn Update 
  ""
  [ self status ]
  (py/call-attr self "Update"  self status ))

(defn as-concatenated-status 
  ""
  [ self  ]
  (py/call-attr self "as_concatenated_status"  self  ))

(defn as-summary-status 
  ""
  [ self  ]
  (py/call-attr self "as_summary_status"  self  ))

(defn ok 
  ""
  [ self  ]
  (py/call-attr self "ok"  self  ))
