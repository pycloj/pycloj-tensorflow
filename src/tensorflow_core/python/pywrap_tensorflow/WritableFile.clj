(ns tensorflow-core.python.pywrap-tensorflow.WritableFile
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

(defn WritableFile 
  ""
  [  ]
  (py/call-attr pywrap-tensorflow "WritableFile"  ))

(defn Close 
  ""
  [ self  ]
  (py/call-attr self "Close"  self  ))

(defn Flush 
  ""
  [ self  ]
  (py/call-attr self "Flush"  self  ))
