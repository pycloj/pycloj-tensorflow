(ns tensorflow-core.python.pywrap-tensorflow.TF-Output
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

(defn TF-Output 
  ""
  [  ]
  (py/call-attr pywrap-tensorflow "TF_Output"  ))

(defn index 
  ""
  [ self ]
    (py/call-attr self "index"))

(defn oper 
  ""
  [ self ]
    (py/call-attr self "oper"))
