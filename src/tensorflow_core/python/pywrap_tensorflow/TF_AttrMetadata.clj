(ns tensorflow-core.python.pywrap-tensorflow.TF-AttrMetadata
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

(defn TF-AttrMetadata 
  ""
  [  ]
  (py/call-attr pywrap-tensorflow "TF_AttrMetadata"  ))

(defn is-list 
  ""
  [ self ]
    (py/call-attr self "is_list"))

(defn list-size 
  ""
  [ self ]
    (py/call-attr self "list_size"))

(defn total-size 
  ""
  [ self ]
    (py/call-attr self "total_size"))

(defn type 
  ""
  [ self ]
    (py/call-attr self "type"))
