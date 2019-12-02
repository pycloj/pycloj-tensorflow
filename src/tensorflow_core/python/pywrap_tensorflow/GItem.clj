(ns tensorflow-core.python.pywrap-tensorflow.GItem
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

(defn GItem 
  ""
  [  ]
  (py/call-attr pywrap-tensorflow "GItem"  ))

(defn item- 
  ""
  [ self ]
    (py/call-attr self "item_"))
