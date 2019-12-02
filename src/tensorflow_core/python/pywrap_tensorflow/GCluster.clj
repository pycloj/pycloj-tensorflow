(ns tensorflow-core.python.pywrap-tensorflow.GCluster
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

(defn GCluster 
  ""
  [  ]
  (py/call-attr pywrap-tensorflow "GCluster"  ))

(defn cluster- 
  ""
  [ self ]
    (py/call-attr self "cluster_"))
