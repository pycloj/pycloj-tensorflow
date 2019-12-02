(ns tensorflow.contrib.specs.python.Idx
  "Implements the identity function in network specifications."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce python (import-module "tensorflow.contrib.specs.python"))

(defn Idx 
  "Implements the identity function in network specifications."
  [  ]
  (py/call-attr specs "Idx"  ))

(defn funcall 
  ""
  [ self x ]
  (py/call-attr self "funcall"  self x ))
