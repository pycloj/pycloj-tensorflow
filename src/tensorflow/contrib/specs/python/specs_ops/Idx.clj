(ns tensorflow.contrib.specs.python.specs-ops.Idx
  "Implements the identity function in network specifications."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce specs-ops (import-module "tensorflow.contrib.specs.python.specs_ops"))

(defn Idx 
  "Implements the identity function in network specifications."
  [  ]
  (py/call-attr specs-ops "Idx"  ))

(defn funcall 
  ""
  [ self x ]
  (py/call-attr self "funcall"  self x ))
