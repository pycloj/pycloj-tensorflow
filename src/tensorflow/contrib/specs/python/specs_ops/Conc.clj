(ns tensorflow.contrib.specs.python.specs-ops.Conc
  "Implements tensor concatenation in network specifications."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce specs-ops (import-module "tensorflow.contrib.specs.python.specs_ops"))

(defn Conc 
  "Implements tensor concatenation in network specifications."
  [ dim ]
  (py/call-attr specs-ops "Conc"  dim ))

(defn funcall 
  ""
  [ self x ]
  (py/call-attr self "funcall"  self x ))
