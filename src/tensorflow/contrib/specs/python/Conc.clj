(ns tensorflow.contrib.specs.python.Conc
  "Implements tensor concatenation in network specifications."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce python (import-module "tensorflow.contrib.specs.python"))

(defn Conc 
  "Implements tensor concatenation in network specifications."
  [ dim ]
  (py/call-attr specs "Conc"  dim ))

(defn funcall 
  ""
  [ self x ]
  (py/call-attr self "funcall"  self x ))
