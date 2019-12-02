(ns tensorflow.python.platform.flags.ArgumentSerializer
  "Base class for generating string representations of a flag value."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce flags (import-module "tensorflow.python.platform.flags"))

(defn ArgumentSerializer 
  "Base class for generating string representations of a flag value."
  [  ]
  (py/call-attr flags "ArgumentSerializer"  ))

(defn serialize 
  "Returns a serialized string of the value."
  [ self value ]
  (py/call-attr self "serialize"  self value ))
