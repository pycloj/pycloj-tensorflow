(ns tensorflow.python.platform.flags.BooleanParser
  "Parser of boolean values."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce flags (import-module "tensorflow.python.platform.flags"))

(defn BooleanParser 
  "Parser of boolean values."
  [  ]
  (py/call-attr flags "BooleanParser"  ))

(defn flag-type 
  "See base class."
  [ self  ]
  (py/call-attr self "flag_type"  self  ))

(defn parse 
  "See base class."
  [ self argument ]
  (py/call-attr self "parse"  self argument ))
