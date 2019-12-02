(ns tensorflow.python.platform.flags.ListParser
  "Parser for a comma-separated list of strings."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce flags (import-module "tensorflow.python.platform.flags"))

(defn ListParser 
  "Parser for a comma-separated list of strings."
  [  ]
  (py/call-attr flags "ListParser"  ))

(defn flag-type 
  "See base class."
  [ self  ]
  (py/call-attr self "flag_type"  self  ))

(defn parse 
  "Parses argument as comma-separated list of strings."
  [ self argument ]
  (py/call-attr self "parse"  self argument ))
