(ns tensorflow.python.platform.flags.ListSerializer
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce flags (import-module "tensorflow.python.platform.flags"))

(defn ListSerializer 
  ""
  [ list_sep ]
  (py/call-attr flags "ListSerializer"  list_sep ))

(defn serialize 
  "See base class."
  [ self value ]
  (py/call-attr self "serialize"  self value ))
