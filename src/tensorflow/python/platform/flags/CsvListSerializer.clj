(ns tensorflow.python.platform.flags.CsvListSerializer
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

(defn CsvListSerializer 
  ""
  [ list_sep ]
  (py/call-attr flags "CsvListSerializer"  list_sep ))

(defn serialize 
  "Serializes a list as a CSV string or unicode."
  [ self value ]
  (py/call-attr self "serialize"  self value ))
