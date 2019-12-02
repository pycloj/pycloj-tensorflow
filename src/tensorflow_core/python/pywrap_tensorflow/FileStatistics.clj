(ns tensorflow-core.python.pywrap-tensorflow.FileStatistics
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

(defn FileStatistics 
  ""
  [  ]
  (py/call-attr pywrap-tensorflow "FileStatistics"  ))

(defn is-directory 
  ""
  [ self ]
    (py/call-attr self "is_directory"))

(defn length 
  ""
  [ self ]
    (py/call-attr self "length"))

(defn mtime-nsec 
  ""
  [ self ]
    (py/call-attr self "mtime_nsec"))
