(ns tensorflow.contrib.learn.python.learn.utils.gc.Path
  "Path(path, export_version)"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce gc (import-module "tensorflow.contrib.learn.python.learn.utils.gc"))

(defn Path 
  "Path(path, export_version)"
  [ path export_version ]
  (py/call-attr gc "Path"  path export_version ))

(defn export-version 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "export_version"))

(defn path 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "path"))
