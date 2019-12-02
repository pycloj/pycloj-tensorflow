(ns tensorflow.-api.v1.compat.v1.summary.FileWriterCache
  "Cache for file writers.

  This class caches file writers, one per directory.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce summary (import-module "tensorflow._api.v1.compat.v1.summary"))

(defn FileWriterCache 
  "Cache for file writers.

  This class caches file writers, one per directory.
  "
  [  ]
  (py/call-attr summary "FileWriterCache"  ))

(defn clear 
  "Clear cached summary writers. Currently only used for unit tests."
  [ self  ]
  (py/call-attr self "clear"  self  ))

(defn get 
  "Returns the FileWriter for the specified directory.

    Args:
      logdir: str, name of the directory.

    Returns:
      A `FileWriter`.
    "
  [ self logdir ]
  (py/call-attr self "get"  self logdir ))
