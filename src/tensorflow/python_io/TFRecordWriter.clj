(ns tensorflow.python-io.TFRecordWriter
  "A class to write records to a TFRecords file.

  This class implements `__enter__` and `__exit__`, and can be used
  in `with` blocks like a normal file.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce python-io (import-module "tensorflow.python_io"))

(defn TFRecordWriter 
  "A class to write records to a TFRecords file.

  This class implements `__enter__` and `__exit__`, and can be used
  in `with` blocks like a normal file.
  "
  [ path options ]
  (py/call-attr python-io "TFRecordWriter"  path options ))

(defn close 
  "Close the file."
  [ self  ]
  (py/call-attr self "close"  self  ))

(defn flush 
  "Flush the file."
  [ self  ]
  (py/call-attr self "flush"  self  ))

(defn write 
  "Write a string record to the file.

    Args:
      record: str
    "
  [ self record ]
  (py/call-attr self "write"  self record ))
