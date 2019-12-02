(ns tensorflow.python-io.TFRecordCompressionType
  "The type of compression for the record."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce python-io (import-module "tensorflow.python_io"))

(defn TFRecordCompressionType 
  "The type of compression for the record."
  [  ]
  (py/call-attr python-io "TFRecordCompressionType"  ))
