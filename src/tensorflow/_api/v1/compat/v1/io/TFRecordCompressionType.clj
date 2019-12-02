(ns tensorflow.-api.v1.compat.v1.io.TFRecordCompressionType
  "The type of compression for the record."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce io (import-module "tensorflow._api.v1.compat.v1.io"))

(defn TFRecordCompressionType 
  "The type of compression for the record."
  [  ]
  (py/call-attr io "TFRecordCompressionType"  ))
