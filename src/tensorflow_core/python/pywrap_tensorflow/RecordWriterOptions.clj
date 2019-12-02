(ns tensorflow-core.python.pywrap-tensorflow.RecordWriterOptions
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

(defn RecordWriterOptions 
  ""
  [  ]
  (py/call-attr pywrap-tensorflow "RecordWriterOptions"  ))

(defn zlib-options 
  ""
  [ self ]
    (py/call-attr self "zlib_options"))
