(ns tensorflow.contrib.slim.python.slim.data.tfexample-decoder.BackupHandler
  "An ItemHandler that tries two ItemHandlers in order."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tfexample-decoder (import-module "tensorflow.contrib.slim.python.slim.data.tfexample_decoder"))

(defn BackupHandler 
  "An ItemHandler that tries two ItemHandlers in order."
  [ handler backup ]
  (py/call-attr tfexample-decoder "BackupHandler"  handler backup ))

(defn keys 
  ""
  [ self ]
    (py/call-attr self "keys"))

(defn tensors-to-item 
  ""
  [ self keys_to_tensors ]
  (py/call-attr self "tensors_to_item"  self keys_to_tensors ))
