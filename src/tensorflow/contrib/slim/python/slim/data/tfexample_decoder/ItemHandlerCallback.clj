(ns tensorflow.contrib.slim.python.slim.data.tfexample-decoder.ItemHandlerCallback
  "An ItemHandler that converts the parsed tensors via a given function.

  Unlike other ItemHandlers, the ItemHandlerCallback resolves its item via
  a callback function rather than using prespecified behavior.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tfexample-decoder (import-module "tensorflow.contrib.slim.python.slim.data.tfexample_decoder"))

(defn ItemHandlerCallback 
  "An ItemHandler that converts the parsed tensors via a given function.

  Unlike other ItemHandlers, the ItemHandlerCallback resolves its item via
  a callback function rather than using prespecified behavior.
  "
  [ keys func ]
  (py/call-attr tfexample-decoder "ItemHandlerCallback"  keys func ))

(defn keys 
  ""
  [ self ]
    (py/call-attr self "keys"))

(defn tensors-to-item 
  ""
  [ self keys_to_tensors ]
  (py/call-attr self "tensors_to_item"  self keys_to_tensors ))
