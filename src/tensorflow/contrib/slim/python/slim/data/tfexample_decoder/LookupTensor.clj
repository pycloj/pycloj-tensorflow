(ns tensorflow.contrib.slim.python.slim.data.tfexample-decoder.LookupTensor
  "An ItemHandler that returns a parsed Tensor, the result of a lookup."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tfexample-decoder (import-module "tensorflow.contrib.slim.python.slim.data.tfexample_decoder"))
(defn LookupTensor 
  "An ItemHandler that returns a parsed Tensor, the result of a lookup."
  [tensor_key table shape_keys shape  & {:keys [default_value]} ]
    (py/call-attr-kw tfexample-decoder "LookupTensor" [tensor_key table shape_keys shape] {:default_value default_value }))

(defn keys 
  ""
  [ self ]
    (py/call-attr self "keys"))

(defn tensors-to-item 
  ""
  [ self keys_to_tensors ]
  (py/call-attr self "tensors_to_item"  self keys_to_tensors ))
