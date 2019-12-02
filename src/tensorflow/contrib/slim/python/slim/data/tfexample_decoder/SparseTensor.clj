(ns tensorflow.contrib.slim.python.slim.data.tfexample-decoder.SparseTensor
  "An ItemHandler for SparseTensors."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tfexample-decoder (import-module "tensorflow.contrib.slim.python.slim.data.tfexample_decoder"))
(defn SparseTensor 
  "An ItemHandler for SparseTensors."
  [indices_key values_key shape_key shape  & {:keys [densify default_value]} ]
    (py/call-attr-kw tfexample-decoder "SparseTensor" [indices_key values_key shape_key shape] {:densify densify :default_value default_value }))

(defn keys 
  ""
  [ self ]
    (py/call-attr self "keys"))

(defn tensors-to-item 
  ""
  [ self keys_to_tensors ]
  (py/call-attr self "tensors_to_item"  self keys_to_tensors ))
