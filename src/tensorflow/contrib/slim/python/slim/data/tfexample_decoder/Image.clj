(ns tensorflow.contrib.slim.python.slim.data.tfexample-decoder.Image
  "An ItemHandler that decodes a parsed Tensor as an image."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tfexample-decoder (import-module "tensorflow.contrib.slim.python.slim.data.tfexample_decoder"))
(defn Image 
  "An ItemHandler that decodes a parsed Tensor as an image."
  [image_key format_key shape  & {:keys [channels dtype repeated dct_method]} ]
    (py/call-attr-kw tfexample-decoder "Image" [image_key format_key shape] {:channels channels :dtype dtype :repeated repeated :dct_method dct_method }))

(defn keys 
  ""
  [ self ]
    (py/call-attr self "keys"))

(defn tensors-to-item 
  "See base class."
  [ self keys_to_tensors ]
  (py/call-attr self "tensors_to_item"  self keys_to_tensors ))
