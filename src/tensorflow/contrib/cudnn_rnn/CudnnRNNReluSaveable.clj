(ns tensorflow.contrib.cudnn-rnn.CudnnRNNReluSaveable
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cudnn-rnn (import-module "tensorflow.contrib.cudnn_rnn"))

(defn CudnnRNNReluSaveable 
  ""
  [opaque_params num_layers num_units input_size & {:keys [input_mode direction scope name]
                       :or {scope None}} ]
    (py/call-attr-kw cudnn-rnn "CudnnRNNReluSaveable" [opaque_params num_layers num_units input_size] {:input_mode input_mode :direction direction :scope scope :name name }))

(defn device 
  "The device for SaveSpec Tensors."
  [ self ]
    (py/call-attr self "device"))

(defn format-converter 
  ""
  [ self ]
    (py/call-attr self "format_converter"))

(defn optional-restore 
  "A hint to restore assertions that this object is optional."
  [ self ]
    (py/call-attr self "optional_restore"))

(defn restore 
  ""
  [ self restored_tensors restored_shapes ]
  (py/call-attr self "restore"  self restored_tensors restored_shapes ))
