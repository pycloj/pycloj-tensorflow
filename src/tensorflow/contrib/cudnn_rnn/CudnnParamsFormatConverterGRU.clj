(ns tensorflow.contrib.cudnn-rnn.CudnnParamsFormatConverterGRU
  "Helper class that converts between params of Cudnn and TF GRU."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cudnn-rnn (import-module "tensorflow.contrib.cudnn_rnn"))
(defn CudnnParamsFormatConverterGRU 
  "Helper class that converts between params of Cudnn and TF GRU."
  [num_layers num_units input_size num_proj  & {:keys [input_mode direction]} ]
    (py/call-attr-kw cudnn-rnn "CudnnParamsFormatConverterGRU" [num_layers num_units input_size num_proj] {:input_mode input_mode :direction direction }))

(defn opaque-to-tf-canonical 
  "Converts cudnn opaque param to tf canonical weights."
  [ self opaque_param ]
  (py/call-attr self "opaque_to_tf_canonical"  self opaque_param ))

(defn tf-canonical-to-opaque 
  "Converts tf canonical weights to cudnn opaque param."
  [ self tf_canonicals weights_proj ]
  (py/call-attr self "tf_canonical_to_opaque"  self tf_canonicals weights_proj ))
