(ns tensorflow.-api.v1.compat.v2.io.TFRecordOptions
  "Options used for manipulating TFRecord files."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce io (import-module "tensorflow._api.v1.compat.v2.io"))

(defn TFRecordOptions 
  "Options used for manipulating TFRecord files."
  [ compression_type flush_mode input_buffer_size output_buffer_size window_bits compression_level compression_method mem_level compression_strategy ]
  (py/call-attr io "TFRecordOptions"  compression_type flush_mode input_buffer_size output_buffer_size window_bits compression_level compression_method mem_level compression_strategy ))
