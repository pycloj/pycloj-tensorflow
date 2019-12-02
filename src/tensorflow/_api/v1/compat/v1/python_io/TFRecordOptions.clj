(ns tensorflow.-api.v1.compat.v1.python-io.TFRecordOptions
  "Options used for manipulating TFRecord files."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce python-io (import-module "tensorflow._api.v1.compat.v1.python_io"))

(defn TFRecordOptions 
  "Options used for manipulating TFRecord files."
  [ compression_type flush_mode input_buffer_size output_buffer_size window_bits compression_level compression_method mem_level compression_strategy ]
  (py/call-attr python-io "TFRecordOptions"  compression_type flush_mode input_buffer_size output_buffer_size window_bits compression_level compression_method mem_level compression_strategy ))
