(ns tensorflow-core.python.pywrap-tensorflow.ZlibCompressionOptions
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

(defn ZlibCompressionOptions 
  ""
  [  ]
  (py/call-attr pywrap-tensorflow "ZlibCompressionOptions"  ))

(defn compression-level 
  ""
  [ self ]
    (py/call-attr self "compression_level"))

(defn compression-method 
  ""
  [ self ]
    (py/call-attr self "compression_method"))

(defn compression-strategy 
  ""
  [ self ]
    (py/call-attr self "compression_strategy"))

(defn flush-mode 
  ""
  [ self ]
    (py/call-attr self "flush_mode"))

(defn input-buffer-size 
  ""
  [ self ]
    (py/call-attr self "input_buffer_size"))

(defn mem-level 
  ""
  [ self ]
    (py/call-attr self "mem_level"))

(defn output-buffer-size 
  ""
  [ self ]
    (py/call-attr self "output_buffer_size"))

(defn window-bits 
  ""
  [ self ]
    (py/call-attr self "window_bits"))
