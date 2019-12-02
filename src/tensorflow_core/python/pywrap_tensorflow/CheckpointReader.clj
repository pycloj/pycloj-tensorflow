(ns tensorflow-core.python.pywrap-tensorflow.CheckpointReader
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

(defn CheckpointReader 
  ""
  [ filename ]
  (py/call-attr pywrap-tensorflow "CheckpointReader"  filename ))

(defn debug-string 
  ""
  [ self  ]
  (py/call-attr self "debug_string"  self  ))

(defn get-tensor 
  ""
  [ self tensor_str ]
  (py/call-attr self "get_tensor"  self tensor_str ))

(defn get-variable-to-dtype-map 
  ""
  [ self  ]
  (py/call-attr self "get_variable_to_dtype_map"  self  ))

(defn get-variable-to-shape-map 
  ""
  [ self  ]
  (py/call-attr self "get_variable_to_shape_map"  self  ))

(defn has-tensor 
  ""
  [ self tensor_str ]
  (py/call-attr self "has_tensor"  self tensor_str ))
