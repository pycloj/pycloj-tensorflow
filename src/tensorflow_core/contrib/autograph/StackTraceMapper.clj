(ns tensorflow-core.contrib.autograph.StackTraceMapper
  "Remaps generated code to code it originated from."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce autograph (import-module "tensorflow_core.contrib.autograph"))

(defn StackTraceMapper 
  "Remaps generated code to code it originated from."
  [ converted_fn ]
  (py/call-attr autograph "StackTraceMapper"  converted_fn ))

(defn get-effective-source-map 
  ""
  [ self  ]
  (py/call-attr self "get_effective_source_map"  self  ))

(defn reset 
  ""
  [ self  ]
  (py/call-attr self "reset"  self  ))
