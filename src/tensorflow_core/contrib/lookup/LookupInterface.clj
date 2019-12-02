(ns tensorflow-core.contrib.lookup.LookupInterface
  "Represent a lookup table that persists across different steps."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce lookup (import-module "tensorflow_core.contrib.lookup"))

(defn LookupInterface 
  "Represent a lookup table that persists across different steps."
  [ key_dtype value_dtype ]
  (py/call-attr lookup "LookupInterface"  key_dtype value_dtype ))

(defn key-dtype 
  "The table key dtype."
  [ self ]
    (py/call-attr self "key_dtype"))

(defn lookup 
  "Looks up `keys` in a table, outputs the corresponding values."
  [ self keys name ]
  (py/call-attr self "lookup"  self keys name ))

(defn name 
  "The name of the table."
  [ self ]
    (py/call-attr self "name"))

(defn resource-handle 
  "Returns the resource handle associated with this Resource."
  [ self ]
    (py/call-attr self "resource_handle"))

(defn size 
  "Compute the number of elements in this table."
  [ self name ]
  (py/call-attr self "size"  self name ))

(defn value-dtype 
  "The table value dtype."
  [ self ]
    (py/call-attr self "value_dtype"))
