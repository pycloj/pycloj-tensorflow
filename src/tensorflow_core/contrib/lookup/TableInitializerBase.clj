(ns tensorflow-core.contrib.lookup.TableInitializerBase
  "Base class for lookup table initializers."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce lookup (import-module "tensorflow_core.contrib.lookup"))

(defn TableInitializerBase 
  "Base class for lookup table initializers."
  [ key_dtype value_dtype ]
  (py/call-attr lookup "TableInitializerBase"  key_dtype value_dtype ))

(defn initialize 
  "Returns the table initialization op."
  [ self table ]
  (py/call-attr self "initialize"  self table ))

(defn key-dtype 
  "The expected table key dtype."
  [ self ]
    (py/call-attr self "key_dtype"))

(defn value-dtype 
  "The expected table value dtype."
  [ self ]
    (py/call-attr self "value_dtype"))
