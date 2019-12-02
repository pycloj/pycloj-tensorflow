(ns tensorflow-core.contrib.lookup.KeyValueTensorInitializer
  "Table initializers given `keys` and `values` tensors."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce lookup (import-module "tensorflow_core.contrib.lookup"))

(defn KeyValueTensorInitializer 
  "Table initializers given `keys` and `values` tensors."
  [ keys values key_dtype value_dtype name ]
  (py/call-attr lookup "KeyValueTensorInitializer"  keys values key_dtype value_dtype name ))

(defn initialize 
  "Initializes the given `table` with `keys` and `values` tensors.

    Args:
      table: The table to initialize.

    Returns:
      The operation that initializes the table.

    Raises:
      TypeError: when the keys and values data types do not match the table
      key and value data types.
    "
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
