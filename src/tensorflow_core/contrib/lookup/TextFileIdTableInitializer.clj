(ns tensorflow-core.contrib.lookup.TextFileIdTableInitializer
  "Table initializer for string to `int64` IDs tables from a text file."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce lookup (import-module "tensorflow_core.contrib.lookup"))

(defn TextFileIdTableInitializer 
  "Table initializer for string to `int64` IDs tables from a text file."
  [filename & {:keys [key_column_index value_column_index vocab_size delimiter name key_dtype]
                       :or {vocab_size None}} ]
    (py/call-attr-kw lookup "TextFileIdTableInitializer" [filename] {:key_column_index key_column_index :value_column_index value_column_index :vocab_size vocab_size :delimiter delimiter :name name :key_dtype key_dtype }))

(defn initialize 
  "Initializes the table from a text file.

    Args:
      table: The table to be initialized.

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
