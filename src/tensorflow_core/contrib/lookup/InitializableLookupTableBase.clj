(ns tensorflow-core.contrib.lookup.InitializableLookupTableBase
  "Initializable lookup table interface.

  An initializable lookup tables persist across different steps.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce lookup (import-module "tensorflow_core.contrib.lookup"))

(defn InitializableLookupTableBase 
  "Initializable lookup table interface.

  An initializable lookup tables persist across different steps.
  "
  [ default_value initializer ]
  (py/call-attr lookup "InitializableLookupTableBase"  default_value initializer ))

(defn default-value 
  "The default value of the table."
  [ self ]
    (py/call-attr self "default_value"))

(defn key-dtype 
  "The table key dtype."
  [ self ]
    (py/call-attr self "key_dtype"))

(defn lookup 
  "Looks up `keys` in a table, outputs the corresponding values.

    The `default_value` is used for keys not present in the table.

    Args:
      keys: Keys to look up. May be either a `SparseTensor` or dense `Tensor`.
      name: A name for the operation (optional).

    Returns:
      A `SparseTensor` if keys are sparse, otherwise a dense `Tensor`.

    Raises:
      TypeError: when `keys` or `default_value` doesn't match the table data
        types.
    "
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
  "Compute the number of elements in this table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A scalar tensor containing the number of elements in this table.
    "
  [ self name ]
  (py/call-attr self "size"  self name ))

(defn value-dtype 
  "The table value dtype."
  [ self ]
    (py/call-attr self "value_dtype"))
