(ns tensorflow-core.contrib.lookup.HashTable
  "A generic hash table implementation.

  Example usage:

  ```python
  table = tf.HashTable(
      tf.KeyValueTensorInitializer(keys, values), -1)
  out = table.lookup(input_tensor)
  table.init.run()
  print(out.eval())
  ```
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

(defn HashTable 
  "A generic hash table implementation.

  Example usage:

  ```python
  table = tf.HashTable(
      tf.KeyValueTensorInitializer(keys, values), -1)
  out = table.lookup(input_tensor)
  table.init.run()
  print(out.eval())
  ```
  "
  [ initializer default_value shared_name name ]
  (py/call-attr lookup "HashTable"  initializer default_value shared_name name ))

(defn default-value 
  "The default value of the table."
  [ self ]
    (py/call-attr self "default_value"))

(defn export 
  "Returns tensors of all keys and values in the table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A pair of tensors with the first tensor containing all keys and the
        second tensors containing all values in the table.
    "
  [ self name ]
  (py/call-attr self "export"  self name ))

(defn init 
  ""
  [ self ]
    (py/call-attr self "init"))

(defn initializer 
  ""
  [ self ]
    (py/call-attr self "initializer"))

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
  ""
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
