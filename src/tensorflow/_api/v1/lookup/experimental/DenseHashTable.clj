(ns tensorflow.-api.v1.lookup.experimental.DenseHashTable
  "A generic mutable hash table implementation using tensors as backing store.

  Data can be inserted by calling the insert method and removed by calling the
  remove method. It does not support initialization via the init method.

  It uses \"open addressing\" with quadratic reprobing to resolve collisions.
  Compared to `MutableHashTable` the insert, remove and lookup operations in a
  `DenseHashTable` are typically faster, but memory usage can be higher.
  However, `DenseHashTable` does not require additional memory for
  temporary tensors created during checkpointing and restore operations.

  Example usage:

  ```python
  table = tf.lookup.DenseHashTable(key_dtype=tf.int64,
                                   value_dtype=tf.int64,
                                   default_value=-1,
                                   empty_key=0,
                                   deleted_key=-1)

  sess.run(table.insert(keys, values))
  out = table.lookup(query_keys)
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
(defonce experimental (import-module "tensorflow._api.v1.lookup.experimental"))
(defn DenseHashTable 
  "A generic mutable hash table implementation using tensors as backing store.

  Data can be inserted by calling the insert method and removed by calling the
  remove method. It does not support initialization via the init method.

  It uses \"open addressing\" with quadratic reprobing to resolve collisions.
  Compared to `MutableHashTable` the insert, remove and lookup operations in a
  `DenseHashTable` are typically faster, but memory usage can be higher.
  However, `DenseHashTable` does not require additional memory for
  temporary tensors created during checkpointing and restore operations.

  Example usage:

  ```python
  table = tf.lookup.DenseHashTable(key_dtype=tf.int64,
                                   value_dtype=tf.int64,
                                   default_value=-1,
                                   empty_key=0,
                                   deleted_key=-1)

  sess.run(table.insert(keys, values))
  out = table.lookup(query_keys)
  print(out.eval())
  ```
  "
  [key_dtype value_dtype default_value empty_key deleted_key initial_num_buckets  & {:keys [name checkpoint]} ]
    (py/call-attr-kw experimental "DenseHashTable" [key_dtype value_dtype default_value empty_key deleted_key initial_num_buckets] {:name name :checkpoint checkpoint }))

(defn erase 
  "Removes `keys` and its associated values from the table.

    If a key is not present in the table, it is silently ignored.

    Args:
      keys: Keys to remove. Can be a tensor of any shape. Must match the table's
        key type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` do not match the table data types.
    "
  [ self keys name ]
  (py/call-attr self "erase"  self keys name ))

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

(defn insert 
  "Associates `keys` with `values`.

    Args:
      keys: Keys to insert. Can be a tensor of any shape. Must match the table's
        key type.
      values: Values to be associated with keys. Must be a tensor of the same
        shape as `keys` and match the table's value type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` or `values` doesn't match the table data
        types.
    "
  [ self keys values name ]
  (py/call-attr self "insert"  self keys values name ))

(defn insert-or-assign 
  "Associates `keys` with `values`.

    Args:
      keys: Keys to insert. Can be a tensor of any shape. Must match the table's
        key type.
      values: Values to be associated with keys. Must be a tensor of the same
        shape as `keys` and match the table's value type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` or `values` doesn't match the table data
        types.
    "
  [ self keys values name ]
  (py/call-attr self "insert_or_assign"  self keys values name ))

(defn key-dtype 
  "The table key dtype."
  [ self ]
    (py/call-attr self "key_dtype"))

(defn lookup 
  "Looks up `keys` in a table, outputs the corresponding values.

    The `default_value` is used for keys not present in the table.

    Args:
      keys: Keys to look up. Can be a tensor of any shape. Must match the
        table's key_dtype.
      name: A name for the operation (optional).

    Returns:
      A tensor containing the values in the same shape as `keys` using the
        table's value type.

    Raises:
      TypeError: when `keys` do not match the table data types.
    "
  [ self keys name ]
  (py/call-attr self "lookup"  self keys name ))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))

(defn remove 
  "Removes `keys` and its associated values from the table.

    If a key is not present in the table, it is silently ignored.

    Args:
      keys: Keys to remove. Can be a tensor of any shape. Must match the table's
        key type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` do not match the table data types.
    "
  [ self keys name ]
  (py/call-attr self "remove"  self keys name ))

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
