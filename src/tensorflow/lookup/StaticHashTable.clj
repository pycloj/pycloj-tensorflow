(ns tensorflow.lookup.StaticHashTable
  "A generic hash table that is immutable once initialized.

  When running in graph mode, you must evaluate the tensor returned by
  `tf.tables_initializer()` before evaluating the tensor returned by
  this class's `lookup()` method. Example usage in graph mode:

  ```python
  keys_tensor = tf.constant([1, 2])
  vals_tensor = tf.constant([3, 4])
  input_tensor = tf.constant([1, 5])
  table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
  out = table.lookup(input_tensor)
  with tf.Session() as sess:
      sess.run(tf.tables_initializer())
      print(sess.run(out))
  ```

  In eager mode, no special code is needed to initialize the table.
  Example usage in eager mode:

  ```python
  tf.enable_eager_execution()
  keys_tensor = tf.constant([1, 2])
  vals_tensor = tf.constant([3, 4])
  input_tensor = tf.constant([1, 5])
  table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
  print(table.lookup(input_tensor))
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
(defonce lookup (import-module "tensorflow.lookup"))

(defn StaticHashTable 
  "A generic hash table that is immutable once initialized.

  When running in graph mode, you must evaluate the tensor returned by
  `tf.tables_initializer()` before evaluating the tensor returned by
  this class's `lookup()` method. Example usage in graph mode:

  ```python
  keys_tensor = tf.constant([1, 2])
  vals_tensor = tf.constant([3, 4])
  input_tensor = tf.constant([1, 5])
  table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
  out = table.lookup(input_tensor)
  with tf.Session() as sess:
      sess.run(tf.tables_initializer())
      print(sess.run(out))
  ```

  In eager mode, no special code is needed to initialize the table.
  Example usage in eager mode:

  ```python
  tf.enable_eager_execution()
  keys_tensor = tf.constant([1, 2])
  vals_tensor = tf.constant([3, 4])
  input_tensor = tf.constant([1, 5])
  table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
  print(table.lookup(input_tensor))
  ```
  "
  [ initializer default_value name ]
  (py/call-attr lookup "StaticHashTable"  initializer default_value name ))

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
