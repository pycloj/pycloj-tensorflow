(ns tensorflow-core.contrib.lookup.TextFileInitializer
  "Table initializers from a text file.

  This initializer assigns one entry in the table for each line in the file.

  The key and value type of the table to initialize is given by `key_dtype` and
  `value_dtype`.

  The key and value content to get from each line is specified by
  the `key_index` and `value_index`.

  * `TextFileIndex.LINE_NUMBER` means use the line number starting from zero,
    expects data type int64.
  * `TextFileIndex.WHOLE_LINE` means use the whole line content, expects data
    type string.
  * A value `>=0` means use the index (starting at zero) of the split line based
      on `delimiter`.

  For example if we have a file with the following content:

  ```
  emerson 10
  lake 20
  palmer 30
  ```

  The following snippet initializes a table with the first column as keys and
  second column as values:

  * `emerson -> 10`
  * `lake -> 20`
  * `palmer -> 30`

  ```python
  table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
      \"test.txt\", tf.string, 0, tf.int64, 1, delimiter=\" \"), -1)
  ...
  table.init.run()
  ```

  Similarly to initialize the whole line as keys and the line number as values.

  * `emerson 10 -> 0`
  * `lake 20 -> 1`
  * `palmer 30 -> 2`

  ```python
  table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
      \"test.txt\", tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
      tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER, delimiter=\" \"), -1)
  ...
  table.init.run()
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

(defn TextFileInitializer 
  "Table initializers from a text file.

  This initializer assigns one entry in the table for each line in the file.

  The key and value type of the table to initialize is given by `key_dtype` and
  `value_dtype`.

  The key and value content to get from each line is specified by
  the `key_index` and `value_index`.

  * `TextFileIndex.LINE_NUMBER` means use the line number starting from zero,
    expects data type int64.
  * `TextFileIndex.WHOLE_LINE` means use the whole line content, expects data
    type string.
  * A value `>=0` means use the index (starting at zero) of the split line based
      on `delimiter`.

  For example if we have a file with the following content:

  ```
  emerson 10
  lake 20
  palmer 30
  ```

  The following snippet initializes a table with the first column as keys and
  second column as values:

  * `emerson -> 10`
  * `lake -> 20`
  * `palmer -> 30`

  ```python
  table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
      \"test.txt\", tf.string, 0, tf.int64, 1, delimiter=\" \"), -1)
  ...
  table.init.run()
  ```

  Similarly to initialize the whole line as keys and the line number as values.

  * `emerson 10 -> 0`
  * `lake 20 -> 1`
  * `palmer 30 -> 2`

  ```python
  table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
      \"test.txt\", tf.string, tf.lookup.TextFileIndex.WHOLE_LINE,
      tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER, delimiter=\" \"), -1)
  ...
  table.init.run()
  ```
  "
  [filename key_dtype key_index value_dtype value_index vocab_size & {:keys [delimiter name]
                       :or {name None}} ]
    (py/call-attr-kw lookup "TextFileInitializer" [filename key_dtype key_index value_dtype value_index vocab_size] {:delimiter delimiter :name name }))

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
