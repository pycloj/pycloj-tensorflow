(ns tensorflow.contrib.cloud.BigtableTable
  "Entry point for reading and writing data in Cloud Bigtable.

  This BigtableTable class is the Python representation of the Cloud Bigtable
  table within TensorFlow. Methods on this class allow data to be read from and
  written to the Cloud Bigtable service in flexible and high performance
  manners.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cloud (import-module "tensorflow.contrib.cloud"))

(defn BigtableTable 
  "Entry point for reading and writing data in Cloud Bigtable.

  This BigtableTable class is the Python representation of the Cloud Bigtable
  table within TensorFlow. Methods on this class allow data to be read from and
  written to the Cloud Bigtable service in flexible and high performance
  manners.
  "
  [ name snapshot resource ]
  (py/call-attr cloud "BigtableTable"  name snapshot resource ))

(defn keys-by-prefix-dataset 
  "Retrieves the row keys matching a given prefix.

    Args:
      prefix: All row keys that begin with `prefix` in the table will be
        retrieved.

    Returns:
      A `tf.data.Dataset`. containing `tf.string` Tensors corresponding to all
      of the row keys matching that prefix.
    "
  [ self prefix ]
  (py/call-attr self "keys_by_prefix_dataset"  self prefix ))

(defn keys-by-range-dataset 
  "Retrieves all row keys between start and end.

    Note: it does NOT retrieve the values of columns.

    Args:
      start: The start row key. The row keys for rows after start (inclusive)
        will be retrieved.
      end: (Optional.) The end row key. Rows up to (but not including) end will
        be retrieved. If end is None, all subsequent row keys will be retrieved.

    Returns:
      A `tf.data.Dataset` containing `tf.string` Tensors corresponding to all
      of the row keys between `start` and `end`.
    "
  [ self start end ]
  (py/call-attr self "keys_by_range_dataset"  self start end ))

(defn lookup-columns 
  "Retrieves the values of columns for a dataset of keys.

    Example usage:

    ```python
    table = bigtable_client.table(\"my_table\")
    key_dataset = table.get_keys_prefix(\"imagenet\")
    images = key_dataset.apply(table.lookup_columns((\"cf1\", \"image\"),
                                                    (\"cf2\", \"label\"),
                                                    (\"cf2\", \"boundingbox\")))
    training_data = images.map(parse_and_crop, num_parallel_calls=64).batch(128)
    ```

    Alternatively, you can use keyword arguments to specify the columns to
    capture. Example (same as above, rewritten):

    ```python
    table = bigtable_client.table(\"my_table\")
    key_dataset = table.get_keys_prefix(\"imagenet\")
    images = key_dataset.apply(table.lookup_columns(
        cf1=\"image\", cf2=(\"label\", \"boundingbox\")))
    training_data = images.map(parse_and_crop, num_parallel_calls=64).batch(128)
    ```

    Note: certain `kwargs` keys are reserved, and thus, some column families
    cannot be identified using the `kwargs` syntax. Instead, please use the
    `args` syntax. This list includes:

      - 'name'

    Note: this list can change at any time.

    Args:
      *args: A list of tuples containing (column family, column name) pairs.
      **kwargs: Column families (keys) and column qualifiers (values).

    Returns:
      A function that can be passed to `tf.data.Dataset.apply` to retrieve the
      values of columns for the rows.
    "
  [ self  ]
  (py/call-attr self "lookup_columns"  self  ))

(defn parallel-scan-prefix 
  "Retrieves row (including values) from the Bigtable service at high speed.

    Rows with row-key prefixed by `prefix` will be retrieved. This method is
    similar to `scan_prefix`, but by contrast performs multiple sub-scans in
    parallel in order to achieve higher performance.

    Note: The dataset produced by this method is not deterministic!

    Specifying the columns to retrieve for each row is done by either using
    kwargs or in the columns parameter. To retrieve values of the columns \"c1\",
    and \"c2\" from the column family \"cfa\", and the value of the column \"c3\"
    from column family \"cfb\", the following datasets (`ds1`, and `ds2`) are
    equivalent:

    ```
    table = # ...
    ds1 = table.parallel_scan_prefix(\"row_prefix\", columns=[(\"cfa\", \"c1\"),
                                                            (\"cfa\", \"c2\"),
                                                            (\"cfb\", \"c3\")])
    ds2 = table.parallel_scan_prefix(\"row_prefix\", cfa=[\"c1\", \"c2\"], cfb=\"c3\")
    ```

    Note: only the latest value of a cell will be retrieved.

    Args:
      prefix: The prefix all row keys must match to be retrieved for prefix-
        based scans.
      num_parallel_scans: (Optional.) The number of concurrent scans against the
        Cloud Bigtable instance.
      probability: (Optional.) A float between 0 (exclusive) and 1 (inclusive).
        A non-1 value indicates to probabilistically sample rows with the
        provided probability.
      columns: The columns to read. Note: most commonly, they are expressed as
        kwargs. Use the columns value if you are using column families that are
        reserved. The value of columns and kwargs are merged. Columns is a list
        of tuples of strings (\"column_family\", \"column_qualifier\").
      **kwargs: The column families and columns to read. Keys are treated as
        column_families, and values can be either lists of strings, or strings
        that are treated as the column qualifier (column name).

    Returns:
      A `tf.data.Dataset` returning the row keys and the cell contents.

    Raises:
      ValueError: If the configured probability is unexpected.
    "
  [ self prefix num_parallel_scans probability columns ]
  (py/call-attr self "parallel_scan_prefix"  self prefix num_parallel_scans probability columns ))

(defn parallel-scan-range 
  "Retrieves rows (including values) from the Bigtable service.

    Rows with row-keys between `start` and `end` will be retrieved. This method
    is similar to `scan_range`, but by contrast performs multiple sub-scans in
    parallel in order to achieve higher performance.

    Note: The dataset produced by this method is not deterministic!

    Specifying the columns to retrieve for each row is done by either using
    kwargs or in the columns parameter. To retrieve values of the columns \"c1\",
    and \"c2\" from the column family \"cfa\", and the value of the column \"c3\"
    from column family \"cfb\", the following datasets (`ds1`, and `ds2`) are
    equivalent:

    ```
    table = # ...
    ds1 = table.parallel_scan_range(\"row_start\",
                                    \"row_end\",
                                    columns=[(\"cfa\", \"c1\"),
                                             (\"cfa\", \"c2\"),
                                             (\"cfb\", \"c3\")])
    ds2 = table.parallel_scan_range(\"row_start\", \"row_end\",
                                    cfa=[\"c1\", \"c2\"], cfb=\"c3\")
    ```

    Note: only the latest value of a cell will be retrieved.

    Args:
      start: The start of the range when scanning by range.
      end: (Optional.) The end of the range when scanning by range.
      num_parallel_scans: (Optional.) The number of concurrent scans against the
        Cloud Bigtable instance.
      probability: (Optional.) A float between 0 (exclusive) and 1 (inclusive).
        A non-1 value indicates to probabilistically sample rows with the
        provided probability.
      columns: The columns to read. Note: most commonly, they are expressed as
        kwargs. Use the columns value if you are using column families that are
        reserved. The value of columns and kwargs are merged. Columns is a list
        of tuples of strings (\"column_family\", \"column_qualifier\").
      **kwargs: The column families and columns to read. Keys are treated as
        column_families, and values can be either lists of strings, or strings
        that are treated as the column qualifier (column name).

    Returns:
      A `tf.data.Dataset` returning the row keys and the cell contents.

    Raises:
      ValueError: If the configured probability is unexpected.
    "
  [ self start end num_parallel_scans probability columns ]
  (py/call-attr self "parallel_scan_range"  self start end num_parallel_scans probability columns ))

(defn sample-keys 
  "Retrieves a sampling of row keys from the Bigtable table.

    This dataset is most often used in conjunction with
    `tf.data.experimental.parallel_interleave` to construct a set of ranges for
    scanning in parallel.

    Returns:
      A `tf.data.Dataset` returning string row keys.
    "
  [ self  ]
  (py/call-attr self "sample_keys"  self  ))

(defn scan-prefix 
  "Retrieves row (including values) from the Bigtable service.

    Rows with row-key prefixed by `prefix` will be retrieved.

    Specifying the columns to retrieve for each row is done by either using
    kwargs or in the columns parameter. To retrieve values of the columns \"c1\",
    and \"c2\" from the column family \"cfa\", and the value of the column \"c3\"
    from column family \"cfb\", the following datasets (`ds1`, and `ds2`) are
    equivalent:

    ```
    table = # ...
    ds1 = table.scan_prefix(\"row_prefix\", columns=[(\"cfa\", \"c1\"),
                                                   (\"cfa\", \"c2\"),
                                                   (\"cfb\", \"c3\")])
    ds2 = table.scan_prefix(\"row_prefix\", cfa=[\"c1\", \"c2\"], cfb=\"c3\")
    ```

    Note: only the latest value of a cell will be retrieved.

    Args:
      prefix: The prefix all row keys must match to be retrieved for prefix-
        based scans.
      probability: (Optional.) A float between 0 (exclusive) and 1 (inclusive).
        A non-1 value indicates to probabilistically sample rows with the
        provided probability.
      columns: The columns to read. Note: most commonly, they are expressed as
        kwargs. Use the columns value if you are using column families that are
        reserved. The value of columns and kwargs are merged. Columns is a list
        of tuples of strings (\"column_family\", \"column_qualifier\").
      **kwargs: The column families and columns to read. Keys are treated as
        column_families, and values can be either lists of strings, or strings
        that are treated as the column qualifier (column name).

    Returns:
      A `tf.data.Dataset` returning the row keys and the cell contents.

    Raises:
      ValueError: If the configured probability is unexpected.
    "
  [ self prefix probability columns ]
  (py/call-attr self "scan_prefix"  self prefix probability columns ))

(defn scan-range 
  "Retrieves rows (including values) from the Bigtable service.

    Rows with row-keys between `start` and `end` will be retrieved.

    Specifying the columns to retrieve for each row is done by either using
    kwargs or in the columns parameter. To retrieve values of the columns \"c1\",
    and \"c2\" from the column family \"cfa\", and the value of the column \"c3\"
    from column family \"cfb\", the following datasets (`ds1`, and `ds2`) are
    equivalent:

    ```
    table = # ...
    ds1 = table.scan_range(\"row_start\", \"row_end\", columns=[(\"cfa\", \"c1\"),
                                                            (\"cfa\", \"c2\"),
                                                            (\"cfb\", \"c3\")])
    ds2 = table.scan_range(\"row_start\", \"row_end\", cfa=[\"c1\", \"c2\"], cfb=\"c3\")
    ```

    Note: only the latest value of a cell will be retrieved.

    Args:
      start: The start of the range when scanning by range.
      end: (Optional.) The end of the range when scanning by range.
      probability: (Optional.) A float between 0 (exclusive) and 1 (inclusive).
        A non-1 value indicates to probabilistically sample rows with the
        provided probability.
      columns: The columns to read. Note: most commonly, they are expressed as
        kwargs. Use the columns value if you are using column families that are
        reserved. The value of columns and kwargs are merged. Columns is a list
        of tuples of strings (\"column_family\", \"column_qualifier\").
      **kwargs: The column families and columns to read. Keys are treated as
        column_families, and values can be either lists of strings, or strings
        that are treated as the column qualifier (column name).

    Returns:
      A `tf.data.Dataset` returning the row keys and the cell contents.

    Raises:
      ValueError: If the configured probability is unexpected.
    "
  [ self start end probability columns ]
  (py/call-attr self "scan_range"  self start end probability columns ))

(defn write 
  "Writes a dataset to the table.

    Args:
      dataset: A `tf.data.Dataset` to be written to this table. It must produce
        a list of number-of-columns+1 elements, all of which must be strings.
        The first value will be used as the row key, and subsequent values will
        be used as cell values for the corresponding columns from the
        corresponding column_families and columns entries.
      column_families: A `tf.Tensor` of `tf.string`s corresponding to the
        column names to store the dataset's elements into.
      columns: A `tf.Tensor` of `tf.string`s corresponding to the column names
        to store the dataset's elements into.
      timestamp: (Optional.) An int64 timestamp to write all the values at.
        Leave as None to use server-provided timestamps.

    Returns:
      A `tf.Operation` that can be run to perform the write.

    Raises:
      ValueError: If there are unexpected or incompatible types, or if the
        number of columns and column_families does not match the output of
        `dataset`.
    "
  [ self dataset column_families columns timestamp ]
  (py/call-attr self "write"  self dataset column_families columns timestamp ))
