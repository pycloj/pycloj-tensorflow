(ns tensorflow.-api.v1.data.experimental.TFRecordWriter
  "Writes data to a TFRecord file.

  To write a `dataset` to a single TFRecord file:

  ```python
  dataset = ... # dataset to be written
  writer = tf.data.experimental.TFRecordWriter(PATH)
  writer.write(dataset)
  ```

  To shard a `dataset` across multiple TFRecord files:

  ```python
  dataset = ... # dataset to be written

  def reduce_func(key, dataset):
    filename = tf.strings.join([PATH_PREFIX, tf.strings.as_string(key)])
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset.map(lambda _, x: x))
    return tf.data.Dataset.from_tensors(filename)

  dataset = dataset.enumerate()
  dataset = dataset.apply(tf.data.experimental.group_by_window(
    lambda i, _: i % NUM_SHARDS, reduce_func, tf.int64.max
  ))
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
(defonce experimental (import-module "tensorflow._api.v1.data.experimental"))

(defn TFRecordWriter 
  "Writes data to a TFRecord file.

  To write a `dataset` to a single TFRecord file:

  ```python
  dataset = ... # dataset to be written
  writer = tf.data.experimental.TFRecordWriter(PATH)
  writer.write(dataset)
  ```

  To shard a `dataset` across multiple TFRecord files:

  ```python
  dataset = ... # dataset to be written

  def reduce_func(key, dataset):
    filename = tf.strings.join([PATH_PREFIX, tf.strings.as_string(key)])
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(dataset.map(lambda _, x: x))
    return tf.data.Dataset.from_tensors(filename)

  dataset = dataset.enumerate()
  dataset = dataset.apply(tf.data.experimental.group_by_window(
    lambda i, _: i % NUM_SHARDS, reduce_func, tf.int64.max
  ))
  ```
  "
  [ filename compression_type ]
  (py/call-attr experimental "TFRecordWriter"  filename compression_type ))

(defn write 
  "Returns a `tf.Operation` to write a dataset to a file.

    Args:
      dataset: a `tf.data.Dataset` whose elements are to be written to a file

    Returns:
      A `tf.Operation` that, when run, writes contents of `dataset` to a file.
    "
  [ self dataset ]
  (py/call-attr self "write"  self dataset ))
