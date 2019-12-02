(ns tensorflow.contrib.data.TFRecordWriter
  "Writes data to a TFRecord file."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce data (import-module "tensorflow.contrib.data"))

(defn TFRecordWriter 
  "Writes data to a TFRecord file."
  [ filename compression_type ]
  (py/call-attr data "TFRecordWriter"  filename compression_type ))

(defn write 
  "Returns a `tf.Operation` to write a dataset to a file.

    Args:
      dataset: a `tf.data.Dataset` whose elements are to be written to a file

    Returns:
      A `tf.Operation` that, when run, writes contents of `dataset` to a file.
    "
  [ self dataset ]
  (py/call-attr self "write"  self dataset ))
