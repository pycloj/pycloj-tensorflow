(ns tensorflow.-api.v1.compat.v1.python-io
  "Python functions for directly manipulating TFRecord-formatted files.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce python-io (import-module "tensorflow._api.v1.compat.v1.python_io"))

(defn tf-record-iterator 
  "An iterator that read the records from a TFRecords file. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use eager execution and: 
`tf.data.TFRecordDataset(path)`

Args:
  path: The path to the TFRecords file.
  options: (optional) A TFRecordOptions object.

Yields:
  Strings.

Raises:
  IOError: If `path` cannot be opened for reading."
  [ path options ]
  (py/call-attr python-io "tf_record_iterator"  path options ))
