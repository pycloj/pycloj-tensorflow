(ns tensorflow.lookup.TextFileIndex
  "The key and value content to get from each line.

  This class defines the key and value used for tf.lookup.TextFileInitializer.

  The key and value content to get from each line is specified either
  by the following, or a value `>=0`.
  * `TextFileIndex.LINE_NUMBER` means use the line number starting from zero,
    expects data type int64.
  * `TextFileIndex.WHOLE_LINE` means use the whole line content, expects data
    type string.

  A value `>=0` means use the index (starting at zero) of the split line based
      on `delimiter`.
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

(defn TextFileIndex 
  "The key and value content to get from each line.

  This class defines the key and value used for tf.lookup.TextFileInitializer.

  The key and value content to get from each line is specified either
  by the following, or a value `>=0`.
  * `TextFileIndex.LINE_NUMBER` means use the line number starting from zero,
    expects data type int64.
  * `TextFileIndex.WHOLE_LINE` means use the whole line content, expects data
    type string.

  A value `>=0` means use the index (starting at zero) of the split line based
      on `delimiter`.
  "
  [  ]
  (py/call-attr lookup "TextFileIndex"  ))
