(ns tensorflow.contrib.framework.RecordInput
  "RecordInput asynchronously reads and randomly yields TFRecords.

  A RecordInput Op will continuously read a batch of records asynchronously
  into a buffer of some fixed capacity. It can also asynchronously yield
  random records from this buffer.

  It will not start yielding until at least `buffer_size / 2` elements have been
  placed into the buffer so that sufficient randomization can take place.

  The order the files are read will be shifted each epoch by `shift_amount` so
  that the data is presented in a different order every epoch.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce framework (import-module "tensorflow.contrib.framework"))

(defn RecordInput 
  "RecordInput asynchronously reads and randomly yields TFRecords.

  A RecordInput Op will continuously read a batch of records asynchronously
  into a buffer of some fixed capacity. It can also asynchronously yield
  random records from this buffer.

  It will not start yielding until at least `buffer_size / 2` elements have been
  placed into the buffer so that sufficient randomization can take place.

  The order the files are read will be shifted each epoch by `shift_amount` so
  that the data is presented in a different order every epoch.
  "
  [file_pattern & {:keys [batch_size buffer_size parallelism shift_ratio seed name batches compression_type]
                       :or {name None batches None compression_type None}} ]
    (py/call-attr-kw framework "RecordInput" [file_pattern] {:batch_size batch_size :buffer_size buffer_size :parallelism parallelism :shift_ratio shift_ratio :seed seed :name name :batches batches :compression_type compression_type }))

(defn get-yield-op 
  "Adds a node that yields a group of records every time it is executed.
    If RecordInput `batches` parameter is not None, it yields a list of
    record batches with the specified `batch_size`.
    "
  [ self  ]
  (py/call-attr self "get_yield_op"  self  ))
