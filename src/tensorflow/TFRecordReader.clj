(ns tensorflow.TFRecordReader
  "A Reader that outputs the records from a TFRecords file.

  See ReaderBase for supported methods.

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tensorflow (import-module "tensorflow"))

(defn TFRecordReader 
  "A Reader that outputs the records from a TFRecords file.

  See ReaderBase for supported methods.

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  "
  [ name options ]
  (py/call-attr tensorflow "TFRecordReader"  name options ))

(defn num-records-produced 
  "Returns the number of records this reader has produced.

    This is the same as the number of Read executions that have
    succeeded.

    Args:
      name: A name for the operation (optional).

    Returns:
      An int64 Tensor.

    "
  [ self name ]
  (py/call-attr self "num_records_produced"  self name ))

(defn num-work-units-completed 
  "Returns the number of work units this reader has finished processing.

    Args:
      name: A name for the operation (optional).

    Returns:
      An int64 Tensor.
    "
  [ self name ]
  (py/call-attr self "num_work_units_completed"  self name ))

(defn read 
  "Returns the next record (key, value) pair produced by a reader.

    Will dequeue a work unit from queue if necessary (e.g. when the
    Reader needs to start reading from a new file since it has
    finished with the previous file).

    Args:
      queue: A Queue or a mutable string Tensor representing a handle
        to a Queue, with string work items.
      name: A name for the operation (optional).

    Returns:
      A tuple of Tensors (key, value).
      key: A string scalar Tensor.
      value: A string scalar Tensor.
    "
  [ self queue name ]
  (py/call-attr self "read"  self queue name ))

(defn read-up-to 
  "Returns up to num_records (key, value) pairs produced by a reader.

    Will dequeue a work unit from queue if necessary (e.g., when the
    Reader needs to start reading from a new file since it has
    finished with the previous file).
    It may return less than num_records even before the last batch.

    Args:
      queue: A Queue or a mutable string Tensor representing a handle
        to a Queue, with string work items.
      num_records: Number of records to read.
      name: A name for the operation (optional).

    Returns:
      A tuple of Tensors (keys, values).
      keys: A 1-D string Tensor.
      values: A 1-D string Tensor.
    "
  [ self queue num_records name ]
  (py/call-attr self "read_up_to"  self queue num_records name ))

(defn reader-ref 
  "Op that implements the reader."
  [ self ]
    (py/call-attr self "reader_ref"))

(defn reset 
  "Restore a reader to its initial clean state.

    Args:
      name: A name for the operation (optional).

    Returns:
      The created Operation.
    "
  [ self name ]
  (py/call-attr self "reset"  self name ))

(defn restore-state 
  "Restore a reader to a previously saved state.

    Not all Readers support being restored, so this can produce an
    Unimplemented error.

    Args:
      state: A string Tensor.
        Result of a SerializeState of a Reader with matching type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.
    "
  [ self state name ]
  (py/call-attr self "restore_state"  self state name ))

(defn serialize-state 
  "Produce a string tensor that encodes the state of a reader.

    Not all Readers support being serialized, so this can produce an
    Unimplemented error.

    Args:
      name: A name for the operation (optional).

    Returns:
      A string Tensor.
    "
  [ self name ]
  (py/call-attr self "serialize_state"  self name ))

(defn supports-serialize 
  "Whether the Reader implementation can serialize its state."
  [ self ]
    (py/call-attr self "supports_serialize"))
