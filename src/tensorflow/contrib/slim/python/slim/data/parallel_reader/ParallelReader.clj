(ns tensorflow.contrib.slim.python.slim.data.parallel-reader.ParallelReader
  "Reader class that uses multiple readers in parallel to improve speed.

  See ReaderBase for supported methods.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce parallel-reader (import-module "tensorflow.contrib.slim.python.slim.data.parallel_reader"))

(defn ParallelReader 
  "Reader class that uses multiple readers in parallel to improve speed.

  See ReaderBase for supported methods.
  "
  [reader_class common_queue & {:keys [num_readers reader_kwargs]
                       :or {reader_kwargs None}} ]
    (py/call-attr-kw parallel-reader "ParallelReader" [reader_class common_queue] {:num_readers num_readers :reader_kwargs reader_kwargs }))

(defn common-queue 
  ""
  [ self ]
    (py/call-attr self "common_queue"))

(defn num-readers 
  ""
  [ self ]
    (py/call-attr self "num_readers"))

(defn num-records-produced 
  "Returns the number of records this reader has produced.

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
  "Returns the next record (key, value pair) produced by the reader.

    The multiple reader instances are all configured to `read()` from the
    filenames listed in `queue` and enqueue their output into the `common_queue`
    passed to the constructor, and this method returns the next record dequeued
    from that `common_queue`.


    Readers dequeue a work unit from `queue` if necessary (e.g. when a
    reader needs to start reading from a new file since it has finished with
    the previous file).

    A queue runner for enqueuing in the `common_queue` is automatically added
    to the TF QueueRunners collection.

    Args:
      queue: A Queue or a mutable string Tensor representing a handle to a
        Queue, with string work items.
      name: A name for the operation (optional).

    Returns:
      The next record (i.e. (key, value pair)) from the common_queue.
    "
  [ self queue name ]
  (py/call-attr self "read"  self queue name ))

(defn read-up-to 
  "Returns up to num_records (key, value pairs) produced by a reader.

    Will dequeue a work unit from queue if necessary (e.g., when the
    Reader needs to start reading from a new file since it has
    finished with the previous file).
    It may return less than num_records even before the last batch.

    **Note** This operation is not supported by all types of `common_queue`s.
    If a `common_queue` does not support `dequeue_up_to()`, then a
    `tf.errors.UnimplementedError` is raised.

    Args:
      queue: A Queue or a mutable string Tensor representing a handle to a
        Queue, with string work items.
      num_records: Number of records to read.
      name: A name for the operation (optional).

    Returns:
      A tuple of Tensors (keys, values) from common_queue.
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
