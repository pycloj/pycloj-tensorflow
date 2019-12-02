(ns tensorflow.contrib.slim.python.slim.data.parallel-reader
  "Implements a parallel data reader with queues and optional shuffling."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce parallel-reader (import-module "tensorflow.contrib.slim.python.slim.data.parallel_reader"))

(defn get-data-files 
  "Get data_files from data_sources.

  Args:
    data_sources: a list/tuple of files or the location of the data, i.e.
      /path/to/train@128, /path/to/train* or /tmp/.../train*

  Returns:
    a list of data_files.

  Raises:
    ValueError: if data files are not found

  "
  [ data_sources ]
  (py/call-attr parallel-reader "get_data_files"  data_sources ))

(defn parallel-read 
  "Reads multiple records in parallel from data_sources using n readers.

  It uses a ParallelReader to read from multiple files in parallel using
  multiple readers created using `reader_class` with `reader_kwargs'.

  If shuffle is True the common_queue would be a RandomShuffleQueue otherwise
  it would be a FIFOQueue.

  Usage:
      data_sources = ['path_to/train*']
      key, value = parallel_read(data_sources, tf.CSVReader, num_readers=4)

  Args:
    data_sources: a list/tuple of files or the location of the data, i.e.
      /path/to/train@128, /path/to/train* or /tmp/.../train*
    reader_class: one of the io_ops.ReaderBase subclasses ex: TFRecordReader
    num_epochs: The number of times each data source is read. If left as None,
      the data will be cycled through indefinitely.
    num_readers: a integer, number of Readers to create.
    reader_kwargs: an optional dict, of kwargs for the reader.
    shuffle: boolean, whether should shuffle the files and the records by using
      RandomShuffleQueue as common_queue.
    dtypes:  A list of types.  The length of dtypes must equal the number of
      elements in each record. If it is None it will default to [tf.string,
      tf.string] for (key, value).
    capacity: integer, capacity of the common_queue.
    min_after_dequeue: integer, minimum number of records in the common_queue
      after dequeue. Needed for a good shuffle.
    seed: A seed for RandomShuffleQueue.
    scope: Optional name scope for the ops.

  Returns:
    key, value: a tuple of keys and values from the data_source.
  "
  [data_sources reader_class num_epochs & {:keys [num_readers reader_kwargs shuffle dtypes capacity min_after_dequeue seed scope]
                       :or {reader_kwargs None dtypes None seed None scope None}} ]
    (py/call-attr-kw parallel-reader "parallel_read" [data_sources reader_class num_epochs] {:num_readers num_readers :reader_kwargs reader_kwargs :shuffle shuffle :dtypes dtypes :capacity capacity :min_after_dequeue min_after_dequeue :seed seed :scope scope }))

(defn single-pass-read 
  "Reads sequentially the data_sources using the reader, doing a single pass.

  Args:
    data_sources: a list/tuple of files or the location of the data, i.e.
      /path/to/train@128, /path/to/train* or /tmp/.../train*
    reader_class: one of the io_ops.ReaderBase subclasses ex: TFRecordReader.
    reader_kwargs: an optional dict, of kwargs for the reader.
    scope: Optional name scope for the ops.

  Returns:
    key, value: a tuple of keys and values from the data_source.
  "
  [ data_sources reader_class reader_kwargs scope ]
  (py/call-attr parallel-reader "single_pass_read"  data_sources reader_class reader_kwargs scope ))
