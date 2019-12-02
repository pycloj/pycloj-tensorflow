(ns tensorflow.contrib.timeseries.CSVReader
  "Reads from a collection of CSV-formatted files."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce timeseries (import-module "tensorflow.contrib.timeseries"))

(defn CSVReader 
  "Reads from a collection of CSV-formatted files."
  [filenames & {:keys [column_names column_dtypes skip_header_lines read_num_records_hint]
                       :or {column_dtypes None skip_header_lines None}} ]
    (py/call-attr-kw timeseries "CSVReader" [filenames] {:column_names column_names :column_dtypes column_dtypes :skip_header_lines skip_header_lines :read_num_records_hint read_num_records_hint }))

(defn check-dataset-size 
  "When possible, raises an error if the dataset is too small.

    This method allows TimeSeriesReaders to raise informative error messages if
    the user has selected a window size in their TimeSeriesInputFn which is
    larger than the dataset size. However, many TimeSeriesReaders will not have
    access to a dataset size, in which case they do not need to override this
    method.

    Args:
      minimum_dataset_size: The minimum number of records which should be
        contained in the dataset. Readers should attempt to raise an error when
        possible if an epoch of data contains fewer records.
    "
  [ self minimum_dataset_size ]
  (py/call-attr self "check_dataset_size"  self minimum_dataset_size ))

(defn read 
  "Reads a chunk of data from the `tf.compat.v1.ReaderBase` for later re-chunking."
  [ self  ]
  (py/call-attr self "read"  self  ))

(defn read-full 
  "Reads a full epoch of data into memory."
  [ self  ]
  (py/call-attr self "read_full"  self  ))
