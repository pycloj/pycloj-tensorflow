(ns tensorflow.contrib.learn.python.learn.datasets.base
  "Base utilities for loading datasets (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce base (import-module "tensorflow.contrib.learn.python.learn.datasets.base"))
(defn deprecated 
  "Decorator for marking functions or methods deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called. It has the following format:

    <function> (from <module>) is deprecated and will be removed after <date>.
    Instructions for updating:
    <instructions>

  If `date` is None, 'after <date>' is replaced with 'in a future version'.
  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated)' is appended
  to the first line of the docstring and a deprecation notice is prepended
  to the rest of the docstring.

  Args:
    date: String or None. The date the function is scheduled to be removed.
      Must be ISO 8601 (YYYY-MM-DD), or None.
    instructions: String. Instructions on how to update code using the
      deprecated function.
    warn_once: Boolean. Set to `True` to warn only the first time the decorated
      function is called. Otherwise, every call will log a warning.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not None or in ISO 8601 format, or instructions are
      empty.
  "
  [date instructions  & {:keys [warn_once]} ]
    (py/call-attr-kw base "deprecated" [date instructions] {:warn_once warn_once }))

(defn load-boston 
  "Load Boston housing dataset. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use scikits.learn.datasets.

Args:
    data_path: string, path to boston dataset (optional)

Returns:
  Dataset object containing data in-memory."
  [ data_path ]
  (py/call-attr base "load_boston"  data_path ))
(defn load-csv-with-header 
  "Load dataset from CSV file with a header row. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.data instead."
  [filename target_dtype features_dtype  & {:keys [target_column]} ]
    (py/call-attr-kw base "load_csv_with_header" [filename target_dtype features_dtype] {:target_column target_column }))
(defn load-csv-without-header 
  "Load dataset from CSV file without a header row. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.data instead."
  [filename target_dtype features_dtype  & {:keys [target_column]} ]
    (py/call-attr-kw base "load_csv_without_header" [filename target_dtype features_dtype] {:target_column target_column }))

(defn load-iris 
  "Load Iris dataset. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use scikits.learn.datasets.

Args:
    data_path: string, path to iris dataset (optional)

Returns:
  Dataset object containing data in-memory."
  [ data_path ]
  (py/call-attr base "load_iris"  data_path ))

(defn maybe-download 
  "Download the data from source url, unless it's already here. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please write your own downloading logic.

Args:
    filename: string, name of the file in the directory.
    work_directory: string, path to working directory.
    source_url: url to download from if file doesn't exist.

Returns:
    Path to resulting file."
  [ filename work_directory source_url ]
  (py/call-attr base "maybe_download"  filename work_directory source_url ))

(defn retry 
  "Simple decorator for wrapping retriable functions. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use the retry module or similar alternatives.

Args:
  initial_delay: the initial delay.
  max_delay: the maximum delay allowed (actual max is
      max_delay * (1 + jitter).
  factor: each subsequent retry, the delay is multiplied by this value.
      (must be >= 1).
  jitter: to avoid lockstep, the returned delay is multiplied by a random
      number between (1-jitter) and (1+jitter). To add a 20% jitter, set
      jitter = 0.2. Must be < 1.
  is_retriable: (optional) a function that takes an Exception as an argument
      and returns true if retry should be applied.

Returns:
  A function that wraps another function to automatically retry it."
  [initial_delay max_delay & {:keys [factor jitter is_retriable]
                       :or {is_retriable None}} ]
    (py/call-attr-kw base "retry" [initial_delay max_delay] {:factor factor :jitter jitter :is_retriable is_retriable }))

(defn shrink-csv 
  "Create a smaller dataset of only 1/ratio of original data. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.data instead."
  [ filename ratio ]
  (py/call-attr base "shrink_csv"  filename ratio ))

(defn urlretrieve-with-retry 
  "The actual wrapper function that applies the retry logic. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use urllib or similar directly."
  [  ]
  (py/call-attr base "urlretrieve_with_retry"  ))
