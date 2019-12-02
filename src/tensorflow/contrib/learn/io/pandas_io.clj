(ns tensorflow.contrib.learn.python.learn.learn-io.pandas-io
  "Methods to allow pandas.DataFrame (deprecated).

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
(defonce pandas-io (import-module "tensorflow.contrib.learn.python.learn.learn_io.pandas_io"))

(defn core-pandas-input-fn 
  "Returns input function that would feed Pandas DataFrame into the model.

  Note: `y`'s index must match `x`'s index.

  Args:
    x: pandas `DataFrame` object.
    y: pandas `Series` object or `DataFrame`. `None` if absent.
    batch_size: int, size of batches to return.
    num_epochs: int, number of epochs to iterate over data. If not `None`,
      read attempts that would exceed this value will raise `OutOfRangeError`.
    shuffle: bool, whether to read the records in random order.
    queue_capacity: int, size of the read queue. If `None`, it will be set
      roughly to the size of `x`.
    num_threads: Integer, number of threads used for reading and enqueueing. In
      order to have predicted and repeatable order of reading and enqueueing,
      such as in prediction and evaluation mode, `num_threads` should be 1.
    target_column: str, name to give the target column `y`. This parameter
      is not used when `y` is a `DataFrame`.

  Returns:
    Function, that has signature of ()->(dict of `features`, `target`)

  Raises:
    ValueError: if `x` already contains a column with the same name as `y`, or
      if the indexes of `x` and `y` don't match.
    ValueError: if 'shuffle' is not provided or a bool.
  "
  [x y & {:keys [batch_size num_epochs shuffle queue_capacity num_threads target_column]
                       :or {shuffle None}} ]
    (py/call-attr-kw pandas-io "core_pandas_input_fn" [x y] {:batch_size batch_size :num_epochs num_epochs :shuffle shuffle :queue_capacity queue_capacity :num_threads num_threads :target_column target_column }))
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
    (py/call-attr-kw pandas-io "deprecated" [date instructions] {:warn_once warn_once }))

(defn extract-pandas-data 
  "Extract data from pandas.DataFrame for predictors. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please access pandas data directly.

Given a DataFrame, will extract the values and cast them to float. The
DataFrame is expected to contain values of type int, float or bool.

Args:
  data: `pandas.DataFrame` containing the data to be extracted.

Returns:
  A numpy `ndarray` of the DataFrame's values as floats.

Raises:
  ValueError: if data contains types other than int, float or bool."
  [ data ]
  (py/call-attr pandas-io "extract_pandas_data"  data ))

(defn extract-pandas-labels 
  "Extract data from pandas.DataFrame for labels. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please access pandas data directly.

Args:
  labels: `pandas.DataFrame` or `pandas.Series` containing one column of
    labels to be extracted.

Returns:
  A numpy `ndarray` of labels from the DataFrame.

Raises:
  ValueError: if more than one column is found or type is not int, float or
    bool."
  [ labels ]
  (py/call-attr pandas-io "extract_pandas_labels"  labels ))

(defn extract-pandas-matrix 
  "Extracts numpy matrix from pandas DataFrame. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please access pandas data directly.

Args:
  data: `pandas.DataFrame` containing the data to be extracted.

Returns:
  A numpy `ndarray` of the DataFrame's values."
  [ data ]
  (py/call-attr pandas-io "extract_pandas_matrix"  data ))
(defn pandas-input-fn 
  "This input_fn diffs from the core version with default `shuffle`. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use tf.compat.v1.estimator.inputs.pandas_input_fn"
  [x y  & {:keys [batch_size num_epochs shuffle queue_capacity num_threads target_column]} ]
    (py/call-attr-kw pandas-io "pandas_input_fn" [x y] {:batch_size batch_size :num_epochs num_epochs :shuffle shuffle :queue_capacity queue_capacity :num_threads num_threads :target_column target_column }))
