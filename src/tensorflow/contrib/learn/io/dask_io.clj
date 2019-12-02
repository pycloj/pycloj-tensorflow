(ns tensorflow.contrib.learn.python.learn.learn-io.dask-io
  "Methods to allow dask.DataFrame (deprecated).

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
(defonce dask-io (import-module "tensorflow.contrib.learn.python.learn.learn_io.dask_io"))
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
    (py/call-attr-kw dask-io "deprecated" [date instructions] {:warn_once warn_once }))

(defn extract-dask-data 
  "Extract data from dask.Series or dask.DataFrame for predictors. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please feed input to tf.data to support dask.

Given a distributed dask.DataFrame or dask.Series containing columns or names
for one or more predictors, this operation returns a single dask.DataFrame or
dask.Series that can be iterated over.

Args:
  data: A distributed dask.DataFrame or dask.Series.

Returns:
  A dask.DataFrame or dask.Series that can be iterated over.
  If the supplied argument is neither a dask.DataFrame nor a dask.Series this
  operation returns it without modification."
  [ data ]
  (py/call-attr dask-io "extract_dask_data"  data ))

(defn extract-dask-labels 
  "Extract data from dask.Series or dask.DataFrame for labels. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please feed input to tf.data to support dask.

Given a distributed dask.DataFrame or dask.Series containing exactly one
column or name, this operation returns a single dask.DataFrame or dask.Series
that can be iterated over.

Args:
  labels: A distributed dask.DataFrame or dask.Series with exactly one
          column or name.

Returns:
  A dask.DataFrame or dask.Series that can be iterated over.
  If the supplied argument is neither a dask.DataFrame nor a dask.Series this
  operation returns it without modification.

Raises:
  ValueError: If the supplied dask.DataFrame contains more than one
              column or the supplied dask.Series contains more than
              one name."
  [ labels ]
  (py/call-attr dask-io "extract_dask_labels"  labels ))
