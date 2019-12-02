(ns tensorflow.contrib.tfprof.tfprof-logger
  "Logging tensorflow::tfprof::OpLogProto.

OpLogProto is used to add extra model information for offline analysis.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tfprof-logger (import-module "tensorflow.contrib.tfprof.tfprof_logger"))
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
    (py/call-attr-kw tfprof-logger "deprecated" [date instructions] {:warn_once warn_once }))
(defn write-op-log 
  "DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2018-01-01.
Instructions for updating:
Use `tf.profiler.write_op_log. go/tfprof`"
  [graph log_dir op_log run_meta  & {:keys [add_trace]} ]
    (py/call-attr-kw tfprof-logger "write_op_log" [graph log_dir op_log run_meta] {:add_trace add_trace }))
