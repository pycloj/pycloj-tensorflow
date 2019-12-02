(ns tensorflow.contrib.eager.python.tfe.ExecutionCallback
  "Valid callback actions.

  These can be passed to `seterr` or `errstate` to create callbacks when
  specific events occur (e.g. an operation produces `NaN`s).

  IGNORE: take no action.
  PRINT:  print a warning to `stdout`.
  RAISE:  raise an error (e.g. `InfOrNanError`).
  WARN:   print a warning using `tf.compat.v1.logging.warn`.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tfe (import-module "tensorflow.contrib.eager.python.tfe"))
(defn ExecutionCallback 
  "Valid callback actions.

  These can be passed to `seterr` or `errstate` to create callbacks when
  specific events occur (e.g. an operation produces `NaN`s).

  IGNORE: take no action.
  PRINT:  print a warning to `stdout`.
  RAISE:  raise an error (e.g. `InfOrNanError`).
  WARN:   print a warning using `tf.compat.v1.logging.warn`.
  "
  [value names module qualname type  & {:keys [start]} ]
    (py/call-attr-kw eager "ExecutionCallback" [value names module qualname type] {:start start }))
