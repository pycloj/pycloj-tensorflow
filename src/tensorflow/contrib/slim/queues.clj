(ns tensorflow.contrib.slim.python.slim.queues
  "Contains a helper context for running queue runners.

@@NestedQueueRunnerError
@@QueueRunners
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce queues (import-module "tensorflow.contrib.slim.python.slim.queues"))

(defn QueueRunners 
  "Creates a context manager that handles starting and stopping queue runners.

  Args:
    session: the currently running session.

  Yields:
    a context in which queues are run.

  Raises:
    NestedQueueRunnerError: if a QueueRunners context is nested within another.
  "
  [ session ]
  (py/call-attr queues "QueueRunners"  session ))

(defn contextmanager 
  "@contextmanager decorator.

    Typical usage:

        @contextmanager
        def some_generator(<arguments>):
            <setup>
            try:
                yield <value>
            finally:
                <cleanup>

    This makes this:

        with some_generator(<arguments>) as <variable>:
            <body>

    equivalent to this:

        <setup>
        try:
            <variable> = <value>
            <body>
        finally:
            <cleanup>
    "
  [ func ]
  (py/call-attr queues "contextmanager"  func ))
