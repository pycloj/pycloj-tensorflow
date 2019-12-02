(ns tensorflow.contrib.training.FeedingQueueRunner
  "A queue runner that allows the feeding of values such as numpy arrays."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce training (import-module "tensorflow.contrib.training"))

(defn FeedingQueueRunner 
  "A queue runner that allows the feeding of values such as numpy arrays."
  [ queue enqueue_ops close_op cancel_op feed_fns queue_closed_exception_types ]
  (py/call-attr training "FeedingQueueRunner"  queue enqueue_ops close_op cancel_op feed_fns queue_closed_exception_types ))

(defn cancel-op 
  ""
  [ self ]
    (py/call-attr self "cancel_op"))

(defn close-op 
  ""
  [ self ]
    (py/call-attr self "close_op"))
(defn create-threads 
  "Create threads to run the enqueue ops for the given session.

    This method requires a session in which the graph was launched.  It creates
    a list of threads, optionally starting them.  There is one thread for each
    op passed in `enqueue_ops`.

    The `coord` argument is an optional coordinator, that the threads will use
    to terminate together and report exceptions.  If a coordinator is given,
    this method starts an additional thread to close the queue when the
    coordinator requests a stop.

    If previously created threads for the given session are still running, no
    new threads will be created.

    Args:
      sess: A `Session`.
      coord: Optional `Coordinator` object for reporting errors and checking
        stop conditions.
      daemon: Boolean.  If `True` make the threads daemon threads.
      start: Boolean.  If `True` starts the threads.  If `False` the
        caller must call the `start()` method of the returned threads.

    Returns:
      A list of threads.
    "
  [self sess coord  & {:keys [daemon start]} ]
    (py/call-attr-kw self "create_threads" [sess coord] {:daemon daemon :start start }))

(defn enqueue-ops 
  ""
  [ self ]
    (py/call-attr self "enqueue_ops"))

(defn exceptions-raised 
  "Exceptions raised but not handled by the `QueueRunner` threads.

    Exceptions raised in queue runner threads are handled in one of two ways
    depending on whether or not a `Coordinator` was passed to
    `create_threads()`:

    * With a `Coordinator`, exceptions are reported to the coordinator and
      forgotten by the `QueueRunner`.
    * Without a `Coordinator`, exceptions are captured by the `QueueRunner` and
      made available in this `exceptions_raised` property.

    Returns:
      A list of Python `Exception` objects.  The list is empty if no exception
      was captured.  (No exceptions are captured when using a Coordinator.)
    "
  [ self ]
    (py/call-attr self "exceptions_raised"))

(defn from-proto 
  "Returns a `QueueRunner` object created from `queue_runner_def`."
  [ self queue_runner_def import_scope ]
  (py/call-attr self "from_proto"  self queue_runner_def import_scope ))

(defn name 
  "The string name of the underlying Queue."
  [ self ]
    (py/call-attr self "name"))

(defn queue 
  ""
  [ self ]
    (py/call-attr self "queue"))

(defn queue-closed-exception-types 
  ""
  [ self ]
    (py/call-attr self "queue_closed_exception_types"))

(defn to-proto 
  ""
  [ self  ]
  (py/call-attr self "to_proto"  self  ))
