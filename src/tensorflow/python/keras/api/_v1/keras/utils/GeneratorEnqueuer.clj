(ns tensorflow.python.keras.api.-v1.keras.utils.GeneratorEnqueuer
  "Builds a queue out of a data generator.

  The provided generator can be finite in which case the class will throw
  a `StopIteration` exception.

  Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

  Arguments:
      generator: a generator function which yields data
      use_multiprocessing: use multiprocessing if True, otherwise threading
      wait_time: time to sleep in-between calls to `put()`
      random_seed: Initial seed for workers,
          will be incremented by one for each worker.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce utils (import-module "tensorflow.python.keras.api._v1.keras.utils"))

(defn GeneratorEnqueuer 
  "Builds a queue out of a data generator.

  The provided generator can be finite in which case the class will throw
  a `StopIteration` exception.

  Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

  Arguments:
      generator: a generator function which yields data
      use_multiprocessing: use multiprocessing if True, otherwise threading
      wait_time: time to sleep in-between calls to `put()`
      random_seed: Initial seed for workers,
          will be incremented by one for each worker.
  "
  [sequence & {:keys [use_multiprocessing random_seed]
                       :or {random_seed None}} ]
    (py/call-attr-kw utils "GeneratorEnqueuer" [sequence] {:use_multiprocessing use_multiprocessing :random_seed random_seed }))

(defn get 
  "Creates a generator to extract data from the queue.

    Skip the data if it is `None`.

    Yields:
        The next element in the queue, i.e. a tuple
        `(inputs, targets)` or
        `(inputs, targets, sample_weights)`.
    "
  [ self  ]
  (py/call-attr self "get"  self  ))

(defn is-running 
  ""
  [ self  ]
  (py/call-attr self "is_running"  self  ))
(defn start 
  "Starts the handler's workers.

    Arguments:
        workers: Number of workers.
        max_queue_size: queue size
            (when full, workers could block on `put()`)
    "
  [self   & {:keys [workers max_queue_size]} ]
    (py/call-attr-kw self "start" [] {:workers workers :max_queue_size max_queue_size }))

(defn stop 
  "Stops running threads and wait for them to exit, if necessary.

    Should be called by the same thread which called `start()`.

    Arguments:
        timeout: maximum time to wait on `thread.join()`
    "
  [ self timeout ]
  (py/call-attr self "stop"  self timeout ))
