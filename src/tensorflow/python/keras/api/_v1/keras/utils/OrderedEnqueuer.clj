(ns tensorflow.python.keras.api.-v1.keras.utils.OrderedEnqueuer
  "Builds a Enqueuer from a Sequence.

  Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

  Arguments:
      sequence: A `tf.keras.utils.data_utils.Sequence` object.
      use_multiprocessing: use multiprocessing if True, otherwise threading
      shuffle: whether to shuffle the data at the beginning of each epoch
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
(defn OrderedEnqueuer 
  "Builds a Enqueuer from a Sequence.

  Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

  Arguments:
      sequence: A `tf.keras.utils.data_utils.Sequence` object.
      use_multiprocessing: use multiprocessing if True, otherwise threading
      shuffle: whether to shuffle the data at the beginning of each epoch
  "
  [sequence  & {:keys [use_multiprocessing shuffle]} ]
    (py/call-attr-kw utils "OrderedEnqueuer" [sequence] {:use_multiprocessing use_multiprocessing :shuffle shuffle }))

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
