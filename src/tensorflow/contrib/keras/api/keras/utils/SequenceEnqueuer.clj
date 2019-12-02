(ns tensorflow.contrib.keras.api.keras.utils.SequenceEnqueuer
  "Base class to enqueue inputs.

  The task of an Enqueuer is to use parallelism to speed up preprocessing.
  This is done with processes or threads.

  Example:

  ```python
      enqueuer = SequenceEnqueuer(...)
      enqueuer.start()
      datas = enqueuer.get()
      for data in datas:
          # Use the inputs; training, evaluating, predicting.
          # ... stop sometime.
      enqueuer.close()
  ```

  The `enqueuer.get()` should be an infinite stream of datas.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce utils (import-module "tensorflow.contrib.keras.api.keras.utils"))
(defn SequenceEnqueuer 
  "Base class to enqueue inputs.

  The task of an Enqueuer is to use parallelism to speed up preprocessing.
  This is done with processes or threads.

  Example:

  ```python
      enqueuer = SequenceEnqueuer(...)
      enqueuer.start()
      datas = enqueuer.get()
      for data in datas:
          # Use the inputs; training, evaluating, predicting.
          # ... stop sometime.
      enqueuer.close()
  ```

  The `enqueuer.get()` should be an infinite stream of datas.
  "
  [sequence  & {:keys [use_multiprocessing]} ]
    (py/call-attr-kw utils "SequenceEnqueuer" [sequence] {:use_multiprocessing use_multiprocessing }))

(defn get 
  "Creates a generator to extract data from the queue.

    Skip the data if it is `None`.
    # Returns
        Generator yielding tuples `(inputs, targets)`
            or `(inputs, targets, sample_weights)`.
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
