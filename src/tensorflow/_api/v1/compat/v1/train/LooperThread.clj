(ns tensorflow.-api.v1.compat.v1.train.LooperThread
  "A thread that runs code repeatedly, optionally on a timer.

  This thread class is intended to be used with a `Coordinator`.  It repeatedly
  runs code specified either as `target` and `args` or by the `run_loop()`
  method.

  Before each run the thread checks if the coordinator has requested stop.  In
  that case the looper thread terminates immediately.

  If the code being run raises an exception, that exception is reported to the
  coordinator and the thread terminates.  The coordinator will then request all
  the other threads it coordinates to stop.

  You typically pass looper threads to the supervisor `Join()` method.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce train (import-module "tensorflow._api.v1.compat.v1.train"))

(defn LooperThread 
  "A thread that runs code repeatedly, optionally on a timer.

  This thread class is intended to be used with a `Coordinator`.  It repeatedly
  runs code specified either as `target` and `args` or by the `run_loop()`
  method.

  Before each run the thread checks if the coordinator has requested stop.  In
  that case the looper thread terminates immediately.

  If the code being run raises an exception, that exception is reported to the
  coordinator and the thread terminates.  The coordinator will then request all
  the other threads it coordinates to stop.

  You typically pass looper threads to the supervisor `Join()` method.
  "
  [ coord timer_interval_secs target args kwargs ]
  (py/call-attr train "LooperThread"  coord timer_interval_secs target args kwargs ))

(defn daemon 
  "A boolean value indicating whether this thread is a daemon thread.

        This must be set before start() is called, otherwise RuntimeError is
        raised. Its initial value is inherited from the creating thread; the
        main thread is not a daemon thread and therefore all threads created in
        the main thread default to daemon = False.

        The entire Python program exits when only daemon threads are left.

        "
  [ self ]
    (py/call-attr self "daemon"))

(defn getName 
  ""
  [ self  ]
  (py/call-attr self "getName"  self  ))

(defn ident 
  "Thread identifier of this thread or None if it has not been started.

        This is a nonzero integer. See the get_ident() function. Thread
        identifiers may be recycled when a thread exits and another thread is
        created. The identifier is available even after the thread has exited.

        "
  [ self ]
    (py/call-attr self "ident"))

(defn isAlive 
  "Return whether the thread is alive.

        This method is deprecated, use is_alive() instead.
        "
  [ self  ]
  (py/call-attr self "isAlive"  self  ))

(defn isDaemon 
  ""
  [ self  ]
  (py/call-attr self "isDaemon"  self  ))

(defn is-alive 
  "Return whether the thread is alive.

        This method returns True just before the run() method starts until just
        after the run() method terminates. The module function enumerate()
        returns a list of all alive threads.

        "
  [ self  ]
  (py/call-attr self "is_alive"  self  ))

(defn join 
  "Wait until the thread terminates.

        This blocks the calling thread until the thread whose join() method is
        called terminates -- either normally or through an unhandled exception
        or until the optional timeout occurs.

        When the timeout argument is present and not None, it should be a
        floating point number specifying a timeout for the operation in seconds
        (or fractions thereof). As join() always returns None, you must call
        is_alive() after join() to decide whether a timeout happened -- if the
        thread is still alive, the join() call timed out.

        When the timeout argument is not present or None, the operation will
        block until the thread terminates.

        A thread can be join()ed many times.

        join() raises a RuntimeError if an attempt is made to join the current
        thread as that would cause a deadlock. It is also an error to join() a
        thread before it has been started and attempts to do so raises the same
        exception.

        "
  [ self timeout ]
  (py/call-attr self "join"  self timeout ))

(defn loop 
  "Start a LooperThread that calls a function periodically.

    If `timer_interval_secs` is None the thread calls `target(args)`
    repeatedly.  Otherwise `target(args)` is called every `timer_interval_secs`
    seconds.  The thread terminates when a stop of the coordinator is
    requested.

    Args:
      coord: A Coordinator.
      timer_interval_secs: Number. Time boundaries at which to call `target`.
      target: A callable object.
      args: Optional arguments to pass to `target` when calling it.
      kwargs: Optional keyword arguments to pass to `target` when calling it.

    Returns:
      The started thread.
    "
  [ self coord timer_interval_secs target args kwargs ]
  (py/call-attr self "loop"  self coord timer_interval_secs target args kwargs ))

(defn name 
  "A string used for identification purposes only.

        It has no semantics. Multiple threads may be given the same name. The
        initial name is set by the constructor.

        "
  [ self ]
    (py/call-attr self "name"))

(defn run 
  ""
  [ self  ]
  (py/call-attr self "run"  self  ))

(defn run-loop 
  "Called at 'timer_interval_secs' boundaries."
  [ self  ]
  (py/call-attr self "run_loop"  self  ))

(defn setDaemon 
  ""
  [ self daemonic ]
  (py/call-attr self "setDaemon"  self daemonic ))

(defn setName 
  ""
  [ self name ]
  (py/call-attr self "setName"  self name ))

(defn start 
  "Start the thread's activity.

        It must be called at most once per thread object. It arranges for the
        object's run() method to be invoked in a separate thread of control.

        This method will raise a RuntimeError if called more than once on the
        same thread object.

        "
  [ self  ]
  (py/call-attr self "start"  self  ))

(defn start-loop 
  "Called when the thread starts."
  [ self  ]
  (py/call-attr self "start_loop"  self  ))

(defn stop-loop 
  "Called when the thread stops."
  [ self  ]
  (py/call-attr self "stop_loop"  self  ))
