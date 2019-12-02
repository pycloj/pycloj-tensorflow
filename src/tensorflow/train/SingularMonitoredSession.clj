(ns tensorflow.train.SingularMonitoredSession
  "Session-like object that handles initialization, restoring, and hooks.

  Please note that this utility is not recommended for distributed settings.
  For distributed settings, please use `tf.compat.v1.train.MonitoredSession`.
  The
  differences between `MonitoredSession` and `SingularMonitoredSession` are:

  * `MonitoredSession` handles `AbortedError` and `UnavailableError` for
    distributed settings, but `SingularMonitoredSession` does not.
  * `MonitoredSession` can be created in `chief` or `worker` modes.
    `SingularMonitoredSession` is always created as `chief`.
  * You can access the raw `tf.compat.v1.Session` object used by
    `SingularMonitoredSession`, whereas in MonitoredSession the raw session is
    private. This can be used:
      - To `run` without hooks.
      - To save and restore.
  * All other functionality is identical.

  Example usage:
  ```python
  saver_hook = CheckpointSaverHook(...)
  summary_hook = SummarySaverHook(...)
  with SingularMonitoredSession(hooks=[saver_hook, summary_hook]) as sess:
    while not sess.should_stop():
      sess.run(train_op)
  ```

  Initialization: At creation time the hooked session does following things
  in given order:

  * calls `hook.begin()` for each given hook
  * finalizes the graph via `scaffold.finalize()`
  * create session
  * initializes the model via initialization ops provided by `Scaffold`
  * restores variables if a checkpoint exists
  * launches queue runners

  Run: When `run()` is called, the hooked session does following things:

  * calls `hook.before_run()`
  * calls TensorFlow `session.run()` with merged fetches and feed_dict
  * calls `hook.after_run()`
  * returns result of `session.run()` asked by user

  Exit: At the `close()`, the hooked session does following things in order:

  * calls `hook.end()`
  * closes the queue runners and the session
  * suppresses `OutOfRange` error which indicates that all inputs have been
    processed if the `SingularMonitoredSession` is used as a context.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce train (import-module "tensorflow.train"))

(defn SingularMonitoredSession 
  "Session-like object that handles initialization, restoring, and hooks.

  Please note that this utility is not recommended for distributed settings.
  For distributed settings, please use `tf.compat.v1.train.MonitoredSession`.
  The
  differences between `MonitoredSession` and `SingularMonitoredSession` are:

  * `MonitoredSession` handles `AbortedError` and `UnavailableError` for
    distributed settings, but `SingularMonitoredSession` does not.
  * `MonitoredSession` can be created in `chief` or `worker` modes.
    `SingularMonitoredSession` is always created as `chief`.
  * You can access the raw `tf.compat.v1.Session` object used by
    `SingularMonitoredSession`, whereas in MonitoredSession the raw session is
    private. This can be used:
      - To `run` without hooks.
      - To save and restore.
  * All other functionality is identical.

  Example usage:
  ```python
  saver_hook = CheckpointSaverHook(...)
  summary_hook = SummarySaverHook(...)
  with SingularMonitoredSession(hooks=[saver_hook, summary_hook]) as sess:
    while not sess.should_stop():
      sess.run(train_op)
  ```

  Initialization: At creation time the hooked session does following things
  in given order:

  * calls `hook.begin()` for each given hook
  * finalizes the graph via `scaffold.finalize()`
  * create session
  * initializes the model via initialization ops provided by `Scaffold`
  * restores variables if a checkpoint exists
  * launches queue runners

  Run: When `run()` is called, the hooked session does following things:

  * calls `hook.before_run()`
  * calls TensorFlow `session.run()` with merged fetches and feed_dict
  * calls `hook.after_run()`
  * returns result of `session.run()` asked by user

  Exit: At the `close()`, the hooked session does following things in order:

  * calls `hook.end()`
  * closes the queue runners and the session
  * suppresses `OutOfRange` error which indicates that all inputs have been
    processed if the `SingularMonitoredSession` is used as a context.
  "
  [hooks scaffold & {:keys [master config checkpoint_dir stop_grace_period_secs checkpoint_filename_with_path]
                       :or {config None checkpoint_dir None checkpoint_filename_with_path None}} ]
    (py/call-attr-kw train "SingularMonitoredSession" [hooks scaffold] {:master master :config config :checkpoint_dir checkpoint_dir :stop_grace_period_secs stop_grace_period_secs :checkpoint_filename_with_path checkpoint_filename_with_path }))

(defn close 
  ""
  [ self  ]
  (py/call-attr self "close"  self  ))

(defn graph 
  "The graph that was launched in this session."
  [ self ]
    (py/call-attr self "graph"))

(defn raw-session 
  "Returns underlying `TensorFlow.Session` object."
  [ self  ]
  (py/call-attr self "raw_session"  self  ))

(defn run 
  "Run ops in the monitored session.

    This method is completely compatible with the `tf.Session.run()` method.

    Args:
      fetches: Same as `tf.Session.run()`.
      feed_dict: Same as `tf.Session.run()`.
      options: Same as `tf.Session.run()`.
      run_metadata: Same as `tf.Session.run()`.

    Returns:
      Same as `tf.Session.run()`.
    "
  [ self fetches feed_dict options run_metadata ]
  (py/call-attr self "run"  self fetches feed_dict options run_metadata ))

(defn run-step-fn 
  "Run ops using a step function.

    Args:
      step_fn: A function or a method with a single argument of type
        `StepContext`.  The function may use methods of the argument to perform
        computations with access to a raw session.  The returned value of the
        `step_fn` will be returned from `run_step_fn`, unless a stop is
        requested.  In that case, the next `should_stop` call will return True.
        Example usage:
            ```python
            with tf.Graph().as_default():
              c = tf.compat.v1.placeholder(dtypes.float32)
              v = tf.add(c, 4.0)
              w = tf.add(c, 0.5)
              def step_fn(step_context):
                a = step_context.session.run(fetches=v, feed_dict={c: 0.5})
                if a <= 4.5:
                  step_context.request_stop()
                  return step_context.run_with_hooks(fetches=w,
                                                     feed_dict={c: 0.1})

              with tf.MonitoredSession() as session:
                while not session.should_stop():
                  a = session.run_step_fn(step_fn)
            ```
            Hooks interact with the `run_with_hooks()` call inside the
                 `step_fn` as they do with a `MonitoredSession.run` call.

    Returns:
      Returns the returned value of `step_fn`.

    Raises:
      StopIteration: if `step_fn` has called `request_stop()`.  It may be
        caught by `with tf.MonitoredSession()` to close the session.
      ValueError: if `step_fn` doesn't have a single argument called
        `step_context`. It may also optionally have `self` for cases when it
        belongs to an object.
    "
  [ self step_fn ]
  (py/call-attr self "run_step_fn"  self step_fn ))

(defn should-stop 
  ""
  [ self  ]
  (py/call-attr self "should_stop"  self  ))
