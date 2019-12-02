(ns tensorflow.-api.v1.compat.v1.train.Supervisor
  "A training helper that checkpoints models and computes summaries.

  This class is deprecated. Please use
  `tf.compat.v1.train.MonitoredTrainingSession` instead.

  The Supervisor is a small wrapper around a `Coordinator`, a `Saver`,
  and a `SessionManager` that takes care of common needs of TensorFlow
  training programs.

  #### Use for a single program

  ```python
  with tf.Graph().as_default():
    ...add operations to the graph...
    # Create a Supervisor that will checkpoint the model in '/tmp/mydir'.
    sv = Supervisor(logdir='/tmp/mydir')
    # Get a TensorFlow session managed by the supervisor.
    with sv.managed_session(FLAGS.master) as sess:
      # Use the session to train the graph.
      while not sv.should_stop():
        sess.run(<my_train_op>)
  ```

  Within the `with sv.managed_session()` block all variables in the graph have
  been initialized.  In addition, a few services have been started to
  checkpoint the model and add summaries to the event log.

  If the program crashes and is restarted, the managed session automatically
  reinitialize variables from the most recent checkpoint.

  The supervisor is notified of any exception raised by one of the services.
  After an exception is raised, `should_stop()` returns `True`.  In that case
  the training loop should also stop.  This is why the training loop has to
  check for `sv.should_stop()`.

  Exceptions that indicate that the training inputs have been exhausted,
  `tf.errors.OutOfRangeError`, also cause `sv.should_stop()` to return `True`
  but are not re-raised from the `with` block: they indicate a normal
  termination.

  #### Use for multiple replicas

  To train with replicas you deploy the same program in a `Cluster`.
  One of the tasks must be identified as the *chief*: the task that handles
  initialization, checkpoints, summaries, and recovery.  The other tasks
  depend on the *chief* for these services.

  The only change you have to do to the single program code is to indicate
  if the program is running as the *chief*.

  ```python
  # Choose a task as the chief. This could be based on server_def.task_index,
  # or job_def.name, or job_def.tasks. It's entirely up to the end user.
  # But there can be only one *chief*.
  is_chief = (server_def.task_index == 0)
  server = tf.distribute.Server(server_def)

  with tf.Graph().as_default():
    ...add operations to the graph...
    # Create a Supervisor that uses log directory on a shared file system.
    # Indicate if you are the 'chief'
    sv = Supervisor(logdir='/shared_directory/...', is_chief=is_chief)
    # Get a Session in a TensorFlow server on the cluster.
    with sv.managed_session(server.target) as sess:
      # Use the session to train the graph.
      while not sv.should_stop():
        sess.run(<my_train_op>)
  ```

  In the *chief* task, the `Supervisor` works exactly as in the first example
  above.  In the other tasks `sv.managed_session()` waits for the Model to have
  been initialized before returning a session to the training code.  The
  non-chief tasks depend on the chief task for initializing the model.

  If one of the tasks crashes and restarts, `managed_session()`
  checks if the Model is initialized.  If yes, it just creates a session and
  returns it to the training code that proceeds normally.  If the model needs
  to be initialized, the chief task takes care of reinitializing it; the other
  tasks just wait for the model to have been initialized.

  NOTE: This modified program still works fine as a single program.
  The single program marks itself as the chief.

  #### What `master` string to use

  Whether you are running on your machine or in the cluster you can use the
  following values for the --master flag:

  * Specifying `''` requests an in-process session that does not use RPC.

  * Specifying `'local'` requests a session that uses the RPC-based
    \"Master interface\" to run TensorFlow programs. See
    `tf.train.Server.create_local_server` for
    details.

  * Specifying `'grpc://hostname:port'` requests a session that uses
    the RPC interface to a specific host, and also allows the in-process
    master to access remote tensorflow workers. Often, it is
    appropriate to pass `server.target` (for some `tf.distribute.Server`
    named `server).

  #### Advanced use

  ##### Launching additional services

  `managed_session()` launches the Checkpoint and Summary services (threads).
  If you need more services to run you can simply launch them in the block
  controlled by `managed_session()`.

  Example: Start a thread to print losses.  We want this thread to run
  every 60 seconds, so we launch it with `sv.loop()`.

  ```python
  ...
  sv = Supervisor(logdir='/tmp/mydir')
  with sv.managed_session(FLAGS.master) as sess:
    sv.loop(60, print_loss, (sess, ))
    while not sv.should_stop():
      sess.run(my_train_op)
  ```

  ##### Launching fewer services

  `managed_session()` launches the \"summary\" and \"checkpoint\" threads which use
  either the optionally `summary_op` and `saver` passed to the constructor, or
  default ones created automatically by the supervisor.  If you want to run
  your own summary and checkpointing logic, disable these services by passing
  `None` to the `summary_op` and `saver` parameters.

  Example: Create summaries manually every 100 steps in the chief.

  ```python
  # Create a Supervisor with no automatic summaries.
  sv = Supervisor(logdir='/tmp/mydir', is_chief=is_chief, summary_op=None)
  # As summary_op was None, managed_session() does not start the
  # summary thread.
  with sv.managed_session(FLAGS.master) as sess:
    for step in xrange(1000000):
      if sv.should_stop():
        break
      if is_chief and step % 100 == 0:
        # Create the summary every 100 chief steps.
        sv.summary_computed(sess, sess.run(my_summary_op))
      else:
        # Train normally
        sess.run(my_train_op)
  ```

  ##### Custom model initialization

  `managed_session()` only supports initializing the model by running an
  `init_op` or restoring from the latest checkpoint.  If you have special
  initialization needs, see how to specify a `local_init_op` when creating the
  supervisor.  You can also use the `SessionManager` directly to create a
  session and check if it could be initialized automatically.
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

(defn Supervisor 
  "A training helper that checkpoints models and computes summaries.

  This class is deprecated. Please use
  `tf.compat.v1.train.MonitoredTrainingSession` instead.

  The Supervisor is a small wrapper around a `Coordinator`, a `Saver`,
  and a `SessionManager` that takes care of common needs of TensorFlow
  training programs.

  #### Use for a single program

  ```python
  with tf.Graph().as_default():
    ...add operations to the graph...
    # Create a Supervisor that will checkpoint the model in '/tmp/mydir'.
    sv = Supervisor(logdir='/tmp/mydir')
    # Get a TensorFlow session managed by the supervisor.
    with sv.managed_session(FLAGS.master) as sess:
      # Use the session to train the graph.
      while not sv.should_stop():
        sess.run(<my_train_op>)
  ```

  Within the `with sv.managed_session()` block all variables in the graph have
  been initialized.  In addition, a few services have been started to
  checkpoint the model and add summaries to the event log.

  If the program crashes and is restarted, the managed session automatically
  reinitialize variables from the most recent checkpoint.

  The supervisor is notified of any exception raised by one of the services.
  After an exception is raised, `should_stop()` returns `True`.  In that case
  the training loop should also stop.  This is why the training loop has to
  check for `sv.should_stop()`.

  Exceptions that indicate that the training inputs have been exhausted,
  `tf.errors.OutOfRangeError`, also cause `sv.should_stop()` to return `True`
  but are not re-raised from the `with` block: they indicate a normal
  termination.

  #### Use for multiple replicas

  To train with replicas you deploy the same program in a `Cluster`.
  One of the tasks must be identified as the *chief*: the task that handles
  initialization, checkpoints, summaries, and recovery.  The other tasks
  depend on the *chief* for these services.

  The only change you have to do to the single program code is to indicate
  if the program is running as the *chief*.

  ```python
  # Choose a task as the chief. This could be based on server_def.task_index,
  # or job_def.name, or job_def.tasks. It's entirely up to the end user.
  # But there can be only one *chief*.
  is_chief = (server_def.task_index == 0)
  server = tf.distribute.Server(server_def)

  with tf.Graph().as_default():
    ...add operations to the graph...
    # Create a Supervisor that uses log directory on a shared file system.
    # Indicate if you are the 'chief'
    sv = Supervisor(logdir='/shared_directory/...', is_chief=is_chief)
    # Get a Session in a TensorFlow server on the cluster.
    with sv.managed_session(server.target) as sess:
      # Use the session to train the graph.
      while not sv.should_stop():
        sess.run(<my_train_op>)
  ```

  In the *chief* task, the `Supervisor` works exactly as in the first example
  above.  In the other tasks `sv.managed_session()` waits for the Model to have
  been initialized before returning a session to the training code.  The
  non-chief tasks depend on the chief task for initializing the model.

  If one of the tasks crashes and restarts, `managed_session()`
  checks if the Model is initialized.  If yes, it just creates a session and
  returns it to the training code that proceeds normally.  If the model needs
  to be initialized, the chief task takes care of reinitializing it; the other
  tasks just wait for the model to have been initialized.

  NOTE: This modified program still works fine as a single program.
  The single program marks itself as the chief.

  #### What `master` string to use

  Whether you are running on your machine or in the cluster you can use the
  following values for the --master flag:

  * Specifying `''` requests an in-process session that does not use RPC.

  * Specifying `'local'` requests a session that uses the RPC-based
    \"Master interface\" to run TensorFlow programs. See
    `tf.train.Server.create_local_server` for
    details.

  * Specifying `'grpc://hostname:port'` requests a session that uses
    the RPC interface to a specific host, and also allows the in-process
    master to access remote tensorflow workers. Often, it is
    appropriate to pass `server.target` (for some `tf.distribute.Server`
    named `server).

  #### Advanced use

  ##### Launching additional services

  `managed_session()` launches the Checkpoint and Summary services (threads).
  If you need more services to run you can simply launch them in the block
  controlled by `managed_session()`.

  Example: Start a thread to print losses.  We want this thread to run
  every 60 seconds, so we launch it with `sv.loop()`.

  ```python
  ...
  sv = Supervisor(logdir='/tmp/mydir')
  with sv.managed_session(FLAGS.master) as sess:
    sv.loop(60, print_loss, (sess, ))
    while not sv.should_stop():
      sess.run(my_train_op)
  ```

  ##### Launching fewer services

  `managed_session()` launches the \"summary\" and \"checkpoint\" threads which use
  either the optionally `summary_op` and `saver` passed to the constructor, or
  default ones created automatically by the supervisor.  If you want to run
  your own summary and checkpointing logic, disable these services by passing
  `None` to the `summary_op` and `saver` parameters.

  Example: Create summaries manually every 100 steps in the chief.

  ```python
  # Create a Supervisor with no automatic summaries.
  sv = Supervisor(logdir='/tmp/mydir', is_chief=is_chief, summary_op=None)
  # As summary_op was None, managed_session() does not start the
  # summary thread.
  with sv.managed_session(FLAGS.master) as sess:
    for step in xrange(1000000):
      if sv.should_stop():
        break
      if is_chief and step % 100 == 0:
        # Create the summary every 100 chief steps.
        sv.summary_computed(sess, sess.run(my_summary_op))
      else:
        # Train normally
        sess.run(my_train_op)
  ```

  ##### Custom model initialization

  `managed_session()` only supports initializing the model by running an
  `init_op` or restoring from the latest checkpoint.  If you have special
  initialization needs, see how to specify a `local_init_op` when creating the
  supervisor.  You can also use the `SessionManager` directly to create a
  session and check if it could be initialized automatically.
  "
  [graph & {:keys [ready_op ready_for_local_init_op is_chief init_op init_feed_dict local_init_op logdir summary_op saver global_step save_summaries_secs save_model_secs recovery_wait_secs stop_grace_secs checkpoint_basename session_manager summary_writer init_fn local_init_run_options]
                       :or {init_feed_dict None logdir None session_manager None init_fn None local_init_run_options None}} ]
    (py/call-attr-kw train "Supervisor" [graph] {:ready_op ready_op :ready_for_local_init_op ready_for_local_init_op :is_chief is_chief :init_op init_op :init_feed_dict init_feed_dict :local_init_op local_init_op :logdir logdir :summary_op summary_op :saver saver :global_step global_step :save_summaries_secs save_summaries_secs :save_model_secs save_model_secs :recovery_wait_secs recovery_wait_secs :stop_grace_secs stop_grace_secs :checkpoint_basename checkpoint_basename :session_manager session_manager :summary_writer summary_writer :init_fn init_fn :local_init_run_options local_init_run_options }))

(defn Loop 
  "Start a LooperThread that calls a function periodically.

    If `timer_interval_secs` is None the thread calls `target(*args, **kwargs)`
    repeatedly.  Otherwise it calls it every `timer_interval_secs`
    seconds.  The thread terminates when a stop is requested.

    The started thread is added to the list of threads managed by the supervisor
    so it does not need to be passed to the `stop()` method.

    Args:
      timer_interval_secs: Number. Time boundaries at which to call `target`.
      target: A callable object.
      args: Optional arguments to pass to `target` when calling it.
      kwargs: Optional keyword arguments to pass to `target` when calling it.

    Returns:
      The started thread.
    "
  [ self timer_interval_secs target args kwargs ]
  (py/call-attr self "Loop"  self timer_interval_secs target args kwargs ))

(defn PrepareSession 
  "Make sure the model is ready to be used.

    Create a session on 'master', recovering or initializing the model as
    needed, or wait for a session to be ready.  If running as the chief
    and `start_standard_service` is set to True, also call the session
    manager to start the standard services.

    Args:
      master: name of the TensorFlow master to use.  See the
        `tf.compat.v1.Session` constructor for how this is interpreted.
      config: Optional ConfigProto proto used to configure the session, which is
        passed as-is to create the session.
      wait_for_checkpoint: Whether we should wait for the availability of a
        checkpoint before creating Session. Defaults to False.
      max_wait_secs: Maximum time to wait for the session to become available.
      start_standard_services: Whether to start the standard services and the
        queue runners.

    Returns:
      A Session object that can be used to drive the model.
    "
  [self  & {:keys [master config wait_for_checkpoint max_wait_secs start_standard_services]
                       :or {config None}} ]
    (py/call-attr-kw self "PrepareSession" [] {:master master :config config :wait_for_checkpoint wait_for_checkpoint :max_wait_secs max_wait_secs :start_standard_services start_standard_services }))

(defn RequestStop 
  "Request that the coordinator stop the threads.

    See `Coordinator.request_stop()`.

    Args:
      ex: Optional `Exception`, or Python `exc_info` tuple as returned by
        `sys.exc_info()`.  If this is the first call to `request_stop()` the
        corresponding exception is recorded and re-raised from `join()`.
    "
  [ self ex ]
  (py/call-attr self "RequestStop"  self ex ))

(defn ShouldStop 
  "Check if the coordinator was told to stop.

    See `Coordinator.should_stop()`.

    Returns:
      True if the coordinator was told to stop, False otherwise.
    "
  [ self  ]
  (py/call-attr self "ShouldStop"  self  ))

(defn StartQueueRunners 
  "Start threads for `QueueRunners`.

    Note that the queue runners collected in the graph key `QUEUE_RUNNERS`
    are already started automatically when you create a session with the
    supervisor, so unless you have non-collected queue runners to start
    you do not need to call this explicitly.

    Args:
      sess: A `Session`.
      queue_runners: A list of `QueueRunners`. If not specified, we'll use the
        list of queue runners gathered in the graph under the key
        `GraphKeys.QUEUE_RUNNERS`.

    Returns:
      The list of threads started for the `QueueRunners`.

    Raises:
      RuntimeError: If called with eager execution enabled.

    @compatibility(eager)
    Queues are not compatible with eager execution. To ingest data when eager
    execution is enabled, use the `tf.data` API.
    @end_compatibility
    "
  [ self sess queue_runners ]
  (py/call-attr self "StartQueueRunners"  self sess queue_runners ))

(defn StartStandardServices 
  "Start the standard services for 'sess'.

    This starts services in the background.  The services started depend
    on the parameters to the constructor and may include:

      - A Summary thread computing summaries every save_summaries_secs.
      - A Checkpoint thread saving the model every save_model_secs.
      - A StepCounter thread measure step time.

    Args:
      sess: A Session.

    Returns:
      A list of threads that are running the standard services.  You can use
      the Supervisor's Coordinator to join these threads with:
        sv.coord.Join(<list of threads>)

    Raises:
      RuntimeError: If called with a non-chief Supervisor.
      ValueError: If not `logdir` was passed to the constructor as the
        services need a log directory.
    "
  [ self sess ]
  (py/call-attr self "StartStandardServices"  self sess ))
(defn Stop 
  "Stop the services and the coordinator.

    This does not close the session.

    Args:
      threads: Optional list of threads to join with the coordinator.  If
        `None`, defaults to the threads running the standard services, the
        threads started for `QueueRunners`, and the threads started by the
        `loop()` method.  To wait on additional threads, pass the list in this
        parameter.
      close_summary_writer: Whether to close the `summary_writer`.  Defaults to
        `True` if the summary writer was created by the supervisor, `False`
        otherwise.
      ignore_live_threads: If `True` ignores threads that remain running after a
        grace period when joining threads via the coordinator, instead of
        raising a RuntimeError.
    "
  [self threads  & {:keys [close_summary_writer ignore_live_threads]} ]
    (py/call-attr-kw self "Stop" [threads] {:close_summary_writer close_summary_writer :ignore_live_threads ignore_live_threads }))

(defn StopOnException 
  "Context handler to stop the supervisor when an exception is raised.

    See `Coordinator.stop_on_exception()`.

    Returns:
      A context handler.
    "
  [ self  ]
  (py/call-attr self "StopOnException"  self  ))

(defn SummaryComputed 
  "Indicate that a summary was computed.

    Args:
      sess: A `Session` object.
      summary: A Summary proto, or a string holding a serialized summary proto.
      global_step: Int. global step this summary is associated with. If `None`,
        it will try to fetch the current step.

    Raises:
      TypeError: if 'summary' is not a Summary proto or a string.
      RuntimeError: if the Supervisor was created without a `logdir`.
    "
  [ self sess summary global_step ]
  (py/call-attr self "SummaryComputed"  self sess summary global_step ))

(defn WaitForStop 
  "Block waiting for the coordinator to stop."
  [ self  ]
  (py/call-attr self "WaitForStop"  self  ))

(defn coord 
  "Return the Coordinator used by the Supervisor.

    The Coordinator can be useful if you want to run multiple threads
    during your training.

    Returns:
      A Coordinator object.
    "
  [ self ]
    (py/call-attr self "coord"))

(defn global-step 
  "Return the global_step Tensor used by the supervisor.

    Returns:
      An integer Tensor for the global_step.
    "
  [ self ]
    (py/call-attr self "global_step"))

(defn init-feed-dict 
  "Return the feed dictionary used when evaluating the `init_op`.

    Returns:
      A feed dictionary or `None`.
    "
  [ self ]
    (py/call-attr self "init_feed_dict"))

(defn init-op 
  "Return the Init Op used by the supervisor.

    Returns:
      An Op or `None`.
    "
  [ self ]
    (py/call-attr self "init_op"))

(defn is-chief 
  "Return True if this is a chief supervisor.

    Returns:
      A bool.
    "
  [ self ]
    (py/call-attr self "is_chief"))

(defn loop 
  "Start a LooperThread that calls a function periodically.

    If `timer_interval_secs` is None the thread calls `target(*args, **kwargs)`
    repeatedly.  Otherwise it calls it every `timer_interval_secs`
    seconds.  The thread terminates when a stop is requested.

    The started thread is added to the list of threads managed by the supervisor
    so it does not need to be passed to the `stop()` method.

    Args:
      timer_interval_secs: Number. Time boundaries at which to call `target`.
      target: A callable object.
      args: Optional arguments to pass to `target` when calling it.
      kwargs: Optional keyword arguments to pass to `target` when calling it.

    Returns:
      The started thread.
    "
  [ self timer_interval_secs target args kwargs ]
  (py/call-attr self "loop"  self timer_interval_secs target args kwargs ))

(defn managed-session 
  "Returns a context manager for a managed session.

    This context manager creates and automatically recovers a session.  It
    optionally starts the standard services that handle checkpoints and
    summaries.  It monitors exceptions raised from the `with` block or from the
    services and stops the supervisor as needed.

    The context manager is typically used as follows:

    ```python
    def train():
      sv = tf.compat.v1.train.Supervisor(...)
      with sv.managed_session(<master>) as sess:
        for step in xrange(..):
          if sv.should_stop():
            break
          sess.run(<my training op>)
          ...do other things needed at each training step...
    ```

    An exception raised from the `with` block or one of the service threads is
    raised again when the block exits.  This is done after stopping all threads
    and closing the session.  For example, an `AbortedError` exception, raised
    in case of preemption of one of the workers in a distributed model, is
    raised again when the block exits.

    If you want to retry the training loop in case of preemption you can do it
    as follows:

    ```python
    def main(...):
      while True
        try:
          train()
        except tf.errors.Aborted:
          pass
    ```

    As a special case, exceptions used for control flow, such as
    `OutOfRangeError` which reports that input queues are exhausted, are not
    raised again from the `with` block: they indicate a clean termination of
    the training loop and are considered normal termination.

    Args:
      master: name of the TensorFlow master to use.  See the
        `tf.compat.v1.Session` constructor for how this is interpreted.
      config: Optional `ConfigProto` proto used to configure the session. Passed
        as-is to create the session.
      start_standard_services: Whether to start the standard services, such as
        checkpoint, summary and step counter.
      close_summary_writer: Whether to close the summary writer when closing the
        session.  Defaults to True.

    Returns:
      A context manager that yields a `Session` restored from the latest
      checkpoint or initialized from scratch if not checkpoint exists.  The
      session is closed when the `with` block exits.
    "
  [self  & {:keys [master config start_standard_services close_summary_writer]
                       :or {config None}} ]
    (py/call-attr-kw self "managed_session" [] {:master master :config config :start_standard_services start_standard_services :close_summary_writer close_summary_writer }))

(defn prepare-or-wait-for-session 
  "Make sure the model is ready to be used.

    Create a session on 'master', recovering or initializing the model as
    needed, or wait for a session to be ready.  If running as the chief
    and `start_standard_service` is set to True, also call the session
    manager to start the standard services.

    Args:
      master: name of the TensorFlow master to use.  See the
        `tf.compat.v1.Session` constructor for how this is interpreted.
      config: Optional ConfigProto proto used to configure the session, which is
        passed as-is to create the session.
      wait_for_checkpoint: Whether we should wait for the availability of a
        checkpoint before creating Session. Defaults to False.
      max_wait_secs: Maximum time to wait for the session to become available.
      start_standard_services: Whether to start the standard services and the
        queue runners.

    Returns:
      A Session object that can be used to drive the model.
    "
  [self  & {:keys [master config wait_for_checkpoint max_wait_secs start_standard_services]
                       :or {config None}} ]
    (py/call-attr-kw self "prepare_or_wait_for_session" [] {:master master :config config :wait_for_checkpoint wait_for_checkpoint :max_wait_secs max_wait_secs :start_standard_services start_standard_services }))

(defn ready-for-local-init-op 
  ""
  [ self ]
    (py/call-attr self "ready_for_local_init_op"))

(defn ready-op 
  "Return the Ready Op used by the supervisor.

    Returns:
      An Op or `None`.
    "
  [ self ]
    (py/call-attr self "ready_op"))

(defn request-stop 
  "Request that the coordinator stop the threads.

    See `Coordinator.request_stop()`.

    Args:
      ex: Optional `Exception`, or Python `exc_info` tuple as returned by
        `sys.exc_info()`.  If this is the first call to `request_stop()` the
        corresponding exception is recorded and re-raised from `join()`.
    "
  [ self ex ]
  (py/call-attr self "request_stop"  self ex ))

(defn save-model-secs 
  "Return the delay between checkpoints.

    Returns:
      A timestamp.
    "
  [ self ]
    (py/call-attr self "save_model_secs"))

(defn save-path 
  "Return the save path used by the supervisor.

    Returns:
      A string.
    "
  [ self ]
    (py/call-attr self "save_path"))

(defn save-summaries-secs 
  "Return the delay between summary computations.

    Returns:
      A timestamp.
    "
  [ self ]
    (py/call-attr self "save_summaries_secs"))

(defn saver 
  "Return the Saver used by the supervisor.

    Returns:
      A Saver object.
    "
  [ self ]
    (py/call-attr self "saver"))

(defn session-manager 
  "Return the SessionManager used by the Supervisor.

    Returns:
      A SessionManager object.
    "
  [ self ]
    (py/call-attr self "session_manager"))

(defn should-stop 
  "Check if the coordinator was told to stop.

    See `Coordinator.should_stop()`.

    Returns:
      True if the coordinator was told to stop, False otherwise.
    "
  [ self  ]
  (py/call-attr self "should_stop"  self  ))

(defn start-queue-runners 
  "Start threads for `QueueRunners`.

    Note that the queue runners collected in the graph key `QUEUE_RUNNERS`
    are already started automatically when you create a session with the
    supervisor, so unless you have non-collected queue runners to start
    you do not need to call this explicitly.

    Args:
      sess: A `Session`.
      queue_runners: A list of `QueueRunners`. If not specified, we'll use the
        list of queue runners gathered in the graph under the key
        `GraphKeys.QUEUE_RUNNERS`.

    Returns:
      The list of threads started for the `QueueRunners`.

    Raises:
      RuntimeError: If called with eager execution enabled.

    @compatibility(eager)
    Queues are not compatible with eager execution. To ingest data when eager
    execution is enabled, use the `tf.data` API.
    @end_compatibility
    "
  [ self sess queue_runners ]
  (py/call-attr self "start_queue_runners"  self sess queue_runners ))

(defn start-standard-services 
  "Start the standard services for 'sess'.

    This starts services in the background.  The services started depend
    on the parameters to the constructor and may include:

      - A Summary thread computing summaries every save_summaries_secs.
      - A Checkpoint thread saving the model every save_model_secs.
      - A StepCounter thread measure step time.

    Args:
      sess: A Session.

    Returns:
      A list of threads that are running the standard services.  You can use
      the Supervisor's Coordinator to join these threads with:
        sv.coord.Join(<list of threads>)

    Raises:
      RuntimeError: If called with a non-chief Supervisor.
      ValueError: If not `logdir` was passed to the constructor as the
        services need a log directory.
    "
  [ self sess ]
  (py/call-attr self "start_standard_services"  self sess ))
(defn stop 
  "Stop the services and the coordinator.

    This does not close the session.

    Args:
      threads: Optional list of threads to join with the coordinator.  If
        `None`, defaults to the threads running the standard services, the
        threads started for `QueueRunners`, and the threads started by the
        `loop()` method.  To wait on additional threads, pass the list in this
        parameter.
      close_summary_writer: Whether to close the `summary_writer`.  Defaults to
        `True` if the summary writer was created by the supervisor, `False`
        otherwise.
      ignore_live_threads: If `True` ignores threads that remain running after a
        grace period when joining threads via the coordinator, instead of
        raising a RuntimeError.
    "
  [self threads  & {:keys [close_summary_writer ignore_live_threads]} ]
    (py/call-attr-kw self "stop" [threads] {:close_summary_writer close_summary_writer :ignore_live_threads ignore_live_threads }))

(defn stop-on-exception 
  "Context handler to stop the supervisor when an exception is raised.

    See `Coordinator.stop_on_exception()`.

    Returns:
      A context handler.
    "
  [ self  ]
  (py/call-attr self "stop_on_exception"  self  ))

(defn summary-computed 
  "Indicate that a summary was computed.

    Args:
      sess: A `Session` object.
      summary: A Summary proto, or a string holding a serialized summary proto.
      global_step: Int. global step this summary is associated with. If `None`,
        it will try to fetch the current step.

    Raises:
      TypeError: if 'summary' is not a Summary proto or a string.
      RuntimeError: if the Supervisor was created without a `logdir`.
    "
  [ self sess summary global_step ]
  (py/call-attr self "summary_computed"  self sess summary global_step ))

(defn summary-op 
  "Return the Summary Tensor used by the chief supervisor.

    Returns:
      A string Tensor for the summary or `None`.
    "
  [ self ]
    (py/call-attr self "summary_op"))

(defn summary-writer 
  "Return the SummaryWriter used by the chief supervisor.

    Returns:
      A SummaryWriter.
    "
  [ self ]
    (py/call-attr self "summary_writer"))

(defn wait-for-stop 
  "Block waiting for the coordinator to stop."
  [ self  ]
  (py/call-attr self "wait_for_stop"  self  ))
