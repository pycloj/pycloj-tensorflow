(ns tensorflow.train.SessionManager
  "Training helper that restores from checkpoint and creates session.

  This class is a small wrapper that takes care of session creation and
  checkpoint recovery. It also provides functions that to facilitate
  coordination among multiple training threads or processes.

  * Checkpointing trained variables as the training progresses.
  * Initializing variables on startup, restoring them from the most recent
    checkpoint after a crash, or wait for checkpoints to become available.

  ### Usage:

  ```python
  with tf.Graph().as_default():
     ...add operations to the graph...
    # Create a SessionManager that will checkpoint the model in '/tmp/mydir'.
    sm = SessionManager()
    sess = sm.prepare_session(master, init_op, saver, checkpoint_dir)
    # Use the session to train the graph.
    while True:
      sess.run(<my_train_op>)
  ```

  `prepare_session()` initializes or restores a model. It requires `init_op`
  and `saver` as an argument.

  A second process could wait for the model to be ready by doing the following:

  ```python
  with tf.Graph().as_default():
     ...add operations to the graph...
    # Create a SessionManager that will wait for the model to become ready.
    sm = SessionManager()
    sess = sm.wait_for_session(master)
    # Use the session to train the graph.
    while True:
      sess.run(<my_train_op>)
  ```

  `wait_for_session()` waits for a model to be initialized by other processes.

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

(defn SessionManager 
  "Training helper that restores from checkpoint and creates session.

  This class is a small wrapper that takes care of session creation and
  checkpoint recovery. It also provides functions that to facilitate
  coordination among multiple training threads or processes.

  * Checkpointing trained variables as the training progresses.
  * Initializing variables on startup, restoring them from the most recent
    checkpoint after a crash, or wait for checkpoints to become available.

  ### Usage:

  ```python
  with tf.Graph().as_default():
     ...add operations to the graph...
    # Create a SessionManager that will checkpoint the model in '/tmp/mydir'.
    sm = SessionManager()
    sess = sm.prepare_session(master, init_op, saver, checkpoint_dir)
    # Use the session to train the graph.
    while True:
      sess.run(<my_train_op>)
  ```

  `prepare_session()` initializes or restores a model. It requires `init_op`
  and `saver` as an argument.

  A second process could wait for the model to be ready by doing the following:

  ```python
  with tf.Graph().as_default():
     ...add operations to the graph...
    # Create a SessionManager that will wait for the model to become ready.
    sm = SessionManager()
    sess = sm.wait_for_session(master)
    # Use the session to train the graph.
    while True:
      sess.run(<my_train_op>)
  ```

  `wait_for_session()` waits for a model to be initialized by other processes.

  "
  [local_init_op ready_op ready_for_local_init_op graph & {:keys [recovery_wait_secs local_init_run_options]
                       :or {local_init_run_options None}} ]
    (py/call-attr-kw train "SessionManager" [local_init_op ready_op ready_for_local_init_op graph] {:recovery_wait_secs recovery_wait_secs :local_init_run_options local_init_run_options }))

(defn prepare-session 
  "Creates a `Session`. Makes sure the model is ready to be used.

    Creates a `Session` on 'master'. If a `saver` object is passed in, and
    `checkpoint_dir` points to a directory containing valid checkpoint
    files, then it will try to recover the model from checkpoint. If
    no checkpoint files are available, and `wait_for_checkpoint` is
    `True`, then the process would check every `recovery_wait_secs`,
    up to `max_wait_secs`, for recovery to succeed.

    If the model cannot be recovered successfully then it is initialized by
    running the `init_op` and calling `init_fn` if they are provided.
    The `local_init_op` is also run after init_op and init_fn, regardless of
    whether the model was recovered successfully, but only if
    `ready_for_local_init_op` passes.

    If the model is recovered from a checkpoint it is assumed that all
    global variables have been initialized, in particular neither `init_op`
    nor `init_fn` will be executed.

    It is an error if the model cannot be recovered and no `init_op`
    or `init_fn` or `local_init_op` are passed.

    Args:
      master: `String` representation of the TensorFlow master to use.
      init_op: Optional `Operation` used to initialize the model.
      saver: A `Saver` object used to restore a model.
      checkpoint_dir: Path to the checkpoint files. The latest checkpoint in the
        dir will be used to restore.
      checkpoint_filename_with_path: Full file name path to the checkpoint file.
      wait_for_checkpoint: Whether to wait for checkpoint to become available.
      max_wait_secs: Maximum time to wait for checkpoints to become available.
      config: Optional `ConfigProto` proto used to configure the session.
      init_feed_dict: Optional dictionary that maps `Tensor` objects to feed
        values.  This feed dictionary is passed to the session `run()` call when
        running the init op.
      init_fn: Optional callable used to initialize the model. Called after the
        optional `init_op` is called.  The callable must accept one argument,
        the session being initialized.

    Returns:
      A `Session` object that can be used to drive the model.

    Raises:
      RuntimeError: If the model cannot be initialized or recovered.
      ValueError: If both checkpoint_dir and checkpoint_filename_with_path are
        set.
    "
  [self master init_op saver checkpoint_dir checkpoint_filename_with_path & {:keys [wait_for_checkpoint max_wait_secs config init_feed_dict init_fn]
                       :or {config None init_feed_dict None init_fn None}} ]
    (py/call-attr-kw self "prepare_session" [master init_op saver checkpoint_dir checkpoint_filename_with_path] {:wait_for_checkpoint wait_for_checkpoint :max_wait_secs max_wait_secs :config config :init_feed_dict init_feed_dict :init_fn init_fn }))

(defn recover-session 
  "Creates a `Session`, recovering if possible.

    Creates a new session on 'master'.  If the session is not initialized
    and can be recovered from a checkpoint, recover it.

    Args:
      master: `String` representation of the TensorFlow master to use.
      saver: A `Saver` object used to restore a model.
      checkpoint_dir: Path to the checkpoint files. The latest checkpoint in the
        dir will be used to restore.
      checkpoint_filename_with_path: Full file name path to the checkpoint file.
      wait_for_checkpoint: Whether to wait for checkpoint to become available.
      max_wait_secs: Maximum time to wait for checkpoints to become available.
      config: Optional `ConfigProto` proto used to configure the session.

    Returns:
      A pair (sess, initialized) where 'initialized' is `True` if
      the session could be recovered and initialized, `False` otherwise.

    Raises:
      ValueError: If both checkpoint_dir and checkpoint_filename_with_path are
        set.
    "
  [self master saver checkpoint_dir checkpoint_filename_with_path & {:keys [wait_for_checkpoint max_wait_secs config]
                       :or {config None}} ]
    (py/call-attr-kw self "recover_session" [master saver checkpoint_dir checkpoint_filename_with_path] {:wait_for_checkpoint wait_for_checkpoint :max_wait_secs max_wait_secs :config config }))
(defn wait-for-session 
  "Creates a new `Session` and waits for model to be ready.

    Creates a new `Session` on 'master'.  Waits for the model to be
    initialized or recovered from a checkpoint.  It's expected that
    another thread or process will make the model ready, and that this
    is intended to be used by threads/processes that participate in a
    distributed training configuration where a different thread/process
    is responsible for initializing or recovering the model being trained.

    NB: The amount of time this method waits for the session is bounded
    by max_wait_secs. By default, this function will wait indefinitely.

    Args:
      master: `String` representation of the TensorFlow master to use.
      config: Optional ConfigProto proto used to configure the session.
      max_wait_secs: Maximum time to wait for the session to become available.

    Returns:
      A `Session`. May be None if the operation exceeds the timeout
      specified by config.operation_timeout_in_ms.

    Raises:
      tf.DeadlineExceededError: if the session is not available after
        max_wait_secs.
    "
  [self master config  & {:keys [max_wait_secs]} ]
    (py/call-attr-kw self "wait_for_session" [master config] {:max_wait_secs max_wait_secs }))
