(ns tensorflow.-api.v1.compat.v2.train
  "Support for training models.

See the [Training](https://tensorflow.org/api_guides/python/train) guide.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce train (import-module "tensorflow._api.v1.compat.v2.train"))

(defn checkpoints-iterator 
  "Continuously yield new checkpoint files as they appear.

  The iterator only checks for new checkpoints when control flow has been
  reverted to it. This means it can miss checkpoints if your code takes longer
  to run between iterations than `min_interval_secs` or the interval at which
  new checkpoints are written.

  The `timeout` argument is the maximum number of seconds to block waiting for
  a new checkpoint.  It is used in combination with the `timeout_fn` as
  follows:

  * If the timeout expires and no `timeout_fn` was specified, the iterator
    stops yielding.
  * If a `timeout_fn` was specified, that function is called and if it returns
    a true boolean value the iterator stops yielding.
  * If the function returns a false boolean value then the iterator resumes the
    wait for new checkpoints.  At this point the timeout logic applies again.

  This behavior gives control to callers on what to do if checkpoints do not
  come fast enough or stop being generated.  For example, if callers have a way
  to detect that the training has stopped and know that no new checkpoints
  will be generated, they can provide a `timeout_fn` that returns `True` when
  the training has stopped.  If they know that the training is still going on
  they return `False` instead.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    min_interval_secs: The minimum number of seconds between yielding
      checkpoints.
    timeout: The maximum number of seconds to wait between checkpoints. If left
      as `None`, then the process will wait indefinitely.
    timeout_fn: Optional function to call after a timeout.  If the function
      returns True, then it means that no new checkpoints will be generated and
      the iterator will exit.  The function is called with no arguments.

  Yields:
    String paths to latest checkpoint files as they arrive.
  "
  [checkpoint_dir & {:keys [min_interval_secs timeout timeout_fn]
                       :or {timeout None timeout_fn None}} ]
    (py/call-attr-kw train "checkpoints_iterator" [checkpoint_dir] {:min_interval_secs min_interval_secs :timeout timeout :timeout_fn timeout_fn }))

(defn get-checkpoint-state 
  "Returns CheckpointState proto from the \"checkpoint\" file.

  If the \"checkpoint\" file contains a valid CheckpointState
  proto, returns it.

  Args:
    checkpoint_dir: The directory of checkpoints.
    latest_filename: Optional name of the checkpoint file.  Default to
      'checkpoint'.

  Returns:
    A CheckpointState if the state was available, None
    otherwise.

  Raises:
    ValueError: if the checkpoint read doesn't have model_checkpoint_path set.
  "
  [ checkpoint_dir latest_filename ]
  (py/call-attr train "get_checkpoint_state"  checkpoint_dir latest_filename ))

(defn latest-checkpoint 
  "Finds the filename of latest saved checkpoint file.

  Args:
    checkpoint_dir: Directory where the variables were saved.
    latest_filename: Optional name for the protocol buffer file that
      contains the list of most recent checkpoint filenames.
      See the corresponding argument to `Saver.save()`.

  Returns:
    The full path to the latest checkpoint or `None` if no checkpoint was found.
  "
  [ checkpoint_dir latest_filename ]
  (py/call-attr train "latest_checkpoint"  checkpoint_dir latest_filename ))

(defn list-variables 
  "Returns list of all variables in the checkpoint.

  Args:
    ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint.

  Returns:
    List of tuples `(name, shape)`.
  "
  [ ckpt_dir_or_file ]
  (py/call-attr train "list_variables"  ckpt_dir_or_file ))

(defn load-checkpoint 
  "Returns `CheckpointReader` for checkpoint found in `ckpt_dir_or_file`.

  If `ckpt_dir_or_file` resolves to a directory with multiple checkpoints,
  reader for the latest checkpoint is returned.

  Args:
    ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint
      file.

  Returns:
    `CheckpointReader` object.

  Raises:
    ValueError: If `ckpt_dir_or_file` resolves to a directory with no
      checkpoints.
  "
  [ ckpt_dir_or_file ]
  (py/call-attr train "load_checkpoint"  ckpt_dir_or_file ))

(defn load-variable 
  "Returns the tensor value of the given variable in the checkpoint.

  Args:
    ckpt_dir_or_file: Directory with checkpoints file or path to checkpoint.
    name: Name of the variable to return.

  Returns:
    A numpy `ndarray` with a copy of the value of this variable.
  "
  [ ckpt_dir_or_file name ]
  (py/call-attr train "load_variable"  ckpt_dir_or_file name ))
