(ns tensorflow.-api.v1.compat.v2.train.CheckpointManager
  "Deletes old checkpoints.

  Example usage:

  ```python
  import tensorflow as tf
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  manager = tf.contrib.checkpoint.CheckpointManager(
      checkpoint, directory=\"/tmp/model\", max_to_keep=5)
  status = checkpoint.restore(manager.latest_checkpoint)
  while True:
    # train
    manager.save()
  ```

  `CheckpointManager` preserves its own state across instantiations (see the
  `__init__` documentation for details). Only one should be active in a
  particular directory at a time.
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
(defn CheckpointManager 
  "Deletes old checkpoints.

  Example usage:

  ```python
  import tensorflow as tf
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  manager = tf.contrib.checkpoint.CheckpointManager(
      checkpoint, directory=\"/tmp/model\", max_to_keep=5)
  status = checkpoint.restore(manager.latest_checkpoint)
  while True:
    # train
    manager.save()
  ```

  `CheckpointManager` preserves its own state across instantiations (see the
  `__init__` documentation for details). Only one should be active in a
  particular directory at a time.
  "
  [checkpoint directory max_to_keep keep_checkpoint_every_n_hours  & {:keys [checkpoint_name]} ]
    (py/call-attr-kw train "CheckpointManager" [checkpoint directory max_to_keep keep_checkpoint_every_n_hours] {:checkpoint_name checkpoint_name }))

(defn checkpoints 
  "A list of managed checkpoints.

    Note that checkpoints saved due to `keep_checkpoint_every_n_hours` will not
    show up in this list (to avoid ever-growing filename lists).

    Returns:
      A list of filenames, sorted from oldest to newest.
    "
  [ self ]
    (py/call-attr self "checkpoints"))

(defn latest-checkpoint 
  "The prefix of the most recent checkpoint in `directory`.

    Equivalent to `tf.train.latest_checkpoint(directory)` where `directory` is
    the constructor argument to `CheckpointManager`.

    Suitable for passing to `tf.train.Checkpoint.restore` to resume training.

    Returns:
      The checkpoint prefix. If there are no checkpoints, returns `None`.
    "
  [ self ]
    (py/call-attr self "latest_checkpoint"))

(defn save 
  "Creates a new checkpoint and manages it.

    Args:
      checkpoint_number: An optional integer, or an integer-dtype `Variable` or
        `Tensor`, used to number the checkpoint. If `None` (default),
        checkpoints are numbered using `checkpoint.save_counter`. Even if
        `checkpoint_number` is provided, `save_counter` is still incremented. A
        user-provided `checkpoint_number` is not incremented even if it is a
        `Variable`.

    Returns:
      The path to the new checkpoint. It is also recorded in the `checkpoints`
      and `latest_checkpoint` properties.
    "
  [ self checkpoint_number ]
  (py/call-attr self "save"  self checkpoint_number ))
