(ns tensorflow.contrib.eager.python.tfe.Saver
  "A tf.compat.v1.train.Saver adapter for use when eager execution is enabled.

  `Saver`'s name-based checkpointing strategy is fragile. Please switch to
  `tf.train.Checkpoint` or `tf.keras.Model.save_weights`, which perform a more
  robust object-based saving. These APIs will load checkpoints written by
  `Saver`.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tfe (import-module "tensorflow.contrib.eager.python.tfe"))

(defn Saver 
  "A tf.compat.v1.train.Saver adapter for use when eager execution is enabled.

  `Saver`'s name-based checkpointing strategy is fragile. Please switch to
  `tf.train.Checkpoint` or `tf.keras.Model.save_weights`, which perform a more
  robust object-based saving. These APIs will load checkpoints written by
  `Saver`.
  "
  [ var_list ]
  (py/call-attr eager "Saver"  var_list ))

(defn restore 
  "Restores previously saved variables.

    Args:
      file_prefix: Path prefix where parameters were previously saved.
        Typically obtained from a previous `save()` call, or from
        `tf.train.latest_checkpoint`.
    "
  [ self file_prefix ]
  (py/call-attr self "restore"  self file_prefix ))

(defn save 
  "Saves variables.

    Args:
      file_prefix: Path prefix of files created for the checkpoint.
      global_step: If provided the global step number is appended to file_prefix
        to create the checkpoint filename. The optional argument can be a
        Tensor, a Variable, or an integer.

    Returns:
      A string: prefix of filenames created for the checkpoint. This may be
       an extension of file_prefix that is suitable to pass as an argument
       to a subsequent call to `restore()`.
    "
  [ self file_prefix global_step ]
  (py/call-attr self "save"  self file_prefix global_step ))
