(ns tensorflow.-api.v1.compat.v2.train.Checkpoint
  "Groups trackable objects, saving and restoring them.

  `Checkpoint`'s constructor accepts keyword arguments whose values are types
  that contain trackable state, such as `tf.keras.optimizers.Optimizer`
  implementations, `tf.Variable`, `tf.keras.Layer` implementations, or
  `tf.keras.Model` implementations. It saves these values with a checkpoint, and
  maintains a `save_counter` for numbering checkpoints.

  Example usage:

  ```python
  import tensorflow as tf
  import os

  checkpoint_directory = \"/tmp/training_checkpoints\"
  checkpoint_prefix = os.path.join(checkpoint_directory, \"ckpt\")

  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
  for _ in range(num_training_steps):
    optimizer.minimize( ... )  # Variables will be restored on creation.
  status.assert_consumed()  # Optional sanity checks.
  checkpoint.save(file_prefix=checkpoint_prefix)
  ```

  `Checkpoint.save` and `Checkpoint.restore` write and read object-based
  checkpoints, in contrast to TensorFlow 1.x's `tf.compat.v1.train.Saver` which
  writes and
  reads `variable.name` based checkpoints. Object-based checkpointing saves a
  graph of dependencies between Python objects (`Layer`s, `Optimizer`s,
  `Variable`s, etc.) with named edges, and this graph is used to match variables
  when restoring a checkpoint. It can be more robust to changes in the Python
  program, and helps to support restore-on-create for variables.

  `Checkpoint` objects have dependencies on the objects passed as keyword
  arguments to their constructors, and each dependency is given a name that is
  identical to the name of the keyword argument for which it was created.
  TensorFlow classes like `Layer`s and `Optimizer`s will automatically add
  dependencies on their variables (e.g. \"kernel\" and \"bias\" for
  `tf.keras.layers.Dense`). Inheriting from `tf.keras.Model` makes managing
  dependencies easy in user-defined classes, since `Model` hooks into attribute
  assignment. For example:

  ```python
  class Regress(tf.keras.Model):

    def __init__(self):
      super(Regress, self).__init__()
      self.input_transform = tf.keras.layers.Dense(10)
      # ...

    def call(self, inputs):
      x = self.input_transform(inputs)
      # ...
  ```

  This `Model` has a dependency named \"input_transform\" on its `Dense` layer,
  which in turn depends on its variables. As a result, saving an instance of
  `Regress` using `tf.train.Checkpoint` will also save all the variables created
  by the `Dense` layer.

  When variables are assigned to multiple workers, each worker writes its own
  section of the checkpoint. These sections are then merged/re-indexed to behave
  as a single checkpoint. This avoids copying all variables to one worker, but
  does require that all workers see a common filesystem.

  While `tf.keras.Model.save_weights` and `tf.train.Checkpoint.save` save in the
  same format, note that the root of the resulting checkpoint is the object the
  save method is attached to. This means saving a `tf.keras.Model` using
  `save_weights` and loading into a `tf.train.Checkpoint` with a `Model`
  attached (or vice versa) will not match the `Model`'s variables. See the
  [guide to training
  checkpoints](https://www.tensorflow.org/alpha/guide/checkpoints) for
  details. Prefer `tf.train.Checkpoint` over `tf.keras.Model.save_weights` for
  training checkpoints.

  Attributes:
    save_counter: Incremented when `save()` is called. Used to number
      checkpoints.
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

(defn Checkpoint 
  "Groups trackable objects, saving and restoring them.

  `Checkpoint`'s constructor accepts keyword arguments whose values are types
  that contain trackable state, such as `tf.keras.optimizers.Optimizer`
  implementations, `tf.Variable`, `tf.keras.Layer` implementations, or
  `tf.keras.Model` implementations. It saves these values with a checkpoint, and
  maintains a `save_counter` for numbering checkpoints.

  Example usage:

  ```python
  import tensorflow as tf
  import os

  checkpoint_directory = \"/tmp/training_checkpoints\"
  checkpoint_prefix = os.path.join(checkpoint_directory, \"ckpt\")

  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
  for _ in range(num_training_steps):
    optimizer.minimize( ... )  # Variables will be restored on creation.
  status.assert_consumed()  # Optional sanity checks.
  checkpoint.save(file_prefix=checkpoint_prefix)
  ```

  `Checkpoint.save` and `Checkpoint.restore` write and read object-based
  checkpoints, in contrast to TensorFlow 1.x's `tf.compat.v1.train.Saver` which
  writes and
  reads `variable.name` based checkpoints. Object-based checkpointing saves a
  graph of dependencies between Python objects (`Layer`s, `Optimizer`s,
  `Variable`s, etc.) with named edges, and this graph is used to match variables
  when restoring a checkpoint. It can be more robust to changes in the Python
  program, and helps to support restore-on-create for variables.

  `Checkpoint` objects have dependencies on the objects passed as keyword
  arguments to their constructors, and each dependency is given a name that is
  identical to the name of the keyword argument for which it was created.
  TensorFlow classes like `Layer`s and `Optimizer`s will automatically add
  dependencies on their variables (e.g. \"kernel\" and \"bias\" for
  `tf.keras.layers.Dense`). Inheriting from `tf.keras.Model` makes managing
  dependencies easy in user-defined classes, since `Model` hooks into attribute
  assignment. For example:

  ```python
  class Regress(tf.keras.Model):

    def __init__(self):
      super(Regress, self).__init__()
      self.input_transform = tf.keras.layers.Dense(10)
      # ...

    def call(self, inputs):
      x = self.input_transform(inputs)
      # ...
  ```

  This `Model` has a dependency named \"input_transform\" on its `Dense` layer,
  which in turn depends on its variables. As a result, saving an instance of
  `Regress` using `tf.train.Checkpoint` will also save all the variables created
  by the `Dense` layer.

  When variables are assigned to multiple workers, each worker writes its own
  section of the checkpoint. These sections are then merged/re-indexed to behave
  as a single checkpoint. This avoids copying all variables to one worker, but
  does require that all workers see a common filesystem.

  While `tf.keras.Model.save_weights` and `tf.train.Checkpoint.save` save in the
  same format, note that the root of the resulting checkpoint is the object the
  save method is attached to. This means saving a `tf.keras.Model` using
  `save_weights` and loading into a `tf.train.Checkpoint` with a `Model`
  attached (or vice versa) will not match the `Model`'s variables. See the
  [guide to training
  checkpoints](https://www.tensorflow.org/alpha/guide/checkpoints) for
  details. Prefer `tf.train.Checkpoint` over `tf.keras.Model.save_weights` for
  training checkpoints.

  Attributes:
    save_counter: Incremented when `save()` is called. Used to number
      checkpoints.
  "
  [  ]
  (py/call-attr train "Checkpoint"  ))

(defn restore 
  "Restore a training checkpoint.

    Restores this `Checkpoint` and any objects it depends on.

    Either assigns values immediately if variables to restore have been created
    already, or defers restoration until the variables are created. Dependencies
    added after this call will be matched if they have a corresponding object in
    the checkpoint (the restore request will queue in any trackable object
    waiting for the expected dependency to be added).

    To ensure that loading is complete and no more assignments will take place,
    use the `assert_consumed()` method of the status object returned by
    `restore`:

    ```python
    checkpoint = tf.train.Checkpoint( ... )
    checkpoint.restore(path).assert_consumed()
    ```

    An exception will be raised if any Python objects in the dependency graph
    were not found in the checkpoint, or if any checkpointed values do not have
    a matching Python object.

    Name-based `tf.compat.v1.train.Saver` checkpoints from TensorFlow 1.x can be
    loaded
    using this method. Names are used to match variables. Re-encode name-based
    checkpoints using `tf.train.Checkpoint.save` as soon as possible.

    Args:
      save_path: The path to the checkpoint, as returned by `save` or
        `tf.train.latest_checkpoint`. If None (as when there is no latest
        checkpoint for `tf.train.latest_checkpoint` to return), returns an
        object which may run initializers for objects in the dependency graph.
        If the checkpoint was written by the name-based
        `tf.compat.v1.train.Saver`, names are used to match variables.

    Returns:
      A load status object, which can be used to make assertions about the
      status of a checkpoint restoration.

      The returned status object has the following methods:

      * `assert_consumed()`:
          Raises an exception if any variables/objects are unmatched: either
          checkpointed values which don't have a matching Python object or
          Python objects in the dependency graph with no values in the
          checkpoint. This method returns the status object, and so may be
          chained with other assertions.

      * `assert_existing_objects_matched()`:
          Raises an exception if any existing Python objects in the dependency
          graph are unmatched. Unlike `assert_consumed`, this assertion will
          pass if values in the checkpoint have no corresponding Python
          objects. For example a `tf.keras.Layer` object which has not yet been
          built, and so has not created any variables, will pass this assertion
          but fail `assert_consumed`. Useful when loading part of a larger
          checkpoint into a new Python program, e.g. a training checkpoint with
          a `tf.compat.v1.train.Optimizer` was saved but only the state required
          for
          inference is being loaded. This method returns the status object, and
          so may be chained with other assertions.

      * `assert_nontrivial_match()`: Asserts that something aside from the root
          object was matched. This is a very weak assertion, but is useful for
          sanity checking in library code where objects may exist in the
          checkpoint which haven't been created in Python and some Python
          objects may not have a checkpointed value.

      * `expect_partial()`: Silence warnings about incomplete checkpoint
          restores. Warnings are otherwise printed for unused parts of the
          checkpoint file or object when the `Checkpoint` object is deleted
          (often at program shutdown).
    "
  [ self save_path ]
  (py/call-attr self "restore"  self save_path ))

(defn save 
  "Saves a training checkpoint and provides basic checkpoint management.

    The saved checkpoint includes variables created by this object and any
    trackable objects it depends on at the time `Checkpoint.save()` is
    called.

    `save` is a basic convenience wrapper around the `write` method,
    sequentially numbering checkpoints using `save_counter` and updating the
    metadata used by `tf.train.latest_checkpoint`. More advanced checkpoint
    management, for example garbage collection and custom numbering, may be
    provided by other utilities which also wrap `write`
    (`tf.contrib.checkpoint.CheckpointManager` for example).

    Args:
      file_prefix: A prefix to use for the checkpoint filenames
        (/path/to/directory/and_a_prefix). Names are generated based on this
        prefix and `Checkpoint.save_counter`.

    Returns:
      The full path to the checkpoint.
    "
  [ self file_prefix ]
  (py/call-attr self "save"  self file_prefix ))

(defn save-counter 
  "An integer variable which starts at zero and is incremented on save.

    Used to number checkpoints.

    Returns:
      The save counter variable.
    "
  [ self ]
    (py/call-attr self "save_counter"))

(defn write 
  "Writes a training checkpoint.

    The checkpoint includes variables created by this object and any
    trackable objects it depends on at the time `Checkpoint.write()` is
    called.

    `write` does not number checkpoints, increment `save_counter`, or update the
    metadata used by `tf.train.latest_checkpoint`. It is primarily intended for
    use by higher level checkpoint management utilities. `save` provides a very
    basic implementation of these features.

    Args:
      file_prefix: A prefix to use for the checkpoint filenames
        (/path/to/directory/and_a_prefix).

    Returns:
      The full path to the checkpoint (i.e. `file_prefix`).
    "
  [ self file_prefix ]
  (py/call-attr self "write"  self file_prefix ))
