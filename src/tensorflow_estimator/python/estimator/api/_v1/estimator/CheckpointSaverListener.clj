(ns tensorflow-estimator.python.estimator.api.-v1.estimator.CheckpointSaverListener
  "Interface for listeners that take action before or after checkpoint save.

  `CheckpointSaverListener` triggers only in steps when `CheckpointSaverHook` is
  triggered, and provides callbacks at the following points:
   - before using the session
   - before each call to `Saver.save()`
   - after each call to `Saver.save()`
   - at the end of session

  To use a listener, implement a class and pass the listener to a
  `CheckpointSaverHook`, as in this example:

  ```python
  class ExampleCheckpointSaverListener(CheckpointSaverListener):
    def begin(self):
      # You can add ops to the graph here.
      print('Starting the session.')
      self.your_tensor = ...

    def before_save(self, session, global_step_value):
      print('About to write a checkpoint')

    def after_save(self, session, global_step_value):
      print('Done writing checkpoint.')
      if decided_to_stop_training():
        return True

    def end(self, session, global_step_value):
      print('Done with the session.')

  ...
  listener = ExampleCheckpointSaverListener()
  saver_hook = tf.estimator.CheckpointSaverHook(
      checkpoint_dir, listeners=[listener])
  with
  tf.compat.v1.train.MonitoredTrainingSession(chief_only_hooks=[saver_hook]):
    ...
  ```

  A `CheckpointSaverListener` may simply take some action after every
  checkpoint save. It is also possible for the listener to use its own schedule
  to act less frequently, e.g. based on global_step_value. In this case,
  implementors should implement the `end()` method to handle actions related to
  the last checkpoint save. But the listener should not act twice if
  `after_save()` already handled this last checkpoint save.

  A `CheckpointSaverListener` can request training to be stopped, by returning
  True in `after_save`. Please note that, in replicated distributed training
  setting, only `chief` should use this behavior. Otherwise each worker will do
  their own evaluation, which may be wasteful of resources.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce estimator (import-module "tensorflow_estimator.python.estimator.api._v1.estimator"))

(defn CheckpointSaverListener 
  "Interface for listeners that take action before or after checkpoint save.

  `CheckpointSaverListener` triggers only in steps when `CheckpointSaverHook` is
  triggered, and provides callbacks at the following points:
   - before using the session
   - before each call to `Saver.save()`
   - after each call to `Saver.save()`
   - at the end of session

  To use a listener, implement a class and pass the listener to a
  `CheckpointSaverHook`, as in this example:

  ```python
  class ExampleCheckpointSaverListener(CheckpointSaverListener):
    def begin(self):
      # You can add ops to the graph here.
      print('Starting the session.')
      self.your_tensor = ...

    def before_save(self, session, global_step_value):
      print('About to write a checkpoint')

    def after_save(self, session, global_step_value):
      print('Done writing checkpoint.')
      if decided_to_stop_training():
        return True

    def end(self, session, global_step_value):
      print('Done with the session.')

  ...
  listener = ExampleCheckpointSaverListener()
  saver_hook = tf.estimator.CheckpointSaverHook(
      checkpoint_dir, listeners=[listener])
  with
  tf.compat.v1.train.MonitoredTrainingSession(chief_only_hooks=[saver_hook]):
    ...
  ```

  A `CheckpointSaverListener` may simply take some action after every
  checkpoint save. It is also possible for the listener to use its own schedule
  to act less frequently, e.g. based on global_step_value. In this case,
  implementors should implement the `end()` method to handle actions related to
  the last checkpoint save. But the listener should not act twice if
  `after_save()` already handled this last checkpoint save.

  A `CheckpointSaverListener` can request training to be stopped, by returning
  True in `after_save`. Please note that, in replicated distributed training
  setting, only `chief` should use this behavior. Otherwise each worker will do
  their own evaluation, which may be wasteful of resources.
  "
  [  ]
  (py/call-attr estimator "CheckpointSaverListener"  ))

(defn after-save 
  ""
  [ self session global_step_value ]
  (py/call-attr self "after_save"  self session global_step_value ))

(defn before-save 
  ""
  [ self session global_step_value ]
  (py/call-attr self "before_save"  self session global_step_value ))

(defn begin 
  ""
  [ self  ]
  (py/call-attr self "begin"  self  ))

(defn end 
  ""
  [ self session global_step_value ]
  (py/call-attr self "end"  self session global_step_value ))
