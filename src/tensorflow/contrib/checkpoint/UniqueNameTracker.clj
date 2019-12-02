(ns tensorflow.contrib.checkpoint.UniqueNameTracker
  "Adds dependencies on trackable objects with name hints.

  Useful for creating dependencies with locally unique names.

  Example usage:
  ```python
  class SlotManager(tf.contrib.checkpoint.Checkpointable):

    def __init__(self):
      # Create a dependency named \"slotdeps\" on the container.
      self.slotdeps = tf.contrib.checkpoint.UniqueNameTracker()
      slotdeps = self.slotdeps
      slots = []
      slots.append(slotdeps.track(tf.Variable(3.), \"x\"))  # Named \"x\"
      slots.append(slotdeps.track(tf.Variable(4.), \"y\"))
      slots.append(slotdeps.track(tf.Variable(5.), \"x\"))  # Named \"x_1\"
  ```
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce checkpoint (import-module "tensorflow.contrib.checkpoint"))

(defn UniqueNameTracker 
  "Adds dependencies on trackable objects with name hints.

  Useful for creating dependencies with locally unique names.

  Example usage:
  ```python
  class SlotManager(tf.contrib.checkpoint.Checkpointable):

    def __init__(self):
      # Create a dependency named \"slotdeps\" on the container.
      self.slotdeps = tf.contrib.checkpoint.UniqueNameTracker()
      slotdeps = self.slotdeps
      slots = []
      slots.append(slotdeps.track(tf.Variable(3.), \"x\"))  # Named \"x\"
      slots.append(slotdeps.track(tf.Variable(4.), \"y\"))
      slots.append(slotdeps.track(tf.Variable(5.), \"x\"))  # Named \"x_1\"
  ```
  "
  [  ]
  (py/call-attr checkpoint "UniqueNameTracker"  ))

(defn layers 
  ""
  [ self ]
    (py/call-attr self "layers"))

(defn losses 
  "Aggregate losses from any `Layer` instances."
  [ self ]
    (py/call-attr self "losses"))

(defn non-trainable-variables 
  ""
  [ self ]
    (py/call-attr self "non_trainable_variables"))

(defn non-trainable-weights 
  ""
  [ self ]
    (py/call-attr self "non_trainable_weights"))

(defn track 
  "Add a dependency on `trackable`.

    Args:
      trackable: An object to add a checkpoint dependency on.
      base_name: A name hint, which is uniquified to determine the dependency
        name.
    Returns:
      `trackable`, for chaining.
    Raises:
      ValueError: If `trackable` is not a trackable object.
    "
  [ self trackable base_name ]
  (py/call-attr self "track"  self trackable base_name ))

(defn trainable 
  ""
  [ self ]
    (py/call-attr self "trainable"))

(defn trainable-variables 
  ""
  [ self ]
    (py/call-attr self "trainable_variables"))

(defn trainable-weights 
  ""
  [ self ]
    (py/call-attr self "trainable_weights"))

(defn updates 
  "Aggregate updates from any `Layer` instances."
  [ self ]
    (py/call-attr self "updates"))

(defn variables 
  ""
  [ self ]
    (py/call-attr self "variables"))

(defn weights 
  ""
  [ self ]
    (py/call-attr self "weights"))
