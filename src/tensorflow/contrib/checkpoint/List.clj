(ns tensorflow.contrib.checkpoint.List
  "An append-only sequence type which is trackable.

  Maintains checkpoint dependencies on its contents (which must also be
  trackable), and forwards any `Layer` metadata such as updates and losses.

  Note that `List` is purely a container. It lets a `tf.keras.Model` or
  other trackable object know about its contents, but does not call any
  `Layer` instances which are added to it. To indicate a sequence of `Layer`
  instances which should be called sequentially, use `tf.keras.Sequential`.

  Example usage:
  ```python
  class HasList(tf.keras.Model):

    def __init__(self):
      super(HasList, self).__init__()
      self.layer_list = tf.contrib.checkpoint.List([layers.Dense(3)])
      self.layer_list.append(layers.Dense(4))

    def call(self, x):
      aggregation = 0.
      for l in self.layer_list:
        x = l(x)
        aggregation += tf.reduce_sum(x)
      return aggregation
  ```

  This kind of wrapping is necessary because `Trackable` objects do not
  (yet) deeply inspect regular Python data structures, so for example assigning
  a regular list (`self.layer_list = [layers.Dense(3)]`) does not create a
  checkpoint dependency and does not add the `Layer` instance's weights to its
  parent `Model`.
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

(defn List 
  "An append-only sequence type which is trackable.

  Maintains checkpoint dependencies on its contents (which must also be
  trackable), and forwards any `Layer` metadata such as updates and losses.

  Note that `List` is purely a container. It lets a `tf.keras.Model` or
  other trackable object know about its contents, but does not call any
  `Layer` instances which are added to it. To indicate a sequence of `Layer`
  instances which should be called sequentially, use `tf.keras.Sequential`.

  Example usage:
  ```python
  class HasList(tf.keras.Model):

    def __init__(self):
      super(HasList, self).__init__()
      self.layer_list = tf.contrib.checkpoint.List([layers.Dense(3)])
      self.layer_list.append(layers.Dense(4))

    def call(self, x):
      aggregation = 0.
      for l in self.layer_list:
        x = l(x)
        aggregation += tf.reduce_sum(x)
      return aggregation
  ```

  This kind of wrapping is necessary because `Trackable` objects do not
  (yet) deeply inspect regular Python data structures, so for example assigning
  a regular list (`self.layer_list = [layers.Dense(3)]`) does not create a
  checkpoint dependency and does not add the `Layer` instance's weights to its
  parent `Model`.
  "
  [  ]
  (py/call-attr checkpoint "List"  ))

(defn append 
  "Add a new trackable value."
  [ self value ]
  (py/call-attr self "append"  self value ))

(defn copy 
  ""
  [ self  ]
  (py/call-attr self "copy"  self  ))

(defn count 
  "S.count(value) -> integer -- return number of occurrences of value"
  [ self value ]
  (py/call-attr self "count"  self value ))

(defn extend 
  "Add a sequence of trackable values."
  [ self values ]
  (py/call-attr self "extend"  self values ))

(defn index 
  "S.index(value, [start, [stop]]) -> integer -- return first index of value.
           Raises ValueError if the value is not present.

           Supporting start and stop arguments is optional, but
           recommended.
        "
  [self value & {:keys [start stop]
                       :or {stop None}} ]
    (py/call-attr-kw self "index" [value] {:start start :stop stop }))

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
