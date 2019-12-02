(ns tensorflow.python.keras.api.-v1.keras.optimizers
  "Built-in optimizer classes.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce optimizers (import-module "tensorflow.python.keras.api._v1.keras.optimizers"))

(defn deserialize 
  "Inverse of the `serialize` function.

  Arguments:
      config: Optimizer configuration dictionary.
      custom_objects: Optional dictionary mapping names (strings) to custom
        objects (classes and functions) to be considered during deserialization.

  Returns:
      A Keras Optimizer instance.
  "
  [ config custom_objects ]
  (py/call-attr optimizers "deserialize"  config custom_objects ))

(defn get 
  "Retrieves a Keras Optimizer instance.

  Arguments:
      identifier: Optimizer identifier, one of
          - String: name of an optimizer
          - Dictionary: configuration dictionary. - Keras Optimizer instance (it
            will be returned unchanged). - TensorFlow Optimizer instance (it
            will be wrapped as a Keras Optimizer).

  Returns:
      A Keras Optimizer instance.

  Raises:
      ValueError: If `identifier` cannot be interpreted.
  "
  [ identifier ]
  (py/call-attr optimizers "get"  identifier ))

(defn serialize 
  ""
  [ optimizer ]
  (py/call-attr optimizers "serialize"  optimizer ))
