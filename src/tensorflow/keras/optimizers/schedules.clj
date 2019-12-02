(ns tensorflow.python.keras.api.-v1.keras.optimizers.schedules
  "Public API for tf.keras.optimizers.schedules namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce schedules (import-module "tensorflow.python.keras.api._v1.keras.optimizers.schedules"))

(defn deserialize 
  ""
  [ config custom_objects ]
  (py/call-attr schedules "deserialize"  config custom_objects ))

(defn serialize 
  ""
  [ learning_rate_schedule ]
  (py/call-attr schedules "serialize"  learning_rate_schedule ))
