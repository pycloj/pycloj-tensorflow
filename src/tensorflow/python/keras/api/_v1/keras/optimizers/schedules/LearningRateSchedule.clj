(ns tensorflow.python.keras.api.-v1.keras.optimizers.schedules.LearningRateSchedule
  "A serializable learning rate decay schedule.

  `LearningRateSchedule`s can be passed in as the learning rate of optimizers in
  `tf.keras.optimizers`. They can be serialized and deserialized using
  `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.
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

(defn LearningRateSchedule 
  "A serializable learning rate decay schedule.

  `LearningRateSchedule`s can be passed in as the learning rate of optimizers in
  `tf.keras.optimizers`. They can be serialized and deserialized using
  `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.
  "
  [  ]
  (py/call-attr schedules "LearningRateSchedule"  ))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
