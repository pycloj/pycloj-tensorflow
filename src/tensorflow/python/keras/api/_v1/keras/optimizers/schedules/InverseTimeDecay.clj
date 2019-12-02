(ns tensorflow.python.keras.api.-v1.keras.optimizers.schedules.InverseTimeDecay
  "A LearningRateSchedule that uses an inverse time decay schedule."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce schedules (import-module "tensorflow.python.keras.api._v1.keras.optimizers.schedules"))

(defn InverseTimeDecay 
  "A LearningRateSchedule that uses an inverse time decay schedule."
  [initial_learning_rate decay_steps decay_rate & {:keys [staircase name]
                       :or {name None}} ]
    (py/call-attr-kw schedules "InverseTimeDecay" [initial_learning_rate decay_steps decay_rate] {:staircase staircase :name name }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
