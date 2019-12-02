(ns tensorflow.python.keras.api.-v1.keras.optimizers.schedules.PolynomialDecay
  "A LearningRateSchedule that uses a polynomial decay schedule."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce schedules (import-module "tensorflow.python.keras.api._v1.keras.optimizers.schedules"))

(defn PolynomialDecay 
  "A LearningRateSchedule that uses a polynomial decay schedule."
  [initial_learning_rate decay_steps & {:keys [end_learning_rate power cycle name]
                       :or {name None}} ]
    (py/call-attr-kw schedules "PolynomialDecay" [initial_learning_rate decay_steps] {:end_learning_rate end_learning_rate :power power :cycle cycle :name name }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
