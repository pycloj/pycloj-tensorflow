(ns tensorflow.python.keras.api.-v1.keras.optimizers.schedules.PiecewiseConstantDecay
  "A LearningRateSchedule that uses a piecewise constant decay schedule."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce schedules (import-module "tensorflow.python.keras.api._v1.keras.optimizers.schedules"))

(defn PiecewiseConstantDecay 
  "A LearningRateSchedule that uses a piecewise constant decay schedule."
  [ boundaries values name ]
  (py/call-attr schedules "PiecewiseConstantDecay"  boundaries values name ))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
