(ns tensorflow.python.keras.api.-v1.keras.experimental.LinearCosineDecay
  "A LearningRateSchedule that uses a linear cosine decay schedule."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow.python.keras.api._v1.keras.experimental"))

(defn LinearCosineDecay 
  "A LearningRateSchedule that uses a linear cosine decay schedule."
  [initial_learning_rate decay_steps & {:keys [num_periods alpha beta name]
                       :or {name None}} ]
    (py/call-attr-kw experimental "LinearCosineDecay" [initial_learning_rate decay_steps] {:num_periods num_periods :alpha alpha :beta beta :name name }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
