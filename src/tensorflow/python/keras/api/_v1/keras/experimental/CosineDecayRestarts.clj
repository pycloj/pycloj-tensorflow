(ns tensorflow.python.keras.api.-v1.keras.experimental.CosineDecayRestarts
  "A LearningRateSchedule that uses a cosine decay schedule with restarts."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow.python.keras.api._v1.keras.experimental"))

(defn CosineDecayRestarts 
  "A LearningRateSchedule that uses a cosine decay schedule with restarts."
  [initial_learning_rate first_decay_steps & {:keys [t_mul m_mul alpha name]
                       :or {name None}} ]
    (py/call-attr-kw experimental "CosineDecayRestarts" [initial_learning_rate first_decay_steps] {:t_mul t_mul :m_mul m_mul :alpha alpha :name name }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
