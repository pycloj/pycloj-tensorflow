(ns tensorflow.python.keras.api.-v1.keras.experimental.CosineDecay
  "A LearningRateSchedule that uses a cosine decay schedule."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow.python.keras.api._v1.keras.experimental"))

(defn CosineDecay 
  "A LearningRateSchedule that uses a cosine decay schedule."
  [initial_learning_rate decay_steps & {:keys [alpha name]
                       :or {name None}} ]
    (py/call-attr-kw experimental "CosineDecay" [initial_learning_rate decay_steps] {:alpha alpha :name name }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
