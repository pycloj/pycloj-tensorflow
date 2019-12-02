(ns tensorflow.python.keras.api.-v1.keras.experimental.NoisyLinearCosineDecay
  "A LearningRateSchedule that uses a noisy linear cosine decay schedule."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow.python.keras.api._v1.keras.experimental"))

(defn NoisyLinearCosineDecay 
  "A LearningRateSchedule that uses a noisy linear cosine decay schedule."
  [initial_learning_rate decay_steps & {:keys [initial_variance variance_decay num_periods alpha beta name]
                       :or {name None}} ]
    (py/call-attr-kw experimental "NoisyLinearCosineDecay" [initial_learning_rate decay_steps] {:initial_variance initial_variance :variance_decay variance_decay :num_periods num_periods :alpha alpha :beta beta :name name }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
