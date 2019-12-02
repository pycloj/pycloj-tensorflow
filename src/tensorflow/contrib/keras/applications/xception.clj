(ns tensorflow.contrib.keras.api.keras.applications.xception
  "Xception Keras application."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce xception (import-module "tensorflow.contrib.keras.api.keras.applications.xception"))

(defn Xception 
  ""
  [  ]
  (py/call-attr xception "Xception"  ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr xception "decode_predictions"  ))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr xception "preprocess_input"  ))
