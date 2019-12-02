(ns tensorflow.python.keras.api.-v1.keras.applications.xception
  "Xception V1 model for Keras.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce xception (import-module "tensorflow.python.keras.api._v1.keras.applications.xception"))

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
