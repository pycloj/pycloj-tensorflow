(ns tensorflow.contrib.keras.api.keras.applications.mobilenet
  "MobileNet Keras application."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce mobilenet (import-module "tensorflow.contrib.keras.api.keras.applications.mobilenet"))

(defn MobileNet 
  ""
  [  ]
  (py/call-attr mobilenet "MobileNet"  ))

(defn decode-predictions 
  ""
  [  ]
  (py/call-attr mobilenet "decode_predictions"  ))

(defn preprocess-input 
  ""
  [  ]
  (py/call-attr mobilenet "preprocess_input"  ))
